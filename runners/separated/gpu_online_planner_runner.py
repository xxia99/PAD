import json
import os
import time
from pathlib import Path

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from algorithms.happo_policy import HAPPO_Policy
from algorithms.happo_trainer_gpu import HAPPOGPU
from envs.env_utils import data_configs
from envs.gpu_online_planner import GPUOnlinePlannerEnv
from envs.om_env_wrappers import MultiDiscrete
from runners.separated.gpu_detector_surrogate import GPUDetectorResponseSurrogate
from runners.separated.gpu_pad_evaluator import simulate_planner_lengths_gpu
from runners.separated.parallel_detectors import ParallelDetectorExecutor
from utils.gpu_separated_buffer import GpuSeparatedReplayBuffer
from utils.swanlab_compat import swanlab


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu().item())
    return float(value)


def _parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    return [float(value)]


class GPUOnlinePlannerRunner:
    """Single-process, GPU-vectorized Stage II planner training runner."""

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = Path(config["run_dir"])
        self.checkpoint_dir = Path(config.get("checkpoint_dir") or self.run_dir)

        self.num_regions = int(data_configs.agent_num)
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.episode_length = int(self.all_args.episode_length)
        self.hidden_size = int(self.all_args.hidden_size)
        self.recurrent_N = int(self.all_args.recurrent_N)
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_eval = bool(self.all_args.use_eval)
        self.use_linear_lr_decay = bool(self.all_args.use_linear_lr_decay)
        self.log_interval = int(self.all_args.log_interval)
        self.save_interval = int(self.all_args.save_interval)
        self.planner_eval_interval = int(getattr(self.all_args, "planner_eval_interval", 100))
        self.planner_eval_episodes = int(getattr(self.all_args, "planner_eval_episodes", 2000))
        self.planner_eval_horizon = int(getattr(self.all_args, "planner_eval_horizon", 2000))
        self.planner_eval_delta = float(getattr(self.all_args, "planner_eval_delta", 1.0))
        self.planner_eval_start_index = int(getattr(self.all_args, "planner_eval_start_index", 0))
        self.planner_eval_random_broken_streams = bool(
            getattr(self.all_args, "planner_eval_random_broken_streams", False)
        )
        self.planner_select_eval_interval = int(getattr(self.all_args, "planner_select_eval_interval", 0) or 0)
        self.planner_select_eval_episodes = int(
            getattr(self.all_args, "planner_select_eval_episodes", self.planner_eval_episodes)
            or self.planner_eval_episodes
        )
        self.planner_select_eval_horizon = int(
            getattr(self.all_args, "planner_select_eval_horizon", self.planner_eval_horizon)
            or self.planner_eval_horizon
        )
        self.planner_select_eval_start_index = int(
            getattr(self.all_args, "planner_select_eval_start_index", self.planner_eval_start_index)
        )
        self.alarm_threshold = float(getattr(self.all_args, "planner_alarm_threshold", 50.0))
        self.alpha_bins = int(getattr(self.all_args, "planner_alpha_bins", 11))
        self.train_t_break = int(getattr(self.all_args, "gpu_stage2_train_t_break", 50))
        self.train_stop_active_fraction = float(
            getattr(self.all_args, "gpu_stage2_train_stop_active_fraction", 0.0) or 0.0
        )
        self.surrogate_pretrain_updates = int(getattr(self.all_args, "gpu_stage2_surrogate_pretrain_updates", 0) or 0)
        self.mixed_surrogate = bool(getattr(self.all_args, "gpu_stage2_mixed_surrogate", False))
        self.cf_alpha_baseline = bool(getattr(self.all_args, "gpu_stage2_cf_alpha_baseline", False))
        self.use_surrogate = bool(getattr(self.all_args, "gpu_stage2_use_surrogate", False)) or self.mixed_surrogate or (
            self.surrogate_pretrain_updates > 0
        )
        self.build_detector_surrogate = self.use_surrogate or self.cf_alpha_baseline
        self.current_train_uses_surrogate = False
        self.current_surrogate_fraction = 0.0
        self.cf_lambda = float(getattr(self.all_args, "gpu_stage2_cf_lambda", 0.0) or 0.0)
        self.cf_horizon = int(getattr(self.all_args, "gpu_stage2_cf_horizon", 30) or 30)
        self.cf_stride = max(1, int(getattr(self.all_args, "gpu_stage2_cf_stride", 5) or 5))
        self.cf_max_states = max(0, int(getattr(self.all_args, "gpu_stage2_cf_max_states", 0) or 0))
        self.cf_chunk_size = max(1, int(getattr(self.all_args, "gpu_stage2_cf_chunk_size", 65536) or 65536))
        self.cf_alpha_candidates = torch.as_tensor(
            _parse_float_list(getattr(self.all_args, "gpu_stage2_cf_alpha_candidates", "0,0.25,0.5,0.75,1.0")),
            device=self.device,
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)
        if self.cf_alpha_candidates.numel() == 0:
            self.cf_alpha_candidates = torch.as_tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=self.device)
        self.stat_progress_reward = bool(getattr(self.all_args, "gpu_stage2_stat_progress_reward", False))
        self.stat_progress_beta_start = float(
            getattr(self.all_args, "gpu_stage2_stat_progress_beta_start", 5.0) or 0.0
        )
        self.stat_progress_beta_end = float(getattr(self.all_args, "gpu_stage2_stat_progress_beta_end", 0.0) or 0.0)
        self.stat_progress_anneal_updates = max(
            1,
            int(getattr(self.all_args, "gpu_stage2_stat_progress_anneal_updates", 100) or 100),
        )
        self.stat_progress_clip = max(0.0, float(getattr(self.all_args, "gpu_stage2_stat_progress_clip", 0.05) or 0.0))
        self.current_stat_progress_beta = 0.0
        self.best_eval_running_length = float("inf")
        self.target_minibatch_size = int(getattr(self.all_args, "gpu_stage2_minibatch_size", 0) or 0)

        self.batch_size = int(self.all_args.gpu_stage2_batch_size or self.all_args.n_rollout_threads)
        self.all_args.n_rollout_threads = self.batch_size
        self.n_rollout_threads = self.batch_size
        self.all_args.use_policy_active_masks = True
        self.all_args.use_value_active_masks = True

        if self.target_minibatch_size > 0:
            samples_per_update = self.episode_length * self.batch_size
            target_batches = int(np.ceil(samples_per_update / float(self.target_minibatch_size)))
            self.all_args.num_mini_batch = max(int(self.all_args.num_mini_batch), target_batches)
        if self.all_args.use_naive_recurrent_policy:
            self.all_args.num_mini_batch = min(int(self.all_args.num_mini_batch), self.batch_size)

        self.envs = GPUOnlinePlannerEnv(self.all_args, self.device, batch_size=self.batch_size)
        self.log_dir = self.run_dir / "logs"
        self.save_dir = self.checkpoint_dir / "models"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writter = SummaryWriter(str(self.log_dir))

        self.detector_policies = self._build_and_load_detectors()
        self.parallel_detector_executor = ParallelDetectorExecutor(
            self.detector_policies,
            self.device,
            disabled=bool(getattr(self.all_args, "disable_parallel_detectors", False)),
            backend=getattr(self.all_args, "parallel_detector_backend", "fused"),
            strict_parity=bool(getattr(self.all_args, "fused_detector_strict_parity", True)),
            fused_disable_tf32=bool(getattr(self.all_args, "fused_detector_disable_tf32", True)),
        )
        print(f"Parallel detector executor: {self.parallel_detector_executor.reason}")
        self.detector_surrogate = (
            GPUDetectorResponseSurrogate(self.all_args, self.device, self.detector_policies)
            if self.build_detector_surrogate
            else None
        )
        self.detector_rnn_states = torch.zeros(
            (self.batch_size, self.num_regions, self.recurrent_N, self.hidden_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.detector_masks = torch.ones((self.batch_size, self.num_regions, 1), device=self.device, dtype=torch.float32)

        planner_obs_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(2 * self.num_regions,),
            dtype=np.float32,
        )
        planner_act_space = MultiDiscrete([[0, 1]] * self.alpha_bins)
        self.policy = HAPPO_Policy(
            self.all_args,
            planner_obs_space,
            planner_obs_space,
            planner_act_space,
            device=self.device,
        )
        self.trainer = HAPPOGPU(self.all_args, self.policy, device=self.device, action_budget=self.alpha_bins)
        self._maybe_load_planner()
        self.buffer = GpuSeparatedReplayBuffer(
            self.all_args,
            planner_obs_space,
            planner_obs_space,
            planner_act_space,
            self.device,
        )

        print(
            "GPU Stage II config:",
            {
                "batch_size": self.batch_size,
                "episode_length": self.episode_length,
                "num_mini_batch": int(self.all_args.num_mini_batch),
                "ppo_epoch": int(self.all_args.ppo_epoch),
                "device": str(self.device),
                "alpha_bins": self.alpha_bins,
                "threshold": self.alarm_threshold,
                "region_aggregator": getattr(self.all_args, "planner_region_aggregator", "max"),
                "region_topk": int(getattr(self.all_args, "planner_region_topk", 3) or 3),
                "train_t_break": self.train_t_break,
                "train_stop_active_fraction": self.train_stop_active_fraction,
                "train_alarm_stat": "top_l_global_qcusum_after_update",
                "planner_reward": "-1_per_active_step",
                "use_policy_active_masks": bool(self.all_args.use_policy_active_masks),
                "use_value_active_masks": bool(self.all_args.use_value_active_masks),
                "train_broken_streams": (
                    "random"
                    if bool(getattr(self.all_args, "gpu_stage2_random_broken_streams", False))
                    else "stratified_cycle"
                ),
                "reset_planner_rnn_each_episode": bool(
                    getattr(self.all_args, "gpu_stage2_reset_rnn_each_episode", False)
                ),
                "eval_start_index": self.planner_eval_start_index,
                "eval_random_broken_streams": self.planner_eval_random_broken_streams,
                "select_eval_interval": self.planner_select_eval_interval,
                "select_eval_episodes": self.planner_select_eval_episodes,
                "surrogate_enabled": self.use_surrogate,
                "surrogate_table_enabled": self.build_detector_surrogate,
                "surrogate_pretrain_updates": self.surrogate_pretrain_updates,
                "surrogate_stat_bins": int(getattr(self.all_args, "gpu_stage2_surrogate_stat_bins", 1) or 1),
                "mixed_surrogate": self.mixed_surrogate,
                "mix_rho": [
                    float(getattr(self.all_args, "gpu_stage2_mix_rho_start", 0.7)),
                    float(getattr(self.all_args, "gpu_stage2_mix_rho_mid", 0.2)),
                    float(getattr(self.all_args, "gpu_stage2_mix_rho_end", 0.0)),
                ],
                "stat_progress_reward": self.stat_progress_reward,
                "stat_progress_beta": [self.stat_progress_beta_start, self.stat_progress_beta_end],
                "stat_progress_anneal_updates": self.stat_progress_anneal_updates,
                "stat_progress_clip": self.stat_progress_clip,
                "cf_alpha_baseline": self.cf_alpha_baseline,
                "cf_lambda": self.cf_lambda,
                "cf_horizon": self.cf_horizon,
                "cf_stride": self.cf_stride,
                "cf_max_states": self.cf_max_states,
                "cf_alpha_candidates": [float(v) for v in self.cf_alpha_candidates.detach().cpu().tolist()],
            },
        )

    @torch.no_grad()
    def _detector_action_list(self, detector_obs, allocations, real_rows=None):
        parallel = self.parallel_detector_executor.act(
            detector_obs,
            self.detector_rnn_states,
            self.detector_masks,
            allocations,
            deterministic=True,
            force_m=True,
        )
        if parallel is not None:
            actions_tensor, next_rnn = parallel
            if real_rows is None:
                self.detector_rnn_states.copy_(next_rnn)
            elif bool(real_rows.any()):
                self.detector_rnn_states[real_rows] = next_rnn[real_rows]
            return [actions_tensor[:, region_id].long() for region_id in range(self.num_regions)]

        detector_actions = []
        for region_id, detector_policy in enumerate(self.detector_policies):
            action, _, next_rnn = detector_policy.actor(
                detector_obs[:, region_id],
                self.detector_rnn_states[:, region_id],
                self.detector_masks[:, region_id],
                None,
                deterministic=True,
                force_m=True,
                m=allocations[:, region_id],
            )
            if real_rows is None:
                self.detector_rnn_states[:, region_id] = next_rnn
            elif bool(real_rows.any()):
                self.detector_rnn_states[real_rows, region_id] = next_rnn[real_rows]
            detector_actions.append(action.long())
        return detector_actions

    def _build_and_load_detectors(self):
        stage1_dir = self.all_args.stage1_model_dir
        if stage1_dir is None:
            stage1_dir = f"checkpoints/detectors/M={self.all_args.M_total}_N={self.all_args.agent_num}"
        stage1_dir = Path(stage1_dir)

        policies = []
        for agent_id in range(self.num_regions):
            share_space = (
                self.envs.share_observation_space[agent_id]
                if self.all_args.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            policy = HAPPO_Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            actor_path = self._detector_actor_path(stage1_dir, agent_id)
            actor_sd = torch.load(str(actor_path), map_location=self.device)
            policy.actor.load_state_dict(actor_sd)
            policy.actor.eval()
            policies.append(policy)
            print(f"Loaded detector actor {agent_id}: {actor_path}")
        return policies

    @staticmethod
    def _detector_actor_path(stage1_dir, agent_id):
        candidates = [
            stage1_dir / f"actor_agent{agent_id}.pt",
            stage1_dir / f"actor_agent{agent_id}_latest.pt",
            stage1_dir / "models" / f"actor_agent{agent_id}.pt",
            stage1_dir / "models" / f"actor_agent{agent_id}_latest.pt",
            stage1_dir / "run_seed_2" / "models" / f"actor_agent{agent_id}.pt",
            stage1_dir / "run_seed_2" / "models" / f"actor_agent{agent_id}_latest.pt",
            stage1_dir / "run_seed_1" / "models" / f"actor_agent{agent_id}.pt",
            stage1_dir / "run_seed_1" / "models" / f"actor_agent{agent_id}_latest.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find actor_agent{agent_id}.pt under {stage1_dir}. "
            f"Checked: {[str(p) for p in candidates]}"
        )

    def _maybe_load_planner(self):
        model_dir = getattr(self.all_args, "model_dir", None)
        planner_model_dir = getattr(self.all_args, "planner_model_dir", None)
        load_dir = model_dir or planner_model_dir
        if not load_dir:
            return

        load_dir = Path(load_dir)
        actor_name = getattr(self.all_args, "planner_model_name", "planner_best_actor.pt")
        actor_candidates = [
            load_dir / actor_name,
            load_dir / "planner_best_actor.pt",
            load_dir / "planner_actor_latest.pt",
            load_dir / "planner_actor.pt",
        ]
        actor_path = next((path for path in actor_candidates if path.exists()), None)
        if actor_path is None:
            raise FileNotFoundError(f"Could not find planner actor under {load_dir}.")
        self.policy.actor.load_state_dict(torch.load(str(actor_path), map_location=self.device))
        print(f"Loaded planner actor: {actor_path}")

        critic_candidates = [
            load_dir / actor_path.name.replace("actor", "critic"),
            load_dir / "planner_best_critic.pt",
            load_dir / "planner_critic_latest.pt",
            load_dir / "planner_critic.pt",
        ]
        critic_path = next((path for path in critic_candidates if path.exists()), None)
        if critic_path is not None:
            self.policy.critic.load_state_dict(torch.load(str(critic_path), map_location=self.device))
            print(f"Loaded planner critic: {critic_path}")

    def run(self):
        start = time.time()
        episodes = max(1, self.num_env_steps // self.episode_length // self.n_rollout_threads)

        if self.use_eval:
            pre_eval_mean, pre_eval_stderr = self.evaluate()
            print(f"[GPU PlannerEval] pre-train mean_rl={pre_eval_mean:.4f}, stderr={pre_eval_stderr:.4f}")
            self.save_eval_checkpoint("eval", 0, pre_eval_mean, pre_eval_stderr)
            if self.planner_select_eval_interval > 0:
                select_mean, select_stderr = self.evaluate(
                    num_episodes=self.planner_select_eval_episodes,
                    horizon=self.planner_select_eval_horizon,
                    start_index=self.planner_select_eval_start_index,
                    log_prefix="GPUPlannerSelectEval",
                    print_name="GPU Stage II select eval",
                )
                print(f"[GPU PlannerSelectEval] pre-train mean_rl={select_mean:.4f}, stderr={select_stderr:.4f}")
                self.save_eval_checkpoint("select_eval", 0, select_mean, select_stderr)
                if select_mean <= self.best_eval_running_length:
                    self.best_eval_running_length = select_mean
                    self.save_best()
            elif pre_eval_mean <= self.best_eval_running_length:
                self.best_eval_running_length = pre_eval_mean
                self.save_best()

        for episode in range(episodes):
            if self.use_eval and episode > 0 and episode % self.planner_eval_interval == 0:
                eval_mean, eval_stderr = self.evaluate()
                print(f"[GPU PlannerEval] episode={episode}, mean_rl={eval_mean:.4f}, stderr={eval_stderr:.4f}")
                self.save_eval_checkpoint("eval", episode, eval_mean, eval_stderr)
                if self.planner_select_eval_interval <= 0 and eval_mean <= self.best_eval_running_length:
                    self.best_eval_running_length = eval_mean
                    self.save_best()
            if (
                self.use_eval
                and self.planner_select_eval_interval > 0
                and episode > 0
                and episode % self.planner_select_eval_interval == 0
            ):
                select_mean, select_stderr = self.evaluate(
                    num_episodes=self.planner_select_eval_episodes,
                    horizon=self.planner_select_eval_horizon,
                    start_index=self.planner_select_eval_start_index,
                    log_prefix="GPUPlannerSelectEval",
                    print_name="GPU Stage II select eval",
                )
                print(f"[GPU PlannerSelectEval] episode={episode}, mean_rl={select_mean:.4f}, stderr={select_stderr:.4f}")
                self.save_eval_checkpoint("select_eval", episode, select_mean, select_stderr)
                if select_mean <= self.best_eval_running_length:
                    self.best_eval_running_length = select_mean
                    self.save_best()

            self.warmup(episode)
            self.current_train_uses_surrogate = self._episode_uses_surrogate(episode)
            self.current_stat_progress_beta = self._stat_progress_beta(episode)
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            episode_reward_total = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
            episode_base_reward_total = torch.zeros_like(episode_reward_total)
            episode_stat_progress_total = torch.zeros_like(episode_reward_total)
            self.rollout_active = torch.ones((self.batch_size,), device=self.device, dtype=torch.bool)
            self.rollout_alarmed = torch.zeros((self.batch_size,), device=self.device, dtype=torch.bool)
            self.rollout_detection_delay = torch.full(
                (self.batch_size,),
                max(self.episode_length - self.train_t_break, 0),
                device=self.device,
                dtype=torch.float32,
            )
            self.rollout_alarm_step = torch.full(
                (self.batch_size,),
                -1.0,
                device=self.device,
                dtype=torch.float32,
            )
            self.rollout_last_stat = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)

            rollout_steps = self.episode_length
            tail_cutoff = False
            for step in range(self.episode_length):
                value, action, action_log_prob, rnn_state, rnn_state_critic = self.collect(step)
                rewards, dones, next_planner_obs = self.env_step(action, step)
                episode_reward_total += rewards[:, 0]
                episode_base_reward_total += self.last_base_rewards[:, 0]
                episode_stat_progress_total += self.last_stat_progress_rewards[:, 0]

                masks = torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)
                masks[~self.rollout_active] = 0.0
                active_masks = self.rollout_active.float().reshape(-1, 1)

                self.buffer.insert(
                    next_planner_obs,
                    next_planner_obs,
                    rnn_state,
                    rnn_state_critic,
                    action,
                    action_log_prob,
                    value,
                    rewards,
                    masks,
                    active_masks=active_masks,
                )
                active_fraction = _scalar(self.rollout_active.float().mean())
                if (not bool(self.rollout_active.any())) or (
                    self.train_stop_active_fraction > 0.0
                    and step >= self.train_t_break
                    and active_fraction <= self.train_stop_active_fraction
                ):
                    rollout_steps = step + 1
                    tail_cutoff = bool(self.rollout_active.any())
                    break

            self.buffer.set_active_length(rollout_steps)
            tail_info = self._apply_tail_cutoff_penalty(
                rollout_steps,
                tail_cutoff,
                episode_reward_total,
                episode_base_reward_total,
            )
            cf_info = self.compute_counterfactual_advantages(rollout_steps)
            self.compute()
            train_info = self.train()
            train_info.update(cf_info)
            train_info.update(tail_info)
            avg_episode_return = _scalar(episode_reward_total.mean())
            train_info["average_episode_return"] = avg_episode_return
            train_info["average_base_episode_return"] = _scalar(episode_base_reward_total.mean())
            train_info["average_stat_progress_return"] = _scalar(episode_stat_progress_total.mean())
            train_info["stat_progress_beta"] = float(self.current_stat_progress_beta)
            rewards_for_loss = self.buffer.rewards[:rollout_steps]
            active_for_loss = self.buffer.active_masks[:rollout_steps]
            active_denom = active_for_loss.sum().clamp_min(1.0)
            train_info["average_step_rewards"] = _scalar((rewards_for_loss * active_for_loss).sum() / active_denom)
            train_info["raw_average_step_rewards"] = _scalar(rewards_for_loss.mean())
            train_info["active_transition_fraction"] = _scalar(active_for_loss.mean())
            if rollout_steps > 0:
                train_info["average_base_step_rewards"] = _scalar(episode_base_reward_total.mean() / float(rollout_steps))
                train_info["average_stat_progress_step_rewards"] = _scalar(
                    episode_stat_progress_total.mean() / float(rollout_steps)
                )
            train_info["rollout_steps"] = float(rollout_steps)
            train_info["rollout_fraction"] = float(rollout_steps) / float(self.episode_length)
            train_info["rollout_stop_active_fraction"] = _scalar(self.rollout_active.float().mean())
            train_info["average_detection_delay"] = _scalar(self.rollout_detection_delay.mean())
            if bool(self.rollout_alarmed.any()):
                train_info["detected_detection_delay"] = _scalar(
                    self.rollout_detection_delay[self.rollout_alarmed].mean()
                )
                train_info["average_alarm_step"] = _scalar(self.rollout_alarm_step[self.rollout_alarmed].mean())
            else:
                train_info["detected_detection_delay"] = float(max(self.episode_length - self.train_t_break, 0))
                train_info["average_alarm_step"] = -1.0
            train_info["alarm_rate"] = _scalar(self.rollout_alarmed.float().mean())
            train_info["censored_rate"] = _scalar((~self.rollout_alarmed).float().mean())
            train_info["average_terminal_stat"] = _scalar(self.rollout_last_stat.mean())
            train_info["surrogate_train"] = 1.0 if self.current_train_uses_surrogate else 0.0
            train_info["surrogate_fraction"] = float(self.current_surrogate_fraction)
            self.log_train(train_info, (episode + 1) * self.episode_length * self.n_rollout_threads)

            if episode % self.log_interval == 0:
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                elapsed = max(time.time() - start, 1e-8)
                reserved_gb = self._gpu_mem_gb("reserved")
                allocated_gb = self._gpu_mem_gb("allocated")
                max_allocated_gb = self._gpu_mem_gb("max_allocated")
                print(
                    f"\nAlgo {self.algorithm_name} Exp {self.experiment_name} GPU Planner updates "
                    f"{episode}/{episodes}, total timesteps {total_num_steps}/{self.num_env_steps}, "
                    f"FPS {int(total_num_steps / elapsed)}, GPU allocated/reserved/max "
                    f"{allocated_gb:.2f}/{reserved_gb:.2f}/{max_allocated_gb:.2f} GB.\n"
                )
                swanlab.log(
                    {
                        "GPUPlanner/memory_allocated_gb": allocated_gb,
                        "GPUPlanner/memory_reserved_gb": reserved_gb,
                        "GPUPlanner/max_memory_allocated_gb": max_allocated_gb,
                        "GPUPlanner/fps": total_num_steps / elapsed,
                    }
                )

            if self.save_interval > 0 and (episode + 1) % self.save_interval == 0:
                self.save_latest()
            if self.use_surrogate and self.surrogate_pretrain_updates > 0 and (episode + 1) == self.surrogate_pretrain_updates:
                self.save_surrogate_pretrain()

        self.save()

    @torch.no_grad()
    def _apply_tail_cutoff_penalty(self, rollout_steps, tail_cutoff, episode_reward_total, episode_base_reward_total=None):
        """Make early tail truncation equivalent to continuing active rows to horizon.

        The rollout may stop once only a small active tail remains. Without this
        correction, censored rows are counted as full-horizon delays in logging
        but receive only the truncated reward in PPO.
        """
        rollout_steps = int(rollout_steps)
        remaining = max(int(self.episode_length) - rollout_steps, 0)
        active_tail = self.rollout_active.clone()
        if (not tail_cutoff) or remaining <= 0 or rollout_steps <= 0 or (not bool(active_tail.any())):
            return {
                "tail_cutoff_applied": 0.0,
                "tail_censored_rate": _scalar(active_tail.float().mean()),
                "tail_remaining_steps": float(remaining),
                "tail_discounted_penalty_mean": 0.0,
                "tail_undiscounted_penalty_mean": 0.0,
            }

        gamma = float(getattr(self.all_args, "gamma", 1.0))
        if abs(gamma - 1.0) < 1e-8:
            discounted_tail = float(remaining)
        else:
            discounted_tail = float(gamma * (1.0 - gamma ** remaining) / (1.0 - gamma))
        undiscounted_tail = float(remaining)

        last_step = rollout_steps - 1
        self.buffer.rewards[last_step, active_tail, 0] -= discounted_tail
        self.buffer.masks[rollout_steps, active_tail, 0] = 0.0
        episode_reward_total[active_tail] -= undiscounted_tail
        if episode_base_reward_total is not None:
            episode_base_reward_total[active_tail] -= undiscounted_tail

        tail_rate = _scalar(active_tail.float().mean())
        return {
            "tail_cutoff_applied": 1.0,
            "tail_censored_rate": tail_rate,
            "tail_remaining_steps": float(remaining),
            "tail_discounted_penalty_mean": tail_rate * discounted_tail,
            "tail_undiscounted_penalty_mean": tail_rate * undiscounted_tail,
        }

    def _episode_uses_surrogate(self, episode):
        if self.mixed_surrogate:
            return self._surrogate_mix_fraction(episode) > 0.0
        if not self.use_surrogate:
            return False
        if self.surrogate_pretrain_updates <= 0:
            return True
        return int(episode) < self.surrogate_pretrain_updates

    def _surrogate_mix_fraction(self, episode):
        if not self.mixed_surrogate:
            return 1.0 if self._episode_uses_surrogate_hard(episode) else 0.0
        start = float(getattr(self.all_args, "gpu_stage2_mix_rho_start", 0.7))
        mid = float(getattr(self.all_args, "gpu_stage2_mix_rho_mid", 0.2))
        end = float(getattr(self.all_args, "gpu_stage2_mix_rho_end", 0.0))
        warm = int(getattr(self.all_args, "gpu_stage2_mix_warm_updates", 20))
        mid_update = int(getattr(self.all_args, "gpu_stage2_mix_mid_update", 50))
        end_update = int(getattr(self.all_args, "gpu_stage2_mix_end_update", 100))
        episode = int(episode)
        if episode < warm:
            rho = start
        elif episode < mid_update:
            frac = (episode - warm) / max(mid_update - warm, 1)
            rho = start + frac * (mid - start)
        elif episode < end_update:
            frac = (episode - mid_update) / max(end_update - mid_update, 1)
            rho = mid + frac * (end - mid)
        else:
            rho = end
        return float(min(max(rho, 0.0), 1.0))

    def _episode_uses_surrogate_hard(self, episode):
        if not self.use_surrogate:
            return False
        if self.surrogate_pretrain_updates <= 0:
            return True
        return int(episode) < self.surrogate_pretrain_updates

    def warmup(self, episode):
        if bool(getattr(self.all_args, "gpu_stage2_random_broken_streams", False)):
            broken_streams = torch.randint(
                int(data_configs.M_total),
                (self.batch_size,),
                device=self.device,
                generator=self.envs.generator,
                dtype=torch.long,
            )
        elif bool(getattr(self.all_args, "gpu_stage2_diverse_bm", True)):
            broken_streams = (
                torch.arange(self.batch_size, device=self.device, dtype=torch.long) + episode * self.batch_size
            ) % int(data_configs.M_total)
        else:
            broken_streams = torch.full(
                (self.batch_size,),
                int(episode % data_configs.M_total),
                device=self.device,
                dtype=torch.long,
            )
        if self.mixed_surrogate:
            self.current_surrogate_fraction = self._surrogate_mix_fraction(episode)
            surrogate_count = int(round(self.current_surrogate_fraction * self.batch_size))
            row_ids = torch.arange(self.batch_size, device=self.device)
            self.surrogate_row_mask = row_ids < surrogate_count
            self.real_row_mask = ~self.surrogate_row_mask
            self.current_train_uses_surrogate = surrogate_count > 0
        else:
            self.current_train_uses_surrogate = self._episode_uses_surrogate(episode)
            self.current_surrogate_fraction = 1.0 if self.current_train_uses_surrogate else 0.0
            self.surrogate_row_mask = torch.full(
                (self.batch_size,), self.current_train_uses_surrogate, device=self.device, dtype=torch.bool
            )
            self.real_row_mask = ~self.surrogate_row_mask
        planner_obs = self.envs.reset(broken_streams=broken_streams)
        if self.current_train_uses_surrogate or self.cf_alpha_baseline:
            surrogate_obs = self._reset_surrogate_state(broken_streams)
            if self.current_train_uses_surrogate:
                planner_obs = torch.where(self.surrogate_row_mask.unsqueeze(-1), surrogate_obs, planner_obs)
        self.detector_rnn_states.zero_()
        self.detector_masks.fill_(1.0)
        self.buffer.share_obs[0].copy_(planner_obs)
        self.buffer.obs[0].copy_(planner_obs)
        if bool(getattr(self.all_args, "gpu_stage2_reset_rnn_each_episode", False)):
            self.buffer.rnn_states[0].zero_()
            self.buffer.rnn_states_critic[0].zero_()
        self.buffer.masks[0].fill_(1.0)
        self.buffer.active_masks[0].fill_(1.0)
        self.buffer.bad_masks[0].fill_(1.0)

    def _stat_progress_beta(self, episode):
        if not self.stat_progress_reward:
            return 0.0
        frac = min(max(float(episode) / float(self.stat_progress_anneal_updates), 0.0), 1.0)
        return self.stat_progress_beta_start + frac * (self.stat_progress_beta_end - self.stat_progress_beta_start)

    def _surrogate_global_stat(self, region_stats):
        return torch.topk(
            region_stats,
            k=min(int(data_configs.l), self.num_regions),
            dim=1,
        ).values.sum(dim=1)

    def _add_stat_progress_reward(self, base_rewards, active_before, step, stat_before, stat_after):
        self.last_base_rewards = base_rewards.clone()
        shaping = torch.zeros_like(base_rewards)
        if (
            self.stat_progress_reward
            and abs(float(self.current_stat_progress_beta)) > 1e-12
            and self.stat_progress_clip > 0.0
            and int(step) >= self.train_t_break
        ):
            denom = max(abs(float(self.alarm_threshold)), 1e-6)
            before = (stat_before.float() / denom).clamp(0.0, 1.0)
            after = (stat_after.float() / denom).clamp(0.0, 1.0)
            progress = (after - before).clamp(-self.stat_progress_clip, self.stat_progress_clip)
            shaping[:, 0] = float(self.current_stat_progress_beta) * progress * active_before.float()
        self.last_stat_progress_rewards = shaping
        return base_rewards + shaping

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = self.policy.get_actions(
            self.buffer.share_obs[step],
            self.buffer.obs[step],
            self.buffer.rnn_states[step],
            self.buffer.rnn_states_critic[step],
            self.buffer.masks[step],
            None,
            deterministic=False,
            force_m=False,
            m=self.alpha_bins,
        )
        return value, action.long(), action_log_prob, rnn_state, rnn_state_critic

    @torch.no_grad()
    def env_step(self, planner_action, step):
        if self.mixed_surrogate:
            return self._env_step_mixed(planner_action, step)
        if self.current_train_uses_surrogate:
            return self._env_step_surrogate(planner_action, step)

        active_before = self.rollout_active.clone()
        region_stats = self.envs.region_statistics()
        lambda_prev = self.envs.global_stat()
        alpha = planner_action.float().mean(dim=1, keepdim=True)
        allocations = self.envs.alpha_to_allocation(alpha, region_stats)
        detector_obs = self.envs.detector_obs(allocations)
        detector_actions = self._detector_action_list(detector_obs, allocations)

        next_planner_obs, realized_usage = self.envs.step(allocations, detector_actions, active_mask=active_before)
        lambda_t = self.envs.global_stat()
        self.rollout_last_stat.copy_(lambda_t)
        new_alarm = active_before & (int(step) >= self.train_t_break) & (lambda_t >= self.alarm_threshold)
        if bool(new_alarm.any()):
            self.rollout_alarmed[new_alarm] = True
            self.rollout_active[new_alarm] = False
            self.rollout_detection_delay[new_alarm] = float(max(int(step) - self.train_t_break, 0))
            self.rollout_alarm_step[new_alarm] = float(int(step))
        dones = new_alarm.reshape(-1, 1)

        base_rewards = -torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)
        base_rewards[~active_before] = 0.0
        base_rewards[dones[:, 0]] = 0.0
        rewards = self._add_stat_progress_reward(base_rewards, active_before, step, lambda_prev, lambda_t)

        self.detector_masks.fill_(1.0)
        inactive_rows = ~self.rollout_active
        if bool(inactive_rows.any()):
            done_rows = inactive_rows
            self.detector_masks[done_rows] = 0.0
            self.detector_rnn_states[done_rows] = 0.0
        return rewards, dones, next_planner_obs

    @torch.no_grad()
    def _env_step_mixed(self, planner_action, step):
        active_before = self.rollout_active.clone()
        real_rows = self.real_row_mask
        surrogate_rows = self.surrogate_row_mask

        real_region_stats = self.envs.region_statistics()
        lambda_real_prev = self.envs.global_stat()
        region_stats = real_region_stats.clone()
        if bool(surrogate_rows.any()):
            region_stats[surrogate_rows] = self.surrogate_region_stats[surrogate_rows]
        lambda_prev_mix = lambda_real_prev.clone()
        if bool(surrogate_rows.any()):
            lambda_prev_mix[surrogate_rows] = self._surrogate_global_stat(
                self.surrogate_region_stats[surrogate_rows]
            )
        alpha = planner_action.float().mean(dim=1, keepdim=True)
        allocations = self.envs.alpha_to_allocation(alpha, region_stats)

        detector_obs = self.envs.detector_obs(allocations)
        detector_actions = self._detector_action_list(detector_obs, allocations, real_rows=real_rows)

        active_real = active_before & real_rows
        next_planner_obs, _ = self.envs.step(allocations, detector_actions, active_mask=active_real)
        lambda_real = self.envs.global_stat()

        lambda_mix = lambda_real.clone()
        if bool(surrogate_rows.any()):
            next_region_stats = self.surrogate_region_stats.clone()
            if self.detector_surrogate.baseline_increment > 0.0:
                next_region_stats[active_before & surrogate_rows] += self.detector_surrogate.baseline_increment
            update_rows = active_before & surrogate_rows & (int(step) >= self.train_t_break)
            if bool(update_rows.any()):
                row_ids = torch.arange(self.batch_size, device=self.device)[update_rows]
                broken_streams = self.surrogate_broken_streams[update_rows]
                increments = self.detector_surrogate.response_increment(
                    broken_streams,
                    allocations[update_rows],
                    int(step),
                    current_region_stats=self.surrogate_region_stats[update_rows],
                )
                broken_regions = self.detector_surrogate.global_to_region[broken_streams]
                next_region_stats[row_ids, broken_regions] += increments

            active_surrogate = active_before & surrogate_rows
            self.surrogate_region_stats[active_surrogate] = next_region_stats[active_surrogate]
            self.surrogate_prev_usage[active_surrogate] = allocations[active_surrogate].float()
            lambda_surrogate = self._surrogate_global_stat(self.surrogate_region_stats)
            lambda_mix[surrogate_rows] = lambda_surrogate[surrogate_rows]
            surrogate_obs = self._surrogate_planner_obs()
            next_planner_obs = torch.where(surrogate_rows.unsqueeze(-1), surrogate_obs, next_planner_obs)

        self.rollout_last_stat.copy_(lambda_mix)
        new_alarm = active_before & (int(step) >= self.train_t_break) & (lambda_mix >= self.alarm_threshold)
        if bool(new_alarm.any()):
            self.rollout_alarmed[new_alarm] = True
            self.rollout_active[new_alarm] = False
            self.rollout_detection_delay[new_alarm] = float(max(int(step) - self.train_t_break, 0))
            self.rollout_alarm_step[new_alarm] = float(int(step))
        dones = new_alarm.reshape(-1, 1)

        base_rewards = -torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)
        base_rewards[~active_before] = 0.0
        base_rewards[dones[:, 0]] = 0.0
        rewards = self._add_stat_progress_reward(base_rewards, active_before, step, lambda_prev_mix, lambda_mix)

        self.detector_masks.fill_(1.0)
        inactive_real = (~self.rollout_active) & real_rows
        if bool(inactive_real.any()):
            self.detector_masks[inactive_real] = 0.0
            self.detector_rnn_states[inactive_real] = 0.0
        return rewards, dones, next_planner_obs

    def _reset_surrogate_state(self, broken_streams):
        self.surrogate_broken_streams = broken_streams.to(device=self.device, dtype=torch.long).reshape(-1)
        self.surrogate_region_stats = torch.zeros(
            (self.batch_size, self.num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self.surrogate_prev_usage = torch.as_tensor(
            data_configs.m,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.batch_size, -1).clone()
        return self._surrogate_planner_obs()

    def _surrogate_planner_obs(self):
        return torch.cat([self.surrogate_prev_usage, self.surrogate_region_stats], dim=1)

    @torch.no_grad()
    def _env_step_surrogate(self, planner_action, step):
        active_before = self.rollout_active.clone()
        region_stats = self.surrogate_region_stats
        lambda_prev = self._surrogate_global_stat(self.surrogate_region_stats)
        alpha = planner_action.float().mean(dim=1, keepdim=True)
        allocations = self.envs.alpha_to_allocation(alpha, region_stats)

        next_region_stats = self.surrogate_region_stats.clone()
        if self.detector_surrogate.baseline_increment > 0.0:
            next_region_stats[active_before] += self.detector_surrogate.baseline_increment

        update_rows = active_before & (int(step) >= self.train_t_break)
        if bool(update_rows.any()):
            row_ids = torch.arange(self.batch_size, device=self.device)[update_rows]
            broken_streams = self.surrogate_broken_streams[update_rows]
            increments = self.detector_surrogate.response_increment(
                broken_streams,
                allocations[update_rows],
                int(step),
                current_region_stats=self.surrogate_region_stats[update_rows],
            )
            broken_regions = self.detector_surrogate.global_to_region[broken_streams]
            next_region_stats[row_ids, broken_regions] += increments

        self.surrogate_region_stats[active_before] = next_region_stats[active_before]
        self.surrogate_prev_usage[active_before] = allocations[active_before].float()
        lambda_t = self._surrogate_global_stat(self.surrogate_region_stats)
        self.rollout_last_stat.copy_(lambda_t)
        new_alarm = active_before & (int(step) >= self.train_t_break) & (lambda_t >= self.alarm_threshold)
        if bool(new_alarm.any()):
            self.rollout_alarmed[new_alarm] = True
            self.rollout_active[new_alarm] = False
            self.rollout_detection_delay[new_alarm] = float(max(int(step) - self.train_t_break, 0))
            self.rollout_alarm_step[new_alarm] = float(int(step))
        dones = new_alarm.reshape(-1, 1)

        base_rewards = -torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)
        base_rewards[~active_before] = 0.0
        base_rewards[dones[:, 0]] = 0.0
        rewards = self._add_stat_progress_reward(base_rewards, active_before, step, lambda_prev, lambda_t)
        return rewards, dones, self._surrogate_planner_obs()

    @torch.no_grad()
    def compute_counterfactual_advantages(self, active_length):
        self.buffer.cf_advantages.zero_()
        self.buffer.cf_masks.zero_()
        self.buffer.cf_lambda = 0.0
        if (not self.cf_alpha_baseline) or self.cf_lambda <= 0.0 or self.detector_surrogate is None:
            return {
                "cf_alpha_enabled": 0.0,
                "cf_alpha_coverage": 0.0,
                "cf_alpha_adv_mean": 0.0,
                "cf_alpha_adv_std": 0.0,
            }

        active_length = int(active_length)
        step_ids = torch.arange(0, active_length, self.cf_stride, device=self.device, dtype=torch.long)
        if step_ids.numel() == 0:
            return {
                "cf_alpha_enabled": 1.0,
                "cf_alpha_coverage": 0.0,
                "cf_alpha_adv_mean": 0.0,
                "cf_alpha_adv_std": 0.0,
            }
        total_candidates = int(step_ids.numel()) * self.batch_size
        if self.cf_max_states > 0 and total_candidates > self.cf_max_states:
            selected = torch.linspace(
                0,
                total_candidates - 1,
                steps=self.cf_max_states,
                device=self.device,
                dtype=torch.float32,
            ).long()
        else:
            selected = torch.arange(total_candidates, device=self.device, dtype=torch.long)

        cf_values = []
        for start in range(0, int(selected.numel()), self.cf_chunk_size):
            flat = selected[start : start + self.cf_chunk_size]
            step_lookup = flat // self.batch_size
            row_ids = flat % self.batch_size
            rollout_steps = step_ids[step_lookup]
            region_stats = self.buffer.obs[rollout_steps, row_ids, self.num_regions :].clone()
            broken_streams = self.envs.broken_streams[row_ids].long()
            sampled_alpha = self.buffer.actions[rollout_steps, row_ids].float().mean(dim=1, keepdim=True)

            q_sampled = self._counterfactual_alpha_returns(
                region_stats,
                broken_streams,
                sampled_alpha,
                rollout_steps,
            )
            q_candidates = []
            for candidate in self.cf_alpha_candidates:
                alpha = torch.full_like(sampled_alpha, float(candidate))
                q_candidates.append(
                    self._counterfactual_alpha_returns(
                        region_stats,
                        broken_streams,
                        alpha,
                        rollout_steps,
                    )
                )
            q_baseline = torch.stack(q_candidates, dim=0).mean(dim=0)
            cf_adv = q_sampled - q_baseline
            self.buffer.cf_advantages[rollout_steps, row_ids, 0] = cf_adv
            self.buffer.cf_masks[rollout_steps, row_ids, 0] = 1.0
            cf_values.append(cf_adv)

        self.buffer.cf_lambda = self.cf_lambda
        if cf_values:
            all_cf = torch.cat(cf_values)
            mean_cf = _scalar(all_cf.mean())
            std_cf = _scalar(all_cf.std(unbiased=False))
        else:
            mean_cf, std_cf = 0.0, 0.0
        coverage = float(self.buffer.cf_masks[:active_length].sum().detach().cpu().item()) / float(
            max(active_length * self.batch_size, 1)
        )
        return {
            "cf_alpha_enabled": 1.0,
            "cf_alpha_coverage": coverage,
            "cf_alpha_adv_mean": mean_cf,
            "cf_alpha_adv_std": std_cf,
        }

    @torch.no_grad()
    def _counterfactual_alpha_returns(self, region_stats, broken_streams, alpha, rollout_steps):
        stats = region_stats.clone()
        active = torch.ones((stats.shape[0],), device=self.device, dtype=torch.bool)
        returns = torch.zeros((stats.shape[0],), device=self.device, dtype=torch.float32)
        discount = torch.ones_like(returns)
        gamma = float(getattr(self.all_args, "gamma", 0.99))
        for offset in range(self.cf_horizon):
            if not bool(active.any()):
                break
            absolute_steps = rollout_steps + int(offset)
            allocations = self.envs.alpha_to_allocation(alpha, stats)
            update_rows = active & (absolute_steps >= self.train_t_break)
            if bool(update_rows.any()):
                row_ids = torch.arange(stats.shape[0], device=self.device)[update_rows]
                increments = self.detector_surrogate.response_increment(
                    broken_streams[update_rows],
                    allocations[update_rows],
                    int(self.train_t_break + min(max(int(offset), 0), self.cf_horizon - 1)),
                    current_region_stats=stats[update_rows],
                )
                broken_regions = self.detector_surrogate.global_to_region[broken_streams[update_rows]]
                stats[row_ids, broken_regions] += increments

            lambda_t = torch.topk(
                stats,
                k=min(int(data_configs.l), self.num_regions),
                dim=1,
            ).values.sum(dim=1)
            alarm = active & (absolute_steps >= self.train_t_break) & (lambda_t >= self.alarm_threshold)
            reward = -torch.ones_like(returns)
            reward[~active] = 0.0
            reward[alarm] = 0.0
            returns += discount * reward
            active[alarm] = False
            discount *= gamma
        return returns

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        active_length = int(getattr(self.buffer, "active_length", self.episode_length))
        next_value = self.policy.get_values(
            self.buffer.share_obs[active_length],
            self.buffer.rnn_states_critic[active_length],
            self.buffer.masks[active_length],
        )
        self.buffer.compute_returns(next_value, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        active_length = int(getattr(self.buffer, "active_length", self.episode_length))
        factor = torch.ones(
            (active_length, self.n_rollout_threads, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.buffer.update_factor(factor)
        old_num_mini_batch = int(self.trainer.num_mini_batch)
        if self.target_minibatch_size > 0:
            effective_samples = active_length * self.n_rollout_threads
            self.trainer.num_mini_batch = max(1, int(np.ceil(effective_samples / float(self.target_minibatch_size))))
        train_info = self.trainer.train(self.buffer)
        self.trainer.num_mini_batch = old_num_mini_batch
        self.buffer.after_update()
        return train_info

    @torch.inference_mode()
    def evaluate(
        self,
        num_episodes=None,
        horizon=None,
        start_index=None,
        log_prefix="GPUPlannerEval",
        print_name="GPU Stage II eval",
    ):
        num_episodes = int(num_episodes or self.planner_eval_episodes)
        horizon = int(horizon or self.planner_eval_horizon)
        start_index = int(self.planner_eval_start_index if start_index is None else start_index)
        eval_start = time.time()
        lengths = simulate_planner_lengths_gpu(
            self,
            num_episodes,
            self.planner_eval_delta,
            self.alarm_threshold,
            horizon,
            force_m=True,
            eval_train=True,
            start_index=start_index,
            random_broken_stream=self.planner_eval_random_broken_streams,
        )
        mean_rl = float(np.mean(lengths))
        stderr_rl = float(np.std(lengths) / np.sqrt(max(len(lengths), 1)))
        elapsed = time.time() - eval_start
        swanlab.log(
            {
                f"{log_prefix}/running_length_mean": mean_rl,
                f"{log_prefix}/running_length_stderr": stderr_rl,
                f"{log_prefix}/delta": self.planner_eval_delta,
                f"{log_prefix}/threshold": self.alarm_threshold,
                f"{log_prefix}/time_sec": elapsed,
                f"{log_prefix}/gpu_vectorized": 1.0,
            }
        )
        print(
            f"{print_name}:",
            {
                "runs": int(num_episodes),
                "batch_size": int(getattr(self.all_args, "gpu_eval_batch_size", 1024)),
                "delta": float(self.planner_eval_delta),
                "threshold": float(self.alarm_threshold),
                "horizon": int(horizon),
                "start_index": int(start_index),
                "seconds": round(elapsed, 3),
            },
        )
        return mean_rl, stderr_rl

    def log_train(self, train_info, total_num_steps):
        for key, value in train_info.items():
            metric = f"GPUPlanner/{key}"
            value_float = _scalar(value)
            self.writter.add_scalars(metric, {metric: value_float}, total_num_steps)
            print(metric, {metric: value_float}, total_num_steps)
            swanlab.log({metric: value_float})

    def save(self):
        torch.save(self.policy.actor.state_dict(), str(self.save_dir / "planner_actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.save_dir / "planner_critic.pt"))

    def save_best(self):
        torch.save(self.policy.actor.state_dict(), str(self.save_dir / "planner_best_actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.save_dir / "planner_best_critic.pt"))

    def save_latest(self):
        torch.save(self.policy.actor.state_dict(), str(self.save_dir / "planner_actor_latest.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.save_dir / "planner_critic_latest.pt"))

    def save_eval_checkpoint(self, prefix, episode, mean_rl, stderr_rl):
        if not bool(getattr(self.all_args, "save_eval_checkpoints", False)):
            return
        eval_dir = self.save_dir / "eval_checkpoints"
        eval_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = str(prefix).replace("/", "_")
        stem = f"{safe_prefix}_episode_{int(episode):06d}"

        metrics = {
            "prefix": str(prefix),
            "episode": int(episode),
            "running_length_mean": float(mean_rl),
            "running_length_stderr": float(stderr_rl),
            "best_eval_running_length": float(self.best_eval_running_length),
        }
        metrics_path = eval_dir / f"{stem}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        actor_ok = self._save_eval_state_dict_with_retries(
            self.policy.actor.state_dict(),
            eval_dir / f"{stem}_actor.pt",
        )
        critic_ok = self._save_eval_state_dict_with_retries(
            self.policy.critic.state_dict(),
            eval_dir / f"{stem}_critic.pt",
        )
        if not (actor_ok and critic_ok):
            metrics["actor_checkpoint_saved"] = bool(actor_ok)
            metrics["critic_checkpoint_saved"] = bool(critic_ok)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    def _save_eval_state_dict_with_retries(self, state_dict, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(3):
            try:
                torch.save(state_dict, str(path))
                return True
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        try:
            torch.save(state_dict, str(path), _use_new_zipfile_serialization=False)
            return True
        except Exception as exc:
            last_error = exc
        print(f"[GPU PlannerEval] warning: failed to save checkpoint {path}: {last_error}", flush=True)
        swanlab.log({"GPUPlanner/eval_checkpoint_save_failed": 1.0})
        try:
            with open(path.with_suffix(path.suffix + ".error.txt"), "w", encoding="utf-8") as f:
                f.write(str(last_error))
        except Exception:
            pass
        return False

    def save_surrogate_pretrain(self):
        torch.save(self.policy.actor.state_dict(), str(self.save_dir / "planner_surrogate_actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.save_dir / "planner_surrogate_critic.pt"))

    def _gpu_mem_gb(self, kind):
        if self.device.type != "cuda":
            return 0.0
        idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        scale = 1024 ** 3
        if kind == "allocated":
            return torch.cuda.memory_allocated(idx) / scale
        if kind == "reserved":
            return torch.cuda.memory_reserved(idx) / scale
        if kind == "max_allocated":
            return torch.cuda.max_memory_allocated(idx) / scale
        raise ValueError(kind)
