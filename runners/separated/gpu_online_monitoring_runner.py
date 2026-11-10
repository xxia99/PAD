import os
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from algorithms.happo_policy import HAPPO_Policy
from algorithms.happo_trainer_gpu import HAPPOGPU
from envs.env_utils import data_configs
from envs.gpu_online_monitoring import GPUOnlineMonitoringEnv
from runners.separated.parallel_detectors import ParallelDetectorExecutor
from utils.gpu_separated_buffer import GpuSeparatedReplayBuffer
from utils.swanlab_compat import swanlab


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu().item())
    return float(value)


class GPUOnlineMonitoringRunner:
    """Single-process, GPU-vectorized Stage I detector training runner."""

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = Path(config["run_dir"])
        self.num_agents = int(data_configs.agent_num)
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_single_network = self.all_args.use_single_network
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.episode_length = int(self.all_args.episode_length)
        self.hidden_size = int(self.all_args.hidden_size)
        self.recurrent_N = int(self.all_args.recurrent_N)
        self.use_eval = bool(self.all_args.use_eval)
        self.eval_interval = int(self.all_args.eval_interval)
        self.log_interval = int(self.all_args.log_interval)
        self.save_interval = int(self.all_args.save_interval)
        self.n_warmup_evaluations = int(self.all_args.n_warmup_evaluations)
        self.n_no_improvement_thres = int(self.all_args.n_no_improvement_thres)
        self.model_dir = self.all_args.model_dir
        self.batch_size = int(self.all_args.gpu_stage1_batch_size or self.all_args.n_rollout_threads)
        self.all_args.n_rollout_threads = self.batch_size
        self.n_rollout_threads = self.batch_size

        target_minibatch = int(getattr(self.all_args, "gpu_stage1_minibatch_size", 0) or 0)
        if target_minibatch > 0:
            samples_per_update = self.episode_length * self.batch_size
            target_batches = int(np.ceil(samples_per_update / float(target_minibatch)))
            self.all_args.num_mini_batch = max(int(self.all_args.num_mini_batch), target_batches)
        if self.all_args.use_naive_recurrent_policy:
            self.all_args.num_mini_batch = min(int(self.all_args.num_mini_batch), self.batch_size)

        self.envs = GPUOnlineMonitoringEnv(self.all_args, self.device, batch_size=self.batch_size)
        self.log_dir = self.run_dir / "logs"
        self.save_dir = self.run_dir / "models"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writter = SummaryWriter(str(self.log_dir))

        print(f"There are {self.num_agents} detectors in the GPU environment.")
        print("share_observation_space:", self.envs.share_observation_space)
        print("observation_space:", self.envs.observation_space)
        print("action_space:", self.envs.action_space)
        print(
            "GPU Stage I config:",
            {
                "batch_size": self.batch_size,
                "episode_length": self.episode_length,
                "num_mini_batch": int(self.all_args.num_mini_batch),
                "ppo_epoch": int(self.all_args.ppo_epoch),
                "device": str(self.device),
            },
        )
        allocation_pool_size = (
            int(self.envs.allocation_pool.shape[0])
            if getattr(self.envs, "allocation_pool", None) is not None
            else 1
        )
        print(
            "GPU Stage I resource allocation:",
            {
                "random": bool(getattr(self.all_args, "gpu_stage1_random_allocations", False)),
                "m_total": int(data_configs.m_total),
                "fixed_m": list(data_configs.m),
                "upper_bound": int(getattr(self.all_args, "gpu_stage1_resource_upper_bound", 10)),
                "valid_allocation_count": allocation_pool_size,
            },
        )

        self.policy = []
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            share_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            policy = HAPPO_Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            trainer = HAPPOGPU(self.all_args, policy, device=self.device)
            buffer = GpuSeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_space,
                self.envs.action_space[agent_id],
                self.device,
            )
            self.policy.append(policy)
            self.trainer.append(trainer)
            self.buffer.append(buffer)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        start = time.time()
        episodes = max(1, self.num_env_steps // self.episode_length // self.n_rollout_threads)
        best_arl = float("inf")
        best_arl_by_m = {}
        best_arl_by_mode_m = {}
        best_select_arl_by_mode_m = {}
        best_select_arl_by_mode = {}
        no_improvement = 0

        for episode in range(episodes):
            if self.use_eval and episode % self.eval_interval == 0:
                rng_state = self._capture_torch_rng() if bool(getattr(self.all_args, "gpu_stage1_eval_preserve_rng", False)) else None
                try:
                    eval_modes = self._eval_force_modes()
                    legacy_eval_logging = (
                        bool(getattr(self.all_args, "gpu_stage1_legacy_eval_logging", False))
                        and eval_modes == ["F"]
                    )
                    eval_log = {}
                    mode_results = {}
                    for eval_mode in eval_modes:
                        force_m = eval_mode == "F"
                        std, arl = self.eval(
                            threshold=float(self.all_args.gpu_stage1_eval_threshold),
                            test_delta=float(self.all_args.delta),
                            force_m=force_m,
                        )
                        mode_results[eval_mode] = (std, arl)
                        print("=====GPU PAD Stage I Evaluation=====")
                        if legacy_eval_logging:
                            print("ARL:", arl, "Std:", std)
                        else:
                            print("Mode:", eval_mode, "ARL:", arl, "Std:", std)
                        print("====================================")
                        if legacy_eval_logging:
                            eval_log["GPU Eval ARL"] = arl
                            eval_log["GPU Eval Std"] = std
                        else:
                            prefix = f"GPU Eval {eval_mode}"
                            eval_log[f"{prefix} ARL"] = arl
                            eval_log[f"{prefix} Std"] = std
                        for m_value, metrics in getattr(self, "last_eval_by_m", {}).items():
                            if legacy_eval_logging:
                                continue
                            eval_log[f"{prefix} ARL/m={m_value}"] = metrics["arl"]
                            eval_log[f"{prefix} N/m={m_value}"] = metrics["n"]
                            key = (eval_mode, int(m_value))
                            previous = best_arl_by_mode_m.get(key, float("inf"))
                            if metrics["arl"] <= previous and episode >= self.n_warmup_evaluations:
                                best_arl_by_mode_m[key] = float(metrics["arl"])
                                self.save_budget_best(int(m_value), mode=eval_mode)
                                print(f"A better GPU Stage I model is saved for mode={eval_mode}, m={int(m_value)}.")
                            eval_log[f"{prefix} Best ARL/m={m_value}"] = best_arl_by_mode_m.get(key, metrics["arl"])
                        if eval_mode == "F":
                            for m_value, metrics in getattr(self, "last_eval_by_m", {}).items():
                                previous = best_arl_by_m.get(int(m_value), float("inf"))
                                if metrics["arl"] <= previous and episode >= self.n_warmup_evaluations:
                                    best_arl_by_m[int(m_value)] = float(metrics["arl"])
                                    self.save_budget_best(int(m_value))
                                eval_log[f"GPU Eval Best ARL/m={m_value}"] = best_arl_by_m.get(int(m_value), metrics["arl"])
                    swanlab.log(eval_log)
                finally:
                    if rng_state is not None:
                        self._restore_torch_rng(rng_state)

                std, arl = mode_results.get("F", next(iter(mode_results.values())))
                if arl <= best_arl:
                    best_arl = arl
                    no_improvement = 0
                    if episode >= self.n_warmup_evaluations:
                        self.save()
                        print("A better GPU Stage I model is saved.")
                elif episode > self.n_warmup_evaluations:
                    no_improvement += 1
                    self.save_latest(episode)
                    if no_improvement == self.n_no_improvement_thres:
                        print(f"Training stopped after {self.n_no_improvement_thres} evaluations without improvement.")
                        return best_arl
                select_interval = int(getattr(self.all_args, "gpu_stage1_select_eval_interval", 0) or 0)
                if (
                    select_interval > 0
                    and episode % select_interval == 0
                    and episode >= self.n_warmup_evaluations
                ):
                    select_log = {}
                    for eval_mode in self._select_eval_force_modes():
                        force_m = eval_mode == "F"
                        std, arl = self.eval(
                            threshold=float(self.all_args.gpu_stage1_eval_threshold),
                            test_delta=float(self.all_args.delta),
                            force_m=force_m,
                            num_runs=int(getattr(self.all_args, "gpu_stage1_select_eval_episodes", 5000)),
                            horizon=int(getattr(self.all_args, "gpu_stage1_select_eval_horizon", 2000)),
                            resource_choices=getattr(self.all_args, "gpu_stage1_select_eval_single_m_choices", None),
                        )
                        prefix = f"GPU Select Eval {eval_mode}"
                        select_log[f"{prefix} ARL"] = arl
                        select_log[f"{prefix} Std"] = std
                        previous_global = best_select_arl_by_mode.get(eval_mode, float("inf"))
                        if arl <= previous_global:
                            best_select_arl_by_mode[eval_mode] = float(arl)
                            self.save_named_best(f"best_SELECT_{eval_mode}")
                            print(f"A better GPU Stage I selection model is saved for mode={eval_mode}.")
                        select_log[f"{prefix} Best ARL"] = best_select_arl_by_mode.get(eval_mode, arl)
                        for m_value, metrics in getattr(self, "last_eval_by_m", {}).items():
                            key = (eval_mode, int(m_value))
                            previous = best_select_arl_by_mode_m.get(key, float("inf"))
                            if metrics["arl"] <= previous:
                                best_select_arl_by_mode_m[key] = float(metrics["arl"])
                                self.save_budget_best(int(m_value), mode=f"SELECT_{eval_mode}")
                                print(
                                    "A better GPU Stage I selection model is saved "
                                    f"for mode={eval_mode}, m={int(m_value)}."
                                )
                            select_log[f"{prefix} ARL/m={m_value}"] = metrics["arl"]
                            select_log[f"{prefix} N/m={m_value}"] = metrics["n"]
                            select_log[f"{prefix} Best ARL/m={m_value}"] = best_select_arl_by_mode_m.get(
                                key, metrics["arl"]
                            )
                    swanlab.log(select_log)

            self.warmup(episode)
            if self.use_linear_lr_decay:
                for policy in self.policy:
                    policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones = self.envs.step(actions)
                self.insert(
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

            self._apply_stage1_loss_mask()
            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if bool(getattr(self.all_args, "gpu_stage1_legacy_metrics", False)):
                legacy_denom = float(max(self.episode_length - data_configs.epoch_buffer, 1))
                hit_by_traj = self.envs.hit_count / legacy_denom
                hit_rate = _scalar(self.envs.hit_count.mean() / legacy_denom)
            else:
                post_denom = self.envs.post_count.clamp_min(1.0)
                hit_by_traj = self.envs.hit_count / post_denom
                hit_rate = _scalar(hit_by_traj.mean())
            for train_info in train_infos:
                train_info["hit_rate"] = hit_rate
            if self.num_agents == 1:
                resources = self.envs.resources[:, 0].long()
                reward_by_traj = self.buffer[0].rewards.mean(dim=0).reshape(-1)
                for m_value in torch.unique(resources).detach().cpu().tolist():
                    mask = resources == int(m_value)
                    if bool(mask.any()):
                        for train_info in train_infos:
                            train_info[f"hit_rate_m{int(m_value)}"] = _scalar(hit_by_traj[mask].mean())
                            train_info[f"average_step_rewards_m{int(m_value)}"] = _scalar(reward_by_traj[mask].mean())
            self.log_train(train_infos, total_num_steps)

            if episode % self.log_interval == 0:
                elapsed = max(time.time() - start, 1e-8)
                reserved_gb = self._gpu_mem_gb("reserved")
                allocated_gb = self._gpu_mem_gb("allocated")
                max_allocated_gb = self._gpu_mem_gb("max_allocated")
                print(
                    f"\nAlgo {self.algorithm_name} Exp {self.experiment_name} GPU updates "
                    f"{episode}/{episodes}, total timesteps {total_num_steps}/{self.num_env_steps}, "
                    f"FPS {int(total_num_steps / elapsed)}, GPU allocated/reserved/max "
                    f"{allocated_gb:.2f}/{reserved_gb:.2f}/{max_allocated_gb:.2f} GB.\n"
                )
                swanlab.log(
                    {
                        "GPU/memory_allocated_gb": allocated_gb,
                        "GPU/memory_reserved_gb": reserved_gb,
                        "GPU/max_memory_allocated_gb": max_allocated_gb,
                        "GPU/fps": total_num_steps / elapsed,
                    }
                )

            if self.save_interval > 0 and (episode + 1) % self.save_interval == 0:
                self.save_latest(episode)

        return best_arl

    def warmup(self, episode):
        if bool(getattr(self.all_args, "gpu_stage1_random_broken_streams", False)):
            broken_streams = torch.randint(
                int(data_configs.M_total),
                (self.batch_size,),
                device=self.device,
                generator=self.envs.generator,
                dtype=torch.long,
            )
        elif bool(getattr(self.all_args, "gpu_stage1_diverse_bm", True)):
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
        train_start_step = getattr(self.all_args, "gpu_stage1_train_start_step", None)
        if train_start_step is None:
            train_start_step = self.envs.t_break
        obs, share_obs = self.envs.reset(
            broken_streams=broken_streams,
            start_step=int(train_start_step),
            episode_steps=self.episode_length,
            t_breaks=self._sample_train_t_breaks(),
        )
        for agent_id in range(self.num_agents):
            share = share_obs if self.use_centralized_V else obs[:, agent_id]
            self.buffer[agent_id].share_obs[0].copy_(share)
            self.buffer[agent_id].obs[0].copy_(obs[:, agent_id])
            if bool(getattr(self.all_args, "gpu_stage1_reset_rnn_each_episode", False)):
                self.buffer[agent_id].rnn_states[0].zero_()
                self.buffer[agent_id].rnn_states_critic[0].zero_()
                self.buffer[agent_id].masks[0].fill_(1.0)
                self.buffer[agent_id].bad_masks[0].fill_(1.0)
                self.buffer[agent_id].active_masks[0].fill_(1.0)

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
                None,
                deterministic=False,
                force_m=False,
                m=self.buffer[agent_id].obs[step][:, 0],
            )
            values.append(value)
            actions.append(action.long())
            action_log_probs.append(action_log_prob)
            rnn_states.append(rnn_state)
            rnn_states_critic.append(rnn_state_critic)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(
        self,
        obs,
        share_obs,
        rewards,
        dones,
        values,
        actions,
        action_log_probs,
        rnn_states,
        rnn_states_critic,
    ):
        dones_env = torch.all(dones, dim=1)
        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1), device=self.device)
        active_masks = torch.ones_like(masks)
        if bool(dones_env.any()):
            masks[dones_env] = 0.0
            active_masks[dones_env] = 1.0
            for agent_id in range(self.num_agents):
                rnn_states[agent_id][dones_env] = 0.0
                rnn_states_critic[agent_id][dones_env] = 0.0
        bad_masks = torch.ones_like(masks)

        for agent_id in range(self.num_agents):
            share = share_obs if self.use_centralized_V else obs[:, agent_id]
            self.buffer[agent_id].insert(
                share,
                obs[:, agent_id],
                rnn_states[agent_id],
                rnn_states_critic[agent_id],
                actions[agent_id],
                action_log_probs[agent_id],
                values[agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks[:, agent_id],
                active_masks[:, agent_id],
            )

    def _apply_stage1_loss_mask(self):
        if not bool(getattr(self.all_args, "gpu_stage1_postchange_only_loss", False)):
            return
        train_start_step = getattr(self.all_args, "gpu_stage1_train_start_step", None)
        if train_start_step is None:
            train_start_step = self.envs.t_break
        loss_start = int(self.envs.t_break) + int(data_configs.epoch_buffer) - int(train_start_step)
        loss_start = max(0, min(int(loss_start), int(self.episode_length)))
        if loss_start <= 0:
            return
        for buffer in self.buffer:
            buffer.active_masks[:loss_start].zero_()

    def _capture_torch_rng(self):
        state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _restore_torch_rng(self, state):
        torch.set_rng_state(state["cpu"])
        if "cuda" in state and torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], self.device)

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        factor = torch.ones((self.episode_length, self.n_rollout_threads, 1), device=self.device)
        for agent_id in torch.randperm(self.num_agents, device=self.device).tolist():
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            old_log_probs = self._flat_action_log_probs(agent_id)
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            new_log_probs = self._flat_action_log_probs(agent_id)
            ratio = torch.prod(torch.exp(new_log_probs - old_log_probs), dim=-1).reshape(
                self.episode_length,
                self.n_rollout_threads,
                1,
            )
            factor = (factor * ratio).detach()
            train_infos.append((agent_id, train_info))
            self.buffer[agent_id].after_update()

        ordered = [None] * self.num_agents
        for agent_id, train_info in train_infos:
            ordered[agent_id] = train_info
        return ordered

    @torch.no_grad()
    def _flat_action_log_probs(self, agent_id):
        buffer = self.buffer[agent_id]
        chunk_size = int(getattr(self.all_args, "gpu_stage1_logprob_chunk_size", 4096))
        episode_length, batch_size = buffer.actions.shape[:2]
        log_probs = torch.empty_like(buffer.action_log_probs)

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            obs_chunk = buffer.obs[:-1, start:end].reshape(-1, *buffer.obs.shape[2:])
            log_prob_chunk, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                obs_chunk,
                buffer.rnn_states[0, start:end].reshape(-1, *buffer.rnn_states.shape[2:]),
                buffer.actions[:, start:end].reshape(-1, buffer.actions.shape[-1]),
                buffer.masks[:-1, start:end].reshape(-1, 1),
                None,
                buffer.active_masks[:-1, start:end].reshape(-1, 1),
                m=obs_chunk[:, 0],
            )
            log_probs[:, start:end].copy_(log_prob_chunk.reshape(episode_length, end - start, -1))

        return log_probs.reshape(-1, log_probs.shape[-1]).detach()

    @torch.inference_mode()
    def eval(self, threshold=50.0, test_delta=1.0, force_m=True, num_runs=None, horizon=None, resource_choices=None):
        eval_start = time.time()
        num_runs = int(num_runs if num_runs is not None else getattr(self.all_args, "gpu_stage1_eval_episodes", 1000))
        eval_batch_size = int(getattr(self.all_args, "gpu_stage1_eval_batch_size", None) or self.batch_size)
        horizon = int(horizon if horizon is not None else getattr(self.all_args, "gpu_stage1_eval_horizon", 2050))
        random_broken = bool(getattr(self.all_args, "gpu_stage1_eval_random_broken_streams", False))
        skip_prechange = bool(getattr(self.all_args, "gpu_stage1_eval_skip_prechange_alarms", False))
        stratified_resources = bool(getattr(self.all_args, "gpu_stage1_eval_stratified_resources", False))
        running_lengths = []
        lengths_by_m = {}
        skipped_prechange = 0
        skipped_censored = 0
        valid_replications = 0

        eval_env = GPUOnlineMonitoringEnv(self.all_args, self.device, batch_size=min(eval_batch_size, num_runs))
        eval_resource_pool = self._eval_resource_pool(eval_env=eval_env, resource_choices=resource_choices)
        use_iot_eval = getattr(data_configs, "eval_data_mode", "synthetic") == "eval_pi"
        if not use_iot_eval:
            pool_seed = getattr(self.all_args, "gpu_stage1_eval_pool_seed", None)
            if pool_seed is None and bool(getattr(self.all_args, "gpu_stage1_eval_common_pools", False)):
                pool_seed = int(getattr(self.all_args, "seed", 1)) + 100000 + int(round(float(test_delta) * 1000))
            eval_env._build_pools(float(test_delta), seed=pool_seed)
        for trainer in self.trainer:
            trainer.prep_rollout()
        fused_executor = None
        if bool(getattr(self.all_args, "gpu_stage1_eval_use_fused_detectors", False)):
            fused_executor = ParallelDetectorExecutor(
                self.policy,
                self.device,
                disabled=bool(getattr(self.all_args, "disable_parallel_detectors", False)),
                backend=getattr(self.all_args, "parallel_detector_backend", "fused"),
                strict_parity=bool(getattr(self.all_args, "fused_detector_strict_parity", True)),
                fused_disable_tf32=bool(getattr(self.all_args, "fused_detector_disable_tf32", True)),
            )
            print(f"GPU Stage I eval detector executor: {fused_executor.reason}")

        for offset in range(0, num_runs, eval_batch_size):
            batch_size = min(eval_batch_size, num_runs - offset)
            batch_horizon = horizon
            if use_iot_eval:
                batch_horizon = eval_env.set_iot_eval_batch(
                    float(test_delta),
                    offset,
                    batch_size,
                    horizon,
                )
            if random_broken:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(getattr(self.all_args, "seed", 1)) + int(offset))
                broken_streams = torch.randint(
                    int(data_configs.M_total),
                    (batch_size,),
                    device=self.device,
                    generator=generator,
                    dtype=torch.long,
                )
            else:
                broken_streams = (
                    torch.arange(offset, offset + batch_size, device=self.device, dtype=torch.long)
                    % int(data_configs.M_total)
                )
            eval_resources = None
            if stratified_resources and eval_resource_pool is not None:
                pool = eval_resource_pool
                pool_indices = torch.arange(
                    offset,
                    offset + batch_size,
                    device=self.device,
                    dtype=torch.long,
                ) % int(pool.shape[0])
                eval_resources = pool.index_select(0, pool_indices)
            obs, _ = eval_env.reset(
                broken_streams=broken_streams,
                start_step=0,
                episode_steps=batch_horizon,
                batch_size=batch_size,
                resources=eval_resources,
            )
            resource_labels = eval_env.resources[:, 0].long().detach().cpu().numpy()
            rnn_states = torch.zeros(
                (batch_size, self.num_agents, self.recurrent_N, self.hidden_size),
                device=self.device,
            )
            masks = torch.ones((batch_size, self.num_agents, 1), device=self.device)
            active = torch.ones((batch_size,), device=self.device, dtype=torch.bool)
            alarmed = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)
            prechange_alarm = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)
            lengths = torch.full(
                (batch_size,),
                max(batch_horizon - eval_env.t_break, 0),
                device=self.device,
                dtype=torch.float32,
            )

            for step in range(batch_horizon):
                parallel_actions = None
                if fused_executor is not None:
                    parallel_actions = fused_executor.act(
                        obs,
                        rnn_states,
                        masks,
                        obs[:, :, 0],
                        deterministic=True,
                        force_m=bool(force_m),
                    )
                if parallel_actions is not None:
                    actions_tensor, rnn_states = parallel_actions
                    actions = [actions_tensor[:, agent_id].long() for agent_id in range(self.num_agents)]
                else:
                    actions = []
                    for agent_id in range(self.num_agents):
                        action, next_rnn = self.trainer[agent_id].policy.act(
                            obs[:, agent_id],
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            None,
                            deterministic=True,
                            force_m=bool(force_m),
                            m=obs[:, agent_id][:, 0],
                        )
                        rnn_states[:, agent_id] = next_rnn
                        actions.append(action.long())

                obs, _, _, _ = eval_env.step(actions)
                raw_alarm = eval_env.global_stat() >= float(threshold)
                if skip_prechange:
                    early_alarm = active & (step < eval_env.t_break) & raw_alarm
                    if bool(early_alarm.any()):
                        prechange_alarm[early_alarm] = True
                        active[early_alarm] = False
                        masks[early_alarm] = 0.0
                        rnn_states[early_alarm] = 0.0
                alarm = active & (step >= eval_env.t_break) & raw_alarm
                if bool(alarm.any()):
                    lengths[alarm] = float(step - eval_env.t_break)
                    alarmed[alarm] = True
                    active[alarm] = False
                    masks[alarm] = 0.0
                    rnn_states[alarm] = 0.0
                if not bool(active.any()):
                    break

            if skip_prechange:
                valid = alarmed & ~prechange_alarm
                valid_replications += int(valid.sum().detach().cpu().item())
                skipped_prechange += int(prechange_alarm.sum().detach().cpu().item())
                skipped_censored += int((~alarmed & ~prechange_alarm).sum().detach().cpu().item())
                if bool(valid.any()):
                    valid_lengths = lengths[valid].detach().cpu().numpy()
                    valid_resources = resource_labels[valid.detach().cpu().numpy().astype(bool)]
                    running_lengths.append(valid_lengths)
                    for m_value in np.unique(valid_resources):
                        lengths_by_m.setdefault(int(m_value), []).append(valid_lengths[valid_resources == m_value])
            else:
                batch_lengths = lengths.detach().cpu().numpy()
                running_lengths.append(batch_lengths)
                for m_value in np.unique(resource_labels):
                    lengths_by_m.setdefault(int(m_value), []).append(batch_lengths[resource_labels == m_value])

        if running_lengths:
            lengths_np = np.concatenate(running_lengths).astype(float, copy=False)
        else:
            lengths_np = np.asarray([float("inf")], dtype=float)
        self.last_eval_by_m = {
            int(m_value): {
                "arl": float(np.mean(np.concatenate(chunks).astype(float, copy=False))),
                "n": int(sum(len(chunk) for chunk in chunks)),
            }
            for m_value, chunks in sorted(lengths_by_m.items())
            if chunks
        }
        elapsed = time.time() - eval_start
        print(
            "GPU Stage I eval:",
            {
                "runs": int(num_runs),
                "batch_size": int(eval_batch_size),
                "delta": float(test_delta),
                "threshold": float(threshold),
                "horizon": int(horizon),
                "random_broken": bool(random_broken),
                "force_m": bool(force_m),
                "skip_prechange": bool(skip_prechange),
                "stratified_resources": bool(stratified_resources),
                "resource_choices": self._eval_resource_choices_for_log(resource_choices),
                "detector_backend": None if fused_executor is None else fused_executor.reason,
                "valid_replications": int(valid_replications) if skip_prechange else int(num_runs),
                "skipped_prechange": int(skipped_prechange),
                "skipped_censored": int(skipped_censored),
                "by_m": self.last_eval_by_m,
                "seconds": round(elapsed, 3),
            },
        )
        return float(np.std(lengths_np) / np.sqrt(max(len(lengths_np), 1))), float(np.mean(lengths_np))

    def _eval_force_modes(self):
        raw = str(getattr(self.all_args, "gpu_stage1_eval_force_modes", "F") or "F")
        return self._parse_force_modes(raw)

    def _select_eval_force_modes(self):
        raw = getattr(self.all_args, "gpu_stage1_select_eval_force_modes", None)
        if raw is None or str(raw).strip() == "":
            raw = getattr(self.all_args, "gpu_stage1_eval_force_modes", "F")
        return self._parse_force_modes(str(raw))

    def _parse_force_modes(self, raw):
        modes = []
        for item in raw.split(","):
            mode = item.strip().upper()
            if not mode:
                continue
            if mode not in {"E", "F"}:
                raise ValueError(f"Unknown GPU Stage I eval force mode {mode!r}; expected E or F.")
            if mode not in modes:
                modes.append(mode)
        return modes or ["F"]

    def _eval_resource_pool(self, eval_env, resource_choices=None):
        if self.num_agents != 1:
            return getattr(eval_env, "allocation_pool", None) if eval_env is not None else None
        choices_raw = resource_choices
        if choices_raw is None or str(choices_raw).strip() == "":
            choices_raw = getattr(self.all_args, "gpu_stage1_eval_single_m_choices", None)
        if choices_raw is None or str(choices_raw).strip() == "":
            return getattr(eval_env, "allocation_pool", None) if eval_env is not None else None
        choices = [int(item.strip()) for item in str(choices_raw).split(",") if item.strip()]
        if not choices:
            return getattr(eval_env, "allocation_pool", None) if eval_env is not None else None
        invalid = [m for m in choices if m < 1 or m > int(data_configs.M_total)]
        if invalid:
            raise ValueError(f"N=1 eval resource choices must be in [1, {int(data_configs.M_total)}], got {invalid}.")
        return torch.as_tensor([[m] for m in sorted(set(choices))], device=self.device, dtype=torch.float32)

    def _eval_resource_choices_for_log(self, resource_choices=None):
        choices_raw = resource_choices
        if choices_raw is None or str(choices_raw).strip() == "":
            choices_raw = getattr(self.all_args, "gpu_stage1_eval_single_m_choices", None)
        return None if choices_raw is None or str(choices_raw).strip() == "" else str(choices_raw)

    def _sample_train_t_breaks(self):
        choices_raw = getattr(self.all_args, "gpu_stage1_train_t_break_choices", None)
        if not choices_raw:
            return None
        choices = [int(item.strip()) for item in str(choices_raw).split(",") if item.strip()]
        if not choices:
            return None
        choices_t = torch.as_tensor(choices, device=self.device, dtype=torch.long)
        probs_raw = getattr(self.all_args, "gpu_stage1_train_t_break_probs", None)
        if probs_raw:
            probs = torch.as_tensor(
                [float(item.strip()) for item in str(probs_raw).split(",") if item.strip()],
                device=self.device,
                dtype=torch.float32,
            )
            if probs.numel() != choices_t.numel():
                raise ValueError(
                    "--gpu_stage1_train_t_break_probs must have the same length as "
                    "--gpu_stage1_train_t_break_choices."
                )
            probs = probs / probs.sum().clamp_min(1e-8)
            indices = torch.multinomial(
                probs,
                num_samples=self.batch_size,
                replacement=True,
                generator=self.envs.generator,
            )
        else:
            indices = torch.randint(
                choices_t.numel(),
                (self.batch_size,),
                device=self.device,
                generator=self.envs.generator,
                dtype=torch.long,
            )
        return choices_t.index_select(0, indices)

    def log_train(self, train_infos, total_num_steps):
        for agent_id, train_info in enumerate(train_infos):
            train_info["average_step_rewards"] = _scalar(self.buffer[agent_id].rewards.mean())
            for key, value in train_info.items():
                if agent_id in (0, 1):
                    metric = f"GPU Monitor/{key}"
                    value_float = _scalar(value)
                    self.writter.add_scalars(metric, {metric: value_float}, total_num_steps)
                    print(metric, {metric: value_float}, total_num_steps)
                    swanlab.log({metric: value_float})

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                torch.save(
                    self.trainer[agent_id].policy.model.state_dict(),
                    str(self.save_dir / f"model_agent{agent_id}.pt"),
                )
            else:
                torch.save(
                    self.trainer[agent_id].policy.actor.state_dict(),
                    str(self.save_dir / f"actor_agent{agent_id}.pt"),
                )
                torch.save(
                    self.trainer[agent_id].policy.critic.state_dict(),
                    str(self.save_dir / f"critic_agent{agent_id}.pt"),
                )

    def save_latest(self, episode_idx):
        del episode_idx
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                torch.save(
                    self.trainer[agent_id].policy.model.state_dict(),
                    str(self.save_dir / f"model_agent{agent_id}_latest.pt"),
                )
            else:
                torch.save(
                    self.trainer[agent_id].policy.actor.state_dict(),
                    str(self.save_dir / f"actor_agent{agent_id}_latest.pt"),
                )
                torch.save(
                    self.trainer[agent_id].policy.critic.state_dict(),
                    str(self.save_dir / f"critic_agent{agent_id}_latest.pt"),
                )

    def save_budget_best(self, m_value, mode=None):
        budget_dir = self.save_dir / (f"best_{str(mode).upper()}_m{int(m_value)}" if mode else f"best_m{int(m_value)}")
        budget_dir.mkdir(parents=True, exist_ok=True)
        self._save_to_dir(budget_dir)

    def save_named_best(self, name):
        best_dir = self.save_dir / str(name)
        best_dir.mkdir(parents=True, exist_ok=True)
        self._save_to_dir(best_dir)

    def _save_to_dir(self, out_dir):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                torch.save(
                    self.trainer[agent_id].policy.model.state_dict(),
                    str(out_dir / f"model_agent{agent_id}.pt"),
                )
            else:
                torch.save(
                    self.trainer[agent_id].policy.actor.state_dict(),
                    str(out_dir / f"actor_agent{agent_id}.pt"),
                )
                torch.save(
                    self.trainer[agent_id].policy.critic.state_dict(),
                    str(out_dir / f"critic_agent{agent_id}.pt"),
                )

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                state = torch.load(os.path.join(str(self.model_dir), f"model_agent{agent_id}.pt"), map_location=self.device)
                self.policy[agent_id].model.load_state_dict(state)
            else:
                actor_state = torch.load(
                    os.path.join(str(self.model_dir), f"actor_agent{agent_id}.pt"),
                    map_location=self.device,
                )
                critic_state = torch.load(
                    os.path.join(str(self.model_dir), f"critic_agent{agent_id}.pt"),
                    map_location=self.device,
                )
                self.policy[agent_id].actor.load_state_dict(actor_state)
                self.policy[agent_id].critic.load_state_dict(critic_state)

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
