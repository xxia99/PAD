import gym
import numpy as np
import torch
from gym import spaces

from envs.env_utils import data_configs
from envs.env_utils.state_action_space import all_qd
from envs.om_env_wrappers import MultiDiscrete
from runners.separated.gpu_pad_evaluator import (
    _alpha_to_allocation_torch,
    _make_iot_bootstrap_pools,
    _make_synthetic_pools,
    _make_tep_bootstrap_pool,
    _region_statistics,
    _sample_synthetic_step,
    _sample_tep_step,
    _update_qcusum,
)


class GPUOnlinePlannerEnv:
    """Batched Stage II planner environment resident on one GPU."""

    def __init__(self, all_args, device, batch_size=None):
        self.all_args = all_args
        self.device = device
        self.batch_size = int(batch_size or all_args.n_rollout_threads)
        self.num_regions = int(data_configs.agent_num)
        self.M_total = int(data_configs.M_total)
        self.M_list = list(data_configs.M)
        self.M_max = int(data_configs.M_max)
        self.region_slices = data_configs.region_slices
        self.d = int(data_configs.d)
        self.l = int(data_configs.l)
        self.m_total = int(data_configs.m_total)
        self.t_break = int(getattr(all_args, "gpu_stage2_train_t_break", 50))
        self.train_horizon = int(getattr(all_args, "gpu_stage2_train_horizon", 2000))
        self.region_aggregator = getattr(all_args, "planner_region_aggregator", "max")
        self.region_topk = int(getattr(all_args, "planner_region_topk", 3) or 3)

        self.detector_obs_dim = self.M_max * 2 + 1
        self.planner_obs_dim = 2 * self.num_regions
        self.detector_obs_dims = [self.detector_obs_dim for _ in range(self.num_regions)]
        self.detector_action_dims = list(self.M_list)

        self.observation_space = [
            spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.detector_obs_dim,), dtype=np.float32)
            for _ in range(self.num_regions)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.detector_obs_dim * self.num_regions,),
                dtype=np.float32,
            )
            for _ in range(self.num_regions)
        ]
        self.action_space = [MultiDiscrete([[0, 1]] * action_dim) for action_dim in self.detector_action_dims]
        self.planner_observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.planner_obs_dim,),
            dtype=np.float32,
        )
        self.planner_action_space = MultiDiscrete([[0, 1]] * int(all_args.planner_alpha_bins))

        self.upper_bounds = torch.as_tensor(self.M_list, device=self.device, dtype=torch.long)
        self.qd_total = torch.as_tensor(
            all_qd(self.M_total, data_configs.train_in_data),
            device=self.device,
            dtype=torch.float32,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(_resolve_seed(getattr(all_args, "seed", 1), default_seed=1))
        if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_pi":
            self.in_pool, self.out_pool = _make_iot_bootstrap_pools(
                float(data_configs.delta),
                pool_size=2000,
                seed=_resolve_seed(getattr(all_args, "seed", 1), default_seed=1),
                device=self.device,
            )
        elif getattr(data_configs, "eval_data_mode", "synthetic") == "eval_tep":
            self.in_pool = _make_tep_bootstrap_pool("train", self.device)
            self.out_pool = None
        else:
            self.in_pool, self.out_pool = _make_synthetic_pools(
                float(data_configs.delta),
                pool_size=2000,
                seed=_resolve_seed(getattr(all_args, "seed", 1), default_seed=1),
                device=self.device,
            )
        self.reset()

    def reset(self, broken_streams=None, batch_size=None):
        if batch_size is not None and int(batch_size) != self.batch_size:
            self.batch_size = int(batch_size)

        B = self.batch_size
        self.step_index = 0
        self.absolute_step = 0

        if broken_streams is None:
            broken_streams = torch.arange(B, device=self.device, dtype=torch.long) % self.M_total
        elif torch.is_tensor(broken_streams):
            broken_streams = broken_streams.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            broken_streams = torch.as_tensor(broken_streams, device=self.device, dtype=torch.long).reshape(-1)
        if broken_streams.numel() == 1 and B > 1:
            broken_streams = broken_streams.expand(B)
        if broken_streams.numel() != B:
            raise ValueError(f"Expected {B} broken streams, got {broken_streams.numel()}.")
        self.broken_streams = broken_streams

        self.stats = torch.zeros((B, self.M_total), device=self.device, dtype=torch.float32)
        self.observed_mask = torch.ones((B, self.M_total), device=self.device, dtype=torch.bool)
        self.local_p = torch.zeros((B, self.M_total, self.d - 1, 2), device=self.device, dtype=torch.float32)
        self.local_n = torch.zeros_like(self.local_p)
        self.alpha_counts = torch.ones((B, self.M_total, self.d), device=self.device, dtype=torch.float32) / float(self.d)
        self.prev_usage = torch.as_tensor(data_configs.m, device=self.device, dtype=torch.float32).unsqueeze(0).expand(
            B,
            -1,
        ).clone()
        self.prev_local_actions = torch.ones((B, self.num_regions, self.M_max), device=self.device, dtype=torch.float32)
        return self.planner_obs()

    def region_statistics(self):
        return _region_statistics(self.stats, self.region_slices, self.region_aggregator, self.region_topk)

    def planner_obs(self):
        return torch.cat([self.prev_usage, self.region_statistics()], dim=1)

    def detector_obs(self, allocations):
        allocations = allocations.to(device=self.device, dtype=torch.float32)
        B = allocations.shape[0]
        obs = torch.zeros((B, self.num_regions, self.detector_obs_dim), device=self.device, dtype=torch.float32)
        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            obs[:, region_id, 0] = allocations[:, region_id]
            obs[:, region_id, 1 : 1 + region_len] = self.stats[:, region_slice]
            obs[
                :,
                region_id,
                1 + self.M_max : 1 + self.M_max + region_len,
            ] = self.prev_local_actions[:, region_id, :region_len]
        return obs

    def alpha_to_allocation(self, alpha, region_stats=None):
        if region_stats is None:
            region_stats = self.region_statistics()
        return _alpha_to_allocation_torch(alpha, region_stats, self.m_total, self.upper_bounds)

    def global_stat(self):
        return torch.topk(self.stats, k=self.l, dim=1).values.sum(dim=1)

    def step(self, allocations, detector_actions, active_mask=None):
        allocations = allocations.to(device=self.device, dtype=torch.long)
        if active_mask is None:
            active_mask = torch.ones((self.batch_size,), device=self.device, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
            if active_mask.numel() != self.batch_size:
                raise ValueError(f"Expected active_mask shape {(self.batch_size,)}, got {tuple(active_mask.shape)}.")

        inactive_mask = ~active_mask
        restore_inactive = bool(inactive_mask.any())
        if restore_inactive:
            old_stats = self.stats[inactive_mask].clone()
            old_observed_mask = self.observed_mask[inactive_mask].clone()
            old_local_p = self.local_p[inactive_mask].clone()
            old_local_n = self.local_n[inactive_mask].clone()
            old_alpha_counts = self.alpha_counts[inactive_mask].clone()
            old_prev_usage = self.prev_usage[inactive_mask].clone()
            old_prev_local_actions = self.prev_local_actions[inactive_mask].clone()

        global_actions = torch.zeros((self.batch_size, self.M_total), device=self.device, dtype=torch.long)
        next_local_actions = torch.zeros_like(self.prev_local_actions)
        usage = torch.zeros((self.batch_size, self.num_regions), device=self.device, dtype=torch.float32)

        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            action_i = detector_actions[region_id].to(device=self.device).long()[:, :region_len]
            global_actions[:, region_slice] = action_i
            next_local_actions[:, region_id, :region_len] = action_i.float()
            usage[:, region_id] = action_i.sum(dim=1).float()

        if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_tep":
            data_t = _sample_tep_step(
                self.in_pool,
                self.broken_streams,
                int(self.absolute_step),
                int(self.t_break),
                float(data_configs.delta),
                self.generator,
            )
        else:
            data_t = _sample_synthetic_step(
                self.in_pool,
                self.out_pool,
                self.broken_streams,
                int(self.absolute_step),
                int(self.t_break),
                self.generator,
            )
        self.stats, self.local_p, self.local_n, self.alpha_counts = _update_qcusum(
            data_t,
            self.qd_total,
            self.observed_mask,
            self.local_p,
            self.local_n,
            self.alpha_counts,
            generator=self.generator,
        )

        self.observed_mask = global_actions.bool()
        self.prev_usage = usage
        self.prev_local_actions = next_local_actions
        if restore_inactive:
            self.stats[inactive_mask] = old_stats
            self.observed_mask[inactive_mask] = old_observed_mask
            self.local_p[inactive_mask] = old_local_p
            self.local_n[inactive_mask] = old_local_n
            self.alpha_counts[inactive_mask] = old_alpha_counts
            self.prev_usage[inactive_mask] = old_prev_usage
            self.prev_local_actions[inactive_mask] = old_prev_local_actions
        self.step_index += 1
        self.absolute_step += 1
        return self.planner_obs(), usage


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)
