import gym
import numpy as np
import torch
from gym import spaces

from envs.env_utils import data_configs
from envs.env_utils.data_loader import _synthetic_in_out_data
from envs.env_utils.state_action_space import all_qd
from envs.om_env_wrappers import MultiDiscrete
from runners.separated.gpu_pad_evaluator import _make_tep_bootstrap_pool, _sample_synthetic_step, _sample_tep_step, _update_qcusum
from runners.separated.gpu_pad_evaluator import _load_iot_batch, _make_iot_bootstrap_pools


class GPUOnlineMonitoringEnv:
    """Batched Stage I monitoring environment that lives entirely on one GPU."""

    def __init__(self, all_args, device, batch_size=None):
        self.all_args = all_args
        self.device = device
        self.batch_size = int(batch_size or all_args.n_rollout_threads)
        self.num_agent = int(data_configs.agent_num)
        self.M_total = int(data_configs.M_total)
        self.M_list = list(data_configs.M)
        self.M_max = int(data_configs.M_max)
        self.region_slices = data_configs.region_slices
        self.d = int(data_configs.d)
        self.l = int(data_configs.l)
        self.t_break = int(getattr(all_args, "gpu_stage1_train_t_break", 50))
        self.random_allocations = bool(getattr(all_args, "gpu_stage1_random_allocations", False))
        self.balanced_allocations = bool(getattr(all_args, "gpu_stage1_balanced_allocations", False))
        self.resource_upper_bound = int(getattr(all_args, "gpu_stage1_resource_upper_bound", 10))
        self.obs_dim = self.M_max * 2 + 1
        self.obs_dims = [self.obs_dim for _ in range(self.num_agent)]
        self.action_dims = list(self.M_list)

        self.observation_space = [
            spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.obs_dim * self.num_agent,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]
        self.action_space = [MultiDiscrete([[0, 1]] * action_dim) for action_dim in self.action_dims]

        self.qd_total = torch.as_tensor(
            all_qd(self.M_total, data_configs.train_in_data),
            device=self.device,
            dtype=torch.float32,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(_resolve_seed(getattr(all_args, "seed", 1), default_seed=1))
        self.allocation_pool = self._build_allocation_pool() if self.random_allocations else None
        self.resource_sample_calls = 0
        self.data_batch = None
        train_pool_seed = getattr(all_args, "gpu_stage1_train_pool_seed", None)
        self._build_pools(float(data_configs.delta), seed=train_pool_seed)
        self.reset()

    def _build_pools(self, delta, seed=None):
        self.data_batch = None
        if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_pi":
            self.in_pool, self.out_pool = _make_iot_bootstrap_pools(
                delta,
                pool_size=2000,
                seed=_resolve_seed(getattr(self.all_args, "seed", 1), default_seed=1),
                device=self.device,
            )
            return
        if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_tep":
            self.in_pool = _make_tep_bootstrap_pool("train", self.device)
            self.out_pool = None
            return
        rng = np.random.default_rng(seed) if seed is not None else None
        in_pool_np, out_pool_np = _synthetic_in_out_data(delta=delta, sample_size=2000, rng=rng)
        self.in_pool = torch.as_tensor(in_pool_np, device=self.device, dtype=torch.float32)
        self.out_pool = torch.as_tensor(out_pool_np, device=self.device, dtype=torch.float32)

    def set_iot_eval_batch(self, delta, start_index, batch_size, max_horizon):
        data_batch, _, t_break, horizon = _load_iot_batch(
            delta,
            start_index,
            batch_size,
            max_horizon,
            self.device,
        )
        self.data_batch = data_batch
        self.t_break = int(t_break)
        return int(horizon)

    def reset(
        self,
        broken_streams=None,
        start_step=None,
        episode_steps=None,
        batch_size=None,
        resources=None,
        t_breaks=None,
    ):
        if batch_size is not None and int(batch_size) != self.batch_size:
            self.batch_size = int(batch_size)

        B = self.batch_size
        self.step_index = 0
        self.absolute_step = int(self.t_break if start_step is None else start_step)
        self.episode_steps = int(getattr(self.all_args, "episode_length", 30) if episode_steps is None else episode_steps)
        if t_breaks is None:
            self.t_breaks = torch.full((B,), int(self.t_break), device=self.device, dtype=torch.long)
        elif torch.is_tensor(t_breaks):
            self.t_breaks = t_breaks.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            self.t_breaks = torch.as_tensor(t_breaks, device=self.device, dtype=torch.long).reshape(-1)
        if self.t_breaks.numel() == 1 and B > 1:
            self.t_breaks = self.t_breaks.expand(B)
        if self.t_breaks.numel() != B:
            raise ValueError(f"Expected {B} change points, got {self.t_breaks.numel()}.")

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
        self.prev_local_actions = torch.ones((B, self.num_agent, self.M_max), device=self.device, dtype=torch.float32)
        self.resources = self._sample_resources(B, resources)
        self.hit_count = torch.zeros((B,), device=self.device, dtype=torch.float32)
        self.post_count = torch.zeros((B,), device=self.device, dtype=torch.float32)
        return self.get_obs()

    def _build_allocation_pool(self):
        if self.num_agent == 1:
            max_resource = max(1, int(self.M_list[0]))
            choice_arg = getattr(self.all_args, "gpu_stage1_single_m_choices", None)
            if choice_arg:
                choices = [int(item.strip()) for item in str(choice_arg).split(",") if item.strip()]
                choices = sorted(set(choices))
                if not choices:
                    raise ValueError("--gpu_stage1_single_m_choices did not contain any valid integers.")
                invalid = [m for m in choices if m < 1 or m > max_resource]
                if invalid:
                    raise ValueError(
                        f"N=1 Stage I resource choices must be in [1, {max_resource}], got {invalid}."
                    )
                allocations = [[m] for m in choices]
            else:
                allocations = [[m] for m in range(1, max_resource + 1)]
            return torch.as_tensor(allocations, device=self.device, dtype=torch.float32)

        total = int(data_configs.m_total)
        caps = [min(max(0, self.resource_upper_bound), int(region_m), total) for region_m in self.M_list]
        allocations = _bounded_allocations(total, caps)
        if not allocations:
            raise ValueError(
                f"No feasible Stage I allocations for m_total={total}, caps={caps}. "
                "Increase --gpu_stage1_resource_upper_bound or check M/m/N."
            )
        return torch.as_tensor(allocations, device=self.device, dtype=torch.float32)

    def _sample_resources(self, batch_size, resources=None):
        if resources is not None:
            if torch.is_tensor(resources):
                resources_t = resources.to(device=self.device, dtype=torch.float32)
            else:
                resources_t = torch.as_tensor(resources, device=self.device, dtype=torch.float32)
            if resources_t.ndim == 1:
                resources_t = resources_t.unsqueeze(0)
            if resources_t.shape[0] == 1 and batch_size > 1:
                resources_t = resources_t.expand(batch_size, -1)
            if tuple(resources_t.shape) != (batch_size, self.num_agent):
                raise ValueError(f"Expected resources shape {(batch_size, self.num_agent)}, got {tuple(resources_t.shape)}.")
            return resources_t.clone()

        if self.random_allocations:
            if self.balanced_allocations:
                indices = (
                    torch.arange(batch_size, device=self.device, dtype=torch.long)
                    + int(self.resource_sample_calls) * int(batch_size)
                ) % int(self.allocation_pool.shape[0])
                self.resource_sample_calls += 1
            else:
                indices = torch.randint(
                    self.allocation_pool.shape[0],
                    (batch_size,),
                    device=self.device,
                    generator=self.generator,
                )
            return self.allocation_pool.index_select(0, indices).clone()

        return torch.as_tensor(data_configs.m, device=self.device, dtype=torch.float32).unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()

    def get_obs(self):
        B = self.batch_size
        obs = torch.zeros((B, self.num_agent, self.obs_dim), device=self.device, dtype=torch.float32)
        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            obs[:, region_id, 0] = self.resources[:, region_id]
            obs[:, region_id, 1 : 1 + region_len] = self.stats[:, region_slice]
            obs[
                :,
                region_id,
                1 + self.M_max : 1 + self.M_max + region_len,
            ] = self.prev_local_actions[:, region_id, :region_len]
        share_obs = obs.reshape(B, self.num_agent * self.obs_dim)
        return obs, share_obs

    def _global_actions(self, actions):
        B = self.batch_size
        global_actions = torch.zeros((B, self.M_total), device=self.device, dtype=torch.long)
        next_local_actions = torch.zeros((B, self.num_agent, self.M_max), device=self.device, dtype=torch.float32)
        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            action_i = actions[region_id].to(device=self.device).long()
            action_i = action_i[:, :region_len]
            global_actions[:, region_slice] = action_i
            next_local_actions[:, region_id, :region_len] = action_i.float()
        return global_actions, next_local_actions

    def step(self, actions):
        global_actions, next_local_actions = self._global_actions(actions)
        if self.data_batch is not None:
            data_t = self.data_batch[:, int(self.step_index), :]
        else:
            if bool(torch.all(self.t_breaks == self.t_breaks[0])):
                if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_tep":
                    data_t = _sample_tep_step(
                        self.in_pool,
                        self.broken_streams,
                        int(self.absolute_step),
                        int(self.t_breaks[0].item()),
                        float(data_configs.delta),
                        self.generator,
                    )
                else:
                    data_t = _sample_synthetic_step(
                        self.in_pool,
                        self.out_pool,
                        self.broken_streams,
                        int(self.absolute_step),
                        int(self.t_breaks[0].item()),
                        self.generator,
                    )
            else:
                data_t = self._sample_synthetic_step_with_variable_breaks()
        self.stats, self.local_p, self.local_n, self.alpha_counts = _update_qcusum(
            data_t,
            self.qd_total,
            self.observed_mask,
            self.local_p,
            self.local_n,
            self.alpha_counts,
            generator=self.generator,
        )

        rows = torch.arange(self.batch_size, device=self.device)
        selected_broken = global_actions[rows, self.broken_streams].bool()
        post_change = self.absolute_step >= self.t_breaks + int(data_configs.epoch_buffer)
        if bool(post_change.any()):
            self.hit_count += selected_broken.float() * post_change.float()
            self.post_count += post_change.float()
            reward = torch.where(selected_broken, torch.ones_like(self.hit_count), -torch.ones_like(self.hit_count))
            reward = torch.where(post_change, reward, torch.zeros_like(reward))
        else:
            reward = torch.zeros_like(self.hit_count)

        self.observed_mask = global_actions.bool()
        self.prev_local_actions = next_local_actions
        self.step_index += 1
        self.absolute_step += 1

        obs, share_obs = self.get_obs()
        rewards = reward[:, None, None].expand(-1, self.num_agent, -1).contiguous()
        dones = torch.zeros((self.batch_size, self.num_agent), device=self.device, dtype=torch.bool)
        return obs, share_obs, rewards, dones

    def global_stat(self):
        return torch.topk(self.stats, k=self.l, dim=1).values.sum(dim=1)

    def _sample_synthetic_step_with_variable_breaks(self):
        batch_size, total_streams = self.batch_size, self.M_total
        if getattr(data_configs, "eval_data_mode", "synthetic") == "eval_tep":
            row_idx = torch.randint(
                self.in_pool.shape[0],
                (batch_size,),
                device=self.device,
                generator=self.generator,
            )
            data_t = self.in_pool.index_select(0, row_idx).clone()
            shifted = self.absolute_step >= self.t_breaks
            if bool(shifted.any()):
                rows = torch.arange(batch_size, device=self.device)[shifted]
                streams = self.broken_streams[shifted]
                data_t[rows, streams] += float(data_configs.delta)
            return data_t

        cols = torch.arange(total_streams, device=self.device).unsqueeze(0).expand(batch_size, total_streams)
        in_idx = torch.randint(
            self.in_pool.shape[0],
            (batch_size, total_streams),
            device=self.device,
            generator=self.generator,
        )
        data_t = self.in_pool[in_idx, cols]
        shifted = self.absolute_step >= self.t_breaks
        if bool(shifted.any()) and self.out_pool is not None:
            rows = torch.arange(batch_size, device=self.device)[shifted]
            streams = self.broken_streams[shifted]
            out_idx = torch.randint(
                self.out_pool.shape[0],
                (int(rows.numel()),),
                device=self.device,
                generator=self.generator,
            )
            data_t[rows, streams] = self.out_pool[out_idx, streams]
        return data_t


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


def _bounded_allocations(total, caps):
    caps = [int(c) for c in caps]
    suffix_caps = [0] * (len(caps) + 1)
    for idx in range(len(caps) - 1, -1, -1):
        suffix_caps[idx] = suffix_caps[idx + 1] + caps[idx]

    allocations = []
    current = [0] * len(caps)

    def rec(pos, remaining):
        if pos == len(caps):
            if remaining == 0:
                allocations.append(current.copy())
            return
        low = max(0, remaining - suffix_caps[pos + 1])
        high = min(caps[pos], remaining)
        for value in range(low, high + 1):
            current[pos] = value
            rec(pos + 1, remaining - value)

    rec(0, int(total))
    return allocations
