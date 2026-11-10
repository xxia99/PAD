import time

import torch

from envs.env_utils import data_configs
from envs.env_utils.state_action_space import all_qd
from runners.separated.gpu_pad_evaluator import _make_synthetic_pools, _update_qcusum


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


class GPUDetectorResponseSurrogate:
    """Smooth detector-response table for low-variance planner pretraining.

    The table is built from the frozen Stage I detector actors. It approximates
    the expected positive increment, next statistic, and alarm probability of
    each region's max CUSUM statistic as a function of region, broken local
    stream, local allocation, time since change, and current statistic bin.
    Planner eval still uses the true detector actors.
    """

    def __init__(self, all_args, device, detector_policies):
        self.all_args = all_args
        self.device = device
        self.detector_policies = detector_policies
        self.num_regions = int(data_configs.agent_num)
        self.M_total = int(data_configs.M_total)
        self.M_max = int(data_configs.M_max)
        self.M_list = list(data_configs.M)
        self.region_slices = data_configs.region_slices
        self.d = int(data_configs.d)
        self.t_break = int(getattr(all_args, "gpu_stage2_train_t_break", 50))
        self.reps = int(getattr(all_args, "gpu_stage2_surrogate_reps", 256))
        self.horizon = int(getattr(all_args, "gpu_stage2_surrogate_horizon", 300))
        self.pool_size = int(getattr(all_args, "gpu_stage2_surrogate_pool_size", 2000))
        self.gain = float(getattr(all_args, "gpu_stage2_surrogate_gain", 1.0))
        self.baseline_increment = float(getattr(all_args, "gpu_stage2_surrogate_baseline_increment", 0.0))
        self.region_aggregator = str(getattr(all_args, "planner_region_aggregator", "max") or "max").lower()
        self.region_topk = int(getattr(all_args, "planner_region_topk", 3) or 3)
        self.stat_bins = max(1, int(getattr(all_args, "gpu_stage2_surrogate_stat_bins", 1) or 1))
        stat_max_arg = getattr(all_args, "gpu_stage2_surrogate_stat_max", None)
        self.stat_max = float(
            stat_max_arg if stat_max_arg is not None else getattr(all_args, "planner_alarm_threshold", 50.0)
        )
        if self.stat_bins > 1:
            self.stat_bin_edges = torch.linspace(
                0.0,
                self.stat_max,
                self.stat_bins + 1,
                device=self.device,
                dtype=torch.float32,
            )[1:-1]
        else:
            self.stat_bin_edges = torch.empty((0,), device=self.device, dtype=torch.float32)
        delta_arg = getattr(all_args, "gpu_stage2_surrogate_delta", None)
        self.delta = float(delta_arg if delta_arg is not None else getattr(all_args, "planner_eval_delta", data_configs.delta))
        self.seed = _resolve_seed(getattr(all_args, "seed", 1), default_seed=1) + 7100

        self.increment_table = torch.zeros(
            (self.num_regions, self.M_max, self.M_max + 1, self.horizon, self.stat_bins),
            device=self.device,
            dtype=torch.float32,
        )
        self.next_stat_table = torch.zeros_like(self.increment_table)
        self.alarm_prob_table = torch.zeros_like(self.increment_table)
        self.valid_local_stream = torch.zeros((self.num_regions, self.M_max), device=self.device, dtype=torch.bool)
        self.global_to_region = torch.zeros((self.M_total,), device=self.device, dtype=torch.long)
        self.global_to_local = torch.zeros((self.M_total,), device=self.device, dtype=torch.long)
        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            self.valid_local_stream[region_id, :region_len] = True
            self.global_to_region[region_slice] = region_id
            self.global_to_local[region_slice] = torch.arange(region_len, device=self.device, dtype=torch.long)

        self._build_table()

    @torch.inference_mode()
    def _build_table(self):
        start = time.time()
        qd_total = torch.as_tensor(
            all_qd(self.M_total, data_configs.train_in_data),
            device=self.device,
            dtype=torch.float32,
        )
        in_pool, out_pool = _make_synthetic_pools(self.delta, self.pool_size, self.seed, self.device)

        for policy in self.detector_policies:
            policy.actor.eval()

        table_max = 0.0
        for region_id, region_slice in enumerate(self.region_slices):
            region_len = region_slice.stop - region_slice.start
            if region_len <= 0:
                continue

            local_streams = torch.arange(region_len, device=self.device, dtype=torch.long).repeat_interleave(
                (region_len + 1) * self.reps
            )
            allocs = torch.arange(region_len + 1, device=self.device, dtype=torch.long).repeat_interleave(self.reps)
            allocs = allocs.repeat(region_len)
            batch_size = int(local_streams.numel())

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + region_id * 9973)

            in_local = in_pool[:, region_slice]
            out_local = out_pool[:, region_slice]
            qd_local = qd_total[:, region_slice]

            stats = torch.zeros((batch_size, region_len), device=self.device, dtype=torch.float32)
            observed_mask = torch.ones((batch_size, region_len), device=self.device, dtype=torch.bool)
            local_p = torch.zeros((batch_size, region_len, self.d - 1, 2), device=self.device, dtype=torch.float32)
            local_n = torch.zeros_like(local_p)
            alpha_counts = torch.ones(
                (batch_size, region_len, self.d),
                device=self.device,
                dtype=torch.float32,
            ) / float(self.d)
            prev_local_actions = torch.ones((batch_size, self.M_max), device=self.device, dtype=torch.float32)
            rnn = torch.zeros(
                (batch_size, self.all_args.recurrent_N, self.all_args.hidden_size),
                device=self.device,
                dtype=torch.float32,
            )
            masks = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)

            for _ in range(self.t_break):
                actions, rnn = self._detector_actions(region_id, stats, prev_local_actions, allocs, rnn, masks)
                data_t = self._sample_local_step(in_local, None, local_streams, generator)
                stats, local_p, local_n, alpha_counts = _update_qcusum(
                    data_t,
                    qd_local,
                    observed_mask,
                    local_p,
                    local_n,
                    alpha_counts,
                    generator=generator,
                )
                observed_mask = actions[:, :region_len].bool()
                prev_local_actions[:, :region_len] = actions[:, :region_len].float()

            inc_sum = torch.zeros(
                (region_len, region_len + 1, self.horizon, self.stat_bins),
                device=self.device,
                dtype=torch.float32,
            )
            next_sum = torch.zeros_like(inc_sum)
            alarm_sum = torch.zeros_like(inc_sum)
            counts = torch.zeros_like(inc_sum)
            fallback_inc_sum = torch.zeros((region_len, region_len + 1, self.horizon), device=self.device)
            fallback_next_sum = torch.zeros_like(fallback_inc_sum)
            fallback_alarm_sum = torch.zeros_like(fallback_inc_sum)
            for t_idx in range(self.horizon):
                prev_score_flat = self._aggregate_local_stats(stats)
                actions, rnn = self._detector_actions(region_id, stats, prev_local_actions, allocs, rnn, masks)
                data_t = self._sample_local_step(in_local, out_local, local_streams, generator)
                stats, local_p, local_n, alpha_counts = _update_qcusum(
                    data_t,
                    qd_local,
                    observed_mask,
                    local_p,
                    local_n,
                    alpha_counts,
                    generator=generator,
                )
                observed_mask = actions[:, :region_len].bool()
                prev_local_actions[:, :region_len] = actions[:, :region_len].float()
                next_score_flat = self._aggregate_local_stats(stats)

                prev_score = prev_score_flat.reshape(region_len, region_len + 1, self.reps)
                next_score = next_score_flat.reshape(region_len, region_len + 1, self.reps)
                increment = (next_score - prev_score).clamp_min(0.0)
                alarm = (next_score >= float(getattr(self.all_args, "planner_alarm_threshold", 50.0))).float()
                bin_ids = self._stat_bin_ids(prev_score)

                fallback_inc_sum[:, :, t_idx] = increment.mean(dim=-1)
                fallback_next_sum[:, :, t_idx] = next_score.mean(dim=-1)
                fallback_alarm_sum[:, :, t_idx] = alarm.mean(dim=-1)
                for bin_id in range(self.stat_bins):
                    mask = (bin_ids == bin_id).to(dtype=torch.float32)
                    count = mask.sum(dim=-1)
                    counts[:, :, t_idx, bin_id] = count
                    inc_sum[:, :, t_idx, bin_id] = (increment * mask).sum(dim=-1)
                    next_sum[:, :, t_idx, bin_id] = (next_score * mask).sum(dim=-1)
                    alarm_sum[:, :, t_idx, bin_id] = (alarm * mask).sum(dim=-1)

            fallback_inc = fallback_inc_sum.unsqueeze(-1).expand_as(inc_sum)
            fallback_next = fallback_next_sum.unsqueeze(-1).expand_as(next_sum)
            fallback_alarm = fallback_alarm_sum.unsqueeze(-1).expand_as(alarm_sum)
            increments = torch.where(counts > 0, inc_sum / counts.clamp_min(1.0), fallback_inc)
            next_stats = torch.where(counts > 0, next_sum / counts.clamp_min(1.0), fallback_next)
            alarm_probs = torch.where(counts > 0, alarm_sum / counts.clamp_min(1.0), fallback_alarm)
            increments[:, 0, :, :] = 0.0
            self.increment_table[region_id, :region_len, : region_len + 1, :, :] = increments.clamp_min_(0.0)
            self.next_stat_table[region_id, :region_len, : region_len + 1, :, :] = next_stats.clamp_min_(0.0)
            self.alarm_prob_table[region_id, :region_len, : region_len + 1, :, :] = alarm_probs.clamp_(0.0, 1.0)
            table_max = max(table_max, float(next_stats.max().detach().cpu().item()))

        elapsed = time.time() - start
        print(
            "GPU detector surrogate table:",
            {
                "delta": self.delta,
                "reps": self.reps,
                "horizon": self.horizon,
                "pool_size": self.pool_size,
                "gain": self.gain,
                "stat_bins": self.stat_bins,
                "stat_max": self.stat_max,
                "region_aggregator": self.region_aggregator,
                "region_topk": self.region_topk,
                "max_curve_stat": round(table_max, 4),
                "seconds": round(elapsed, 3),
            },
        )

    def _aggregate_local_stats(self, stats):
        if self.region_aggregator == "max":
            return stats.max(dim=1).values
        if self.region_aggregator == "mean":
            return stats.mean(dim=1)
        if self.region_aggregator == "sum":
            return stats.sum(dim=1)
        if self.region_aggregator == "topk":
            k = min(max(1, int(self.region_topk)), stats.shape[1])
            return torch.topk(stats, k=k, dim=1).values.mean(dim=1)
        raise ValueError(f"Unknown planner_region_aggregator={self.region_aggregator!r}.")

    def _stat_bin_ids(self, values):
        if self.stat_bins <= 1:
            return torch.zeros_like(values, dtype=torch.long)
        values = values.clamp_min(0.0)
        return torch.bucketize(values, self.stat_bin_edges).clamp_(0, self.stat_bins - 1)

    def _detector_actions(self, region_id, stats, prev_local_actions, allocs, rnn, masks):
        region_len = self.M_list[region_id]
        obs = torch.zeros((stats.shape[0], 2 * self.M_max + 1), device=self.device, dtype=torch.float32)
        obs[:, 0] = allocs.float()
        obs[:, 1 : 1 + region_len] = stats
        obs[:, 1 + self.M_max : 1 + self.M_max + region_len] = prev_local_actions[:, :region_len]
        actions, _, next_rnn = self.detector_policies[region_id].actor(
            obs,
            rnn,
            masks,
            None,
            deterministic=True,
            force_m=True,
            m=allocs,
        )
        return actions.long(), next_rnn

    def _sample_local_step(self, in_pool, out_pool, local_streams, generator):
        batch_size, region_len = local_streams.shape[0], in_pool.shape[1]
        cols = torch.arange(region_len, device=self.device).unsqueeze(0).expand(batch_size, region_len)
        in_idx = torch.randint(in_pool.shape[0], (batch_size, region_len), device=self.device, generator=generator)
        data_t = in_pool[in_idx, cols]
        if out_pool is not None:
            rows = torch.arange(batch_size, device=self.device)
            out_idx = torch.randint(out_pool.shape[0], (batch_size,), device=self.device, generator=generator)
            data_t[rows, local_streams] = out_pool[out_idx, local_streams]
        return data_t

    def response_increment(self, broken_streams, allocations, step, current_region_stats=None):
        if step < self.t_break:
            return torch.zeros((broken_streams.shape[0],), device=self.device, dtype=torch.float32)
        t_idx = min(max(int(step) - self.t_break, 0), self.horizon - 1)
        broken_regions = self.global_to_region[broken_streams]
        broken_locals = self.global_to_local[broken_streams]
        rows = torch.arange(broken_streams.shape[0], device=self.device)
        local_allocs = allocations[rows, broken_regions].long().clamp_(0, self.M_max)
        if current_region_stats is None:
            stat_bins = torch.zeros_like(broken_regions)
        else:
            current_stats = current_region_stats[rows, broken_regions]
            stat_bins = self._stat_bin_ids(current_stats)
        inc = self.increment_table[broken_regions, broken_locals, local_allocs, t_idx, stat_bins]
        return inc * self.gain
