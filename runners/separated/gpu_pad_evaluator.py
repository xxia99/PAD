import pickle

import numpy as np
import torch
import torch.nn.functional as F

from envs.env_utils import data_configs
from envs.env_utils.data_loader import _synthetic_in_out_data
from envs.env_utils.state_action_space import all_qd
from envs.env_utils.tep_data import load_standardized_tep_arrays
from runners.separated.parallel_detectors import ParallelDetectorExecutor


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


def _torch_bounded_hare_niemeyer(q, total_resources, upper_bounds):
    upper_bounds = upper_bounds.to(device=q.device, dtype=torch.long)
    floors = torch.floor(q).long()
    allocations = torch.minimum(floors, upper_bounds.unsqueeze(0))
    remaining = int(total_resources) - allocations.sum(dim=1)
    remainders = q - torch.floor(q)
    remainders = remainders.masked_fill(allocations >= upper_bounds.unsqueeze(0), -1e20)

    rows = torch.arange(q.shape[0], device=q.device)
    for _ in range(int(total_resources)):
        active = remaining > 0
        if not bool(active.any()):
            break
        best = torch.argmax(remainders, dim=1)
        can_add = active & (allocations[rows, best] < upper_bounds[best])
        if not bool(can_add.any()):
            break
        add_rows = rows[can_add]
        add_cols = best[can_add]
        allocations[add_rows, add_cols] += 1
        remaining[can_add] -= 1
        saturated = allocations[add_rows, add_cols] >= upper_bounds[add_cols]
        if bool(saturated.any()):
            remainders[add_rows[saturated], add_cols[saturated]] = -1e20
    return allocations


def _alpha_to_allocation_torch(alpha, region_stats, total_resources, upper_bounds):
    num_regions = region_stats.shape[1]
    fair = (float(total_resources) / num_regions) * torch.ones_like(region_stats)
    denom = region_stats.sum(dim=1, keepdim=True)
    greedy = region_stats / denom.clamp_min(1e-8) * float(total_resources)
    greedy = torch.where(denom <= 1e-8, fair, greedy)
    q = alpha * fair + (1.0 - alpha) * greedy
    return _torch_bounded_hare_niemeyer(q, total_resources, upper_bounds)


def _region_statistics(stats, region_slices, aggregator="max", topk=3):
    aggregator = str(aggregator or "max").lower()
    values = []
    for region_slice in region_slices:
        region = stats[:, region_slice]
        if aggregator == "max":
            values.append(region.max(dim=1).values)
        elif aggregator == "mean":
            values.append(region.mean(dim=1))
        elif aggregator == "sum":
            values.append(region.sum(dim=1))
        elif aggregator == "topk":
            k = min(max(1, int(topk)), region.shape[1])
            values.append(torch.topk(region, k=k, dim=1).values.mean(dim=1))
        else:
            raise ValueError(f"Unknown planner_region_aggregator={aggregator!r}.")
    return torch.stack(values, dim=1)


def _stream_to_region_tensor(region_slices, total_streams, device):
    mapping = torch.empty((total_streams,), device=device, dtype=torch.long)
    for region_id, region_slice in enumerate(region_slices):
        mapping[region_slice] = int(region_id)
    return mapping


def _update_qcusum(data_t, qd_total, observed_mask, local_p, local_n, alpha_counts, generator=None):
    d = int(data_configs.d)
    allowance = float(data_configs.allowance)
    device = data_t.device

    interval_idx = (data_t.unsqueeze(1) > qd_total.unsqueeze(0)).sum(dim=1)
    y_actual = F.one_hot(interval_idx.long(), num_classes=d).to(dtype=data_t.dtype)

    theta = alpha_counts / alpha_counts.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sampled_idx = torch.multinomial(theta.reshape(-1, d), num_samples=1, generator=generator).reshape(
        data_t.shape[0],
        data_t.shape[1],
    )
    y_pseudo = F.one_hot(sampled_idx.long(), num_classes=d).to(dtype=data_t.dtype)
    y = torch.where(observed_mask.unsqueeze(-1), y_actual, y_pseudo)

    alpha_new = alpha_counts + torch.where(observed_mask.unsqueeze(-1), y_actual, torch.zeros_like(y_actual))
    an = torch.cumsum(y, dim=-1)[..., :-1]
    ap = 1.0 - an

    g_pos = 1.0 - torch.arange(1, d, device=device, dtype=data_t.dtype) / d
    g_neg = torch.arange(1, d, device=device, dtype=data_t.dtype) / d

    s1p = local_p[..., 0]
    s2p = local_p[..., 1]
    s1n = local_n[..., 0]
    s2n = local_n[..., 1]

    diff_p = s1p - s2p + ap - g_pos
    diff_n = s1n - s2n + an - g_neg
    cp = ((diff_p ** 2) / (s2p + g_pos).clamp_min(1e-8)).sum(dim=-1)
    cn = ((diff_n ** 2) / (s2n + g_neg).clamp_min(1e-8)).sum(dim=-1)

    wp = torch.clamp(cp - allowance, min=0.0)
    wn = torch.clamp(cn - allowance, min=0.0)
    new_stats = torch.maximum(wp, wn)

    next_p = torch.where(observed_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(local_p), local_p)
    next_n = torch.where(observed_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(local_n), local_n)

    update_p = observed_mask & (cp > allowance)
    update_n = observed_mask & (cn > allowance)
    factor_p = torch.where(update_p, (cp - allowance) / cp.clamp_min(1e-8), torch.zeros_like(cp))
    factor_n = torch.where(update_n, (cn - allowance) / cn.clamp_min(1e-8), torch.zeros_like(cn))

    next_p[..., 0] = torch.where(update_p.unsqueeze(-1), (s1p + ap) * factor_p.unsqueeze(-1), next_p[..., 0])
    next_p[..., 1] = torch.where(update_p.unsqueeze(-1), (s2p + g_pos) * factor_p.unsqueeze(-1), next_p[..., 1])
    next_n[..., 0] = torch.where(update_n.unsqueeze(-1), (s1n + an) * factor_n.unsqueeze(-1), next_n[..., 0])
    next_n[..., 1] = torch.where(update_n.unsqueeze(-1), (s2n + g_neg) * factor_n.unsqueeze(-1), next_n[..., 1])

    return new_stats, next_p, next_n, alpha_new


def _make_synthetic_pools(delta, pool_size, seed, device):
    rng = np.random.default_rng(seed)
    in_pool_np, out_pool_np = _synthetic_in_out_data(float(delta), int(pool_size), rng=rng)
    in_pool = torch.as_tensor(in_pool_np, device=device, dtype=torch.float32)
    out_pool = torch.as_tensor(out_pool_np, device=device, dtype=torch.float32)
    return in_pool, out_pool


def _iot_reference_arrays(path=None):
    path = path or getattr(data_configs, "case_data_path", "datasets/IoT_case.pkl")
    with open(path, "rb") as f:
        data_dict = pickle.load(f)

    data_ic = np.asarray(data_dict["data_ic"], dtype=np.float32)
    if data_ic.shape[0] != data_configs.M_total:
        data_ic = data_ic.T
    data_ic = data_ic[: data_configs.M_total, :]
    row_mean = np.mean(data_ic, axis=1, keepdims=True)
    row_std = np.std(data_ic, axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    data_ic = (data_ic - row_mean) / row_std
    return data_dict, data_ic.astype(np.float32, copy=False), row_mean, row_std


def _make_iot_bootstrap_pools(delta, pool_size, seed, device):
    _, data_ic, _, _ = _iot_reference_arrays()
    rng = np.random.default_rng(seed)
    pool_size = int(pool_size)
    if pool_size > 0 and pool_size != data_ic.shape[1]:
        indices = rng.choice(data_ic.shape[1], size=pool_size, replace=True)
        in_pool_np = data_ic[:, indices].T
    else:
        in_pool_np = data_ic.T
    out_pool_np = in_pool_np + float(delta)
    return (
        torch.as_tensor(in_pool_np, device=device, dtype=torch.float32),
        torch.as_tensor(out_pool_np, device=device, dtype=torch.float32),
    )


def _make_tep_bootstrap_pool(split, device):
    train, test, _, _ = load_standardized_tep_arrays(getattr(data_configs, "case_data_path", "datasets/TEP"))
    pool_np = train if str(split).lower() == "train" else test
    return torch.as_tensor(pool_np, device=device, dtype=torch.float32)


def _sample_tep_step(in_pool, broken_streams, step, t_break, delta, generator):
    batch_size = broken_streams.shape[0]
    device = in_pool.device
    row_idx = torch.randint(in_pool.shape[0], (batch_size,), device=device, generator=generator)
    data_t = in_pool.index_select(0, row_idx).clone()
    if step >= int(t_break) and float(delta) != 0.0:
        if broken_streams.ndim == 1:
            broken_streams = broken_streams.unsqueeze(1)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(broken_streams)
        data_t[rows, broken_streams] += float(delta)
    return data_t


def _load_iot_batch(delta, start_index, batch_size, max_horizon, device):
    data_dict, data_ic, row_mean, row_std = _iot_reference_arrays()

    if float(delta) == 0.0:
        pool = torch.as_tensor(data_ic.T, device=device, dtype=torch.float32)
        return None, pool, 30, int(max_horizon)

    shift_key = int(round(float(delta)))
    if shift_key not in {1, 2, 3}:
        raise ValueError(f"eval_pi mode supports mean shift in {{0,1,2,3}}, got {delta}.")

    selected = np.asarray(data_dict.get(f"data_{shift_key}", data_dict.get(f"data{shift_key}")))
    selected = np.asarray(selected, dtype=np.float32).copy()
    if selected.shape[1] != data_configs.M_total and selected.shape[2] == data_configs.M_total:
        selected = np.transpose(selected, (0, 2, 1))
    for stream_id in range(min(selected.shape[1], data_configs.M_total)):
        selected[:, stream_id, :] = (selected[:, stream_id, :] - row_mean[stream_id, 0]) / row_std[stream_id, 0]

    episode_indices = (start_index + np.arange(batch_size)) % selected.shape[0]
    batch = selected[episode_indices]
    if batch.shape[1] != data_configs.M_total:
        batch = np.transpose(batch, (0, 2, 1))
    batch = batch[:, : data_configs.M_total, :]
    batch = np.transpose(batch, (0, 2, 1))
    horizon = min(int(max_horizon), int(batch.shape[1]))
    return torch.as_tensor(batch[:, :horizon], device=device, dtype=torch.float32), None, 30, horizon


def _sample_synthetic_step(in_pool, out_pool, broken_streams, step, t_break, generator):
    batch_size, total_streams = broken_streams.shape[0], in_pool.shape[1]
    device = in_pool.device
    cols = torch.arange(total_streams, device=device).unsqueeze(0).expand(batch_size, total_streams)
    in_idx = torch.randint(in_pool.shape[0], (batch_size, total_streams), device=device, generator=generator)
    data_t = in_pool[in_idx, cols]
    if step >= t_break and out_pool is not None:
        if broken_streams.ndim == 1:
            broken_streams = broken_streams.unsqueeze(1)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(broken_streams)
        out_idx = torch.randint(
            out_pool.shape[0],
            broken_streams.shape,
            device=device,
            generator=generator,
        )
        data_t[rows, broken_streams] = out_pool[out_idx, broken_streams]
    return data_t


@torch.inference_mode()
def simulate_planner_lengths_gpu(
    runner,
    num_runs,
    delta,
    threshold,
    max_horizon,
    force_m,
    eval_train=False,
    start_index=0,
    random_broken_stream=False,
    num_broken_streams=1,
    return_status=False,
    anomaly_pattern="random_streams",
    return_diagnostics=False,
):
    """Batched torch simulator for planner evaluation/testing.

    It mirrors the existing EvalEnv timing convention: planner/detector decisions
    are made from the current CUSUM state, then the Q-CUSUM state is updated using
    the observation mask selected on the previous step.
    """
    device = runner.device
    all_args = runner.all_args
    batch_size_cfg = int(getattr(all_args, "gpu_eval_batch_size", 1024))
    base_seed = _resolve_seed(getattr(all_args, "seed", 1), default_seed=1)
    total_streams = int(data_configs.M_total)
    num_regions = int(data_configs.agent_num)
    d = int(data_configs.d)
    m_total = int(data_configs.m_total)
    m_max = int(data_configs.M_max)
    region_slices = data_configs.region_slices
    upper_bounds = torch.as_tensor(data_configs.M, device=device, dtype=torch.long)
    qd_total = torch.as_tensor(all_qd(total_streams, data_configs.train_in_data), device=device, dtype=torch.float32)
    region_aggregator = getattr(all_args, "planner_region_aggregator", "max")
    region_topk = int(getattr(all_args, "planner_region_topk", 3) or 3)
    stream_to_region = _stream_to_region_tensor(region_slices, total_streams, device)
    region_starts = torch.as_tensor([sl.start for sl in region_slices], device=device, dtype=torch.long)
    region_lengths = torch.as_tensor([sl.stop - sl.start for sl in region_slices], device=device, dtype=torch.long)

    eval_mode = getattr(data_configs, "eval_data_mode", "synthetic")
    if eval_mode in {"case", "eval_case"}:
        raise NotImplementedError("GPU evaluator currently supports synthetic, eval_pi, and eval_tep data modes.")
    tep_pool = _make_tep_bootstrap_pool("train" if eval_train else "test", device) if eval_mode == "eval_tep" else None

    lengths_out = []
    alarmed_out = []
    pre_change_alarm_out = []
    censored_out = []
    affected_alloc_share_out = []
    affected_rank_out = []
    avg_usage_out = []
    total_usage_out = []
    fixed_planner_alpha = getattr(runner, "fixed_planner_alpha", None)
    if fixed_planner_alpha is None:
        runner.policy.actor.eval()
    for policy in runner.detector_policies:
        policy.actor.eval()
    parallel_detector_executor = getattr(runner, "parallel_detector_executor", None)
    if parallel_detector_executor is None:
        parallel_detector_executor = ParallelDetectorExecutor(
            runner.detector_policies,
            device,
            disabled=bool(getattr(all_args, "disable_parallel_detectors", False)),
            backend=getattr(all_args, "parallel_detector_backend", "fused"),
            strict_parity=bool(getattr(all_args, "fused_detector_strict_parity", True)),
            fused_disable_tf32=bool(getattr(all_args, "fused_detector_disable_tf32", True)),
        )
        runner.parallel_detector_executor = parallel_detector_executor

    for offset in range(0, int(num_runs), batch_size_cfg):
        batch_size = min(batch_size_cfg, int(num_runs) - offset)
        seed_offset = int(start_index) + offset
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed + seed_offset)

        data_batch, iot_pool, t_break, horizon = None, None, 50, int(max_horizon)
        in_pool, out_pool = None, None
        if eval_mode == "eval_pi":
            data_batch, iot_pool, t_break, horizon = _load_iot_batch(delta, seed_offset, batch_size, max_horizon, device)
        elif eval_mode == "eval_tep":
            in_pool = tep_pool
        else:
            pool_size = 2000 if eval_train else 1000
            in_pool, out_pool = _make_synthetic_pools(delta, pool_size, base_seed + seed_offset, device)

        stats = torch.zeros((batch_size, total_streams), device=device, dtype=torch.float32)
        observed_mask = torch.ones((batch_size, total_streams), device=device, dtype=torch.bool)
        local_p = torch.zeros((batch_size, total_streams, d - 1, 2), device=device, dtype=torch.float32)
        local_n = torch.zeros_like(local_p)
        alpha_counts = torch.ones((batch_size, total_streams, d), device=device, dtype=torch.float32) / float(d)

        prev_usage = torch.as_tensor(data_configs.m, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).clone()
        prev_local_actions = torch.ones((batch_size, num_regions, m_max), device=device, dtype=torch.float32)
        detector_rnn = torch.zeros((batch_size, num_regions, all_args.recurrent_N, all_args.hidden_size), device=device)
        detector_masks = torch.ones((batch_size, num_regions, 1), device=device)
        planner_rnn = torch.zeros((batch_size, all_args.recurrent_N, all_args.hidden_size), device=device)
        planner_masks = torch.ones((batch_size, 1), device=device)

        run_indices = torch.arange(seed_offset, seed_offset + batch_size, device=device)
        anomaly_count = max(1, min(int(num_broken_streams), total_streams))
        pattern = str(anomaly_pattern or "random_streams")
        if pattern == "dense_within_region" and float(delta) != 0.0:
            affected_regions = (run_indices % num_regions).long()
            max_local = int(max(int(length) for length in data_configs.M))
            local_scores = torch.rand(
                (batch_size, max_local),
                device=device,
                generator=generator,
            )
            valid_local = torch.arange(max_local, device=device).unsqueeze(0) < region_lengths[affected_regions].unsqueeze(1)
            local_scores = local_scores.masked_fill(~valid_local, -1.0)
            local_k = min(anomaly_count, max_local)
            local_idx = local_scores.topk(local_k, dim=1).indices.long()
            broken_streams = region_starts[affected_regions].unsqueeze(1) + local_idx
        elif random_broken_stream and float(delta) != 0.0:
            broken_streams = torch.rand(
                (batch_size, total_streams),
                device=device,
                generator=generator,
            ).topk(anomaly_count, dim=1).indices.long()
        else:
            offsets = torch.arange(anomaly_count, device=device).unsqueeze(0)
            broken_streams = ((run_indices.unsqueeze(1) + offsets) % total_streams).long()
        affected_region_mask = torch.zeros((batch_size, num_regions), device=device, dtype=torch.bool)
        affected_region_mask.scatter_(1, stream_to_region[broken_streams], True)

        in_control = float(delta) == 0.0 and not eval_train
        default_length = int(horizon) if in_control else max(int(horizon) - int(t_break), 0)
        running_lengths = torch.full((batch_size,), default_length, device=device, dtype=torch.float32)
        active = torch.ones((batch_size,), device=device, dtype=torch.bool)
        alarmed = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        pre_change_alarm = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        affected_alloc_sum = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        affected_rank_sum = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        affected_diag_count = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        usage_sum = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        usage_count = torch.zeros((batch_size,), device=device, dtype=torch.float32)

        for step in range(int(horizon)):
            if not bool(active.any()):
                break

            region_stats = _region_statistics(stats, region_slices, region_aggregator, region_topk)
            if fixed_planner_alpha is None:
                planner_obs = torch.cat([prev_usage, region_stats], dim=1)
                planner_actions, planner_rnn_next = runner.policy.act(
                    planner_obs,
                    planner_rnn,
                    planner_masks,
                    None,
                    deterministic=True,
                    force_m=False,
                    m=int(all_args.planner_alpha_bins),
                )
                planner_rnn = planner_rnn_next
                alpha = planner_actions.float().mean(dim=1, keepdim=True)
            else:
                alpha = torch.full(
                    (batch_size, 1),
                    float(fixed_planner_alpha),
                    device=device,
                    dtype=torch.float32,
                )
            allocations = _alpha_to_allocation_torch(alpha, region_stats, m_total, upper_bounds)
            if return_diagnostics and (not in_control) and step >= int(t_break):
                diag_rows = active
                if bool(diag_rows.any()):
                    alloc_share = (allocations.float() * affected_region_mask.float()).sum(dim=1) / float(m_total)
                    ranks = torch.argsort(torch.argsort(-region_stats, dim=1), dim=1).float() + 1.0
                    affected_rank = ranks.masked_fill(~affected_region_mask, float(num_regions + 1)).min(dim=1).values
                    affected_alloc_sum[diag_rows] += alloc_share[diag_rows]
                    affected_rank_sum[diag_rows] += affected_rank[diag_rows]
                    affected_diag_count[diag_rows] += 1.0

            global_actions = torch.zeros((batch_size, total_streams), device=device, dtype=torch.long)
            usage = torch.zeros((batch_size, num_regions), device=device, dtype=torch.float32)
            next_local_actions = torch.zeros_like(prev_local_actions)
            detector_obs = torch.zeros((batch_size, num_regions, 2 * m_max + 1), device=device, dtype=torch.float32)

            for region_id, region_slice in enumerate(region_slices):
                region_len = region_slice.stop - region_slice.start
                detector_obs[:, region_id, 0] = allocations[:, region_id].float()
                detector_obs[:, region_id, 1 : 1 + region_len] = stats[:, region_slice]
                detector_obs[
                    :,
                    region_id,
                    1 + m_max : 1 + m_max + region_len,
                ] = prev_local_actions[:, region_id, :region_len]

            parallel_actions = parallel_detector_executor.act(
                detector_obs,
                detector_rnn,
                detector_masks,
                allocations,
                deterministic=True,
                force_m=force_m,
            )
            if parallel_actions is not None:
                actions_tensor, detector_rnn = parallel_actions
            else:
                actions_tensor, detector_rnn = parallel_detector_executor.act_serial(
                    detector_obs,
                    detector_rnn,
                    detector_masks,
                    allocations,
                    deterministic=True,
                    force_m=force_m,
                )

            for region_id, region_slice in enumerate(region_slices):
                region_len = region_slice.stop - region_slice.start
                det_actions = actions_tensor[:, region_id, :region_len].long()
                global_actions[:, region_slice] = det_actions
                usage[:, region_id] = det_actions.sum(dim=1).float()
                next_local_actions[:, region_id, :region_len] = det_actions.float()

            if bool(active.any()):
                usage_sum[active] += usage.sum(dim=1)[active]
                usage_count[active] += 1.0

            if data_batch is not None:
                data_t = data_batch[:, step, :]
            elif iot_pool is not None:
                cols = torch.arange(total_streams, device=device).unsqueeze(0).expand(batch_size, total_streams)
                in_idx = torch.randint(iot_pool.shape[0], (batch_size, total_streams), device=device, generator=generator)
                data_t = iot_pool[in_idx, cols]
            elif eval_mode == "eval_tep":
                data_t = _sample_tep_step(in_pool, broken_streams, step, int(t_break), delta, generator)
            else:
                data_t = _sample_synthetic_step(in_pool, out_pool, broken_streams, step, int(t_break), generator)

            stats, local_p, local_n, alpha_counts = _update_qcusum(
                data_t,
                qd_total,
                observed_mask,
                local_p,
                local_n,
                alpha_counts,
                generator=generator,
            )
            observed_mask = global_actions.bool()
            prev_usage = usage
            prev_local_actions = next_local_actions

            lambda_t = torch.topk(stats, k=int(data_configs.l), dim=1).values.sum(dim=1)
            raw_alarm = lambda_t >= float(threshold)
            if in_control:
                alarm = raw_alarm
            elif not eval_train:
                early_alarm = active & raw_alarm & (step < int(t_break))
                if bool(early_alarm.any()):
                    pre_change_alarm[early_alarm] = True
                    active[early_alarm] = False
                    detector_masks[early_alarm] = 0.0
                    detector_rnn[early_alarm] = 0.0
                    planner_masks[early_alarm] = 0.0
                    planner_rnn[early_alarm] = 0.0
                alarm = raw_alarm & (step >= int(t_break))
            else:
                alarm = raw_alarm & (step >= int(t_break))
            newly_done = active & alarm
            if bool(newly_done.any()):
                running_lengths[newly_done] = step if in_control else max(step - int(t_break), 0)
                alarmed[newly_done] = True
                active[newly_done] = False
                detector_masks[newly_done] = 0.0
                detector_rnn[newly_done] = 0.0
                planner_masks[newly_done] = 0.0
                planner_rnn[newly_done] = 0.0

        censored = ~alarmed & ~pre_change_alarm
        lengths_out.append(running_lengths.detach().cpu().numpy())
        if return_status:
            alarmed_out.append(alarmed.detach().cpu().numpy())
            pre_change_alarm_out.append(pre_change_alarm.detach().cpu().numpy())
            censored_out.append(censored.detach().cpu().numpy())
        if return_diagnostics:
            diag_denom = affected_diag_count.clamp_min(1.0)
            affected_alloc_share_out.append((affected_alloc_sum / diag_denom).detach().cpu().numpy())
            affected_rank_out.append((affected_rank_sum / diag_denom).detach().cpu().numpy())
        if return_status:
            avg_usage_out.append((usage_sum / usage_count.clamp_min(1.0)).detach().cpu().numpy())
            total_usage_out.append(usage_sum.detach().cpu().numpy())

    lengths = np.concatenate(lengths_out).astype(float, copy=False)
    if not return_status and not return_diagnostics:
        return lengths

    result = {"lengths": lengths}
    if return_status:
        result.update(
            {
                "alarmed": np.concatenate(alarmed_out).astype(bool, copy=False),
                "pre_change_alarm": np.concatenate(pre_change_alarm_out).astype(bool, copy=False),
                "censored": np.concatenate(censored_out).astype(bool, copy=False),
                "avg_usage": np.concatenate(avg_usage_out).astype(float, copy=False),
                "total_usage": np.concatenate(total_usage_out).astype(float, copy=False),
            }
        )
    if return_diagnostics:
        result["affected_allocation_share"] = np.concatenate(affected_alloc_share_out).astype(float, copy=False)
        result["affected_region_rank"] = np.concatenate(affected_rank_out).astype(float, copy=False)
    return result
