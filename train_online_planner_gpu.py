#!/usr/bin/env python
"""Stage II planner training entry point.

This script fixes the trained regional detectors and trains the system-level
planner. The final reviewer archive uses the GPU runner and fused detector
backend so planner rollouts are batched on one GPU.
"""

import datetime
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import get_config
from envs.env_utils import data_configs
from runners.separated.gpu_online_planner_runner import GPUOnlinePlannerRunner
from utils.swanlab_compat import swanlab


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


def parse_args(args, parser):
    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    # 1. Parse CLI options and select the Stage II code path.
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    all_args.gpu_stage2 = True
    data_configs.configure_from_args(all_args)

    # 2. Limit CPU helper threads. Stage II rollout/eval work is GPU-vectorized.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(all_args.n_training_threads))))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, int(all_args.n_training_threads))))

    # 3. Select device and deterministic/precision settings.
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.gpu}")
        torch.cuda.set_device(device)
        torch.set_num_threads(max(1, int(all_args.n_training_threads)))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(max(1, int(all_args.n_training_threads)))

    seed = _resolve_seed(all_args.seed, default_seed=1)
    all_args.seed = seed
    print("-------------------------------------------------GPU Stage II planner training starts---------------------------------------------------")

    # 4. Create run/checkpoint directories. `checkpoint_root` is optional; the
    # regular model directory is always created below `run_root`.
    batch_size = int(all_args.gpu_stage2_batch_size or all_args.n_rollout_threads)
    run_name = (
        f"gpu_stage2_{all_args.experiment_name}"
        if all_args.experiment_name and all_args.experiment_name != "check"
        else f"gpu_stage2_M={all_args.M_total}_m={all_args.m_total}_N={all_args.agent_num}_B={batch_size}"
    )
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
    if all_args.run_root:
        run_dir = Path(all_args.run_root) / run_name / timestamp
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results_planner/" + timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir / f"run_seed_{seed + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("run_dir:", run_dir)

    checkpoint_dir = None
    if all_args.checkpoint_root:
        checkpoint_dir = Path(all_args.checkpoint_root) / run_name / timestamp / f"run_seed_{seed + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print("checkpoint_dir:", checkpoint_dir)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 5. Initialize optional logging. Missing SwanLab is handled by a no-op
    # compatibility wrapper.
    swanlab.init(
        project="PAD",
        name=run_name,
        config={
            "seed": seed,
            "M_total": all_args.M_total,
            "m_total": all_args.m_total,
            "agent_num": all_args.agent_num,
            "data_mode": getattr(all_args, "data_mode", None),
            "case_data_path": getattr(all_args, "case_data_path", None),
            "gpu_stage2_batch_size": batch_size,
            "episode_length": all_args.episode_length,
            "num_env_steps": all_args.num_env_steps,
            "planner_eval_interval": all_args.planner_eval_interval,
            "planner_eval_episodes": all_args.planner_eval_episodes,
            "planner_eval_horizon": all_args.planner_eval_horizon,
            "planner_eval_start_index": all_args.planner_eval_start_index,
            "planner_eval_random_broken_streams": bool(all_args.planner_eval_random_broken_streams),
            "planner_select_eval_interval": all_args.planner_select_eval_interval,
            "planner_select_eval_episodes": all_args.planner_select_eval_episodes,
            "planner_select_eval_horizon": all_args.planner_select_eval_horizon,
            "planner_select_eval_start_index": all_args.planner_select_eval_start_index,
            "planner_alarm_threshold": all_args.planner_alarm_threshold,
            "planner_region_aggregator": all_args.planner_region_aggregator,
            "planner_region_topk": all_args.planner_region_topk,
            "gpu_stage2_train_t_break": all_args.gpu_stage2_train_t_break,
            "gpu_stage2_train_stop_active_fraction": all_args.gpu_stage2_train_stop_active_fraction,
            "gpu_stage2_use_surrogate": all_args.gpu_stage2_use_surrogate,
            "gpu_stage2_surrogate_pretrain_updates": all_args.gpu_stage2_surrogate_pretrain_updates,
            "gpu_stage2_surrogate_reps": all_args.gpu_stage2_surrogate_reps,
            "gpu_stage2_surrogate_horizon": all_args.gpu_stage2_surrogate_horizon,
            "gpu_stage2_surrogate_delta": all_args.gpu_stage2_surrogate_delta,
            "gpu_stage2_surrogate_gain": all_args.gpu_stage2_surrogate_gain,
            "gpu_stage2_surrogate_stat_bins": all_args.gpu_stage2_surrogate_stat_bins,
            "gpu_stage2_surrogate_stat_max": all_args.gpu_stage2_surrogate_stat_max,
            "gpu_stage2_mixed_surrogate": all_args.gpu_stage2_mixed_surrogate,
            "gpu_stage2_mix_rho_start": all_args.gpu_stage2_mix_rho_start,
            "gpu_stage2_mix_rho_mid": all_args.gpu_stage2_mix_rho_mid,
            "gpu_stage2_mix_rho_end": all_args.gpu_stage2_mix_rho_end,
            "gpu_stage2_mix_warm_updates": all_args.gpu_stage2_mix_warm_updates,
            "gpu_stage2_mix_mid_update": all_args.gpu_stage2_mix_mid_update,
            "gpu_stage2_mix_end_update": all_args.gpu_stage2_mix_end_update,
            "gpu_stage2_cf_alpha_baseline": all_args.gpu_stage2_cf_alpha_baseline,
            "gpu_stage2_cf_lambda": all_args.gpu_stage2_cf_lambda,
            "gpu_stage2_cf_horizon": all_args.gpu_stage2_cf_horizon,
            "gpu_stage2_cf_stride": all_args.gpu_stage2_cf_stride,
            "gpu_stage2_cf_max_states": all_args.gpu_stage2_cf_max_states,
            "gpu_stage2_cf_alpha_candidates": all_args.gpu_stage2_cf_alpha_candidates,
            "gpu_stage2_stat_progress_reward": all_args.gpu_stage2_stat_progress_reward,
            "gpu_stage2_stat_progress_beta_start": all_args.gpu_stage2_stat_progress_beta_start,
            "gpu_stage2_stat_progress_beta_end": all_args.gpu_stage2_stat_progress_beta_end,
            "gpu_stage2_stat_progress_anneal_updates": all_args.gpu_stage2_stat_progress_anneal_updates,
            "gpu_stage2_stat_progress_clip": all_args.gpu_stage2_stat_progress_clip,
            "gpu_stage2_random_broken_streams": bool(all_args.gpu_stage2_random_broken_streams),
            "gpu_stage2_reset_rnn_each_episode": bool(all_args.gpu_stage2_reset_rnn_each_episode),
            "checkpoint_root": all_args.checkpoint_root,
            "run_root": all_args.run_root,
            "save_eval_checkpoints": all_args.save_eval_checkpoints,
            "gpu_stage2_planner_reward": "-1_per_active_step",
            "gpu_stage2_train_broken_streams": "random" if all_args.gpu_stage2_random_broken_streams else "stratified_cycle",
            "gpu_stage2_alarm_stat": "top_l_global_qcusum_after_update",
            "stage1_model_dir": all_args.stage1_model_dir,
        },
        dir="../",
    )

    # 6. Build and execute the GPU Stage II planner runner.
    config = {
        "all_args": all_args,
        "device": device,
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
    }
    runner = GPUOnlinePlannerRunner(config)
    try:
        runner.run()
    finally:
        runner.writter.close()
        swanlab.finish()
