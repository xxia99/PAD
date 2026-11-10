#!/usr/bin/env python
"""Stage I detector training entry point.

This script trains the regional detector policies. It is the reviewer-facing
GPU implementation: rollouts are collected as batched tensors on one GPU, and
in-training evaluations also use the GPU evaluator.
"""

import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch

from config import get_config
from envs.env_utils import data_configs
from runners.separated.gpu_online_monitoring_runner import GPUOnlineMonitoringRunner
from utils.swanlab_compat import swanlab


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


def parse_args(args, parser):
    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    # 1. Parse the shared PAD/HAPPO command line. `gpu_stage1=True` selects the
    # detector-training code path inside the runner and environment.
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    all_args.gpu_stage1 = True
    data_configs.configure_from_args(all_args)

    # 2. Keep CPU thread use modest. The expensive work is vectorized on GPU,
    # so oversubscribing CPU threads only makes reviewer machines noisier.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(all_args.n_training_threads))))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, int(all_args.n_training_threads))))

    # 3. Select the requested CUDA device and set PyTorch deterministic/precision
    # controls. CPU fallback exists only for tiny debugging runs.
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.gpu}")
        torch.cuda.set_device(device)
        torch.set_num_threads(max(1, int(all_args.n_training_threads)))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(getattr(all_args, "float32_matmul_precision", "high")))
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(max(1, int(all_args.n_training_threads)))

    seed = _resolve_seed(all_args.seed, default_seed=1)
    print(f"-------------------------------------------------GPU Stage I training starts for seed: {seed}---------------------------------------------------")

    # 4. Create the run directory. Model actors are saved under
    # `<run_dir>/run_seed_*/models/`.
    batch_size = int(all_args.gpu_stage1_batch_size or all_args.n_rollout_threads)
    run_name = (
        f"gpu_stage1_{all_args.experiment_name}"
        if all_args.experiment_name and all_args.experiment_name != "check"
        else f"gpu_stage1_M={all_args.M_total}_m={all_args.m_total}_N={all_args.agent_num}_B={batch_size}"
    )
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
    if all_args.run_root:
        run_dir = Path(all_args.run_root) / run_name / timestamp
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results/" + timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_res_record_file = run_dir / "seed_results.txt"
    seed_res_record_file.touch(exist_ok=True)
    run_dir = run_dir / f"run_seed_{seed + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("run_dir:", run_dir)

    # 5. Initialize optional logging. If SwanLab is unavailable, the compatibility
    # wrapper silently turns these calls into no-ops.
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
            "gpu_stage1_batch_size": batch_size,
            "float32_matmul_precision": str(getattr(all_args, "float32_matmul_precision", "high")),
            "gpu_stage1_random_allocations": bool(all_args.gpu_stage1_random_allocations),
            "gpu_stage1_balanced_allocations": bool(all_args.gpu_stage1_balanced_allocations),
            "gpu_stage1_random_broken_streams": bool(all_args.gpu_stage1_random_broken_streams),
            "gpu_stage1_reset_rnn_each_episode": bool(all_args.gpu_stage1_reset_rnn_each_episode),
            "gpu_stage1_eval_force_modes": str(all_args.gpu_stage1_eval_force_modes),
            "gpu_stage1_eval_common_pools": bool(all_args.gpu_stage1_eval_common_pools),
            "gpu_stage1_train_pool_seed": getattr(all_args, "gpu_stage1_train_pool_seed", None),
            "gpu_stage1_eval_pool_seed": getattr(all_args, "gpu_stage1_eval_pool_seed", None),
            "gpu_stage1_eval_preserve_rng": bool(all_args.gpu_stage1_eval_preserve_rng),
            "gpu_stage1_legacy_metrics": bool(all_args.gpu_stage1_legacy_metrics),
            "gpu_stage1_legacy_eval_logging": bool(all_args.gpu_stage1_legacy_eval_logging),
            "gpu_stage1_eval_use_fused_detectors": bool(all_args.gpu_stage1_eval_use_fused_detectors),
            "gpu_stage1_postchange_only_loss": bool(all_args.gpu_stage1_postchange_only_loss),
            "parallel_detector_backend": str(getattr(all_args, "parallel_detector_backend", "fused")),
            "fused_detector_strict_parity": bool(getattr(all_args, "fused_detector_strict_parity", True)),
            "gpu_stage1_resource_upper_bound": int(all_args.gpu_stage1_resource_upper_bound),
            "gpu_stage1_train_t_break": int(all_args.gpu_stage1_train_t_break),
            "gpu_stage1_train_t_break_choices": getattr(all_args, "gpu_stage1_train_t_break_choices", None),
            "gpu_stage1_train_t_break_probs": getattr(all_args, "gpu_stage1_train_t_break_probs", None),
            "episode_length": all_args.episode_length,
            "num_env_steps": all_args.num_env_steps,
            "eval_interval": all_args.eval_interval,
            "gpu_stage1_eval_episodes": all_args.gpu_stage1_eval_episodes,
            "gpu_stage1_eval_horizon": all_args.gpu_stage1_eval_horizon,
            "gpu_stage1_eval_threshold": all_args.gpu_stage1_eval_threshold,
            "gpu_stage1_eval_single_m_choices": getattr(all_args, "gpu_stage1_eval_single_m_choices", None),
            "gpu_stage1_select_eval_interval": int(getattr(all_args, "gpu_stage1_select_eval_interval", 0)),
            "gpu_stage1_select_eval_episodes": int(getattr(all_args, "gpu_stage1_select_eval_episodes", 0)),
            "gpu_stage1_select_eval_horizon": int(getattr(all_args, "gpu_stage1_select_eval_horizon", 0)),
            "gpu_stage1_select_eval_force_modes": getattr(all_args, "gpu_stage1_select_eval_force_modes", None),
            "gpu_stage1_select_eval_single_m_choices": getattr(
                all_args, "gpu_stage1_select_eval_single_m_choices", None
            ),
        },
        dir="../",
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 6. Build and execute the GPU Stage I runner.
    config = {
        "all_args": all_args,
        "num_agents": data_configs.agent_num,
        "device": device,
        "run_dir": run_dir,
    }

    runner = GPUOnlineMonitoringRunner(config)
    try:
        runner.run()
    finally:
        runner.writter.close()
        swanlab.finish()
