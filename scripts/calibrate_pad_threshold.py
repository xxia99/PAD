#!/usr/bin/env python
"""Utility used by the PAD test scripts to construct actors and calibrate ARL0.

The helper functions in this file are imported by `run_pad_test_protocol.py`
and `reproduce_m50_table.py`. They build a lightweight runner shell containing
only the trained actors needed for GPU-vectorized testing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
import torch

from algorithms.happo_policy import HAPPO_Policy
from config import get_config
from envs.env_utils import data_configs
from envs.om_env_wrappers import MultiDiscrete
from runners.separated.gpu_pad_evaluator import simulate_planner_lengths_gpu


class _RunnerShell:
    pass


def _parse_args():
    parser = argparse.ArgumentParser(description="Calibrate PAD threshold by ARL0.")
    parser.add_argument("--M_total", type=int, default=20)
    parser.add_argument("--m_total", type=int, default=10)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--M_list", type=str, default=None)
    parser.add_argument("--m_list", type=str, default=None)
    parser.add_argument("--data-mode", dest="data_mode", choices=["synthetic", "eval_pi", "eval_tep"], default="synthetic")
    parser.add_argument("--case-data-path", dest="case_data_path", type=str, default="datasets/IoT_case.pkl")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_eval_batch_size", type=int, default=1000)
    parser.add_argument("--planner_model_dir", type=str, default="checkpoints/planner/M=20_N=5/run_seed_1/models")
    parser.add_argument("--planner_model_name", type=str, default="planner_best_actor.pt")
    parser.add_argument("--stage1_model_dir", type=str, default="checkpoints/detectors/M=20_N=5")
    parser.add_argument("--target", type=float, default=370.0)
    parser.add_argument("--arl0_lower", type=float, default=360.0)
    parser.add_argument("--arl0_upper", type=float, default=380.0)
    parser.add_argument("--max_bisect_iters", type=int, default=16)
    parser.add_argument("--force_m", action="store_true", default=False)
    parser.add_argument(
        "--parallel_detector_backend",
        type=str,
        default="fused",
        choices=["serial", "streams", "fused"],
    )
    parser.add_argument("--fused_detector_strict_parity", action="store_true", default=True)
    parser.add_argument("--no_fused_detector_strict_parity", dest="fused_detector_strict_parity", action="store_false")
    parser.add_argument("--fused_detector_disable_tf32", action="store_true", default=True)
    parser.add_argument("--no_fused_detector_disable_tf32", dest="fused_detector_disable_tf32", action="store_false")
    parser.add_argument("--out_dir", type=str, default="calibration_results/PAD_M20_N5_m10_try1000")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _build_base_args(cli_args):
    parser = get_config()
    parser_args = [
        "--M_total", str(cli_args.M_total),
        "--m_total", str(cli_args.m_total),
        "--agent_num", str(cli_args.agent_num),
        "--planner_eval_horizon", str(cli_args.horizon),
        "--test_replications", str(cli_args.replications),
        "--calibration_runs", str(cli_args.replications),
        "--gpu", str(cli_args.gpu),
        "--gpu_eval_batch_size", str(cli_args.gpu_eval_batch_size),
        "--stage1_model_dir", cli_args.stage1_model_dir,
    ]
    if getattr(cli_args, "M_list", None):
        parser_args.extend(["--M_list", str(cli_args.M_list)])
    if getattr(cli_args, "m_list", None):
        parser_args.extend(["--m_list", str(cli_args.m_list)])
    if getattr(cli_args, "data_mode", None):
        parser_args.extend(["--data-mode", str(cli_args.data_mode)])
    if getattr(cli_args, "case_data_path", None):
        parser_args.extend(["--case-data-path", str(cli_args.case_data_path)])
    args = parser.parse_args(parser_args)
    args.seed = int(cli_args.seed)
    args.n_rollout_threads = 1
    args.n_test_rollout_threads = 1
    args.use_gpu_eval = True
    args.force_m = bool(cli_args.force_m)
    args.planner_region_aggregator = getattr(cli_args, "planner_region_aggregator", "max")
    args.planner_region_topk = int(getattr(cli_args, "planner_region_topk", 3) or 3)
    args.parallel_detector_backend = getattr(cli_args, "parallel_detector_backend", "fused")
    args.fused_detector_strict_parity = bool(getattr(cli_args, "fused_detector_strict_parity", True))
    args.fused_detector_disable_tf32 = bool(getattr(cli_args, "fused_detector_disable_tf32", True))
    return args


def _load_policy_actor(policy, path, device):
    state = torch.load(path, map_location=device)
    policy.actor.load_state_dict(state)
    policy.actor.eval()


def _detector_actor_path(stage1_dir, region_id):
    stage1_dir = Path(stage1_dir)
    candidates = [
        stage1_dir / f"actor_agent{region_id}.pt",
        stage1_dir / f"actor_agent{region_id}_latest.pt",
        stage1_dir / "models" / f"actor_agent{region_id}.pt",
        stage1_dir / "models" / f"actor_agent{region_id}_latest.pt",
        stage1_dir / "run_seed_2" / "models" / f"actor_agent{region_id}.pt",
        stage1_dir / "run_seed_2" / "models" / f"actor_agent{region_id}_latest.pt",
        stage1_dir / "run_seed_1" / "models" / f"actor_agent{region_id}.pt",
        stage1_dir / "run_seed_1" / "models" / f"actor_agent{region_id}_latest.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find detector actor_agent{region_id} under {stage1_dir}. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _build_runner(args, cli_args, device):
    # Build one detector actor per region from `actor_agent*.pt`.
    m_max = data_configs.M_max
    num_regions = data_configs.agent_num
    det_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2 * m_max + 1,), dtype=np.float32)
    det_share_obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(num_regions * (2 * m_max + 1),),
        dtype=np.float32,
    )

    detector_policies = []
    for region_id, region_streams in enumerate(data_configs.M):
        det_act_space = MultiDiscrete([[0, 1]] * int(region_streams))
        policy = HAPPO_Policy(args, det_obs_space, det_share_obs_space, det_act_space, device=device)
        actor_path = _detector_actor_path(cli_args.stage1_model_dir, region_id)
        _load_policy_actor(policy, actor_path, device)
        detector_policies.append(policy)

    fixed_planner_alpha = getattr(cli_args, "fixed_planner_alpha", None)
    planner_policy = None
    if fixed_planner_alpha is None:
        # Load the system-level planner actor unless a fixed fair/greedy alpha
        # baseline was requested.
        planner_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2 * num_regions,), dtype=np.float32)
        planner_act_space = MultiDiscrete([[0, 1]] * int(args.planner_alpha_bins))
        planner_policy = HAPPO_Policy(args, planner_obs_space, planner_obs_space, planner_act_space, device=device)
        planner_path = Path(cli_args.planner_model_dir) / cli_args.planner_model_name
        _load_policy_actor(planner_policy, planner_path, device)

    runner = _RunnerShell()
    runner.all_args = args
    runner.device = device
    runner.policy = planner_policy
    runner.detector_policies = detector_policies
    runner.fixed_planner_alpha = fixed_planner_alpha
    return runner


def main():
    cli_args = _parse_args()
    args = _build_base_args(cli_args)
    data_configs.configure_from_args(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(cli_args.gpu))
        device = torch.device(f"cuda:{int(cli_args.gpu)}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(cli_args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    out_dir = Path(cli_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = _build_runner(args, cli_args, device)

    records = []

    def evaluate_threshold(threshold):
        t0 = time.time()
        lengths = simulate_planner_lengths_gpu(
            runner,
            int(cli_args.replications),
            0.0,
            float(threshold),
            int(cli_args.horizon),
            force_m=bool(cli_args.force_m),
            eval_train=False,
        )
        rec = {
            "threshold": float(threshold),
            "ARL0": float(np.mean(lengths)),
            "stderr": float(np.std(lengths) / np.sqrt(max(len(lengths), 1))),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "censored_at_horizon": int(np.sum(lengths >= int(cli_args.horizon))),
            "replications": int(cli_args.replications),
            "horizon": int(cli_args.horizon),
            "seconds": float(time.time() - t0),
        }
        records.append(rec)
        print(json.dumps(rec), flush=True)
        return rec

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "cuda": bool(torch.cuda.is_available()),
                "M": data_configs.M,
                "m": data_configs.m,
                "target_interval": [float(cli_args.arl0_lower), float(cli_args.arl0_upper)],
                "parallel_detector_backend": str(cli_args.parallel_detector_backend),
                "fused_detector_strict_parity": bool(cli_args.fused_detector_strict_parity),
                "fused_detector_disable_tf32": bool(cli_args.fused_detector_disable_tf32),
            }
        ),
        flush=True,
    )

    low = 0.0
    low_rec = evaluate_threshold(low)
    high = 1.0
    high_rec = evaluate_threshold(high)
    target = float(cli_args.target)

    while high_rec["ARL0"] < target and high < 1e6:
        low = high
        low_rec = high_rec
        high *= 2.0
        high_rec = evaluate_threshold(high)

    best = min(records, key=lambda r: abs(r["ARL0"] - target))
    for _ in range(int(cli_args.max_bisect_iters)):
        mid = 0.5 * (low + high)
        mid_rec = evaluate_threshold(mid)
        if float(cli_args.arl0_lower) <= mid_rec["ARL0"] <= float(cli_args.arl0_upper):
            best = mid_rec
            break
        if abs(mid_rec["ARL0"] - target) < abs(best["ARL0"] - target):
            best = mid_rec
        if mid_rec["ARL0"] >= target:
            high = mid
        else:
            low = mid

    output = {
        "protocol": "ARL0 calibration: mean_shift=0, false-alarm run length from start.",
        "M_total": int(cli_args.M_total),
        "N": int(cli_args.agent_num),
        "m_total": int(cli_args.m_total),
        "replications": int(cli_args.replications),
        "horizon": int(cli_args.horizon),
        "target": float(cli_args.target),
        "target_interval": [float(cli_args.arl0_lower), float(cli_args.arl0_upper)],
        "force_m": bool(cli_args.force_m),
        "parallel_detector_backend": str(cli_args.parallel_detector_backend),
        "fused_detector_strict_parity": bool(cli_args.fused_detector_strict_parity),
        "fused_detector_disable_tf32": bool(cli_args.fused_detector_disable_tf32),
        "best": best,
        "records": records,
    }
    out_path = out_dir / "threshold_calibration_arl0_1000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("BEST " + json.dumps(best), flush=True)
    print(f"saved={out_path}", flush=True)


if __name__ == "__main__":
    main()
