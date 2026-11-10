#!/usr/bin/env python
"""PAD threshold calibration and ARL1 testing protocol.

This script is the full paper-style tester: calibrate an IC threshold for
ARL0 ~= 370 unless `--fixed_threshold` is provided, then evaluate OC ARL1 for
the requested mean shifts. Test logging is off by default.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from envs.env_utils import data_configs
from runners.separated.gpu_pad_evaluator import simulate_planner_lengths_gpu
from scripts.calibrate_pad_threshold import _build_base_args, _build_runner
from utils.swanlab_compat import swanlab


def _parse_mean_shifts(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _json_float(value):
    if value is None:
        return None
    return float(value)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run PAD ARL0 calibration and ARL1 tests.")
    parser.add_argument("--M_total", type=int, default=50)
    parser.add_argument("--m_total", type=int, required=True)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--M_list", type=str, default=None, help="Comma-separated per-region stream counts.")
    parser.add_argument("--m_list", type=str, default=None, help="Comma-separated per-region resource counts.")
    parser.add_argument(
        "--data-mode",
        dest="data_mode",
        choices=["synthetic", "eval_pi", "eval_tep"],
        default="synthetic",
        help="Data source: synthetic simulation, IoT case pickle, or TEP case CSVs.",
    )
    parser.add_argument(
        "--case-data-path",
        dest="case_data_path",
        type=str,
        default="datasets/IoT_case.pkl",
        help="Path to IoT_case.pkl for eval_pi, or directory containing TEP CSVs for eval_tep.",
    )
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--target", type=float, default=370.0)
    parser.add_argument("--arl0_lower", type=float, default=360.0)
    parser.add_argument("--arl0_upper", type=float, default=380.0)
    parser.add_argument("--max_bisect_iters", type=int, default=18)
    parser.add_argument(
        "--fixed_threshold",
        type=float,
        default=None,
        help="Skip ARL0 calibration and run ARL1 tests with this threshold.",
    )
    parser.add_argument("--mean_shifts", type=str, default="1,2,3")
    parser.add_argument("--num_anomalies", type=int, default=1)
    parser.add_argument(
        "--anomaly_pattern",
        type=str,
        default="random_streams",
        choices=["random_streams", "dense_within_region"],
        help="ARL1 anomaly placement pattern.",
    )
    parser.add_argument("--planner_region_aggregator", type=str, default="max", choices=["max", "mean", "sum", "topk"])
    parser.add_argument("--planner_region_topk", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_eval_batch_size", type=int, default=5000)
    parser.add_argument("--require_cuda", action="store_true", default=True)
    parser.add_argument("--allow_cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--require_single_gpu_batch", action="store_true", default=True)
    parser.add_argument("--allow_split_batches", dest="require_single_gpu_batch", action="store_false")
    parser.add_argument("--planner_model_dir", type=str, default=None)
    parser.add_argument("--planner_model_name", type=str, default="planner_best_actor.pt")
    parser.add_argument(
        "--fixed_planner_alpha",
        type=float,
        default=None,
        help=(
            "Bypass the learned planner and use a constant alpha. "
            "Current allocator uses alpha=1 for fair and alpha=0 for greedy."
        ),
    )
    parser.add_argument("--stage1_model_dir", type=str, required=True)
    parser.add_argument("--force_m", action="store_true", default=True)
    parser.add_argument("--no_force_m", dest="force_m", action="store_false")
    parser.add_argument(
        "--parallel_detector_backend",
        type=str,
        default="fused",
        choices=["serial", "streams", "fused"],
        help="detector execution backend for PAD testing.",
    )
    parser.add_argument("--fused_detector_strict_parity", action="store_true", default=True)
    parser.add_argument("--no_fused_detector_strict_parity", dest="fused_detector_strict_parity", action="store_false")
    parser.add_argument("--fused_detector_disable_tf32", action="store_true", default=True)
    parser.add_argument("--no_fused_detector_disable_tf32", dest="fused_detector_disable_tf32", action="store_false")
    parser.add_argument("--out_dir", type=str, default="pad_test_results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--enable_swanlab",
        action="store_true",
        default=False,
        help="Enable SwanLab logging for this testing protocol. Disabled by default for test runs.",
    )
    return parser.parse_args()


def _summarize_arl0(lengths, threshold, args, elapsed):
    return {
        "threshold": float(threshold),
        "ARL0": float(np.mean(lengths)),
        "stderr": float(np.std(lengths) / np.sqrt(max(len(lengths), 1))),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "censored_at_horizon": int(np.sum(lengths >= int(args.horizon))),
        "replications": int(args.replications),
        "horizon": int(args.horizon),
        "seconds": float(elapsed),
    }


def _calibrate_threshold(runner, args):
    records = []

    def evaluate(threshold):
        start = time.time()
        lengths = simulate_planner_lengths_gpu(
            runner,
            int(args.replications),
            0.0,
            float(threshold),
            int(args.horizon),
            force_m=bool(args.force_m),
            eval_train=False,
        )
        rec = _summarize_arl0(lengths, threshold, args, time.time() - start)
        records.append(rec)
        if bool(getattr(args, "enable_swanlab", False)):
            swanlab.log(
                {
                    "PADTest/calibration_threshold": rec["threshold"],
                    "PADTest/calibration_ARL0": rec["ARL0"],
                    "PADTest/calibration_stderr": rec["stderr"],
                    "PADTest/calibration_seconds": rec["seconds"],
                    "PADTest/m_total": int(args.m_total),
                }
            )
        print(json.dumps({"event": "calibration_eval", **rec}), flush=True)
        return rec

    low = 0.0
    low_rec = evaluate(low)
    high = 1.0
    high_rec = evaluate(high)
    target = float(args.target)

    while high_rec["ARL0"] < target and high < 1e6:
        low = high
        low_rec = high_rec
        high *= 2.0
        high_rec = evaluate(high)

    best = min(records, key=lambda rec: abs(rec["ARL0"] - target))
    for _ in range(int(args.max_bisect_iters)):
        mid = 0.5 * (low + high)
        mid_rec = evaluate(mid)
        if float(args.arl0_lower) <= mid_rec["ARL0"] <= float(args.arl0_upper):
            best = mid_rec
            break
        if abs(mid_rec["ARL0"] - target) < abs(best["ARL0"] - target):
            best = mid_rec
        if mid_rec["ARL0"] >= target:
            high = mid
        else:
            low = mid

    return best, records


def _test_arl1(runner, args, threshold):
    records = []
    for delta in _parse_mean_shifts(args.mean_shifts):
        start = time.time()
        status = simulate_planner_lengths_gpu(
            runner,
            int(args.replications),
            float(delta),
            float(threshold),
            int(args.horizon),
            force_m=bool(args.force_m),
            eval_train=False,
            random_broken_stream=True,
            num_broken_streams=int(args.num_anomalies),
            return_status=True,
            anomaly_pattern=str(args.anomaly_pattern),
            return_diagnostics=True,
        )
        lengths = status["lengths"]
        valid = status["alarmed"] & ~status["pre_change_alarm"] & ~status["censored"]
        valid_lengths = lengths[valid]
        valid_n = int(valid_lengths.shape[0])
        skipped_pre_change = int(np.sum(status["pre_change_alarm"]))
        skipped_missed = int(np.sum(status["censored"]))
        rec = {
            "mean_shift": float(delta),
            "num_anomalies": int(args.num_anomalies),
            "ARL1": _json_float(np.mean(valid_lengths) if valid_n else None),
            "stderr": _json_float(np.std(valid_lengths) / np.sqrt(valid_n) if valid_n else None),
            "median": _json_float(np.median(valid_lengths) if valid_n else None),
            "p10": _json_float(np.percentile(valid_lengths, 10) if valid_n else None),
            "p90": _json_float(np.percentile(valid_lengths, 90) if valid_n else None),
            "min": _json_float(np.min(valid_lengths) if valid_n else None),
            "max": _json_float(np.max(valid_lengths) if valid_n else None),
            "capped_ARL1": float(np.mean(lengths)),
            "valid_replications": valid_n,
            "skipped_pre_change_alarms": skipped_pre_change,
            "skipped_missed_at_horizon": skipped_missed,
            "skipped_total": skipped_pre_change + skipped_missed,
            "threshold": float(threshold),
            "replications": int(args.replications),
            "horizon": int(args.horizon),
            "change_point": 50,
            "random_broken_stream": True,
            "anomaly_pattern": str(args.anomaly_pattern),
            "affected_allocation_share": _json_float(np.mean(status["affected_allocation_share"][valid]) if valid_n else None),
            "affected_allocation_share_all": _json_float(np.mean(status["affected_allocation_share"]) if len(lengths) else None),
            "affected_region_rank": _json_float(np.mean(status["affected_region_rank"][valid]) if valid_n else None),
            "affected_region_rank_all": _json_float(np.mean(status["affected_region_rank"]) if len(lengths) else None),
            "avg_usage": _json_float(np.mean(status["avg_usage"][valid]) if valid_n and "avg_usage" in status else None),
            "avg_usage_all": _json_float(np.mean(status["avg_usage"]) if "avg_usage" in status and len(lengths) else None),
            "avg_monitoring_cost": _json_float(
                np.mean(status["total_usage"][valid]) if valid_n and "total_usage" in status else None
            ),
            "avg_monitoring_cost_all": _json_float(
                np.mean(status["total_usage"]) if "total_usage" in status and len(lengths) else None
            ),
            "seconds": float(time.time() - start),
        }
        records.append(rec)
        if bool(getattr(args, "enable_swanlab", False)):
            swanlab.log(
                {
                    f"PADTest/ARL1_shift_{delta:g}": rec["ARL1"],
                    f"PADTest/ARL1_shift_{delta:g}_stderr": rec["stderr"],
                    f"PADTest/ARL1_shift_{delta:g}_valid_replications": valid_n,
                    f"PADTest/ARL1_shift_{delta:g}_skipped_total": rec["skipped_total"],
                    f"PADTest/ARL1_shift_{delta:g}_affected_allocation_share": rec["affected_allocation_share"],
                    f"PADTest/ARL1_shift_{delta:g}_affected_region_rank": rec["affected_region_rank"],
                    f"PADTest/ARL1_shift_{delta:g}_seconds": rec["seconds"],
                    "PADTest/m_total": int(args.m_total),
                }
            )
        print(json.dumps({"event": "arl1_eval", **rec}), flush=True)
    return records


def main():
    # 1. Parse CLI options and configure the synthetic/case data dimensions.
    cli_args = _parse_args()
    base_args = _build_base_args(cli_args)
    data_configs.configure_from_args(base_args)

    # 2. Enforce the GPU-vectorized protocol unless the user explicitly allows
    # CPU/split-batch behavior for tiny local debugging.
    if cli_args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("PAD testing protocol requires CUDA; refusing to fall back to CPU.")
    if cli_args.require_single_gpu_batch and int(cli_args.gpu_eval_batch_size) < int(cli_args.replications):
        raise RuntimeError(
            "PAD testing protocol requires all replications in one GPU batch; "
            f"got gpu_eval_batch_size={cli_args.gpu_eval_batch_size} < replications={cli_args.replications}."
        )
    if cli_args.fixed_planner_alpha is None and not cli_args.planner_model_dir:
        raise RuntimeError("--planner_model_dir is required unless --fixed_planner_alpha is set.")

    # 3. Select device and seed all random generators used by the simulator.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cli_args.gpu))
        device = torch.device(f"cuda:{int(cli_args.gpu)}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(cli_args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    run_name = cli_args.run_name or f"pad_test_M{cli_args.M_total}_N{cli_args.agent_num}_m{cli_args.m_total}"
    out_dir = Path(cli_args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "protocol": "PAD ARL0 calibration then ARL1 test",
        "M_total": int(cli_args.M_total),
        "N": int(cli_args.agent_num),
        "m_total": int(cli_args.m_total),
        "M_list": cli_args.M_list,
        "m_list": cli_args.m_list,
        "data_mode": str(cli_args.data_mode),
        "case_data_path": str(cli_args.case_data_path),
        "replications": int(cli_args.replications),
        "horizon": int(cli_args.horizon),
        "target_ARL0": float(cli_args.target),
        "target_interval": [float(cli_args.arl0_lower), float(cli_args.arl0_upper)],
        "fixed_threshold": _json_float(cli_args.fixed_threshold),
        "mean_shifts": _parse_mean_shifts(cli_args.mean_shifts),
        "num_anomalies": int(cli_args.num_anomalies),
        "anomaly_pattern": str(cli_args.anomaly_pattern),
        "planner_region_aggregator": str(cli_args.planner_region_aggregator),
        "planner_region_topk": int(cli_args.planner_region_topk),
        "force_m": bool(cli_args.force_m),
        "random_broken_stream": True,
        "stage1_model_dir": cli_args.stage1_model_dir,
        "planner_model_dir": cli_args.planner_model_dir,
        "planner_model_name": cli_args.planner_model_name,
        "fixed_planner_alpha": _json_float(cli_args.fixed_planner_alpha),
        "parallel_detector_backend": str(cli_args.parallel_detector_backend),
        "fused_detector_strict_parity": bool(cli_args.fused_detector_strict_parity),
        "fused_detector_disable_tf32": bool(cli_args.fused_detector_disable_tf32),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "M": list(map(int, data_configs.M)),
        "m": list(map(int, data_configs.m)),
        "swanlab_enabled": bool(cli_args.enable_swanlab),
    }
    if bool(cli_args.enable_swanlab):
        swanlab.init(project="PAD", name=run_name, config=config)
    print(json.dumps({"event": "start", **config}), flush=True)

    # 4. Load PAD actors, calibrate or accept a fixed threshold, then evaluate
    # ARL1 for each requested mean shift.
    runner = _build_runner(base_args, cli_args, device)
    if cli_args.fixed_threshold is None:
        calibration_best, calibration_records = _calibrate_threshold(runner, cli_args)
        threshold = calibration_best["threshold"]
    else:
        threshold = float(cli_args.fixed_threshold)
        calibration_records = []
        calibration_best = {
            "threshold": threshold,
            "ARL0": None,
            "stderr": None,
            "replications": int(cli_args.replications),
            "horizon": int(cli_args.horizon),
            "fixed_threshold": True,
        }
        if bool(cli_args.enable_swanlab):
            swanlab.log(
                {
                    "PADTest/final_threshold": threshold,
                    "PADTest/m_total": int(cli_args.m_total),
                }
            )
        print(json.dumps({"event": "fixed_threshold", **calibration_best}), flush=True)
    arl1_records = _test_arl1(runner, cli_args, threshold)

    output = {
        **config,
        "calibration_best": calibration_best,
        "calibration_records": calibration_records,
        "arl1_records": arl1_records,
    }
    out_path = out_dir / "pad_test_protocol_results.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    final_log = {"PADTest/final_threshold": calibration_best["threshold"]}
    if calibration_best["ARL0"] is not None:
        final_log["PADTest/final_ARL0"] = calibration_best["ARL0"]
    if calibration_best["stderr"] is not None:
        final_log["PADTest/final_ARL0_stderr"] = calibration_best["stderr"]
    if bool(cli_args.enable_swanlab):
        swanlab.log(final_log)
    print(f"saved={out_path}", flush=True)
    if bool(cli_args.enable_swanlab):
        swanlab.finish()


if __name__ == "__main__":
    main()
