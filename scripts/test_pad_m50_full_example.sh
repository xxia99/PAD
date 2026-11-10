#!/usr/bin/env bash
# Full PAD(F) test command with ARL0 calibration followed by ARL1 evaluation.
#
# This is the direct testing protocol used by the paper-scale experiments. The
# quick reproduction scripts use fixed thresholds to reduce reviewer runtime.
#
# Usage:
#   source scripts/test_pad_m50_full_example.sh 0 path/to/stage1_detector_models path/to/stage2_planner_models 20

set -euo pipefail

GPU_ID="${1:-0}"
M_TOTAL="${4:-20}"
STAGE1_MODEL_DIR="${2:-checkpoints/m50_n5_m${M_TOTAL}_detectors}"
PLANNER_MODEL_DIR="${3:-checkpoints/m50_n5_m${M_TOTAL}_planner}"

python scripts/run_pad_test_protocol.py \
  --M_total 50 \
  --m_total "${M_TOTAL}" \
  --agent_num 5 \
  --gpu "${GPU_ID}" \
  --stage1_model_dir "${STAGE1_MODEL_DIR}" \
  --planner_model_dir "${PLANNER_MODEL_DIR}" \
  --planner_model_name "planner_best_actor.pt" \
  --replications 5000 \
  --gpu_eval_batch_size 5000 \
  --horizon 2000 \
  --mean_shifts "0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0" \
  --force_m \
  --parallel_detector_backend fused \
  --out_dir "results/pad_m50_m${M_TOTAL}_full_test" \
  --run_name "pad_F_m50_n5_m${M_TOTAL}_full"
