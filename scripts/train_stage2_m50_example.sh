#!/usr/bin/env bash
# Example Stage II planner training command for the best GPU protocol.
#
# The detectors are fixed; the planner learns the scalar alpha allocation rule.
# Counterfactual alpha baselines are enabled because they gave the most stable
# planner learning in our final protocol.
#
# Usage:
#   source scripts/train_stage2_m50_example.sh 0 path/to/stage1_detector_models 20
#
# Args:
#   1: CUDA device id
#   2: Stage I detector model directory
#   3: total budget m, default 20

set -euo pipefail

GPU_ID="${1:-0}"
M_TOTAL="${3:-20}"
STAGE1_MODEL_DIR="${2:-checkpoints/m50_n5_m${M_TOTAL}_detectors}"
RUN_ROOT="${PAD_RUN_ROOT:-results_stage2}"

python train_online_planner_gpu.py \
  --M_total 50 \
  --m_total "${M_TOTAL}" \
  --agent_num 5 \
  --gpu "${GPU_ID}" \
  --run_root "${RUN_ROOT}" \
  --stage1_model_dir "${STAGE1_MODEL_DIR}" \
  --experiment_name "stage2_m50_n5_m${M_TOTAL}_best_gpu_protocol" \
  --num_env_steps 819200000 \
  --episode_length 1000 \
  --gpu_stage2_batch_size 8192 \
  --gpu_stage2_train_t_break 50 \
  --planner_alarm_threshold 50 \
  --planner_eval_interval 10 \
  --planner_eval_episodes 1000 \
  --planner_eval_horizon 1050 \
  --planner_eval_random_broken_streams \
  --planner_select_eval_interval 50 \
  --planner_select_eval_episodes 5000 \
  --planner_select_eval_horizon 1050 \
  --planner_region_aggregator max \
  --gamma 0.999 \
  --ppo_epoch 15 \
  --gpu_stage2_random_broken_streams \
  --gpu_stage2_cf_alpha_baseline \
  --gpu_stage2_cf_lambda 0.5 \
  --gpu_stage2_cf_stride 2 \
  --gpu_stage2_cf_max_states 262144 \
  --gpu_stage2_cf_alpha_candidates "0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0" \
  --parallel_detector_backend fused \
  --n_training_threads 1
