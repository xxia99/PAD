#!/usr/bin/env bash
# Example Stage I detector training command for the best GPU protocol.
#
# This is the training recipe used for the current M=50,N=5 detector
# family: fully vectorized GPU rollouts, random regional allocations, random
# broken streams, H=100 training horizon, change point 50, and fused eval.
#
# Usage:
#   source scripts/train_stage1_m50_example.sh 0 20
#
# Args:
#   1: CUDA device id
#   2: total budget m, default 20

set -euo pipefail

GPU_ID="${1:-0}"
M_TOTAL="${2:-20}"
RUN_ROOT="${PAD_RUN_ROOT:-results_stage1}"

python train_online_monitoring_gpu.py \
  --M_total 50 \
  --m_total "${M_TOTAL}" \
  --agent_num 5 \
  --gpu "${GPU_ID}" \
  --run_root "${RUN_ROOT}" \
  --experiment_name "stage1_m50_n5_m${M_TOTAL}_best_gpu_protocol" \
  --num_env_steps 819200000 \
  --episode_length 100 \
  --gamma 0.99 \
  --gpu_stage1_batch_size 16384 \
  --gpu_stage1_random_allocations \
  --gpu_stage1_resource_upper_bound 10 \
  --gpu_stage1_random_broken_streams \
  --gpu_stage1_train_t_break 50 \
  --gpu_stage1_eval_use_fused_detectors \
  --gpu_stage1_eval_preserve_rng \
  --gpu_stage1_eval_episodes 1000 \
  --gpu_stage1_eval_horizon 1050 \
  --gpu_stage1_eval_threshold 50 \
  --eval_interval 10 \
  --save_interval 10 \
  --ppo_epoch 15 \
  --num_mini_batch 1 \
  --n_training_threads 1
