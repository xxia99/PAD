#!/usr/bin/env bash
# Example Stage I detector training command for the single-region M=15 protocol.
#
# This trains one detector with budgets sampled from {2,4,6}. There is no
# Stage II planner when N=1.

set -euo pipefail

GPU_ID="${1:-0}"
RUN_ROOT="${PAD_RUN_ROOT:-results_stage1_m15}"

python train_online_monitoring_gpu.py \
  --algorithm_name happo \
  --seed 0 \
  --M_total 15 \
  --m_total 4 \
  --agent_num 1 \
  --delta 1 \
  --episode_length 100 \
  --gamma 0.99 \
  --num_env_steps 4915200000 \
  --gpu "${GPU_ID}" \
  --gpu_stage1_batch_size 49152 \
  --gpu_stage1_minibatch_size 262144 \
  --gpu_stage1_logprob_chunk_size 4096 \
  --gpu_stage1_train_start_step 0 \
  --gpu_stage1_train_t_break 50 \
  --eval_interval 10 \
  --gpu_stage1_eval_episodes 1000 \
  --gpu_stage1_eval_batch_size 4096 \
  --gpu_stage1_eval_horizon 1050 \
  --gpu_stage1_eval_threshold 50 \
  --gpu_stage1_eval_random_broken_streams \
  --gpu_stage1_eval_skip_prechange_alarms \
  --gpu_stage1_eval_stratified_resources \
  --gpu_stage1_eval_force_modes E,F \
  --log_interval 1 \
  --save_interval 10 \
  --n_training_threads 1 \
  --multi_discrete_budget_mode topk_mask \
  --gpu_stage1_random_allocations \
  --gpu_stage1_single_m_choices 2,4,6 \
  --gpu_stage1_random_broken_streams \
  --gpu_stage1_reset_rnn_each_episode \
  --gpu_stage1_resource_upper_bound 10 \
  --run_root "${RUN_ROOT}" \
  --experiment_name "stage1_m15_n1_randm246_best_gpu_protocol"

