import argparse

def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.
    """
    parser = argparse.ArgumentParser(description='onpolicy_algorithm', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='happo', choices=["happo"])
    parser.add_argument('--scenario_name', type=str, 
                        default='Planner and Detectors', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, 
                        default=3)
    parser.add_argument('--num_agents', type=int, 
                        default=1, help="number of players")
    parser.add_argument("--agent_num", type=int,
                        default=4, help="number of regions/agents for monitoring")
    parser.add_argument("--M_total", type=int,
                        default=20, help="total number of data streams")
    parser.add_argument("--m_total", type=int,
                        default=8, help="total number of resources")
    parser.add_argument("--M_list", type=str,
                        default=None, help="comma-separated per-region streams; sum must equal M_total")
    parser.add_argument("--m_list", type=str,
                        default=None, help="comma-separated per-region resources; sum must equal m_total")
    parser.add_argument("--delta", type=float,
                        default=1.0, help="mean shift delta for data generation.")
    parser.add_argument("--use_torch_stats", action="store_true",
                        default=True, help="use torch (CUDA if available) for stats sampling")
    parser.add_argument("--use_single_network", action='store_true', 
                        default=False)
    parser.add_argument("--experiment_name", type=str, 
                        default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, 
                        default=[0], help="Random seed for numpy/torch")
    parser.add_argument("--seed_specify", action="store_true",
                        default=True, help="Random or specify seed for numpy/torch")
    parser.add_argument("--running_id", type=int, 
                        default=1, help="the running index of experiment")
    parser.add_argument("--cuda", action='store_false', 
                        default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false',
                        default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--float32_matmul_precision", type=str,
                        choices=["highest", "high", "medium"], default="high",
                        help="torch float32 matmul precision; use highest to disable TF32-style approximation for reproduction runs.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, 
                        default=64, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, 
                        default=1, help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_eval_workers", type=int,
                        default=1, help="Number of parallel workers for evaluation runs")
    parser.add_argument("--n_test_rollout_threads", type=int,
                        default=1, help="Number of parallel envs for testing rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, 
                        default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, 
                        default=64e6, help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--n_warmup_evaluations", type=int, 
                        default=10, help="Number of evaluations for warmup")
    parser.add_argument("--n_no_improvement_thres", type=int, 
                        default=10000, help="Threshold number of evaluations with no improvement")
    parser.add_argument("--user_name", type=str, 
                        default='marl', help="[for wandb usage], to specify user's name for simply collecting training data.")
    # env parameters
    parser.add_argument("--env_name", type=str, 
                        default='PAD', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=30, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int,  
                        default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, 
                        default=128, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, 
                        default=2, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_false', 
                        default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', 
                        default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', 
                        default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, 
                        default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=True, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, 
                        default=2, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, 
                        default=10, help="Time length of chunks used to train a recurrent_policy")
    
    # optimizer parameters
    parser.add_argument("--lr", type=float, 
                        default=1e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, 
                        default=1e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, 
                        default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)


    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, 
                        default=15, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', 
                        default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, 
                        default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, 
                        default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, 
                        default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_false', 
                        default=False, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, 
                        default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=True, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', 
                        default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false',
                        default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", action='store_false',
                        default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, 
                        default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--save_interval", type=int, 
                        default=1, help="time duration between contiunous twice models saving.")
    parser.add_argument("--checkpoint_root", type=str,
                        default=None, help="optional root directory for model/eval checkpoints.")
    parser.add_argument("--run_root", type=str,
                        default=None, help="optional root directory for non-checkpoint run outputs such as TensorBoard logs.")
    parser.add_argument("--save_eval_checkpoints", action="store_true",
                        default=False, help="save actor/critic snapshots after every planner eval.")
    parser.add_argument("--log_interval", type=int, 
                        default=1, help="time duration between contiunous twice log printing.")
    parser.add_argument("--model_dir", type=str, 
                        default=None, help="by default None. set the path to pretrained model.")
    parser.add_argument("--stage1_model_dir", type=str,
                        default=None, help="directory containing fixed detector actor weights for stage-2 planner training.")
    parser.add_argument("--planner_alarm_threshold", type=float,
                        default=50.0, help="alarm threshold H used by planner for Lambda_t.")
    parser.add_argument("--planner_alpha_bins", type=int,
                        default=11, help="number of Bernoulli heads used to parameterize planner alpha in [0,1].")
    parser.add_argument("--planner_step_verbose", action="store_true",
                        default=False, help="print per-step planner rollout details.")
    parser.add_argument("--debug_validate_actions", action="store_true",
                        default=False, help="enable expensive action-count validation checks.")
    parser.add_argument("--disable_parallel_detectors", action="store_true",
                        default=False, help="disable batched N-detector inference and use the serial detector loop.")
    parser.add_argument("--parallel_detector_backend", type=str,
                        default="fused", choices=["serial", "streams", "fused"],
                        help="detector execution backend for fixed Stage II/test detectors.")
    parser.add_argument("--fused_detector_strict_parity", action="store_true",
                        default=True, help="check fused detector actions against the serial detector path before using it.")
    parser.add_argument("--no_fused_detector_strict_parity", dest="fused_detector_strict_parity",
                        action="store_false", help="skip the fused detector serial parity guard.")
    parser.add_argument("--fused_detector_disable_tf32", action="store_true",
                        default=True, help="temporarily disable TF32 inside fused detector inference for stable action parity.")
    parser.add_argument("--no_fused_detector_disable_tf32", dest="fused_detector_disable_tf32",
                        action="store_false", help="allow TF32 inside fused detector inference.")
    parser.add_argument("--planner_eval_interval", type=int,
                        default=100, help="evaluate planner every N episodes during training.")
    parser.add_argument("--planner_eval_episodes", type=int,
                        default=2000, help="number of episodes for each planner evaluation.")
    parser.add_argument("--planner_eval_horizon", type=int,
                        default=2000, help="max steps per planner evaluation episode.")
    parser.add_argument("--planner_eval_delta", type=float,
                        default=1.0, help="mean shift used in planner evaluation episodes.")
    parser.add_argument("--planner_eval_start_index", type=int,
                        default=0, help="fixed start index for common-random-number planner eval replications.")
    parser.add_argument("--planner_eval_random_broken_streams", action="store_true",
                        default=False, help="sample random broken streams in planner eval instead of cyclic stream indices.")
    parser.add_argument("--planner_select_eval_interval", type=int,
                        default=0,
                        help="run a larger real-detector GPU eval every N planner updates for best-checkpoint selection. 0 uses the regular eval for selection.")
    parser.add_argument("--planner_select_eval_episodes", type=int,
                        default=5000, help="number of replications for selection-only planner eval.")
    parser.add_argument("--planner_select_eval_horizon", type=int,
                        default=None, help="horizon for selection-only planner eval. Defaults to --planner_eval_horizon.")
    parser.add_argument("--planner_select_eval_start_index", type=int,
                        default=0, help="fixed start index for selection-only common-random-number eval.")
    parser.add_argument("--planner_region_aggregator", type=str,
                        default="max", choices=["max", "mean", "sum", "topk"],
                        help="regional statistic used in planner observations and statistic-driven allocation.")
    parser.add_argument("--planner_region_topk", type=int,
                        default=3, help="k used when --planner_region_aggregator=topk.")
    parser.add_argument("--use_gpu_eval", action="store_true",
                        default=True, help="use batched torch/CUDA simulator for planner eval and non-debug testing.")
    parser.add_argument("--no_gpu_eval", dest="use_gpu_eval", action="store_false",
                        help="disable batched torch/CUDA planner eval/testing and use the legacy env path.")
    parser.add_argument("--gpu_eval_batch_size", type=int,
                        default=1024, help="number of replications per GPU eval/test batch.")
    parser.add_argument("--gpu_stage2", action="store_true",
                        default=False, help="use fully vectorized GPU Stage II planner training.")
    parser.add_argument("--gpu_stage2_batch_size", type=int,
                        default=None, help="number of Stage II rollout replications kept on GPU. Defaults to --n_rollout_threads.")
    parser.add_argument("--gpu_stage2_minibatch_size", type=int,
                        default=262144, help="target PPO minibatch samples for GPU Stage II planner training.")
    parser.add_argument("--gpu_stage2_train_horizon", type=int,
                        default=2000, help="synthetic data horizon used by GPU Stage II training episodes.")
    parser.add_argument("--gpu_stage2_train_t_break", type=int,
                        default=50, help="change point used by GPU Stage II training episodes.")
    parser.add_argument("--gpu_stage2_train_stop_active_fraction", type=float,
                        default=0.0,
                        help="stop a GPU Stage II rollout once the still-active fraction is at or below this value.")
    parser.add_argument("--gpu_stage2_use_surrogate", action="store_true",
                        default=False, help="train GPU Stage II with a detector-response table surrogate.")
    parser.add_argument("--gpu_stage2_surrogate_pretrain_updates", type=int,
                        default=0,
                        help="number of initial GPU Stage II updates trained in the detector-response surrogate before switching to real detectors. If 0 with --gpu_stage2_use_surrogate, use surrogate for all updates.")
    parser.add_argument("--gpu_stage2_surrogate_reps", type=int,
                        default=256, help="replications per region/stream/allocation cell when building the GPU detector-response table.")
    parser.add_argument("--gpu_stage2_surrogate_horizon", type=int,
                        default=300, help="time-since-change horizon stored in the GPU detector-response table.")
    parser.add_argument("--gpu_stage2_surrogate_pool_size", type=int,
                        default=2000, help="synthetic sample pool size used to build the GPU detector-response table.")
    parser.add_argument("--gpu_stage2_surrogate_delta", type=float,
                        default=None, help="mean shift used to build the surrogate table. Defaults to --planner_eval_delta.")
    parser.add_argument("--gpu_stage2_surrogate_gain", type=float,
                        default=1.0, help="multiplicative scale applied to surrogate statistic increments.")
    parser.add_argument("--gpu_stage2_surrogate_baseline_increment", type=float,
                        default=0.0, help="small optional in-control regional statistic drift in surrogate rollouts.")
    parser.add_argument("--gpu_stage2_surrogate_stat_bins", type=int,
                        default=1,
                        help="number of current-statistic bins in the detector-response surrogate table. 1 disables stat conditioning.")
    parser.add_argument("--gpu_stage2_surrogate_stat_max", type=float,
                        default=None,
                        help="upper edge for surrogate statistic bins. Defaults to --planner_alarm_threshold.")
    parser.add_argument("--gpu_stage2_mixed_surrogate", action="store_true",
                        default=False,
                        help="mix surrogate and real-detector rollout rows in each GPU Stage II update.")
    parser.add_argument("--gpu_stage2_mix_rho_start", type=float,
                        default=0.7, help="initial surrogate row fraction for mixed Stage II training.")
    parser.add_argument("--gpu_stage2_mix_rho_mid", type=float,
                        default=0.2, help="middle surrogate row fraction for mixed Stage II training.")
    parser.add_argument("--gpu_stage2_mix_rho_end", type=float,
                        default=0.0, help="final surrogate row fraction for mixed Stage II training.")
    parser.add_argument("--gpu_stage2_mix_warm_updates", type=int,
                        default=20, help="number of updates to keep rho_start before decay.")
    parser.add_argument("--gpu_stage2_mix_mid_update", type=int,
                        default=50, help="update where mixed surrogate rho reaches rho_mid.")
    parser.add_argument("--gpu_stage2_mix_end_update", type=int,
                        default=100, help="update where mixed surrogate rho reaches rho_end.")
    parser.add_argument("--gpu_stage2_cf_alpha_baseline", action="store_true",
                        default=False,
                        help="blend PPO GAE with a short-horizon counterfactual alpha baseline.")
    parser.add_argument("--gpu_stage2_cf_lambda", type=float,
                        default=0.5, help="blend weight for counterfactual alpha advantage.")
    parser.add_argument("--gpu_stage2_cf_horizon", type=int,
                        default=30, help="short horizon used to estimate counterfactual alpha returns.")
    parser.add_argument("--gpu_stage2_cf_stride", type=int,
                        default=5, help="compute counterfactual advantages every N rollout steps.")
    parser.add_argument("--gpu_stage2_cf_max_states", type=int,
                        default=65536, help="max rollout states per update used for counterfactual alpha baseline.")
    parser.add_argument("--gpu_stage2_cf_chunk_size", type=int,
                        default=65536, help="chunk size for vectorized counterfactual alpha baseline.")
    parser.add_argument("--gpu_stage2_cf_alpha_candidates", type=str,
                        default="0,0.25,0.5,0.75,1.0",
                        help="comma-separated scalar alpha candidates for counterfactual baseline.")
    parser.add_argument("--gpu_stage2_stat_progress_reward", action="store_true",
                        default=False,
                        help="add a training-only dense reward for post-change progress of the global statistic toward the threshold.")
    parser.add_argument("--gpu_stage2_stat_progress_beta_start", type=float,
                        default=5.0, help="initial coefficient for the GPU Stage II statistic-progress reward.")
    parser.add_argument("--gpu_stage2_stat_progress_beta_end", type=float,
                        default=0.0, help="final coefficient for the GPU Stage II statistic-progress reward.")
    parser.add_argument("--gpu_stage2_stat_progress_anneal_updates", type=int,
                        default=100, help="number of planner updates over which to linearly anneal the statistic-progress reward coefficient.")
    parser.add_argument("--gpu_stage2_stat_progress_clip", type=float,
                        default=0.05, help="absolute per-step clip for normalized statistic progress before applying beta.")
    parser.add_argument("--gpu_stage2_diverse_bm", action="store_true",
                        default=True, help="cycle broken streams across the GPU Stage II rollout batch.")
    parser.add_argument("--no_gpu_stage2_diverse_bm", dest="gpu_stage2_diverse_bm", action="store_false",
                        help="use one broken stream per GPU Stage II update.")
    parser.add_argument("--gpu_stage2_random_broken_streams", action="store_true",
                        default=False, help="sample broken streams independently for each Stage II trajectory and episode.")
    parser.add_argument("--gpu_stage2_reset_rnn_each_episode", action="store_true",
                        default=False, help="reset planner actor/critic RNN states at every independent GPU Stage II episode.")
    parser.add_argument("--gpu_stage1", action="store_true",
                        default=False, help="use fully vectorized GPU Stage I detector training.")
    parser.add_argument("--gpu_stage1_batch_size", type=int,
                        default=None, help="number of Stage I rollout replications kept on GPU. Defaults to --n_rollout_threads.")
    parser.add_argument("--multi_discrete_budget_mode", type=str,
                        choices=["topk_mask", "all_then_repair"], default="topk_mask",
                        help="budget handling for MultiDiscrete detector actions. topk_mask preserves legacy behavior; "
                             "all_then_repair samples across all valid streams and repairs actions to the budget.")
    parser.add_argument("--gpu_stage1_train_start_step", type=int,
                        default=None,
                        help="absolute data step at which Stage I training rollouts start. "
                             "Default None preserves legacy behavior and starts at the change point.")
    parser.add_argument("--gpu_stage1_eval_batch_size", type=int,
                        default=None, help="GPU batch size for Stage I in-train detector eval. Defaults to --gpu_stage1_batch_size.")
    parser.add_argument("--gpu_stage1_eval_episodes", type=int,
                        default=1000, help="number of GPU-vectorized Stage I eval episodes.")
    parser.add_argument("--gpu_stage1_eval_horizon", type=int,
                        default=2050, help="max horizon for GPU-vectorized Stage I eval.")
    parser.add_argument("--gpu_stage1_eval_threshold", type=float,
                        default=50.0, help="alarm threshold used by GPU-vectorized Stage I eval.")
    parser.add_argument("--gpu_stage1_eval_random_broken_streams", action="store_true",
                        default=False, help="sample random broken streams in GPU Stage I eval instead of cyclic streams.")
    parser.add_argument("--gpu_stage1_eval_skip_prechange_alarms", action="store_true",
                        default=False, help="exclude Stage I eval replications that alarm before the change point.")
    parser.add_argument("--gpu_stage1_eval_stratified_resources", action="store_true",
                        default=False, help="cycle through the Stage I allocation pool during eval instead of sampling resources randomly.")
    parser.add_argument("--gpu_stage1_eval_common_pools", action="store_true",
                        default=False, help="use deterministic synthetic IC/OC pools across Stage I eval checkpoints.")
    parser.add_argument("--gpu_stage1_train_pool_seed", type=int,
                        default=None, help="seed for Stage I synthetic training IC/OC pools. Default None preserves legacy entropy-based pools.")
    parser.add_argument("--gpu_stage1_eval_pool_seed", type=int,
                        default=None, help="seed for Stage I synthetic eval IC/OC pools. Overrides --gpu_stage1_eval_common_pools when set.")
    parser.add_argument("--gpu_stage1_eval_preserve_rng", action="store_true",
                        default=False, help="restore torch RNG state after Stage I eval so eval cannot perturb training rollouts.")
    parser.add_argument("--gpu_stage1_legacy_metrics", action="store_true",
                        default=False, help="use legacy Stage I train metric denominators for old M50 reproduction runs.")
    parser.add_argument("--gpu_stage1_legacy_eval_logging", action="store_true",
                        default=False, help="log Stage I eval as GPU Eval ARL/Std instead of mode-specific keys.")
    parser.add_argument("--gpu_stage1_eval_use_fused_detectors", action="store_true",
                        default=False, help="use fused N-detector execution in Stage I eval when detector dimensions match.")
    parser.add_argument("--gpu_stage1_postchange_only_loss", action="store_true",
                        default=False, help="mask pre-change Stage I samples out of PPO/value losses while keeping rollout logs unchanged.")
    parser.add_argument("--gpu_stage1_eval_force_modes", type=str,
                        default="F", help="comma-separated Stage I eval modes. F uses force_m=True; E uses force_m=False.")
    parser.add_argument("--gpu_stage1_eval_single_m_choices", type=str,
                        default=None, help="comma-separated scalar resource choices for N=1 Stage I eval. "
                                           "Defaults to --gpu_stage1_single_m_choices.")
    parser.add_argument("--gpu_stage1_select_eval_interval", type=int,
                        default=0, help="run a larger Stage I eval every N updates for selection checkpoints.")
    parser.add_argument("--gpu_stage1_select_eval_episodes", type=int,
                        default=5000, help="number of replications for Stage I selection eval.")
    parser.add_argument("--gpu_stage1_select_eval_horizon", type=int,
                        default=2000, help="horizon for Stage I selection eval.")
    parser.add_argument("--gpu_stage1_select_eval_force_modes", type=str,
                        default=None, help="comma-separated selection eval modes. Defaults to --gpu_stage1_eval_force_modes.")
    parser.add_argument("--gpu_stage1_select_eval_single_m_choices", type=str,
                        default=None, help="comma-separated N=1 resource choices for selection eval.")
    parser.add_argument("--gpu_stage1_minibatch_size", type=int,
                        default=262144, help="target PPO minibatch samples for GPU Stage I training.")
    parser.add_argument("--gpu_stage1_logprob_chunk_size", type=int,
                        default=4096, help="env-batch chunk size for GPU Stage I HAPPO factor log-prob recomputation.")
    parser.add_argument("--gpu_stage1_diverse_bm", action="store_true",
                        default=True, help="cycle broken streams across GPU batch instead of using one stream per update.")
    parser.add_argument("--no_gpu_stage1_diverse_bm", dest="gpu_stage1_diverse_bm", action="store_false",
                        help="use one broken stream per GPU Stage I update.")
    parser.add_argument("--gpu_stage1_random_broken_streams", action="store_true",
                        default=False, help="sample broken streams independently for each Stage I rollout trajectory and episode.")
    parser.add_argument("--gpu_stage1_reset_rnn_each_episode", action="store_true",
                        default=False, help="reset detector actor/critic RNN states at every independent GPU Stage I episode.")
    parser.add_argument("--gpu_stage1_random_allocations", action="store_true",
                        default=False, help="sample one valid resource allocation per GPU Stage I rollout trajectory. For N=1, sample scalar m uniformly from [1, M].")
    parser.add_argument("--gpu_stage1_balanced_allocations", action="store_true",
                        default=False, help="cycle through valid Stage I resource allocations so each rollout batch is balanced.")
    parser.add_argument("--gpu_stage1_single_m_choices", type=str,
                        default=None, help="comma-separated scalar resource choices for N=1 GPU Stage I random allocations, e.g. 2,4,6. Defaults to 1..M.")
    parser.add_argument("--gpu_stage1_resource_upper_bound", type=int,
                        default=10, help="per-region upper bound used when sampling random GPU Stage I allocations.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', 
                        default=True, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--no_eval", dest="use_eval", action="store_false",
                        help="disable in-training evaluation.")
    parser.add_argument("--eval_interval", type=int, 
                        default=100, help="time duration between contiunous twice evaluation progress.")
    #parser.add_argument("--eval_episodes", type=int, 
    #                    default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', 
                        default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', 
                        default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, 
                        default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, 
                        default=0.1, help="the play interval of each rendered image in saved video.")
    parser.add_argument("--gpu", type=int,
                        default=0, help="GPU device id to use if cuda is available")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Run planner testing/evaluation only (no training).")
    parser.add_argument("--planner_model_dir", type=str, default=None,
                        help="Directory containing planner actor/critic checkpoints.")
    parser.add_argument("--planner_model_name", type=str, default="planner_best_actor.pt",
                        help="Planner actor checkpoint filename under --planner_model_dir.")
    parser.add_argument("--test_mode", choices=["test", "case"], default="test",
                        help="Testing source: held-out testing data or case data.")
    parser.add_argument("--test_replications", type=int, default=1000,
                        help="Number of testing episodes per mean shift.")
    parser.add_argument("--test_mean_shifts", type=str, default="0,0.2,0.4,0.6,0.8,1.0",
                        help='Comma-separated mean shifts for testing, e.g. "0,0.5,1.0".')
    parser.add_argument("--test_threshold", type=float, default=None,
                        help="Use a fixed alarm threshold for testing. If omitted, calibrate ARL0 to ~370.")
    parser.add_argument("--target_arl0", type=float, default=370.0,
                        help="Target ARL0 used in threshold calibration.")
    parser.add_argument("--arl0_lower", type=float, default=360.0,
                        help="Lower bound of acceptable ARL0.")
    parser.add_argument("--arl0_upper", type=float, default=380.0,
                        help="Upper bound of acceptable ARL0.")
    parser.add_argument("--calibration_iters", type=int, default=18,
                        help="Bisection iterations for threshold calibration.")
    parser.add_argument("--calibration_runs", type=int, default=None,
                        help="Calibration episode count. Default: --test_replications.")
    parser.add_argument("--force_m", action="store_true", default=False,
                        help="Force each detector to use all allocated resources.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode: single worker and save best episode trajectory.")
    # Baseline-style aliases for planner --test mode
    parser.add_argument("--M", type=int, default=None,
                        help="Alias of --M_total for test convenience.")
    parser.add_argument("--N", type=int, default=None,
                        help="Alias of --agent_num for test convenience.")
    parser.add_argument("--m", type=int, default=None,
                        help="Alias of --m_total for test convenience.")
    parser.add_argument("--num-anomalies", dest="num_anomalies", type=int, default=None,
                        help="Baseline-style argument (currently accepted for CLI compatibility).")
    parser.add_argument("--mean-shift", dest="mean_shift", type=float, default=None,
                        help="Alias of --delta / --test_mean_shifts for test convenience.")
    parser.add_argument("--change-point", dest="change_point", type=int, default=None,
                        help="Baseline-style argument (currently accepted for CLI compatibility).")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Alias of --planner_eval_horizon for test convenience.")
    parser.add_argument("--replications", type=int, default=None,
                        help="Alias of --test_replications for test convenience.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Alias of --n_eval_workers for test convenience.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Alias of --test_threshold for test convenience.")
    parser.add_argument("--data-mode", dest="data_mode", choices=["synthetic", "case", "eval_case", "eval_pi", "eval_tep"], default=None,
                        help="Baseline-style data source selector for --test mode.")
    parser.add_argument("--case-data-path", dest="case_data_path", type=str,
                        default="datasets/IoT_case.pkl", help="Path to the case-study data file or directory.")
    parser.add_argument("--gpu_stage1_train_t_break", type=int,
                        default=50, help="change point used by GPU Stage I train/eval episodes.")
    parser.add_argument("--gpu_stage1_train_t_break_choices", type=str,
                        default=None, help="comma-separated per-trajectory Stage I training change-point choices.")
    parser.add_argument("--gpu_stage1_train_t_break_probs", type=str,
                        default=None, help="comma-separated probabilities for --gpu_stage1_train_t_break_choices.")

    return parser
