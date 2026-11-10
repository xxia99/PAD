import torch

from utils.util import get_shape_from_act_space, get_shape_from_obs_space


def _denormalize_on_device(value_normalizer, tensor):
    if hasattr(value_normalizer, "running_mean_var"):
        mean, var = value_normalizer.running_mean_var()
        norm_axes = getattr(value_normalizer, "norm_axes", 1)
        return tensor * torch.sqrt(var)[(None,) * norm_axes] + mean[(None,) * norm_axes]
    value = value_normalizer.denormalize(tensor)
    return torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)


class GpuSeparatedReplayBuffer:
    """Replay buffer kept on the training GPU."""

    def __init__(self, args, obs_space, share_obs_space, act_space, device):
        self.device = device
        self.episode_length = int(args.episode_length)
        self.n_rollout_threads = int(args.n_rollout_threads)
        self.recurrent_N = int(args.recurrent_N)
        self.rnn_hidden_size = int(args.hidden_size)
        self.gamma = float(args.gamma)
        self.gae_lambda = float(args.gae_lambda)
        self._use_gae = bool(args.use_gae)
        self._use_proper_time_limits = bool(args.use_proper_time_limits)

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        act_shape = get_shape_from_act_space(act_space)

        self.share_obs = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            device=device,
            dtype=torch.float32,
        )
        self.obs = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            device=device,
            dtype=torch.float32,
        )
        self.rnn_states = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size),
            device=device,
            dtype=torch.float32,
        )
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)
        self.value_preds = torch.zeros((self.episode_length + 1, self.n_rollout_threads, 1), device=device)
        self.returns = torch.zeros_like(self.value_preds)
        self.actions = torch.zeros((self.episode_length, self.n_rollout_threads, act_shape), device=device)
        self.action_log_probs = torch.zeros_like(self.actions)
        self.rewards = torch.zeros((self.episode_length, self.n_rollout_threads, 1), device=device)
        self.masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, 1), device=device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)
        self.available_actions = None
        self.factor = None
        self.cf_advantages = torch.zeros_like(self.rewards)
        self.cf_masks = torch.zeros_like(self.rewards)
        self.cf_lambda = 0.0
        self.step = 0
        self.active_length = self.episode_length

    def set_active_length(self, length):
        length = int(length)
        if length < 1 or length > self.episode_length:
            raise ValueError(f"active_length must be in [1, {self.episode_length}], got {length}.")
        self.active_length = length

    def update_factor(self, factor):
        self.factor = factor

    def insert(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        del available_actions
        self.share_obs[self.step + 1].copy_(share_obs)
        self.obs[self.step + 1].copy_(obs)
        self.rnn_states[self.step + 1].copy_(rnn_states)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions.float())
        self.action_log_probs[self.step].copy_(action_log_probs.float())
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        if active_masks is not None:
            self.active_masks[self.step + 1].copy_(active_masks)
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        last = int(self.active_length)
        self.share_obs[0].copy_(self.share_obs[last])
        self.obs[0].copy_(self.obs[last])
        self.rnn_states[0].copy_(self.rnn_states[last])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[last])
        self.masks[0].copy_(self.masks[last])
        self.bad_masks[0].copy_(self.bad_masks[last])
        self.active_masks[0].copy_(self.active_masks[last])
        self.cf_advantages.zero_()
        self.cf_masks.zero_()
        self.cf_lambda = 0.0
        self.step = 0
        self.active_length = self.episode_length

    @torch.no_grad()
    def compute_returns(self, next_value, value_normalizer=None):
        active_length = int(self.active_length)
        self.value_preds[active_length].copy_(next_value)
        if self._use_gae:
            gae = torch.zeros_like(next_value)
            for step in reversed(range(active_length)):
                next_value_step = (
                    _denormalize_on_device(value_normalizer, self.value_preds[step + 1])
                    if value_normalizer is not None
                    else self.value_preds[step + 1]
                )
                curr_value_step = (
                    _denormalize_on_device(value_normalizer, self.value_preds[step])
                    if value_normalizer is not None
                    else self.value_preds[step]
                )
                delta = self.rewards[step] + self.gamma * next_value_step * self.masks[step + 1] - curr_value_step
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                if self._use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step].copy_(gae + curr_value_step)
        else:
            self.returns[active_length].copy_(next_value)
            for step in reversed(range(active_length)):
                next_return = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
                if self._use_proper_time_limits:
                    bootstrap = (
                        _denormalize_on_device(value_normalizer, self.value_preds[step])
                        if value_normalizer is not None
                        else self.value_preds[step]
                    )
                    next_return = next_return * self.bad_masks[step + 1] + bootstrap * (1.0 - self.bad_masks[step + 1])
                self.returns[step].copy_(next_return)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length = int(self.active_length)
        n_rollout_threads = self.n_rollout_threads
        batch_size = episode_length * n_rollout_threads
        if mini_batch_size is None and int(num_mini_batch) > batch_size:
            raise ValueError(f"num_mini_batch={num_mini_batch} exceeds flattened batch size={batch_size}.")

        rand = torch.randperm(batch_size, device=self.device)
        share_obs = self.share_obs[:episode_length].reshape(batch_size, *self.share_obs.shape[2:])
        obs = self.obs[:episode_length].reshape(batch_size, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:episode_length].reshape(batch_size, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:episode_length].reshape(batch_size, *self.rnn_states_critic.shape[2:])
        actions = self.actions[:episode_length].reshape(batch_size, self.actions.shape[-1])
        value_preds = self.value_preds[:episode_length].reshape(batch_size, 1)
        returns = self.returns[:episode_length].reshape(batch_size, 1)
        masks = self.masks[:episode_length].reshape(batch_size, 1)
        active_masks = self.active_masks[:episode_length].reshape(batch_size, 1)
        action_log_probs = self.action_log_probs[:episode_length].reshape(batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(batch_size, 1)
        factor = self.factor.reshape(batch_size, self.factor.shape[-1])

        if mini_batch_size is None:
            batches = torch.chunk(rand, int(num_mini_batch))
        else:
            batches = [rand[i : i + int(mini_batch_size)] for i in range(0, batch_size, int(mini_batch_size))]

        for indices in batches:
            if indices.numel() == 0:
                continue
            yield (
                share_obs[indices],
                obs[indices],
                rnn_states[indices],
                rnn_states_critic[indices],
                actions[indices],
                value_preds[indices],
                returns[indices],
                masks[indices],
                active_masks[indices],
                action_log_probs[indices],
                advantages[indices],
                None,
                factor[indices],
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length = int(self.active_length)
        n_rollout_threads = self.n_rollout_threads
        assert n_rollout_threads >= num_mini_batch, (
            f"n_rollout_threads={n_rollout_threads} must be >= num_mini_batch={num_mini_batch}."
        )
        perm = torch.randperm(n_rollout_threads, device=self.device)

        for indices in torch.chunk(perm, int(num_mini_batch)):
            if indices.numel() == 0:
                continue
            n_envs = int(indices.numel())

            share_obs_batch = self.share_obs[:episode_length, indices]
            obs_batch = self.obs[:episode_length, indices]
            actions_batch = self.actions[:episode_length, indices]
            value_preds_batch = self.value_preds[:episode_length, indices]
            return_batch = self.returns[:episode_length, indices]
            masks_batch = self.masks[:episode_length, indices]
            active_masks_batch = self.active_masks[:episode_length, indices]
            old_action_log_probs_batch = self.action_log_probs[:episode_length, indices]
            adv_targ = advantages[:episode_length, indices]
            factor_batch = self.factor[:episode_length, indices]
            rnn_states_batch = self.rnn_states[0, indices]
            rnn_states_critic_batch = self.rnn_states_critic[0, indices]

            yield (
                share_obs_batch.reshape(episode_length * n_envs, *self.share_obs.shape[2:]),
                obs_batch.reshape(episode_length * n_envs, *self.obs.shape[2:]),
                rnn_states_batch.reshape(n_envs, *self.rnn_states.shape[2:]),
                rnn_states_critic_batch.reshape(n_envs, *self.rnn_states_critic.shape[2:]),
                actions_batch.reshape(episode_length * n_envs, self.actions.shape[-1]),
                value_preds_batch.reshape(episode_length * n_envs, 1),
                return_batch.reshape(episode_length * n_envs, 1),
                masks_batch.reshape(episode_length * n_envs, 1),
                active_masks_batch.reshape(episode_length * n_envs, 1),
                old_action_log_probs_batch.reshape(episode_length * n_envs, self.action_log_probs.shape[-1]),
                adv_targ.reshape(episode_length * n_envs, 1),
                None,
                factor_batch.reshape(episode_length * n_envs, self.factor.shape[-1]),
            )

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        raise NotImplementedError("GPU recurrent chunk generator is not implemented; use naive recurrent or feed-forward policy.")
