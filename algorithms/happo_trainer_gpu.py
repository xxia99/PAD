import torch
import torch.nn as nn

from algorithms.happo_trainer import HAPPO
from algorithms.utils.util import check
from utils.util import get_gard_norm


def _denormalize_on_device(value_normalizer, tensor):
    if hasattr(value_normalizer, "running_mean_var"):
        mean, var = value_normalizer.running_mean_var()
        norm_axes = getattr(value_normalizer, "norm_axes", 1)
        return tensor * torch.sqrt(var)[(None,) * norm_axes] + mean[(None,) * norm_axes]
    value = value_normalizer.denormalize(tensor)
    return torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu().item())
    return float(value)


def _normalize_advantages(advantages, valid):
    valid_advantages = advantages[valid.expand_as(advantages)]
    if valid_advantages.numel() == 0:
        mean_advantages = advantages.mean()
        std_advantages = advantages.std(unbiased=False)
    else:
        mean_advantages = valid_advantages.mean()
        std_advantages = valid_advantages.std(unbiased=False)
    return (advantages - mean_advantages) / (std_advantages + 1e-5)


class HAPPOGPU(HAPPO):
    """HAPPO trainer variant that keeps rollout tensors on the target GPU."""

    def __init__(self, args, policy, device=torch.device("cpu"), action_budget="obs0"):
        super().__init__(args, policy, device=device)
        self.action_budget = action_budget

    def _action_budget_for_batch(self, obs_batch):
        if self.action_budget == "obs0":
            return obs_batch[:, 0]
        if self.action_budget in (None, "none"):
            return None
        if torch.is_tensor(self.action_budget):
            return self.action_budget.to(device=obs_batch.device)
        return self.action_budget

    def ppo_update(self, sample, update_actor=True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, factor_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            m=self._action_budget_for_batch(obs_batch),
        )

        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch), dim=-1, keepdim=True)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        active_length = int(getattr(buffer, "active_length", buffer.rewards.shape[0]))
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:active_length] - _denormalize_on_device(
                self.value_normalizer,
                buffer.value_preds[:active_length],
            )
        else:
            advantages = buffer.returns[:active_length] - buffer.value_preds[:active_length]

        valid = buffer.active_masks[:active_length] > 0.0
        valid_advantages = advantages[valid.expand_as(advantages)]
        raw_advantage_mean = _to_float(valid_advantages.mean()) if valid_advantages.numel() > 0 else 0.0
        raw_advantage_std = _to_float(valid_advantages.std(unbiased=False)) if valid_advantages.numel() > 0 else 0.0
        advantages = _normalize_advantages(advantages, valid)

        cf_lambda = float(getattr(buffer, "cf_lambda", 0.0) or 0.0)
        if cf_lambda > 0.0 and hasattr(buffer, "cf_advantages") and hasattr(buffer, "cf_masks"):
            cf_advantages = buffer.cf_advantages[:active_length]
            cf_valid = valid & (buffer.cf_masks[:active_length] > 0.0)
            if bool(cf_valid.any()):
                cf_norm = _normalize_advantages(cf_advantages, cf_valid)
                blended = (1.0 - cf_lambda) * advantages + cf_lambda * cf_norm
                advantages = torch.where(cf_valid, blended, advantages)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
            "active_mask_fraction": _to_float(valid.float().mean()),
            "raw_advantage_mean": raw_advantage_mean,
            "raw_advantage_std": raw_advantage_std,
            "policy_active_masks": float(bool(self._use_policy_active_masks)),
            "value_active_masks": float(bool(self._use_value_active_masks)),
        }

        num_updates = 0
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages,
                    self.num_mini_batch,
                    self.data_chunk_length,
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = (
                    self.ppo_update(sample, update_actor=update_actor)
                )
                train_info["value_loss"] += _to_float(value_loss)
                train_info["policy_loss"] += _to_float(policy_loss)
                train_info["dist_entropy"] += _to_float(dist_entropy)
                train_info["actor_grad_norm"] += _to_float(actor_grad_norm)
                train_info["critic_grad_norm"] += _to_float(critic_grad_norm)
                train_info["ratio"] += _to_float(imp_weights.mean())
                num_updates += 1

        if num_updates > 0:
            averaged_keys = (
                "value_loss",
                "policy_loss",
                "dist_entropy",
                "actor_grad_norm",
                "critic_grad_norm",
                "ratio",
            )
            for key in averaged_keys:
                train_info[key] /= num_updates

        return train_info
