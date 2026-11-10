from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
import torch.nn.functional as F
# from envs.env_utils.data_configs import m
from torch.nn.utils.rnn import pad_sequence


def _m_max(m):
    """Return max selectable count whether m is scalar or iterable."""
    if isinstance(m, int):
        return int(m)
    if torch.is_tensor(m):
        if m.numel() == 1:
            return int(m.item())
        return int(torch.max(m).item())
    return int(max(m))


def _to_count_vector(value, batch_size, default_count, device):
    """Convert scalar/list/tensor counts to a clamped [batch_size] tensor."""
    if value is None:
        counts = torch.full((batch_size,), int(default_count), device=device, dtype=torch.long)
    elif isinstance(value, (int, float)):
        counts = torch.full((batch_size,), int(value), device=device, dtype=torch.long)
    elif torch.is_tensor(value):
        counts = value.to(device=device).reshape(-1).long()
    else:
        counts = torch.as_tensor(value, device=device).reshape(-1).long()

    if counts.numel() == 1 and batch_size > 1:
        counts = counts.expand(batch_size)
    if counts.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} budget counts, got {counts.numel()}.")
    return counts.clamp(min=0, max=int(default_count))


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.debug_validate_actions = getattr(args, "debug_validate_actions", False) if args is not None else False
        self.budget_mode = getattr(args, "multi_discrete_budget_mode", "topk_mask") if args is not None else "topk_mask"
        self.action_type = action_space.__class__.__name__
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            self.hidden_layers = torch.nn.ModuleList()
            for action_dim in action_dims:
                self.hidden_layers.append(torch.nn.Linear(inputs_dim, action_dim))
                # self.action_outs.append(Categorical(action_dim, action_dim, use_orthogonal, gain))
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain, args),
                                              Categorical(inputs_dim, discrete_dim, use_orthogonal, gain)])

    def _multi_discrete_raw_logits(self, x, available_actions=None, m=None):
        action_logits = [F.softplus(head(x)) for head in self.hidden_layers]
        ac_tensor = torch.stack(action_logits, dim=0)
        num_heads, batch_size = ac_tensor.shape[0], ac_tensor.shape[1]
        device = ac_tensor.device

        valid_counts = _to_count_vector(available_actions, batch_size, num_heads, device)
        budgets = _to_count_vector(m, batch_size, num_heads, device)
        budgets = torch.minimum(budgets, valid_counts)

        head_ids = torch.arange(num_heads, device=device).unsqueeze(1)
        valid_head_mask = head_ids < valid_counts.unsqueeze(0)
        return ac_tensor, valid_head_mask, budgets

    def _multi_discrete_logits_and_mask(self, x, available_actions=None, m=None):
        ac_tensor, valid_head_mask, budgets = self._multi_discrete_raw_logits(x, available_actions, m)
        num_heads = ac_tensor.shape[0]
        device = ac_tensor.device

        observe_scores = ac_tensor[:, :, 1].masked_fill(~valid_head_mask, -1e20)
        sorted_indices = torch.argsort(observe_scores, dim=0, descending=True)
        rank_mask = torch.arange(num_heads, device=device).unsqueeze(1) < budgets.unsqueeze(0)
        topk_mask = torch.zeros_like(observe_scores)
        topk_mask.scatter_(0, sorted_indices, rank_mask.to(topk_mask.dtype))
        topk_mask = topk_mask * valid_head_mask.to(topk_mask.dtype)

        ac_tensor[:, :, 1] = ac_tensor[:, :, 1].masked_fill(topk_mask == 0, -1e20)
        ac_tensor[:, :, 1] = ac_tensor[:, :, 1].masked_fill(~valid_head_mask, -1e20)
        return ac_tensor, topk_mask, budgets

    def _repair_to_budget(self, actions, action_scores, budgets):
        """Clamp sampled MultiDiscrete actions to per-row budgets without pre-masking exploration."""
        actions = actions.long()
        budgets = budgets.long()
        counts = actions.sum(dim=1)
        over_mask = counts > budgets
        if not torch.any(over_mask):
            return actions

        repaired = actions.clone()
        max_budget = int(budgets.max().item()) if budgets.numel() else 0
        for budget in range(max_budget + 1):
            rows = over_mask & (budgets == budget)
            if not torch.any(rows):
                continue
            if budget <= 0:
                repaired[rows] = 0
                continue
            row_scores = action_scores[rows].masked_fill(actions[rows] == 0, -1e20)
            keep = torch.topk(row_scores, k=budget, dim=1).indices
            next_actions = torch.zeros_like(repaired[rows])
            next_actions.scatter_(1, keep, 1)
            repaired[rows] = next_actions
        return repaired

    def _forward_all_then_repair(self, x, available_actions=None, deterministic=False, force_m=False, m=None):
        ac_tensor, valid_head_mask, budgets = self._multi_discrete_raw_logits(x, available_actions, m)
        ac_tensor[:, :, 1] = ac_tensor[:, :, 1].masked_fill(~valid_head_mask, -1e20)
        new_logits = F.softmax(ac_tensor, dim=2)
        action_scores = new_logits[:, :, 1].transpose(0, 1)

        if force_m:
            actions = torch.zeros((new_logits.shape[1], new_logits.shape[0]), device=x.device, dtype=torch.long)
            max_budget = int(budgets.max().item()) if budgets.numel() else 0
            for budget in range(max_budget + 1):
                rows = budgets == budget
                if not torch.any(rows) or budget <= 0:
                    continue
                selected = torch.topk(action_scores[rows], k=budget, dim=1).indices
                next_actions = torch.zeros_like(actions[rows])
                next_actions.scatter_(1, selected, 1)
                actions[rows] = next_actions
            log_probs = torch.log(new_logits.clamp_min(1e-20))
            action_log_probs = log_probs.gather(
                2,
                actions.transpose(0, 1).unsqueeze(-1),
            ).squeeze(-1).transpose(0, 1)
            return actions, action_log_probs

        if deterministic:
            actions = torch.argmax(new_logits, dim=2).transpose(0, 1).long()
        else:
            sampled = [
                torch.distributions.Categorical(probs=new_logits[i]).sample()
                for i in range(new_logits.shape[0])
            ]
            actions = torch.stack(sampled, dim=1).long()

        actions = self._repair_to_budget(actions, action_scores, budgets)
        log_probs = torch.log(new_logits.clamp_min(1e-20))
        action_log_probs = log_probs.gather(
            2,
            actions.transpose(0, 1).unsqueeze(-1),
        ).squeeze(-1).transpose(0, 1)
        return actions, action_log_probs
    
    def forward(self, x, available_actions=None, deterministic=False, force_m=False, m=None):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.budget_mode == "all_then_repair":
            return self._forward_all_then_repair(x, available_actions, deterministic, force_m, m)

        ac_tensor, new_tensor, budgets = self._multi_discrete_logits_and_mask(x, available_actions, m)
        if force_m:
            # Fast-path used by fixed detectors in Stage-2 planner training.
            # No categorical sampling/log-prob computation is needed because the
            # final action is the deterministic top-k mask.
            actions = torch.transpose(new_tensor, 0, 1).long()
            action_log_probs = torch.ones_like(actions)
            return actions, action_log_probs

        new_logits = F.softmax(ac_tensor, dim=2)
        if deterministic:
            actions = torch.argmax(new_logits, dim=2).transpose(0, 1).long()
        else:
            sampled = [
                torch.distributions.Categorical(probs=new_logits[i]).sample()
                for i in range(new_logits.shape[0])
            ]
            actions = torch.stack(sampled, dim=1).long()

        # Ensure no more than m elements in each sampled action.
        actions_m = torch.transpose(new_tensor, 0, 1).long()
        sampled_counts = actions.sum(dim=1)
        over_mask = sampled_counts > budgets
        if torch.any(over_mask):
            actions[over_mask] = actions_m[over_mask]

        log_probs = torch.log(new_logits.clamp_min(1e-20))
        action_log_probs = log_probs.gather(
            2,
            actions.transpose(0, 1).unsqueeze(-1),
        ).squeeze(-1).transpose(0, 1)

        if self.debug_validate_actions:
            final_counts = actions.sum(dim=1)
            assert torch.all(final_counts <= budgets), (
                f"Invalid sampled action counts {final_counts.tolist()} with budgets={budgets.tolist()}"
            )

        return actions, action_log_probs

    # def get_probs(self, x, available_actions=None):
    #     """
    #     Compute action probabilities from inputs.
    #     :param x: (torch.Tensor) input to network.
    #     :param available_actions: (torch.Tensor) denotes which actions are available to agent
    #                               (if None, all actions available)
    #
    #     :return action_probs: (torch.Tensor)
    #     """
    #     if self.mixed_action or self.multi_discrete:
    #         action_probs = []
    #         for action_out in self.action_outs:
    #             action_logit = action_out(x)
    #             action_prob = action_logit.probs
    #             action_probs.append(action_prob)
    #         action_probs = torch.cat(action_probs, -1)
    #     else:
    #         action_logits = self.action_out(x, available_actions)
    #         action_probs = action_logits.probs
    #
    #     return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, m=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action = torch.transpose(action.long(), 0, 1)
        action_log_probs = []
        dist_entropy = []

        if self.budget_mode == "all_then_repair":
            ac_tensor, valid_head_mask, _ = self._multi_discrete_raw_logits(x, available_actions, m)
            ac_tensor[:, :, 1] = ac_tensor[:, :, 1].masked_fill(~valid_head_mask, -1e20)
        else:
            ac_tensor, _, _ = self._multi_discrete_logits_and_mask(x, available_actions, m)
        new_logits = F.softmax(ac_tensor, dim=2)

        for act, action_logit in zip(action, new_logits):
            action_dist = torch.distributions.Categorical(probs=action_logit)
            act_log_prob = action_dist.log_prob(act).reshape(act.shape[0], -1)
            action_log_probs.append(act_log_prob)
            entropy = action_dist.entropy()
            if active_masks is not None:
                mask = active_masks.squeeze(-1)
                dist_entropy.append((entropy * mask).sum() / mask.sum().clamp_min(1.0))
            else:
                dist_entropy.append(entropy.mean())

        action_log_probs = torch.cat(action_log_probs, -1)
        dist_entropy = torch.stack(dist_entropy).mean()

        return action_log_probs, dist_entropy

    # def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
    #     """
    #     Compute log probability and entropy of given actions.
    #     :param x: (torch.Tensor) input to network.
    #     :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
    #     :param available_actions: (torch.Tensor) denotes which actions are available to agent
    #                                                           (if None, all actions available)
    #     :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
    #
    #     :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
    #     :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
    #     """
    #     if self.mixed_action:
    #         a, b = action.split((2, 1), -1)
    #         b = b.long()
    #         action = [a, b]
    #         action_log_probs = []
    #         dist_entropy = []
    #         for action_out, act in zip(self.action_outs, action):
    #             action_logit = action_out(x)
    #             action_log_probs.append(action_logit.log_probs(act))
    #             if active_masks is not None:
    #                 if len(action_logit.entropy().shape) == len(active_masks.shape):
    #                     dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum())
    #                 else:
    #                     dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
    #             else:
    #                 dist_entropy.append(action_logit.entropy().mean())
    #
    #         action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
    #         dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98
    #
    #     elif self.multi_discrete:
    #         action = torch.transpose(action, 0, 1)
    #         action_log_probs = []
    #         dist_entropy = []
    #         for action_out, act in zip(self.action_outs, action):
    #             action_logit = action_out(x)
    #             action_log_probs.append(action_logit.log_probs(act))
    #             if active_masks is not None:
    #                 dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
    #             else:
    #                 dist_entropy.append(action_logit.entropy().mean())
    #
    #         action_log_probs = torch.cat(action_log_probs, -1)
    #         dist_entropy = torch.tensor(dist_entropy).mean()
    #
    #     else:
    #         action_logits = self.action_out(x, available_actions)
    #         action_log_probs = action_logits.log_probs(action)
    #         if active_masks is not None:
    #             if self.action_type == "Discrete":
    #                 dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
    #             else:
    #                 dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
    #         else:
    #             dist_entropy = action_logits.entropy().mean()
    #
    #     return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector,-1)
            action_std = torch.cat(std_collector,-1)
            all_probs = torch.cat(probs_collector,-1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
