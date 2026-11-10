import torch
import torch.nn.functional as F
from contextlib import contextmanager


class ParallelDetectorExecutor:
    """Batched executor for fixed regional detector actors with equal dimensions.

    The current Stage II/test path keeps one detector policy per region. This
    helper runs those separate detector weights in one batched tensor program
    when the detector architectures and action dimensions match. It is intended
    for fixed detector inference, not detector training.
    """

    def __init__(
        self,
        detector_policies,
        device,
        disabled=False,
        backend="streams",
        strict_parity=True,
        fused_disable_tf32=True,
    ):
        self.detector_policies = list(detector_policies)
        self.device = device
        self.disabled = bool(disabled)
        self.backend = "serial" if self.disabled else str(backend or "streams").lower()
        self.strict_parity = bool(strict_parity)
        self.fused_disable_tf32 = bool(fused_disable_tf32)
        self.enabled = False
        self.reason = "disabled" if self.disabled else ""
        self.state = {}
        self.streams = []
        self.parity_checked = False
        self.parity_failed = False
        self.last_parity = None
        if not self.disabled:
            self._build()

    def _equal_action_dim(self):
        try:
            dims = [len(policy.actor.act.hidden_layers) for policy in self.detector_policies]
        except AttributeError:
            return False
        if not dims or len(set(dims)) != 1:
            return False
        self.action_dim = int(dims[0])
        self.num_regions = len(dims)
        return True

    def _build(self):
        if not self.detector_policies:
            self.reason = "no detector policies"
            return
        if self.backend == "serial":
            self.reason = "serial"
            return
        if not self._equal_action_dim():
            self.reason = "unequal detector action dimensions"
            return
        if self.backend == "streams":
            if torch.device(self.device).type != "cuda":
                self.reason = "streams require cuda"
                return
            self.streams = [torch.cuda.Stream(device=self.device) for _ in self.detector_policies]
            self.enabled = True
            self.reason = "cuda_streams"
            return
        if self.backend != "fused":
            self.reason = f"unknown backend: {self.backend}"
            return
        self._build_state()

    def _build_state(self):
        if not self.detector_policies:
            self.reason = "no detector policies"
            return

        actors = [policy.actor for policy in self.detector_policies]
        first = actors[0]
        try:
            action_dims = [len(actor.act.hidden_layers) for actor in actors]
            obs_dims = [actor.base.mlp.fc1[0].in_features for actor in actors]
            hidden_sizes = [actor.hidden_size for actor in actors]
            recurrent_layers = [actor._recurrent_N for actor in actors]
            recurrent_flags = [
                bool(actor._use_naive_recurrent_policy or actor._use_recurrent_policy)
                for actor in actors
            ]
        except AttributeError as exc:
            self.reason = f"unsupported actor structure: {exc}"
            return

        if len(set(action_dims)) != 1 or len(set(obs_dims)) != 1:
            self.reason = "unequal detector obs/action dimensions"
            return
        if len(set(hidden_sizes)) != 1 or len(set(recurrent_layers)) != 1 or len(set(recurrent_flags)) != 1:
            self.reason = "unequal detector recurrent settings"
            return
        if not first.act.multi_discrete:
            self.reason = "detector action space is not MultiDiscrete"
            return

        self.num_regions = len(actors)
        self.action_dim = int(action_dims[0])
        self.obs_dim = int(obs_dims[0])
        self.hidden_size = int(hidden_sizes[0])
        self.recurrent_N = int(recurrent_layers[0])
        self.use_recurrent = bool(recurrent_flags[0])
        self.activation = "relu" if bool(first.args.use_ReLU) else "tanh"
        self.feature_norm_eps = (
            float(first.base.feature_norm.eps) if getattr(first.base, "_use_feature_normalization", False) else None
        )
        self.rnn_norm_eps = float(first.rnn.norm.eps) if self.use_recurrent else None
        self.fc1_norm_eps = float(first.base.mlp.fc1[2].eps)
        self.fc2_norm_eps = [float(layer[2].eps) for layer in first.base.mlp.fc2]

        self._stack_parameters(actors)
        self.enabled = True
        self.reason = "fused_pending_parity" if self.strict_parity else "fused"

    @contextmanager
    def _tf32_guard(self):
        if not (self.fused_disable_tf32 and torch.device(self.device).type == "cuda"):
            yield
            return
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def _stack_linear(self, modules, attr_path):
        layers = []
        for module in modules:
            layer = module
            for attr in attr_path:
                layer = layer[attr] if isinstance(attr, int) else getattr(layer, attr)
            layers.append(layer)
        weight = torch.stack([layer.weight.detach() for layer in layers], dim=0).to(self.device)
        bias = torch.stack([layer.bias.detach() for layer in layers], dim=0).to(self.device)
        return weight, bias

    def _stack_layer_norm(self, modules, attr_path):
        layers = []
        for module in modules:
            layer = module
            for attr in attr_path:
                layer = layer[attr] if isinstance(attr, int) else getattr(layer, attr)
            layers.append(layer)
        weight = torch.stack([layer.weight.detach() for layer in layers], dim=0).to(self.device)
        bias = torch.stack([layer.bias.detach() for layer in layers], dim=0).to(self.device)
        return weight, bias

    def _stack_parameters(self, actors):
        state = {}
        if self.feature_norm_eps is not None:
            state["feature_norm"] = self._stack_layer_norm(actors, ["base", "feature_norm"])

        state["fc1"] = self._stack_linear(actors, ["base", "mlp", "fc1", 0])
        state["fc1_norm"] = self._stack_layer_norm(actors, ["base", "mlp", "fc1", 2])
        state["fc2"] = []
        state["fc2_norm"] = []
        for layer_id in range(len(actors[0].base.mlp.fc2)):
            state["fc2"].append(self._stack_linear(actors, ["base", "mlp", "fc2", layer_id, 0]))
            state["fc2_norm"].append(self._stack_layer_norm(actors, ["base", "mlp", "fc2", layer_id, 2]))

        if self.use_recurrent:
            rnn = {"weight_ih": [], "weight_hh": [], "bias_ih": [], "bias_hh": []}
            for layer_id in range(self.recurrent_N):
                rnn["weight_ih"].append(
                    torch.stack(
                        [getattr(actor.rnn.rnn, f"weight_ih_l{layer_id}").detach() for actor in actors],
                        dim=0,
                    ).to(self.device)
                )
                rnn["weight_hh"].append(
                    torch.stack(
                        [getattr(actor.rnn.rnn, f"weight_hh_l{layer_id}").detach() for actor in actors],
                        dim=0,
                    ).to(self.device)
                )
                rnn["bias_ih"].append(
                    torch.stack(
                        [getattr(actor.rnn.rnn, f"bias_ih_l{layer_id}").detach() for actor in actors],
                        dim=0,
                    ).to(self.device)
                )
                rnn["bias_hh"].append(
                    torch.stack(
                        [getattr(actor.rnn.rnn, f"bias_hh_l{layer_id}").detach() for actor in actors],
                        dim=0,
                    ).to(self.device)
                )
            state["rnn"] = rnn
            state["rnn_norm"] = self._stack_layer_norm(actors, ["rnn", "norm"])

        action_weights = []
        action_biases = []
        for actor in actors:
            action_weights.append(torch.stack([head.weight.detach() for head in actor.act.hidden_layers], dim=0))
            action_biases.append(torch.stack([head.bias.detach() for head in actor.act.hidden_layers], dim=0))
        state["action_weight"] = torch.stack(action_weights, dim=0).to(self.device)
        state["action_bias"] = torch.stack(action_biases, dim=0).to(self.device)
        self.state = state

    @staticmethod
    def _linear(x, weight, bias):
        return torch.einsum("bni,noi->bno", x, weight) + bias.unsqueeze(0)

    @staticmethod
    def _layer_norm(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight.unsqueeze(0) + bias.unsqueeze(0)

    def _activate(self, x):
        return F.relu(x) if self.activation == "relu" else torch.tanh(x)

    def _mlp(self, obs):
        x = obs
        if self.feature_norm_eps is not None:
            weight, bias = self.state["feature_norm"]
            x = self._layer_norm(x, weight, bias, self.feature_norm_eps)

        weight, bias = self.state["fc1"]
        x = self._linear(x, weight, bias)
        x = self._activate(x)
        weight, bias = self.state["fc1_norm"]
        x = self._layer_norm(x, weight, bias, self.fc1_norm_eps)

        for layer_id, (weight, bias) in enumerate(self.state["fc2"]):
            x = self._linear(x, weight, bias)
            x = self._activate(x)
            norm_weight, norm_bias = self.state["fc2_norm"][layer_id]
            x = self._layer_norm(x, norm_weight, norm_bias, self.fc2_norm_eps[layer_id])
        return x

    def _gru_one_step(self, x, rnn_states, masks):
        h = rnn_states * masks.unsqueeze(2)
        next_layers = []
        inp = x
        rnn = self.state["rnn"]
        for layer_id in range(self.recurrent_N):
            h_prev = h[:, :, layer_id, :]
            gi = self._linear(inp, rnn["weight_ih"][layer_id], rnn["bias_ih"][layer_id])
            gh = self._linear(h_prev, rnn["weight_hh"][layer_id], rnn["bias_hh"][layer_id])
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            reset_gate = torch.sigmoid(i_r + h_r)
            update_gate = torch.sigmoid(i_z + h_z)
            new_gate = torch.tanh(i_n + reset_gate * h_n)
            h_new = new_gate + update_gate * (h_prev - new_gate)
            next_layers.append(h_new)
            inp = h_new
        next_rnn = torch.stack(next_layers, dim=2)
        weight, bias = self.state["rnn_norm"]
        out = self._layer_norm(inp, weight, bias, self.rnn_norm_eps)
        return out, next_rnn

    def act(self, obs, rnn_states, masks, allocations, deterministic=True, force_m=True):
        if not self.enabled:
            return None
        if self.streams:
            return self._act_streams(obs, rnn_states, masks, allocations, deterministic, force_m)
        return self._act_fused_guarded(obs, rnn_states, masks, allocations, deterministic, force_m)

    def _act_streams(self, obs, rnn_states, masks, allocations, deterministic=True, force_m=True):
        if not deterministic:
            return None
        if obs.dim() != 3 or obs.shape[1] != self.num_regions:
            return None
        current_stream = torch.cuda.current_stream(device=self.device)
        actions = [None] * self.num_regions
        next_rnns = [None] * self.num_regions
        for region_id, detector_policy in enumerate(self.detector_policies):
            stream = self.streams[region_id]
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                action, _, next_rnn = detector_policy.actor(
                    obs[:, region_id],
                    rnn_states[:, region_id],
                    masks[:, region_id],
                    None,
                    deterministic=deterministic,
                    force_m=force_m,
                    m=allocations[:, region_id],
                )
                actions[region_id] = action.long()
                next_rnns[region_id] = next_rnn
        for stream in self.streams:
            current_stream.wait_stream(stream)
        return torch.stack(actions, dim=1), torch.stack(next_rnns, dim=1)

    def _act_fused_guarded(self, obs, rnn_states, masks, allocations, deterministic=True, force_m=True):
        if self.parity_failed:
            return None
        with self._tf32_guard():
            fused = self._act_fused(obs, rnn_states, masks, allocations, deterministic, force_m)
            if fused is None:
                return None
            if self.strict_parity and not self.parity_checked:
                serial_actions, serial_rnn = self.act_serial(
                    obs,
                    rnn_states,
                    masks,
                    allocations,
                    deterministic=deterministic,
                    force_m=force_m,
                )
                fused_actions, fused_rnn = fused
                action_diff = int((serial_actions != fused_actions).sum().item())
                rnn_max_abs = float((serial_rnn - fused_rnn).abs().max().item())
                self.last_parity = {
                    "action_equal": action_diff == 0,
                    "action_diff": action_diff,
                    "rnn_max_abs": rnn_max_abs,
                    "tf32_disabled": bool(self.fused_disable_tf32),
                }
                self.parity_checked = True
                if action_diff != 0:
                    self.parity_failed = True
                    self.enabled = False
                    self.reason = f"fused parity failed: action_diff={action_diff}"
                    return None
                self.reason = "fused"
            return fused

    def _act_fused(self, obs, rnn_states, masks, allocations, deterministic=True, force_m=True):
        if not deterministic:
            return None
        if obs.dim() != 3 or obs.shape[1] != self.num_regions or obs.shape[2] != self.obs_dim:
            return None
        if allocations.shape[:2] != obs.shape[:2]:
            return None

        x = self._mlp(obs)
        if self.use_recurrent:
            x, next_rnn = self._gru_one_step(x, rnn_states, masks)
        else:
            next_rnn = rnn_states

        logits = F.softplus(
            torch.einsum("bnh,naoh->bnao", x, self.state["action_weight"]) + self.state["action_bias"].unsqueeze(0)
        )
        budgets = allocations.long().clamp(min=0, max=self.action_dim)
        observe_scores = logits[..., 1]
        sorted_indices = torch.argsort(observe_scores, dim=2, descending=True)
        rank_mask = torch.arange(self.action_dim, device=obs.device).view(1, 1, -1) < budgets.unsqueeze(-1)
        topk_mask = torch.zeros_like(observe_scores)
        topk_mask.scatter_(2, sorted_indices, rank_mask.to(topk_mask.dtype))

        if force_m:
            actions = topk_mask.long()
        else:
            masked_logits = logits.clone()
            masked_logits[..., 1] = masked_logits[..., 1].masked_fill(topk_mask == 0, -1e20)
            actions = torch.argmax(masked_logits, dim=-1).long()
        return actions, next_rnn

    def act_serial(self, detector_obs, rnn_states, masks, allocations, deterministic=True, force_m=True):
        actions = []
        next_rnns = []
        for region_id, detector_policy in enumerate(self.detector_policies):
            action, _, next_rnn = detector_policy.actor(
                detector_obs[:, region_id],
                rnn_states[:, region_id],
                masks[:, region_id],
                None,
                deterministic=deterministic,
                force_m=force_m,
                m=allocations[:, region_id],
            )
            actions.append(action.long())
            next_rnns.append(next_rnn)
        return torch.stack(actions, dim=1), torch.stack(next_rnns, dim=1)
