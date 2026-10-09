"""含位置/变量编码的因子专家预测器及其独立专家归因。"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .lrp import RelevanceTape
from .backbones import LEGACY_ENCODERS, build_legacy_encoder


@dataclass
class ModelConfig:
    n_variables: int
    lookback: int
    horizon: int
    targets: int = 1
    n_experts: int = 3
    hidden: int = 32
    heads: int = 4
    depth: int = 1
    latent_block: int = 8
    encoder: str = "temporal_transformer"
    patch_size: int = 4
    lrp_epsilon: float = 1e-6
    use_vae: bool = True
    expert_backbone: str = "native"
    official_options: dict | None = None
    target_indices: tuple[int, ...] | None = None
    use_revin: bool = False
    revin_affine: bool = True
    revin_epsilon: float = 1e-5
    official_sampling_seed: int = 42

    def __post_init__(self):
        for key in ("n_variables", "lookback", "horizon", "targets", "hidden", "heads", "depth", "latent_block"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.hidden % self.heads or self.n_experts < 1:
            raise ValueError("hidden must be divisible by heads and n_experts >= 1")
        if any(not isinstance(v, bool) for v in (self.use_vae, self.use_revin, self.revin_affine)):
            raise ValueError("use_vae, use_revin and revin_affine must be boolean")
        if self.revin_epsilon <= 0:
            raise ValueError("revin_epsilon must be positive")
        if self.use_revin and self.expert_backbone != "native":
            raise ValueError("configure native normalization of each official model in official_options; external RevIN would alter its full architecture")
        from .official_backbones import OFFICIAL_MODEL_NAMES
        if self.expert_backbone != "native" and self.expert_backbone not in OFFICIAL_MODEL_NAMES:
            raise ValueError("expert_backbone must be native or a complete official forecaster")
        if self.official_options is not None and not isinstance(self.official_options, dict):
            raise ValueError("official_options must be a mapping")
        if self.target_indices is not None:
            self.target_indices = tuple(self.target_indices)
            if len(self.target_indices) != self.targets or len(set(self.target_indices)) != self.targets or any(
                    not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= self.n_variables for i in self.target_indices):
                raise ValueError("target_indices must identify exactly the target input columns")
        elif self.expert_backbone != "native" and self.targets != self.n_variables:
            raise ValueError("official subset forecasting requires explicit target_indices")
        if self.encoder not in ("temporal_transformer", "inverted_transformer", "patch_transformer") and self.encoder not in LEGACY_ENCODERS:
            raise ValueError("unknown encoder; legacy style encoders are not official backbones")
        if self.patch_size < 1 or self.lrp_epsilon <= 0:
            raise ValueError("patch_size and lrp_epsilon must be positive")


def positions(length, hidden):
    pos = torch.arange(length).float().unsqueeze(1)
    scale = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.) / hidden))
    out = torch.zeros(length, hidden)
    out[:, 0::2] = torch.sin(pos * scale)
    out[:, 1::2] = torch.cos(pos * scale[:out[:, 1::2].shape[-1]])
    return out


class AttentionBlock(nn.Module):
    """显式注意力避免 fused-kernel 限制高阶参数梯度。"""
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.ff1 = nn.Linear(hidden, 4 * hidden)
        self.ff2 = nn.Linear(4 * hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)

    def on_tape(self, tape, node, prefix="", modulations=None):
        x = tape.value(node)
        leading, tokens, hidden = x.shape[:-2], x.shape[-2], x.shape[-1]
        def linear(n, key):
            layer = getattr(self, key)
            name = prefix + key + ".weight"
            weight = layer.weight
            if modulations and name in modulations:
                weight = weight * modulations[name]
            return tape.linear(n, weight, layer.bias, name if prefix.startswith("experts.") else None)
        def split(n):
            n = tape.reshape(n, (*leading, tokens, self.heads, self.width))
            dims = list(range(len(leading) + 3))
            dims[-3], dims[-2] = dims[-2], dims[-3]
            return tape.permute(n, tuple(dims))
        q, k, v = (split(linear(node, key)) for key in ("q", "k", "v"))
        dims = list(range(tape.value(k).ndim))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        score = tape.scale(tape.matmul(q, tape.permute(k, tuple(dims))), 1 / math.sqrt(self.width))
        context = tape.matmul(tape.softmax(score), v)
        dims = list(range(tape.value(context).ndim))
        dims[-3], dims[-2] = dims[-2], dims[-3]
        context = tape.reshape(tape.permute(context, tuple(dims)), x.shape)
        h = tape.layer_norm(tape.add(node, linear(context, "out")), self.norm1)
        ff = linear(tape.gelu(linear(h, "ff1")), "ff2")
        return tape.layer_norm(tape.add(h, ff), self.norm2)

    def forward(self, x, prefix="", modulations=None):
        # 普通预测不保存LRP闭包；归因使用同一参数的on_tape计算。
        def linear(value, key):
            layer = getattr(self, key)
            name = prefix + key + ".weight"
            if modulations and name in modulations:
                weight = layer.weight * modulations[name]
                return torch.einsum("b...i,boi->b...o", value, weight) + layer.bias
            return layer(value)
        shape = (*x.shape[:-1], self.heads, self.width)
        q, k, v = (linear(x, key).reshape(shape).transpose(-3, -2) for key in ("q", "k", "v"))
        attention = (q @ k.transpose(-2, -1) / math.sqrt(self.width)).softmax(-1)
        context = (attention @ v).transpose(-3, -2).reshape_as(x)
        h = self.norm1(x + linear(context, "out"))
        return self.norm2(h + linear(F.gelu(linear(h, "ff1")), "ff2"))


class SummaryEncoder(nn.Module):
    """三种明确命名的原生编码器；不冒称官方八骨干复现。"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.encoder in LEGACY_ENCODERS:
            self.legacy = build_legacy_encoder(config)
            return
        if config.encoder == "temporal_transformer":
            width, tokens = config.n_variables, config.lookback
        elif config.encoder == "inverted_transformer":
            width, tokens = config.lookback, config.n_variables
        else:
            width = config.patch_size
            tokens = config.n_variables * math.ceil(config.lookback / config.patch_size)
        self.embedding = nn.Linear(width, config.hidden)
        self.register_buffer("position", positions(tokens, config.hidden))
        self.blocks = nn.ModuleList([AttentionBlock(config.hidden, config.heads) for _ in range(config.depth)])

    def forward(self, x):
        if self.config.encoder in LEGACY_ENCODERS:
            return self.legacy(x)
        if self.config.encoder == "inverted_transformer":
            x = x.transpose(1, 2)
        elif self.config.encoder == "patch_transformer":
            x = x.transpose(1, 2)
            p = self.config.patch_size
            x = F.pad(x, (0, (-x.shape[-1]) % p), mode="replicate")
            x = x.unfold(-1, p, p).reshape(x.shape[0], -1, p)
        h = self.embedding(x) + self.position
        for block in self.blocks:
            h = block(h)
        return h.mean(1)


class VariableExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(1, config.hidden)
        self.variable_embedding = nn.Parameter(torch.randn(config.n_variables, config.hidden) * .02)
        self.register_buffer("lag_position", positions(config.lookback, config.hidden))
        self.blocks = nn.ModuleList([AttentionBlock(config.hidden, config.heads) for _ in range(config.depth)])

    def on_tape(self, tape, prefix, modulations=None, input_node=0):
        x = tape.value(input_node)
        node = tape.reshape(input_node, (*x.shape, 1))
        name = prefix + "embedding.weight"
        weight = self.embedding.weight
        if modulations and name in modulations:
            weight = weight * modulations[name]
        node = tape.linear(node, weight, self.embedding.bias, name)
        pos = self.lag_position[None, :, None, :] + self.variable_embedding[None, None, :, :]
        node = tape.constant_add(node, pos)
        for i, block in enumerate(self.blocks):
            node = block.on_tape(tape, node, f"{prefix}blocks.{i}.", modulations)
        return tape.mean(node, dim=2)

    def forward(self, x, prefix, modulations=None):
        name = prefix + "embedding.weight"
        if modulations and name in modulations:
            weight = self.embedding.weight * modulations[name]
            h = torch.einsum("b...i,boi->b...o", x.unsqueeze(-1), weight) + self.embedding.bias
        else:
            h = self.embedding(x.unsqueeze(-1))
        h = h + self.lag_position[None, :, None, :] + self.variable_embedding[None, None, :, :]
        for i, block in enumerate(self.blocks):
            h = block(h, f"{prefix}blocks.{i}.", modulations)
        return h.mean(2)


@dataclass
class ForecastOutput:
    prediction: Tensor
    gates: Tensor
    mu: Tensor | None
    log_var: Tensor | None
    reconstruction: Tensor | None


@dataclass
class Attribution:
    maps: Tensor
    channels: dict[str, Tensor]
    residual: Tensor
    conservation_error: Tensor
    residual_by_operation: dict[str, Tensor]


class PaperForecaster(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.use_revin and config.revin_affine:
            self.revin_weight = nn.Parameter(torch.ones(config.n_variables))
            self.revin_bias = nn.Parameter(torch.zeros(config.n_variables))
        self.encoder = SummaryEncoder(config)
        latent = config.n_experts * config.latent_block
        if config.use_vae:
            self.mu = nn.Linear(config.hidden, latent)
            self.log_var = nn.Linear(config.hidden, latent)
            self.decoder = nn.Sequential(nn.Linear(latent, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.lookback * config.n_variables))
        else:
            self.gate_features = nn.Sequential(nn.Linear(config.hidden, latent), nn.GELU())
        self.router = nn.Linear(latent, config.n_experts)
        if config.expert_backbone == "native":
            self.experts = nn.ModuleList([VariableExpert(config) for _ in range(config.n_experts)])
            self.output_head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.horizon * config.targets))
        else:
            from .official_backbones import OfficialForecastConfig, OfficialForecaster
            dimensions = dict(model_name=config.expert_backbone, lookback=config.lookback, horizon=config.horizon,
                              n_variables=config.n_variables, target_indices=config.target_indices or tuple(range(config.targets)),
                              hidden=config.hidden, heads=config.heads, depth=config.depth)
            if set(dimensions) & (config.official_options or {}).keys():
                raise ValueError("official_options cannot override dimensions or the selected backbone")
            official = OfficialForecastConfig(**dimensions, **(config.official_options or {}))
            self.experts = nn.ModuleList([OfficialForecaster(official) for _ in range(config.n_experts)])
            # Complete author forecasts are combined in common target coordinates.
            self.output_head = nn.Identity()

    def _validate_x(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.config.lookback, self.config.n_variables):
            raise ValueError("expected input [B,lookback,n_variables]")

    def _normalize_instance(self, x):
        if not self.config.use_revin:
            return x, None
        mean = x.mean(1, keepdim=True).detach()
        std = (x.var(1, keepdim=True, unbiased=False) + self.config.revin_epsilon).sqrt().detach()
        norm = (x - mean) / std
        if self.config.revin_affine:
            norm = norm * self.revin_weight + self.revin_bias
        return norm, (mean, std)

    def _denormalize_instance(self, value, stats, *, targets=True):
        if stats is None:
            return value
        indices = list(self.config.target_indices or range(self.config.targets)) if targets else list(range(self.config.n_variables))
        mean, std = (item[:, :, indices] for item in stats)
        if self.config.revin_affine:
            value = (value - self.revin_bias[indices]) / (self.revin_weight[indices] + self.config.revin_epsilon ** 2)
        return value * std + mean

    def forward(self, x, stochastic=False, modulations=None):
        self._validate_x(x)
        x, stats = self._normalize_instance(x)
        summary = self.encoder(x)
        if self.config.use_vae:
            mu, log_var = self.mu(summary), self.log_var(summary).clamp(-12, 8)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * log_var) if stochastic else mu
            reconstruction = self._denormalize_instance(self.decoder(z).reshape_as(x), stats, targets=False)
        else:
            mu = log_var = reconstruction = None
            z = self.gate_features(summary)
        gates = self.router(z).softmax(-1)
        if self.config.expert_backbone != "native":
            predictions = torch.stack([self._official_prediction(x, m, modulations) for m in range(len(self.experts))], 1)
            prediction = (predictions * gates[:, :, None, None]).sum(1)
            return ForecastOutput(prediction, gates, mu, log_var, reconstruction)
        embs = torch.stack([expert(x, f"experts.{m}.", modulations) for m, expert in enumerate(self.experts)], 1)
        merged = (embs * gates[:, :, None, None]).sum(1).mean(1)
        prediction = self.output_head(merged).reshape(x.shape[0], self.config.horizon, self.config.targets)
        return ForecastOutput(self._denormalize_instance(prediction, stats), gates, mu, log_var, reconstruction)

    def _official_prediction(self, x, expert_index, modulations=None):
        expert = self.experts[expert_index]
        if not expert.training:
            devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.config.official_sampling_seed + expert_index)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(self.config.official_sampling_seed + expert_index)
                return self._official_forward(x, expert_index, modulations)
        return self._official_forward(x, expert_index, modulations)

    def _official_forward(self, x, expert_index, modulations=None):
        expert = self.experts[expert_index]
        if not modulations:
            return expert(x)
        prefix = f"experts.{expert_index}."
        selected = {key[len(prefix):]: value for key, value in modulations.items() if key.startswith(prefix)}
        if not selected:
            return expert(x)
        if expert.training:
            raise ValueError("temporary official-backbone modulation requires eval mode")
        parameters = dict(expert.named_parameters())
        if set(selected) - parameters.keys():
            raise ValueError("unknown official expert modulation parameter")
        rows = []
        for b in range(len(x)):
            updates = {key: parameters[key] * value[b] for key, value in selected.items()}
            devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.config.official_sampling_seed + expert_index)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(self.config.official_sampling_seed + expert_index)
                rows.append(torch.func.functional_call(expert, updates, (x[b:b + 1],), strict=False))
        return torch.cat(rows)

    def expert_weights(self):
        if self.config.expert_backbone != "native":
            heads = {id(parameter) for expert in self.experts for parameter in expert.head_parameters()}
            return {name + ".weight": module.weight for name, module in self.named_modules()
                    if name.startswith("experts.") and isinstance(module, nn.Linear) and id(module.weight) not in heads}
        return {name: p for name, p in self.named_parameters() if name.startswith("experts.") and name.endswith(".weight") and p.ndim == 2}

    def expert_prediction(self, x, expert_index):
        if self.config.expert_backbone != "native":
            return self._official_prediction(x, expert_index)
        x, stats = self._normalize_instance(x)
        e = self.experts[expert_index](x, f"experts.{expert_index}.").mean(1)
        prediction = self.output_head(e).reshape(x.shape[0], self.config.horizon, self.config.targets)
        return self._denormalize_instance(prediction, stats)

    def attribute(self, x, freeze_head=True):
        """返回每位专家、目标的步均值绝对相关性；head参数按需固定。"""
        self._validate_x(x)
        if self.config.expert_backbone != "native":
            from .official_lrp import attribute_official_mixture
            return attribute_official_mixture(self, x, freeze_head=freeze_head)
        maps, residuals, errors, channels, ledgers = [], [], [], {}, []
        for m, expert in enumerate(self.experts):
            tape = RelevanceTape(x, self.config.lrp_epsilon)
            node, stats = 0, None
            if self.config.use_revin:
                _, stats = self._normalize_instance(x)
                node = tape.scale(tape.constant_add(node, -stats[0]), 1 / stats[1])
                if self.config.revin_affine:
                    node = tape.constant_add(tape.scale(node, self.revin_weight), self.revin_bias)
            node = tape.mean(expert.on_tape(tape, f"experts.{m}.", input_node=node), dim=1)
            for layer in self.output_head:
                if isinstance(layer, nn.Linear):
                    w = layer.weight.detach() if freeze_head else layer.weight
                    b = layer.bias.detach() if freeze_head else layer.bias
                    node = tape.linear(node, w, b)
                elif isinstance(layer, nn.GELU):
                    node = tape.gelu(node)
                else:
                    raise TypeError(f"No LRP rule for output-head layer {type(layer).__name__}")
            if stats is not None:
                indices = list(self.config.target_indices or range(self.config.targets))
                node = tape.reshape(node, (len(x), self.config.horizon, self.config.targets))
                if self.config.revin_affine:
                    node = tape.scale(tape.constant_add(node, -self.revin_bias[indices]),
                                      1 / (self.revin_weight[indices] + self.config.revin_epsilon ** 2))
                node = tape.constant_add(tape.scale(node, stats[1][:, :, indices]), stats[0][:, :, indices])
                node = tape.reshape(node, (len(x), -1))
            output = tape.value(node)
            expert_maps, expert_residual, expert_error = [], [], []
            expert_channels = {}
            expert_ledger = {}
            for i in range(self.config.targets):
                rows, res, err, per_target = [], [], [], {}
                target_ledger = {}
                for j in range(self.config.horizon):
                    index = j * self.config.targets + i
                    seed = torch.zeros_like(output)
                    seed[:, index] = output[:, index]
                    result = tape.propagate(node, seed)
                    rows.append(result.input_relevance.abs().transpose(1, 2))
                    res.append(result.residual)
                    err.append(result.conservation_error.abs())
                    for key, value in result.channels.items():
                        per_target.setdefault(key, []).append(value)
                    for key, value in result.residual_by_operation.items():
                        target_ledger.setdefault(key, []).append(value)
                expert_maps.append(torch.stack(rows).mean(0))
                expert_residual.append(torch.stack(res).mean(0))
                expert_error.append(torch.stack(err).amax(0))
                for key, values in per_target.items():
                    expert_channels.setdefault(key, []).append(torch.stack(values).mean(0))
                for key, values in target_ledger.items():
                    expert_ledger.setdefault(key, []).append(torch.stack(values).mean(0))
            maps.append(torch.stack(expert_maps, 1))
            residuals.append(torch.stack(expert_residual, 1))
            errors.append(torch.stack(expert_error, 1))
            for key, values in expert_channels.items():
                channels[key] = torch.stack(values, 1)
            ledgers.append({key: torch.stack(values, 1) for key, values in expert_ledger.items()})
        ledger = {key: torch.stack([entry[key] for entry in ledgers], 2) for key in ledgers[0]}
        return Attribution(torch.stack(maps, 2), channels, torch.stack(residuals, 2), torch.stack(errors, 2), ledger)


class BranchPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SummaryEncoder(config)
        self.head = nn.Linear(config.hidden, config.horizon * config.targets)
        self.config = config

    def forward(self, x):
        return self.head(self.encoder(x)).reshape(x.shape[0], self.config.horizon, self.config.targets)


class ForecastBranches(nn.Module):
    """两个独立训练的FOIL分支；仅在预测坐标中取差。"""
    def __init__(self, config, mode="foil", contrast="difference"):
        super().__init__()
        if mode not in ("foil", "erm_pair", "variant_only", "disabled"):
            raise ValueError("unknown invariant-module ablation")
        self.mode, self.config = mode, config
        if contrast not in ("difference", "variant", "invariant"):
            raise ValueError("contrast must be difference, variant, or invariant")
        if mode in ("variant_only", "disabled") and contrast == "invariant":
            raise ValueError("invariant contrast requires an invariant branch")
        self.contrast = contrast
        self.invariant = BranchPredictor(config) if mode in ("foil", "erm_pair") else None
        self.variant = BranchPredictor(config) if mode != "disabled" else None

    def forward(self, x):
        if self.variant is None:
            return x.new_zeros((len(x), self.config.horizon * self.config.targets))
        if self.contrast == "invariant":
            return self.invariant(x).flatten(1)
        if self.contrast == "variant":
            return self.variant(x).flatten(1)
        if self.invariant is None:
            return self.variant(x).flatten(1)
        return (self.variant(x) - self.invariant(x)).flatten(1)

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
