from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass
class DeltaSummary:
    delta_param_count: int
    delta_norm_pre_clamp: float
    delta_norm_post_clamp: float
    clamp_fraction: float


class IdentitySignedSeparableAdaptationGenerator(nn.Module):
    """Identity-preserving signed Stage III V2 generator.

    The legacy generator maps zero drift to a nonzero update because of biased
    layers and ``sigmoid + 1`` scaling. This V2 generator is bias-free, so
    ``drift == 0`` always maps to zero scale and therefore zero parameter
    delta. Its additive separable scale is odd around zero, so sign-flipped
    drift can produce sign-flipped deltas instead of the same positive scale.
    """

    def __init__(
        self,
        drift_dim: int,
        hidden_dim: int,
        max_dim: int,
        max_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_dim <= 0:
            raise ValueError("max_dim must be positive")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")

        self.drift_dim = drift_dim
        self.hidden_dim = hidden_dim
        self.max_dim = max_dim
        self.max_scale = max_scale
        self.net = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * max_dim, bias=False),
        )
        self._reset_parameters()

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "drift_dim": self.drift_dim,
            "hidden_dim": self.hidden_dim,
            "max_dim": self.max_dim,
            "max_scale": self.max_scale,
        }

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)

    def forward(
        self,
        drift: torch.Tensor,
        out_dim: int,
        in_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if out_dim > self.max_dim or in_dim > self.max_dim:
            raise ValueError(
                f"parameter shape ({out_dim}, {in_dim}) exceeds max_dim={self.max_dim}"
            )
        drift = drift.flatten()
        if drift.numel() != self.drift_dim:
            raise ValueError(f"drift dim {drift.numel()} does not match {self.drift_dim}")
        raw = self.net(drift)
        row_scale = self.max_scale * torch.tanh(raw[:out_dim])
        col_scale = self.max_scale * torch.tanh(raw[self.max_dim : self.max_dim + in_dim])
        return row_scale, col_scale


class _HyperLoRAParamAdapter(nn.Module):
    def __init__(self, out_dim: int, in_dim: int, rank: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank))
        self.layer_gate = nn.Parameter(torch.ones(rank))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)

    def forward(self, code: torch.Tensor, max_scale: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        code = code.to(device=device, dtype=self.lora_a.dtype) * self.layer_gate
        delta = self.lora_b @ (code.unsqueeze(1) * self.lora_a)
        return (max_scale * delta).to(device=device, dtype=dtype)


class _HyperBiasParamAdapter(nn.Module):
    def __init__(self, out_dim: int, rank: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.rank = rank
        self.basis = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, code: torch.Tensor, max_scale: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        code = code.to(device=device, dtype=self.basis.dtype)
        delta = self.basis @ code
        return (max_scale * delta).to(device=device, dtype=dtype)


class LayerAwareHyperLoRAGenerator(nn.Module):
    """Layer-aware identity HyperLoRA generator for Stage III V2.

    It learns a distinct low-rank adapter for each target matrix while sharing
    a centered drift conditioner. With bias-free layers, zero drift maps to a
    zero conditioning code, so every registered layer produces zero delta.
    ``lora_b`` starts at zero, giving identity initial behavior while nonzero
    drift still yields gradients to ``lora_b``.
    """

    def __init__(
        self,
        named_param_shapes: Mapping[str, tuple[int, int]],
        drift_dim: int,
        hidden_dim: int,
        rank: int,
        max_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if not named_param_shapes:
            raise ValueError("named_param_shapes must not be empty")
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")

        self.drift_dim = drift_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.max_scale = max_scale
        self._named_param_shapes = {
            name: (int(shape[0]), int(shape[1])) for name, shape in named_param_shapes.items()
        }
        self.drift_conditioner = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, rank, bias=False),
            nn.Tanh(),
        )
        self.adapters = nn.ModuleDict()
        self._name_to_key: dict[str, str] = {}
        self._registered_param_names: list[str] = []
        for name, shape in named_param_shapes.items():
            if len(shape) != 2:
                raise ValueError(f"HyperLoRA only supports 2-D parameters: {name} -> {shape}")
            out_dim, in_dim = int(shape[0]), int(shape[1])
            key = self._key_for_name(name)
            self._name_to_key[name] = key
            self._registered_param_names.append(name)
            self.adapters[key] = _HyperLoRAParamAdapter(out_dim=out_dim, in_dim=in_dim, rank=rank)
        self._reset_conditioner()

    @staticmethod
    def _key_for_name(name: str) -> str:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        return f"p_{digest}"

    def _reset_conditioner(self) -> None:
        for module in self.drift_conditioner:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)

    @property
    def registered_param_names(self) -> tuple[str, ...]:
        return tuple(self._registered_param_names)

    @property
    def adapter_count(self) -> int:
        return len(self.adapters)

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "named_param_shapes": self._named_param_shapes,
            "drift_dim": self.drift_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "max_scale": self.max_scale,
        }

    def conditioning_code(self, drift: torch.Tensor) -> torch.Tensor:
        drift = drift.flatten()
        if drift.numel() != self.drift_dim:
            raise ValueError(f"drift dim {drift.numel()} does not match {self.drift_dim}")
        return self.drift_conditioner(drift)

    def delta_for_param(
        self,
        name: str,
        param: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        if name not in self._name_to_key:
            raise KeyError(f"parameter is not registered in HyperLoRA generator: {name}")
        adapter = self.adapters[self._name_to_key[name]]
        if tuple(param.shape) != (adapter.out_dim, adapter.in_dim):
            raise ValueError(
                f"parameter shape mismatch for {name}: "
                f"expected {(adapter.out_dim, adapter.in_dim)}, got {tuple(param.shape)}"
            )
        code = self.conditioning_code(drift.to(device=param.device, dtype=param.dtype))
        return adapter(code, self.max_scale, dtype=param.dtype, device=param.device)


class LayerAwareHyperAdapterGenerator(nn.Module):
    """Layer-aware HyperLoRA generator with additive bias adapters.

    V5/V7 only update 2-D weights. For load forecasting, a substantial part of
    online residual error can be level-like; an output-head bias update is the
    shortest identity-preserving correction path. This opt-in V8 generator keeps
    the existing low-rank weight adapters and adds additive 1-D bias adapters.
    Zero drift still maps to zero delta because the shared conditioner is
    bias-free and bias adapter bases start at zero.
    """

    def __init__(
        self,
        named_param_shapes: Mapping[str, tuple[int, ...]],
        drift_dim: int,
        hidden_dim: int,
        rank: int,
        max_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if not named_param_shapes:
            raise ValueError("named_param_shapes must not be empty")
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")

        self.drift_dim = drift_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.max_scale = max_scale
        self._named_param_shapes = {
            name: tuple(int(dim) for dim in shape) for name, shape in named_param_shapes.items()
        }
        self.drift_conditioner = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, rank, bias=False),
            nn.Tanh(),
        )
        self.weight_adapters = nn.ModuleDict()
        self.bias_adapters = nn.ModuleDict()
        self._name_to_kind_key: dict[str, tuple[str, str]] = {}
        self._registered_param_names: list[str] = []
        for name, shape in self._named_param_shapes.items():
            key = LayerAwareHyperLoRAGenerator._key_for_name(name)
            self._registered_param_names.append(name)
            if len(shape) == 2:
                out_dim, in_dim = shape
                self.weight_adapters[key] = _HyperLoRAParamAdapter(out_dim=out_dim, in_dim=in_dim, rank=rank)
                self._name_to_kind_key[name] = ("weight", key)
            elif len(shape) == 1:
                (out_dim,) = shape
                self.bias_adapters[key] = _HyperBiasParamAdapter(out_dim=out_dim, rank=rank)
                self._name_to_kind_key[name] = ("bias", key)
            else:
                raise ValueError(f"HyperAdapter only supports 1-D or 2-D parameters: {name} -> {shape}")
        self._reset_conditioner()

    def _reset_conditioner(self) -> None:
        for module in self.drift_conditioner:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)

    @property
    def registered_param_names(self) -> tuple[str, ...]:
        return tuple(self._registered_param_names)

    @property
    def adapter_count(self) -> int:
        return len(self.weight_adapters) + len(self.bias_adapters)

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "named_param_shapes": self._named_param_shapes,
            "drift_dim": self.drift_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "max_scale": self.max_scale,
        }

    def conditioning_code(self, drift: torch.Tensor) -> torch.Tensor:
        drift = drift.flatten()
        if drift.numel() != self.drift_dim:
            raise ValueError(f"drift dim {drift.numel()} does not match {self.drift_dim}")
        return self.drift_conditioner(drift)

    def delta_for_param(
        self,
        name: str,
        param: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        if name not in self._name_to_kind_key:
            raise KeyError(f"parameter is not registered in HyperAdapter generator: {name}")
        kind, key = self._name_to_kind_key[name]
        code = self.conditioning_code(drift.to(device=param.device, dtype=param.dtype))
        if kind == "weight":
            adapter = self.weight_adapters[key]
            if tuple(param.shape) != (adapter.out_dim, adapter.in_dim):
                raise ValueError(
                    f"parameter shape mismatch for {name}: "
                    f"expected {(adapter.out_dim, adapter.in_dim)}, got {tuple(param.shape)}"
                )
            return adapter(code, self.max_scale, dtype=param.dtype, device=param.device)
        adapter = self.bias_adapters[key]
        if tuple(param.shape) != (adapter.out_dim,):
            raise ValueError(
                f"parameter shape mismatch for {name}: expected {(adapter.out_dim,)}, got {tuple(param.shape)}"
            )
        return adapter(code, self.max_scale, dtype=param.dtype, device=param.device)


def build_identity_param_delta(
    params: Mapping[str, torch.Tensor],
    drift: torch.Tensor,
    generator: IdentitySignedSeparableAdaptationGenerator,
    *,
    target_prefixes: tuple[str, ...] | None = None,
    target_names: tuple[str, ...] = ("weight",),
) -> dict[str, torch.Tensor]:
    """Build bounded, identity-centered parameter deltas for 2-D weights."""

    deltas: dict[str, torch.Tensor] = {}
    for name, param in params.items():
        if param.ndim != 2:
            continue
        if target_prefixes and not any(name.startswith(prefix) for prefix in target_prefixes):
            continue
        if target_names and not any(name.endswith(target_name) for target_name in target_names):
            continue
        out_dim, in_dim = param.shape
        row_scale, col_scale = generator(drift.to(device=param.device, dtype=param.dtype), out_dim, in_dim)
        signed_scale = 0.5 * (row_scale.unsqueeze(1) + col_scale.unsqueeze(0))
        deltas[name] = signed_scale * param
    return deltas


def _target_selected(
    name: str,
    *,
    target_prefixes: tuple[str, ...] | None,
    target_names: tuple[str, ...],
) -> bool:
    if target_prefixes and not any(name.startswith(prefix) for prefix in target_prefixes):
        return False
    return not target_names or any(name.endswith(target_name) for target_name in target_names)


def build_stage3_v2_param_delta(
    params: Mapping[str, torch.Tensor],
    drift: torch.Tensor,
    generator: nn.Module,
    *,
    target_prefixes: tuple[str, ...] | None = None,
    target_names: tuple[str, ...] = ("weight",),
) -> dict[str, torch.Tensor]:
    if isinstance(generator, (LayerAwareHyperLoRAGenerator, LayerAwareHyperAdapterGenerator)):
        deltas: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            if isinstance(generator, LayerAwareHyperLoRAGenerator) and param.ndim != 2:
                continue
            if isinstance(generator, LayerAwareHyperAdapterGenerator) and param.ndim not in {1, 2}:
                continue
            if not _target_selected(name, target_prefixes=target_prefixes, target_names=target_names):
                continue
            if name not in generator.registered_param_names:
                continue
            deltas[name] = generator.delta_for_param(name, param, drift)
        return deltas
    if isinstance(generator, IdentitySignedSeparableAdaptationGenerator):
        return build_identity_param_delta(
            params,
            drift,
            generator,
            target_prefixes=target_prefixes,
            target_names=target_names,
        )
    raise TypeError(f"unsupported Stage III V2 generator type: {type(generator)!r}")


def summarize_delta(
    delta: Mapping[str, torch.Tensor],
    *,
    max_delta: float | None = None,
) -> dict[str, float | int]:
    flat_parts = [value.detach().flatten() for value in delta.values()]
    if not flat_parts:
        return {
            "delta_param_count": 0,
            "delta_norm_pre_clamp": 0.0,
            "delta_norm_post_clamp": 0.0,
            "clamp_fraction": 0.0,
        }

    flat = torch.cat(flat_parts)
    pre_norm = torch.linalg.vector_norm(flat).item()
    if max_delta is None:
        post = flat
        clamp_fraction = 0.0
    else:
        post = torch.clamp(flat, -max_delta, max_delta)
        clamp_count = int((flat.abs() > max_delta).sum().item())
        clamp_fraction = clamp_count / int(flat.numel())
    return {
        "delta_param_count": int(flat.numel()),
        "delta_norm_pre_clamp": float(pre_norm),
        "delta_norm_post_clamp": float(torch.linalg.vector_norm(post).item()),
        "clamp_fraction": float(clamp_fraction),
    }
