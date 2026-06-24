from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


V13_SIGNAL_MODES = {"level_offset", "fluctuation_delta"}
V13_COUNTERFACTUAL_MODES = {
    "real",
    "zero",
    "shuffled",
    "sign_flipped",
    "amplified_2x",
    "amplified_5x",
    "random_matched_norm",
}


@dataclass
class Stage3SignalResultV13:
    signal: torch.Tensor
    level_offset: torch.Tensor
    fluctuation: torch.Tensor
    raw: torch.Tensor
    mode: str
    counterfactual_mode: str
    drift_dim: int
    protocol_version: str
    first_step_zero_fraction: float
    zero_mask_fraction: float
    level_norm_mean: float
    fluctuation_norm_mean: float
    signal_norm_mean: float
    scaled_abs_mean: float


def _as_sequence_matrix(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim == 1:
        matrix = sequence.unsqueeze(0)
    elif sequence.ndim > 2:
        matrix = sequence.flatten(start_dim=1)
    else:
        matrix = sequence
    return matrix.detach().float()


def compute_v13_sequence_delta(sequence: torch.Tensor, previous_tail: torch.Tensor | None = None) -> torch.Tensor:
    """Compute temporal delta only for a sequence that is actually supplied."""
    matrix = _as_sequence_matrix(sequence)
    if matrix.shape[0] == 0:
        raise ValueError("sequence must contain at least one sample")
    if previous_tail is None:
        previous = torch.cat([matrix[:1], matrix[:-1]], dim=0)
    else:
        tail = _as_sequence_matrix(previous_tail).to(device=matrix.device, dtype=matrix.dtype)
        if tail.shape[1] != matrix.shape[1]:
            raise ValueError("previous_tail feature dimension must match sequence feature dimension")
        previous = torch.cat([tail[-1:], matrix[:-1]], dim=0)
    return matrix - previous


class Stage3SignalBuilderV13:
    protocol_version = "v13_fluctuation_first_signal_contract"

    def __init__(
        self,
        *,
        drift_dim: int,
        mode: Literal["level_offset", "fluctuation_delta"] = "fluctuation_delta",
        normalize: bool = True,
        robust_scale_quantile: float = 0.75,
        eps: float = 1e-6,
        polarity: float = 1.0,
        random_seed: int = 1729,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if mode not in V13_SIGNAL_MODES:
            raise ValueError(f"unsupported V13 signal mode: {mode}")
        if not 0.0 < robust_scale_quantile <= 1.0:
            raise ValueError("robust_scale_quantile must be in (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.drift_dim = int(drift_dim)
        self.mode = mode
        self.normalize = bool(normalize)
        self.robust_scale_quantile = float(robust_scale_quantile)
        self.eps = float(eps)
        self.polarity = float(polarity)
        self.random_seed = int(random_seed)
        self.train_scale: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.fit_split = ""

    def _as_matrix(self, level_offset: torch.Tensor) -> torch.Tensor:
        matrix = _as_sequence_matrix(level_offset)
        if matrix.ndim != 2 or matrix.shape[1] != self.drift_dim:
            raise ValueError(f"level_offset must have shape [N, {self.drift_dim}]")
        return matrix * self.polarity

    def _transition(self, raw: torch.Tensor, previous_level_tail: torch.Tensor | None = None) -> torch.Tensor:
        return compute_v13_sequence_delta(raw, previous_tail=previous_level_tail)

    def _mode_signal(
        self,
        raw: torch.Tensor,
        previous_level_tail: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fluctuation = self._transition(raw, previous_level_tail=previous_level_tail)
        if self.mode == "level_offset":
            return raw, fluctuation
        if self.mode == "fluctuation_delta":
            return fluctuation, fluctuation
        raise ValueError(f"unsupported V13 signal mode: {self.mode}")

    def fit(self, level_offset_train: torch.Tensor, *, split: str = "train") -> "Stage3SignalBuilderV13":
        if split != "train":
            raise ValueError("Stage3SignalBuilderV13 can only be fit on the train split")
        raw = self._as_matrix(level_offset_train)
        signal, _ = self._mode_signal(raw)
        scale = torch.quantile(signal.abs(), self.robust_scale_quantile, dim=0)
        fallback = signal.abs().mean(dim=0)
        scale = torch.where(scale > self.eps, scale, fallback)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        self.train_scale = scale.detach().float()
        self.fit_sample_count = int(raw.shape[0])
        self.fit_split = "train"
        return self

    def transform(
        self,
        level_offset: torch.Tensor,
        *,
        previous_level_tail: torch.Tensor | None = None,
        counterfactual_mode: str = "real",
    ) -> Stage3SignalResultV13:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before transform")
        if counterfactual_mode not in V13_COUNTERFACTUAL_MODES:
            raise ValueError(f"unsupported V13 counterfactual mode: {counterfactual_mode}")
        raw = self._as_matrix(level_offset)
        signal, fluctuation = self._mode_signal(raw, previous_level_tail=previous_level_tail)
        signal = self._apply_counterfactual(signal, mode=counterfactual_mode)
        zero_mask = torch.linalg.vector_norm(signal, dim=1, keepdim=True) <= self.eps
        if self.normalize:
            scale = self.train_scale.to(device=signal.device, dtype=signal.dtype).clamp_min(self.eps)
            scaled = signal / scale.unsqueeze(0)
        else:
            scaled = signal
        scaled = torch.where(zero_mask, torch.zeros_like(scaled), scaled)
        first_zero = float((torch.linalg.vector_norm(fluctuation[:1], dim=1) <= self.eps).float().mean().item())
        return Stage3SignalResultV13(
            signal=scaled.detach().float(),
            level_offset=raw.detach().float(),
            fluctuation=fluctuation.detach().float(),
            raw=raw.detach().float(),
            mode=self.mode,
            counterfactual_mode=counterfactual_mode,
            drift_dim=self.drift_dim,
            protocol_version=self.protocol_version,
            first_step_zero_fraction=first_zero,
            zero_mask_fraction=float(zero_mask.float().mean().item()),
            level_norm_mean=float(torch.linalg.vector_norm(raw, dim=1).mean().item()) if raw.numel() else 0.0,
            fluctuation_norm_mean=float(torch.linalg.vector_norm(fluctuation, dim=1).mean().item())
            if fluctuation.numel()
            else 0.0,
            signal_norm_mean=float(torch.linalg.vector_norm(scaled, dim=1).mean().item()) if scaled.numel() else 0.0,
            scaled_abs_mean=float(scaled.abs().mean().item()) if scaled.numel() else 0.0,
        )

    def _apply_counterfactual(self, signal: torch.Tensor, *, mode: str) -> torch.Tensor:
        if mode == "real":
            return signal
        if mode == "zero":
            return torch.zeros_like(signal)
        if mode == "sign_flipped":
            return -signal
        if mode == "amplified_2x":
            return 2.0 * signal
        if mode == "amplified_5x":
            return 5.0 * signal
        if mode == "shuffled":
            if signal.shape[0] <= 1:
                return signal.clone()
            return signal.flip(0)
        if mode == "random_matched_norm":
            gen = torch.Generator(device=signal.device)
            gen.manual_seed(self.random_seed)
            random = torch.randn(signal.shape, generator=gen, device=signal.device, dtype=signal.dtype)
            random_norm = torch.linalg.vector_norm(random, dim=1, keepdim=True).clamp_min(self.eps)
            signal_norm = torch.linalg.vector_norm(signal, dim=1, keepdim=True)
            return random / random_norm * signal_norm
        raise ValueError(f"unsupported V13 counterfactual mode: {mode}")

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before diagnostics")
        return {
            "signal_contract_version": self.protocol_version,
            "signal_mode": self.mode,
            "signal_normalize": str(self.normalize),
            "signal_fit_split": self.fit_split,
            "signal_fit_sample_count": self.fit_sample_count,
            "signal_scale_abs_mean": float(self.train_scale.abs().mean().item()),
            "signal_scale_min": float(self.train_scale.min().item()),
            "signal_scale_max": float(self.train_scale.max().item()),
        }

    def stop_diagnostic(
        self,
        signal: torch.Tensor,
        *,
        zero_fraction_threshold: float = 0.99,
    ) -> dict[str, object]:
        matrix = _as_sequence_matrix(signal)
        if matrix.shape[0] == 0:
            raise ValueError("signal must contain at least one sample")
        zero_fraction = float((torch.linalg.vector_norm(matrix, dim=1) <= self.eps).float().mean().item())
        triggered = zero_fraction >= float(zero_fraction_threshold)
        return {
            "stop_condition_triggered": triggered,
            "stop_condition_name": "fluctuation_collapsed_to_zero" if triggered else "",
            "zero_fraction": zero_fraction,
            "zero_fraction_threshold": float(zero_fraction_threshold),
        }

    def state_dict(self) -> dict[str, object]:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before serialization")
        return {
            "protocol_version": self.protocol_version,
            "drift_dim": self.drift_dim,
            "mode": self.mode,
            "normalize": self.normalize,
            "robust_scale_quantile": self.robust_scale_quantile,
            "eps": self.eps,
            "polarity": self.polarity,
            "random_seed": self.random_seed,
            "train_scale": self.train_scale.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "fit_split": self.fit_split,
        }

    def load_state_dict(self, state: dict[str, object]) -> "Stage3SignalBuilderV13":
        self.drift_dim = int(state["drift_dim"])
        self.mode = str(state.get("mode", self.mode))
        if self.mode not in V13_SIGNAL_MODES:
            raise ValueError(f"unsupported loaded V13 signal mode: {self.mode}")
        self.normalize = bool(state.get("normalize", self.normalize))
        self.robust_scale_quantile = float(state.get("robust_scale_quantile", self.robust_scale_quantile))
        self.eps = float(state.get("eps", self.eps))
        self.polarity = float(state.get("polarity", self.polarity))
        self.random_seed = int(state.get("random_seed", self.random_seed))
        self.train_scale = torch.as_tensor(state["train_scale"]).detach().float()
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        self.fit_split = str(state.get("fit_split", "train"))
        return self

