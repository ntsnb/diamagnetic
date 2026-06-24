from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


V12B_SIGNAL_MODES = {"level_vector", "delta_vector"}
V12B_COUNTERFACTUAL_MODES = {
    "real",
    "zero",
    "shuffled",
    "sign_flipped",
    "amplified_2x",
    "amplified_5x",
    "random_matched_norm",
}


@dataclass
class Stage3TeacherSignalResultV12b:
    signal: torch.Tensor
    raw: torch.Tensor
    mode: str
    counterfactual_mode: str
    drift_dim: int
    zero_mask_fraction: float
    raw_abs_mean: float
    signal_abs_mean: float
    scaled_abs_mean: float
    signal_norm_mean: float


class Stage3TeacherSignalBuilderV12b:
    """Train-only signal contract shared by V12b aligner, teacher, and audits.

    The legacy Stage III signal path normalized each batch by its own absolute
    mean, which changes scale between fit and evaluation and can amplify tiny
    drift. V12b fits a robust per-dimension scale on the train split, then
    reuses that scale for every split and counterfactual intervention. Zero
    drift is always mapped to exact zero.
    """

    protocol_version = "v12b_train_only_signal_contract"

    def __init__(
        self,
        *,
        drift_dim: int,
        mode: Literal["level_vector", "delta_vector"] = "level_vector",
        normalize: bool = True,
        robust_scale_quantile: float = 0.75,
        eps: float = 1e-6,
        polarity: float = 1.0,
        random_seed: int = 1729,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if mode not in V12B_SIGNAL_MODES:
            raise ValueError(f"unsupported V12b signal mode: {mode}")
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

    def _as_matrix(self, drift: torch.Tensor) -> torch.Tensor:
        if drift.ndim == 1:
            drift = drift.unsqueeze(0)
        elif drift.ndim > 2:
            drift = drift.flatten(start_dim=1)
        if drift.ndim != 2 or drift.shape[1] != self.drift_dim:
            raise ValueError(f"drift must have shape [N, {self.drift_dim}]")
        return drift.detach().float() * self.polarity

    def _sequence_signal(self, raw: torch.Tensor, previous_tail: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "level_vector":
            return raw
        if raw.shape[0] == 0:
            raise ValueError("drift sequence must contain at least one sample")
        if previous_tail is None:
            prev = torch.cat([raw[:1], raw[:-1]], dim=0)
        else:
            prev_first = previous_tail.detach().to(device=raw.device, dtype=raw.dtype).reshape_as(raw[:1])
            prev = torch.cat([prev_first, raw[:-1]], dim=0)
        return raw - prev

    def fit(self, drift_train: torch.Tensor, *, split: str = "train") -> "Stage3TeacherSignalBuilderV12b":
        if split != "train":
            raise ValueError("Stage3TeacherSignalBuilderV12b can only be fit on the train split")
        raw = self._as_matrix(drift_train)
        signal = self._sequence_signal(raw)
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
        drift: torch.Tensor,
        *,
        previous_tail: torch.Tensor | None = None,
        counterfactual_mode: str = "real",
    ) -> Stage3TeacherSignalResultV12b:
        if self.train_scale is None:
            raise RuntimeError("Stage3TeacherSignalBuilderV12b must be fit before transform")
        if counterfactual_mode not in V12B_COUNTERFACTUAL_MODES:
            raise ValueError(f"unsupported V12b counterfactual mode: {counterfactual_mode}")
        raw = self._as_matrix(drift)
        signal = self._sequence_signal(raw, previous_tail=previous_tail)
        signal = self._apply_counterfactual(signal, mode=counterfactual_mode)
        zero_mask = torch.linalg.vector_norm(signal, dim=1, keepdim=True) <= self.eps
        if self.normalize:
            scale = self.train_scale.to(device=signal.device, dtype=signal.dtype).clamp_min(self.eps)
            scaled = signal / scale.unsqueeze(0)
        else:
            scaled = signal
        scaled = torch.where(zero_mask, torch.zeros_like(scaled), scaled)
        return Stage3TeacherSignalResultV12b(
            signal=scaled,
            raw=raw,
            mode=self.mode,
            counterfactual_mode=counterfactual_mode,
            drift_dim=self.drift_dim,
            zero_mask_fraction=float(zero_mask.float().mean().detach().cpu().item()),
            raw_abs_mean=float(raw.abs().mean().detach().cpu().item()) if raw.numel() else 0.0,
            signal_abs_mean=float(signal.abs().mean().detach().cpu().item()) if signal.numel() else 0.0,
            scaled_abs_mean=float(scaled.abs().mean().detach().cpu().item()) if scaled.numel() else 0.0,
            signal_norm_mean=float(torch.linalg.vector_norm(signal, dim=1).mean().detach().cpu().item())
            if signal.numel()
            else 0.0,
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
        raise ValueError(f"unsupported V12b counterfactual mode: {mode}")

    def state_dict(self) -> dict[str, object]:
        if self.train_scale is None:
            raise RuntimeError("Stage3TeacherSignalBuilderV12b must be fit before serialization")
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

    def load_state_dict(self, state: dict[str, object]) -> "Stage3TeacherSignalBuilderV12b":
        self.drift_dim = int(state["drift_dim"])
        self.mode = str(state.get("mode", self.mode))
        if self.mode not in V12B_SIGNAL_MODES:
            raise ValueError(f"unsupported loaded V12b signal mode: {self.mode}")
        self.normalize = bool(state.get("normalize", self.normalize))
        self.robust_scale_quantile = float(state.get("robust_scale_quantile", self.robust_scale_quantile))
        self.eps = float(state.get("eps", self.eps))
        self.polarity = float(state.get("polarity", self.polarity))
        self.random_seed = int(state.get("random_seed", self.random_seed))
        self.train_scale = torch.as_tensor(state["train_scale"]).detach().float()
        if self.train_scale.numel() != self.drift_dim:
            raise ValueError("loaded V12b signal scale has incompatible shape")
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        self.fit_split = str(state.get("fit_split", "train"))
        return self

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.train_scale is None:
            raise RuntimeError("Stage3TeacherSignalBuilderV12b must be fit before diagnostics")
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
