from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch


V12D_CONTEXT_MODES = {"drift_only", "history_only", "drift_plus_history"}


@dataclass
class CausalResidualHistoryFeaturesV12d:
    features: torch.Tensor
    residual_history: torch.Tensor
    availability_mask: torch.Tensor
    residual_tail: torch.Tensor
    lags: tuple[int, ...]
    feedback_delay: int
    include_availability_mask: bool
    standardized: bool


@dataclass
class Stage3V12dContextFeatures:
    features: torch.Tensor
    mode: str
    drift_dim: int
    history_dim: int
    feature_dim: int


class CausalResidualHistoryFeatureBuilderV12d:
    """Causal residual-history feature builder for V12d teacher-only probes.

    For sample ``t``, lag ``1`` means the previous observed residual. The
    current residual is never included in its own feature row. ``feedback_delay``
    can be increased when deployment feedback would arrive later.
    """

    protocol_version = "v12d_causal_residual_history_features"

    def __init__(
        self,
        *,
        output_shape: tuple[int, ...],
        lags: Iterable[int] = (1, 2, 3, 6, 12, 24),
        feedback_delay: int = 0,
        include_availability_mask: bool = True,
        standardize: bool = True,
        robust_scale_quantile: float = 0.75,
        eps: float = 1e-6,
    ) -> None:
        if not output_shape or any(int(dim) <= 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        parsed_lags = tuple(sorted({int(value) for value in lags}))
        if not parsed_lags or any(value <= 0 for value in parsed_lags):
            raise ValueError("lags must contain positive integers")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative")
        if not 0.0 < robust_scale_quantile <= 1.0:
            raise ValueError("robust_scale_quantile must be in (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.lags = parsed_lags
        self.feedback_delay = int(feedback_delay)
        self.include_availability_mask = bool(include_availability_mask)
        self.standardize = bool(standardize)
        self.robust_scale_quantile = float(robust_scale_quantile)
        self.eps = float(eps)
        self.history_scale: torch.Tensor | None = None
        self.fit_split = ""
        self.fit_sample_count = 0

    @property
    def history_value_dim(self) -> int:
        return len(self.lags) * self.output_dim

    @property
    def feature_dim(self) -> int:
        mask_dim = self.history_value_dim if self.include_availability_mask else 0
        return self.history_value_dim + mask_dim

    @property
    def required_tail_length(self) -> int:
        return max(self.lags) + self.feedback_delay

    def _residual_matrix(self, *, y_base: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_base_2d = y_base.detach().float().reshape(y_base.shape[0], -1)
        y_true_2d = y_true.detach().float().reshape(y_true.shape[0], -1)
        if y_base_2d.shape != y_true_2d.shape:
            raise ValueError("y_base and y_true must have the same flattened shape")
        if y_base_2d.shape[1] != self.output_dim:
            raise ValueError(f"residual output dim {y_base_2d.shape[1]} does not match expected {self.output_dim}")
        return y_true_2d - y_base_2d

    def _raw_history(
        self,
        residual: torch.Tensor,
        *,
        previous_residual_tail: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.ndim != 2 or residual.shape[1] != self.output_dim:
            raise ValueError(f"residual must have shape [N, {self.output_dim}]")
        if previous_residual_tail is None:
            tail = torch.empty(0, self.output_dim, dtype=residual.dtype, device=residual.device)
        else:
            tail = previous_residual_tail.detach().float().reshape(-1, self.output_dim).to(device=residual.device)
            if tail.shape[0] > self.required_tail_length:
                tail = tail[-self.required_tail_length :]
        source = torch.cat([tail, residual], dim=0)
        offset = tail.shape[0]
        values = torch.zeros(residual.shape[0], len(self.lags), self.output_dim, dtype=residual.dtype, device=residual.device)
        mask = torch.zeros_like(values)
        for row_idx in range(residual.shape[0]):
            for lag_idx, lag in enumerate(self.lags):
                source_idx = offset + row_idx - self.feedback_delay - lag
                if source_idx >= 0:
                    values[row_idx, lag_idx] = source[source_idx]
                    mask[row_idx, lag_idx] = 1.0
        return values.reshape(residual.shape[0], -1), mask.reshape(residual.shape[0], -1)

    def fit(
        self,
        *,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
    ) -> "CausalResidualHistoryFeatureBuilderV12d":
        if split != "train":
            raise ValueError("CausalResidualHistoryFeatureBuilderV12d can only be fit on the train split")
        residual = self._residual_matrix(y_base=y_base_train, y_true=y_train)
        raw_history, _ = self._raw_history(residual)
        abs_history = raw_history.abs()
        scale = torch.quantile(abs_history, self.robust_scale_quantile, dim=0)
        fallback = abs_history.mean(dim=0)
        scale = torch.where(scale > self.eps, scale, fallback)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        self.history_scale = scale.detach().float()
        self.fit_split = "train"
        self.fit_sample_count = int(residual.shape[0])
        return self

    def transform(
        self,
        *,
        y_base: torch.Tensor,
        y_true: torch.Tensor,
        previous_residual_tail: torch.Tensor | None = None,
    ) -> CausalResidualHistoryFeaturesV12d:
        if self.history_scale is None:
            raise RuntimeError("CausalResidualHistoryFeatureBuilderV12d must be fit before transform")
        residual = self._residual_matrix(y_base=y_base, y_true=y_true)
        raw_history, mask = self._raw_history(residual, previous_residual_tail=previous_residual_tail)
        if self.standardize:
            scale = self.history_scale.to(device=raw_history.device, dtype=raw_history.dtype).clamp_min(self.eps)
            history = raw_history / scale.unsqueeze(0)
        else:
            history = raw_history
        history = torch.where(mask > 0.0, history, torch.zeros_like(history))
        features = torch.cat([history, mask], dim=1) if self.include_availability_mask else history
        return CausalResidualHistoryFeaturesV12d(
            features=features.detach().float(),
            residual_history=history.detach().float(),
            availability_mask=mask.detach().float(),
            residual_tail=self.residual_tail(y_base=y_base, y_true=y_true),
            lags=self.lags,
            feedback_delay=self.feedback_delay,
            include_availability_mask=self.include_availability_mask,
            standardized=self.standardize,
        )

    def residual_tail(self, *, y_base: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        residual = self._residual_matrix(y_base=y_base, y_true=y_true)
        if residual.shape[0] <= self.required_tail_length:
            return residual.detach().float()
        return residual[-self.required_tail_length :].detach().float()

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.history_scale is None:
            raise RuntimeError("CausalResidualHistoryFeatureBuilderV12d must be fit before diagnostics")
        return {
            "history_feature_protocol": self.protocol_version,
            "history_lags": ",".join(str(value) for value in self.lags),
            "history_feedback_delay": self.feedback_delay,
            "history_include_availability_mask": str(self.include_availability_mask),
            "history_standardize": str(self.standardize),
            "history_fit_split": self.fit_split,
            "history_fit_sample_count": self.fit_sample_count,
            "history_value_dim": self.history_value_dim,
            "history_feature_dim": self.feature_dim,
            "history_scale_abs_mean": float(self.history_scale.abs().mean().item()),
            "history_scale_min": float(self.history_scale.min().item()),
            "history_scale_max": float(self.history_scale.max().item()),
        }

    def state_dict(self) -> dict[str, object]:
        if self.history_scale is None:
            raise RuntimeError("CausalResidualHistoryFeatureBuilderV12d must be fit before serialization")
        return {
            "protocol_version": self.protocol_version,
            "output_shape": self.output_shape,
            "lags": self.lags,
            "feedback_delay": self.feedback_delay,
            "include_availability_mask": self.include_availability_mask,
            "standardize": self.standardize,
            "robust_scale_quantile": self.robust_scale_quantile,
            "eps": self.eps,
            "history_scale": self.history_scale.detach().clone(),
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> "CausalResidualHistoryFeatureBuilderV12d":
        self.output_shape = tuple(int(dim) for dim in state["output_shape"])  # type: ignore[index]
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.lags = tuple(int(value) for value in state["lags"])  # type: ignore[index]
        self.feedback_delay = int(state.get("feedback_delay", self.feedback_delay))
        self.include_availability_mask = bool(state.get("include_availability_mask", self.include_availability_mask))
        self.standardize = bool(state.get("standardize", self.standardize))
        self.robust_scale_quantile = float(state.get("robust_scale_quantile", self.robust_scale_quantile))
        self.eps = float(state.get("eps", self.eps))
        self.history_scale = torch.as_tensor(state["history_scale"]).detach().float()  # type: ignore[index]
        if self.history_scale.numel() != self.history_value_dim:
            raise ValueError("loaded V12d residual-history scale has incompatible shape")
        self.fit_split = str(state.get("fit_split", "train"))
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        return self


def build_stage3_v12d_context_features(
    *,
    drift_signal: torch.Tensor,
    history_features: torch.Tensor,
    mode: Literal["drift_only", "history_only", "drift_plus_history"],
) -> Stage3V12dContextFeatures:
    if mode not in V12D_CONTEXT_MODES:
        raise ValueError(f"unsupported V12d context feature mode: {mode}")
    drift = drift_signal.detach().float()
    history = history_features.detach().float()
    if drift.ndim != 2:
        raise ValueError("drift_signal must be a 2-D tensor")
    if history.ndim != 2:
        raise ValueError("history_features must be a 2-D tensor")
    if drift.shape[0] != history.shape[0]:
        raise ValueError("drift_signal and history_features must have the same sample count")
    if mode == "drift_only":
        features = drift
    elif mode == "history_only":
        features = history
    else:
        features = torch.cat([drift, history], dim=1)
    return Stage3V12dContextFeatures(
        features=features.detach().float(),
        mode=mode,
        drift_dim=int(drift.shape[1]),
        history_dim=int(history.shape[1]),
        feature_dim=int(features.shape[1]),
    )
