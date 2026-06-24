from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class PerHorizonResidualHistoryFeaturesV12e:
    features: torch.Tensor
    residual_history: torch.Tensor
    availability_mask: torch.Tensor
    residual_tail: torch.Tensor
    lags: tuple[int, ...]
    horizon_offsets: tuple[int, ...]
    label_delay: int
    include_availability_mask: bool
    standardized: bool


class PerHorizonLabelAvailabilityHistoryBuilderV12e:
    """Per-horizon residual-history features with explicit label arrival.

    A residual from forecast sample ``t - lag`` and horizon ``h`` is available
    for the current sample ``t`` only when:

    ``lag >= horizon_offsets[h] + label_delay``.

    This prevents long-horizon residuals from entering the history buffer before
    their true labels would have arrived in a streaming protocol.
    """

    protocol_version = "v12e_per_horizon_label_availability_history"

    def __init__(
        self,
        *,
        output_shape: tuple[int, ...],
        lags: Iterable[int] = (1, 2, 3, 6, 12, 24),
        horizon_offsets: Iterable[int] | None = None,
        label_delay: int = 0,
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
        if label_delay < 0:
            raise ValueError("label_delay must be non-negative")
        if not 0.0 < robust_scale_quantile <= 1.0:
            raise ValueError("robust_scale_quantile must be in (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.horizon_count = self.output_shape[0]
        self.per_horizon_dim = int(torch.tensor(self.output_shape[1:] or (1,)).prod().item())
        self.output_dim = self.horizon_count * self.per_horizon_dim
        if horizon_offsets is None:
            parsed_offsets = tuple(range(1, self.horizon_count + 1))
        else:
            parsed_offsets = tuple(int(value) for value in horizon_offsets)
        if len(parsed_offsets) != self.horizon_count:
            raise ValueError("horizon_offsets must match the first dimension of output_shape")
        if any(value <= 0 for value in parsed_offsets):
            raise ValueError("horizon_offsets must contain positive integers")

        self.lags = parsed_lags
        self.horizon_offsets = parsed_offsets
        self.label_delay = int(label_delay)
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
        return max(self.lags)

    def _residual_tensor(self, *, y_base: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        sample_count = int(y_base.shape[0])
        y_base_view = y_base.detach().float().reshape(sample_count, self.horizon_count, self.per_horizon_dim)
        y_true_view = y_true.detach().float().reshape(sample_count, self.horizon_count, self.per_horizon_dim)
        if y_base_view.shape != y_true_view.shape:
            raise ValueError("y_base and y_true must have the same flattened shape")
        return y_true_view - y_base_view

    def _raw_history(
        self,
        residual: torch.Tensor,
        *,
        previous_residual_tail: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.ndim != 3 or residual.shape[1:] != (self.horizon_count, self.per_horizon_dim):
            raise ValueError(
                f"residual must have shape [N, {self.horizon_count}, {self.per_horizon_dim}]"
            )
        if previous_residual_tail is None:
            tail = torch.empty(
                0,
                self.horizon_count,
                self.per_horizon_dim,
                dtype=residual.dtype,
                device=residual.device,
            )
        else:
            tail = previous_residual_tail.detach().float().reshape(
                -1,
                self.horizon_count,
                self.per_horizon_dim,
            ).to(device=residual.device)
            if tail.shape[0] > self.required_tail_length:
                tail = tail[-self.required_tail_length :]
        source = torch.cat([tail, residual], dim=0)
        offset = tail.shape[0]
        values = torch.zeros(
            residual.shape[0],
            len(self.lags),
            self.horizon_count,
            self.per_horizon_dim,
            dtype=residual.dtype,
            device=residual.device,
        )
        mask = torch.zeros_like(values)
        horizon_offsets = torch.tensor(self.horizon_offsets, dtype=torch.long, device=residual.device)
        for row_idx in range(residual.shape[0]):
            for lag_idx, lag in enumerate(self.lags):
                source_idx = offset + row_idx - lag
                if source_idx < 0:
                    continue
                available_h = int(lag) >= horizon_offsets + self.label_delay
                if bool(available_h.any()):
                    values[row_idx, lag_idx, available_h] = source[source_idx, available_h]
                    mask[row_idx, lag_idx, available_h] = 1.0
        return values.reshape(residual.shape[0], -1), mask.reshape(residual.shape[0], -1)

    def _fit_scale(self, raw_history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scale = torch.ones(raw_history.shape[1], dtype=raw_history.dtype, device=raw_history.device)
        abs_history = raw_history.abs()
        for idx in range(raw_history.shape[1]):
            available = mask[:, idx] > 0.0
            if not bool(available.any()):
                continue
            values = abs_history[available, idx]
            quantile = torch.quantile(values, self.robust_scale_quantile)
            fallback = values.mean()
            chosen = quantile if quantile > self.eps else fallback
            scale[idx] = chosen if chosen > self.eps else torch.tensor(1.0, dtype=scale.dtype, device=scale.device)
        return scale

    def fit(
        self,
        *,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
    ) -> "PerHorizonLabelAvailabilityHistoryBuilderV12e":
        if split != "train":
            raise ValueError("PerHorizonLabelAvailabilityHistoryBuilderV12e can only be fit on the train split")
        residual = self._residual_tensor(y_base=y_base_train, y_true=y_train)
        raw_history, mask = self._raw_history(residual)
        self.history_scale = self._fit_scale(raw_history, mask).detach().float()
        self.fit_split = "train"
        self.fit_sample_count = int(residual.shape[0])
        return self

    def transform(
        self,
        *,
        y_base: torch.Tensor,
        y_true: torch.Tensor,
        previous_residual_tail: torch.Tensor | None = None,
    ) -> PerHorizonResidualHistoryFeaturesV12e:
        if self.history_scale is None:
            raise RuntimeError("PerHorizonLabelAvailabilityHistoryBuilderV12e must be fit before transform")
        residual = self._residual_tensor(y_base=y_base, y_true=y_true)
        raw_history, mask = self._raw_history(residual, previous_residual_tail=previous_residual_tail)
        if self.standardize:
            scale = self.history_scale.to(device=raw_history.device, dtype=raw_history.dtype).clamp_min(self.eps)
            history = raw_history / scale.unsqueeze(0)
        else:
            history = raw_history
        history = torch.where(mask > 0.0, history, torch.zeros_like(history))
        features = torch.cat([history, mask], dim=1) if self.include_availability_mask else history
        return PerHorizonResidualHistoryFeaturesV12e(
            features=features.detach().float(),
            residual_history=history.detach().float(),
            availability_mask=mask.detach().float(),
            residual_tail=self.residual_tail(y_base=y_base, y_true=y_true),
            lags=self.lags,
            horizon_offsets=self.horizon_offsets,
            label_delay=self.label_delay,
            include_availability_mask=self.include_availability_mask,
            standardized=self.standardize,
        )

    def residual_tail(self, *, y_base: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        residual = self._residual_tensor(y_base=y_base, y_true=y_true)
        if residual.shape[0] <= self.required_tail_length:
            return residual.detach().float()
        return residual[-self.required_tail_length :].detach().float()

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.history_scale is None:
            raise RuntimeError("PerHorizonLabelAvailabilityHistoryBuilderV12e must be fit before diagnostics")
        return {
            "history_feature_protocol": self.protocol_version,
            "history_lags": ",".join(str(value) for value in self.lags),
            "history_horizon_offsets": ",".join(str(value) for value in self.horizon_offsets),
            "history_label_delay": self.label_delay,
            "history_include_availability_mask": str(self.include_availability_mask),
            "history_standardize": str(self.standardize),
            "history_fit_split": self.fit_split,
            "history_fit_sample_count": self.fit_sample_count,
            "history_horizon_count": self.horizon_count,
            "history_per_horizon_dim": self.per_horizon_dim,
            "history_value_dim": self.history_value_dim,
            "history_feature_dim": self.feature_dim,
            "history_scale_abs_mean": float(self.history_scale.abs().mean().item()),
            "history_scale_min": float(self.history_scale.min().item()),
            "history_scale_max": float(self.history_scale.max().item()),
        }

    def state_dict(self) -> dict[str, object]:
        if self.history_scale is None:
            raise RuntimeError("PerHorizonLabelAvailabilityHistoryBuilderV12e must be fit before serialization")
        return {
            "protocol_version": self.protocol_version,
            "output_shape": self.output_shape,
            "lags": self.lags,
            "horizon_offsets": self.horizon_offsets,
            "label_delay": self.label_delay,
            "include_availability_mask": self.include_availability_mask,
            "standardize": self.standardize,
            "robust_scale_quantile": self.robust_scale_quantile,
            "eps": self.eps,
            "history_scale": self.history_scale.detach().clone(),
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> "PerHorizonLabelAvailabilityHistoryBuilderV12e":
        self.output_shape = tuple(int(dim) for dim in state["output_shape"])  # type: ignore[index]
        self.horizon_count = self.output_shape[0]
        self.per_horizon_dim = int(torch.tensor(self.output_shape[1:] or (1,)).prod().item())
        self.output_dim = self.horizon_count * self.per_horizon_dim
        self.lags = tuple(int(value) for value in state["lags"])  # type: ignore[index]
        self.horizon_offsets = tuple(int(value) for value in state["horizon_offsets"])  # type: ignore[index]
        self.label_delay = int(state.get("label_delay", self.label_delay))
        self.include_availability_mask = bool(state.get("include_availability_mask", self.include_availability_mask))
        self.standardize = bool(state.get("standardize", self.standardize))
        self.robust_scale_quantile = float(state.get("robust_scale_quantile", self.robust_scale_quantile))
        self.eps = float(state.get("eps", self.eps))
        self.history_scale = torch.as_tensor(state["history_scale"]).detach().float()  # type: ignore[index]
        if self.history_scale.numel() != self.history_value_dim:
            raise ValueError("loaded V12e residual-history scale has incompatible shape")
        self.fit_split = str(state.get("fit_split", "train"))
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        return self
