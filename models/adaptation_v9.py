from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from models.adaptation_v2 import LayerAwareHyperAdapterGenerator


class ResidualBasisHyperGeneratorV9(LayerAwareHyperAdapterGenerator):
    """Stage III V9 generator with parameter deltas and residual correction.

    The class reuses the tested V8 weight/bias adapter path and adds a
    bias-free residual correction head. Because the residual head has no biases,
    zero drift produces exactly zero parameter delta and zero residual
    correction.
    """

    def __init__(
        self,
        named_param_shapes: Mapping[str, tuple[int, ...]],
        drift_dim: int,
        hidden_dim: int,
        rank: int,
        output_shape: tuple[int, ...],
        max_scale: float = 0.05,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__(
            named_param_shapes=named_param_shapes,
            drift_dim=drift_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            max_scale=max_scale,
        )
        if not output_shape or any(int(dim) <= 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.residual_scale = float(residual_scale)
        self.residual_head = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.output_dim, bias=False),
        )
        self._reset_residual_head()

    def _reset_residual_head(self) -> None:
        for module in self.residual_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)

    def residual_correction(self, drift: torch.Tensor, reference_prediction: torch.Tensor) -> torch.Tensor:
        drift = drift.flatten()
        if drift.numel() != self.drift_dim:
            raise ValueError(f"drift dim {drift.numel()} does not match {self.drift_dim}")
        raw = self.residual_head(drift.to(device=reference_prediction.device, dtype=reference_prediction.dtype))
        correction = self.residual_scale * raw.reshape(1, *self.output_shape)
        return correction.expand_as(reference_prediction)

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "named_param_shapes": self._named_param_shapes,
            "drift_dim": self.drift_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "max_scale": self.max_scale,
            "output_shape": self.output_shape,
            "residual_scale": self.residual_scale,
        }
