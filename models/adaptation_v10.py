from __future__ import annotations

from typing import Mapping

from models.adaptation_v9 import ResidualBasisHyperGeneratorV9


class ResidualBasisHyperGeneratorV10(ResidualBasisHyperGeneratorV9):
    """Stage III V10 generator class.

    V10 intentionally keeps V9 generator capacity fixed while changing the
    upstream alignment and residual-teacher calibration. This makes the
    experiment isolate whether the repaired drift evidence path helps before
    increasing generator expressiveness.
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
            output_shape=output_shape,
            max_scale=max_scale,
            residual_scale=residual_scale,
        )
