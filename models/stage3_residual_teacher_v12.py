from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.stage3_residual_teacher_v11 import TrainOnlyRobustBlockResidualTeacherV11


class TrainOnlyOutcomeAlignedBlockResidualTeacherV12(TrainOnlyRobustBlockResidualTeacherV11):
    """V12 protocol wrapper for the strict V11 block residual teacher.

    The V12 scientific change is the outcome-aligned transport aligner. The
    teacher remains intentionally conservative: train-only fit, zero-drift
    identity, worst-block shrinkage calibration, and reliability-gated
    abstention. A dedicated class/protocol keeps V12 artifacts and diagnostics
    separate from V11 while preserving the strict gate.
    """

    protocol_version = "v12_outcome_aligned_block_residual_teacher"


def save_stage3_v12_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V12 residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyOutcomeAlignedBlockResidualTeacherV12.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyOutcomeAlignedBlockResidualTeacherV12.__name__)
    torch.save(
        {
            "class_name": TrainOnlyOutcomeAlignedBlockResidualTeacherV12.__name__,
            "protocol_version": TrainOnlyOutcomeAlignedBlockResidualTeacherV12.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v12_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyOutcomeAlignedBlockResidualTeacherV12, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    shrinkage_grid = state.get("shrinkage_grid", None)
    if shrinkage_grid is not None:
        shrinkage_grid = torch.as_tensor(shrinkage_grid).detach().cpu().flatten().tolist()
    teacher = TrainOnlyOutcomeAlignedBlockResidualTeacherV12(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        ridge=float(state.get("ridge", 1e-3)),
        eps=float(state.get("eps", 1e-8)),
        inner_fit_fraction=float(state.get("inner_fit_fraction", 0.75)),
        shrinkage_grid=shrinkage_grid,
        min_calibration_gain=float(state.get("min_calibration_gain", 0.0)),
        fallback_alpha=float(state.get("fallback_alpha", 0.0)),
        alpha_mode=str(state.get("alpha_mode", "scalar_nonnegative")),
        allow_negative_alpha=bool(state.get("allow_negative_alpha", False)),
        calibration_mae_weight=float(state.get("calibration_mae_weight", 0.0)),
        calibration_block_count=int(state.get("calibration_block_count", 5)),
        min_block_mse_gain=float(state.get("min_block_mse_gain", 0.0)),
        min_block_mae_gain=float(state.get("min_block_mae_gain", 0.0)),
        min_block_sign_agreement=float(state.get("min_block_sign_agreement", 0.5)),
        reliability_gate_enabled=bool(state.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(state.get("reliability_gate_quantile", 0.9)),
        min_reliable_fraction=float(state.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(state.get("disable_if_unreliable", True)),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V12 residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V12 residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyOutcomeAlignedBlockResidualTeacherV12.protocol_version)
    metadata.setdefault("class_name", TrainOnlyOutcomeAlignedBlockResidualTeacherV12.__name__)
    return teacher, metadata
