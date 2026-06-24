from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from models.stage3_residual_teacher_v12c import TrainOnlyChronologicalOOFResidualTeacherV12c


V12D_CONTEXT_FEATURE_MODES = {"drift_only", "history_only", "drift_plus_history"}


class TrainOnlyContextOOFResidualTeacherV12d(TrainOnlyChronologicalOOFResidualTeacherV12c):
    """V12d context teacher for drift-vs-history teacher-only ablations.

    V12d intentionally keeps V12c's chronological OOF policy logic while
    changing the feature contract from "drift only" to an explicit context
    feature matrix. The context can be drift-only, residual-history-only, or
    drift+history. The class does not build residual-history features itself;
    callers must provide a causally constructed matrix.
    """

    protocol_version = "v12d_context_oof_residual_teacher"

    def __init__(
        self,
        *,
        context_feature_mode: str,
        history_lags: Iterable[int] | None = None,
        feedback_delay: int = 0,
        history_feature_dim: int = 0,
        drift_signal_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        if context_feature_mode not in V12D_CONTEXT_FEATURE_MODES:
            raise ValueError(f"unsupported V12d context feature mode: {context_feature_mode}")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative")
        super().__init__(**kwargs)
        self.context_feature_mode = context_feature_mode
        self.history_lags = tuple(int(value) for value in (history_lags or ()))
        self.feedback_delay = int(feedback_delay)
        self.history_feature_dim = int(history_feature_dim)
        self.drift_signal_dim = int(drift_signal_dim)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        summary = super().diagnostic_summary()
        summary["teacher_method"] = self.protocol_version
        summary["teacher_context_feature_mode"] = self.context_feature_mode
        summary["teacher_history_lags"] = ",".join(str(value) for value in self.history_lags)
        summary["teacher_feedback_delay"] = self.feedback_delay
        summary["teacher_history_feature_dim"] = self.history_feature_dim
        summary["teacher_drift_signal_dim"] = self.drift_signal_dim
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["protocol_version"] = self.protocol_version
        state["context_feature_mode"] = self.context_feature_mode
        state["history_lags"] = self.history_lags
        state["feedback_delay"] = self.feedback_delay
        state["history_feature_dim"] = self.history_feature_dim
        state["drift_signal_dim"] = self.drift_signal_dim
        return state

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyContextOOFResidualTeacherV12d":
        super().load_state_dict(state)
        self.context_feature_mode = str(state.get("context_feature_mode", self.context_feature_mode))
        if self.context_feature_mode not in V12D_CONTEXT_FEATURE_MODES:
            raise ValueError(f"loaded unsupported V12d context feature mode: {self.context_feature_mode}")
        self.history_lags = tuple(int(value) for value in state.get("history_lags", self.history_lags))
        self.feedback_delay = int(state.get("feedback_delay", self.feedback_delay))
        self.history_feature_dim = int(state.get("history_feature_dim", self.history_feature_dim))
        self.drift_signal_dim = int(state.get("drift_signal_dim", self.drift_signal_dim))
        return self


def save_stage3_v12d_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyContextOOFResidualTeacherV12d,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V12d residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyContextOOFResidualTeacherV12d.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyContextOOFResidualTeacherV12d.__name__)
    signal_builder = getattr(teacher, "stage3_signal_builder_v12c", None)
    if signal_builder is not None and hasattr(signal_builder, "state_dict"):
        payload_metadata.setdefault("stage3_signal_builder_v12c_state", signal_builder.state_dict())
    history_builder = getattr(teacher, "stage3_history_builder_v12d", None)
    if history_builder is not None and hasattr(history_builder, "state_dict"):
        payload_metadata.setdefault("stage3_history_builder_v12d_state", history_builder.state_dict())
    torch.save(
        {
            "class_name": TrainOnlyContextOOFResidualTeacherV12d.__name__,
            "protocol_version": TrainOnlyContextOOFResidualTeacherV12d.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v12d_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyContextOOFResidualTeacherV12d, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    teacher = TrainOnlyContextOOFResidualTeacherV12d(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        context_feature_mode=str(state.get("context_feature_mode", "drift_only")),
        history_lags=tuple(int(value) for value in state.get("history_lags", ())),
        feedback_delay=int(state.get("feedback_delay", 0)),
        history_feature_dim=int(state.get("history_feature_dim", 0)),
        drift_signal_dim=int(state.get("drift_signal_dim", 0)),
        ridge_grid=torch.as_tensor(state.get("ridge_grid", [1e-3])).flatten().tolist(),
        alpha_grid=torch.as_tensor(state.get("alpha_grid", [0.0])).flatten().tolist(),
        eps=float(state.get("eps", 1e-8)),
        fold_count=int(state.get("fold_count", 5)),
        min_prefix_samples=int(state.get("min_prefix_samples", 32)),
        purge_gap=int(state.get("purge_gap", 0)),
        min_block_mse_gain=float(state.get("min_block_mse_gain", 0.0)),
        min_block_mae_gain=float(state.get("min_block_mae_gain", 0.0)),
        min_block_sign_agreement=float(state.get("min_block_sign_agreement", 0.5)),
        min_block_pass_fraction=float(state.get("min_block_pass_fraction", 0.6)),
        min_output_gain=float(state.get("min_output_gain", 0.0)),
        calibration_mae_weight=float(state.get("calibration_mae_weight", 0.0)),
        allow_negative_alpha=bool(state.get("allow_negative_alpha", True)),
        reliability_gate_enabled=bool(state.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(state.get("reliability_gate_quantile", 0.95)),
        min_reliable_fraction=float(state.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(state.get("disable_if_unreliable", False)),
        selected_policy=str(state.get("selected_policy", "relaxed")),
        claim_policy=str(state.get("claim_policy", "strict")),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V12d residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V12d residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyContextOOFResidualTeacherV12d.protocol_version)
    metadata.setdefault("class_name", TrainOnlyContextOOFResidualTeacherV12d.__name__)
    return teacher, metadata
