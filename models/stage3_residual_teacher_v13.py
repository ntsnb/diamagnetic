from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.stage3_residual_teacher_v12d import TrainOnlyContextOOFResidualTeacherV12d


V13_CONTEXT_FEATURE_MODES = {
    "fluctuation_only",
    "history_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
}


@dataclass
class Stage3V13ContextFeatures:
    features: torch.Tensor
    mode: str
    fluctuation_dim: int
    history_dim: int
    level_offset_dim: int
    feature_dim: int


def build_stage3_v13_context_features(
    *,
    fluctuation_signal: torch.Tensor,
    history_features: torch.Tensor,
    level_offset: torch.Tensor,
    mode: str,
) -> Stage3V13ContextFeatures:
    if mode not in V13_CONTEXT_FEATURE_MODES:
        raise ValueError(f"unsupported V13 context feature mode: {mode}")
    fluctuation = fluctuation_signal.detach().float()
    history = history_features.detach().float()
    level = level_offset.detach().float()
    if fluctuation.ndim != 2:
        raise ValueError("fluctuation_signal must be a 2-D tensor")
    if history.ndim != 2:
        raise ValueError("history_features must be a 2-D tensor")
    if level.ndim != 2:
        raise ValueError("level_offset must be a 2-D tensor")
    if fluctuation.shape[0] != history.shape[0] or fluctuation.shape[0] != level.shape[0]:
        raise ValueError("fluctuation_signal, history_features, and level_offset must have the same sample count")
    if mode == "fluctuation_only":
        features = fluctuation
    elif mode == "history_only":
        features = history
    elif mode == "fluctuation_plus_history":
        features = torch.cat([fluctuation, history], dim=1)
    elif mode == "level_offset_only":
        features = level
    elif mode == "level_offset_plus_history":
        features = torch.cat([level, history], dim=1)
    else:
        raise ValueError(f"unsupported V13 context feature mode: {mode}")
    return Stage3V13ContextFeatures(
        features=features.detach().float(),
        mode=mode,
        fluctuation_dim=int(fluctuation.shape[1]),
        history_dim=int(history.shape[1]),
        level_offset_dim=int(level.shape[1]),
        feature_dim=int(features.shape[1]),
    )


def _v12d_inherited_context_mode(v13_mode: str) -> str:
    if v13_mode == "history_only":
        return "history_only"
    if v13_mode in {"fluctuation_plus_history", "level_offset_plus_history"}:
        return "drift_plus_history"
    if v13_mode in {"fluctuation_only", "level_offset_only"}:
        return "drift_only"
    raise ValueError(f"unsupported V13 context feature mode: {v13_mode}")


def _inherited_drift_signal_dim(
    *,
    context_feature_mode: str,
    fluctuation_signal_dim: int,
    level_offset_dim: int,
) -> int:
    if context_feature_mode == "history_only":
        return 0
    if context_feature_mode in {"level_offset_only", "level_offset_plus_history"}:
        return int(level_offset_dim)
    return int(fluctuation_signal_dim)


class TrainOnlyFluctuationContextResidualTeacherV13(TrainOnlyContextOOFResidualTeacherV12d):
    protocol_version = "v13_fluctuation_context_oof_residual_teacher"

    def __init__(
        self,
        *,
        context_feature_mode: str,
        fluctuation_signal_dim: int = 0,
        history_feature_dim: int = 0,
        level_offset_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        if context_feature_mode not in V13_CONTEXT_FEATURE_MODES:
            raise ValueError(f"unsupported V13 context feature mode: {context_feature_mode}")
        inherited_context_mode = _v12d_inherited_context_mode(context_feature_mode)
        inherited_drift_dim = _inherited_drift_signal_dim(
            context_feature_mode=context_feature_mode,
            fluctuation_signal_dim=fluctuation_signal_dim,
            level_offset_dim=level_offset_dim,
        )
        super().__init__(
            context_feature_mode=inherited_context_mode,
            history_lags=kwargs.pop("history_lags", ()),
            feedback_delay=int(kwargs.pop("feedback_delay", 0)),
            history_feature_dim=int(history_feature_dim),
            drift_signal_dim=inherited_drift_dim,
            **kwargs,
        )
        self.context_feature_mode = context_feature_mode
        self.v12d_inherited_context_feature_mode = inherited_context_mode
        self.fluctuation_signal_dim = int(fluctuation_signal_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.level_offset_dim = int(level_offset_dim)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        summary = super().diagnostic_summary()
        summary["teacher_method"] = self.protocol_version
        summary["teacher_context_feature_mode"] = self.context_feature_mode
        summary["teacher_v12d_inherited_context_feature_mode"] = self.v12d_inherited_context_feature_mode
        summary["teacher_fluctuation_signal_dim"] = self.fluctuation_signal_dim
        summary["teacher_history_feature_dim"] = self.history_feature_dim
        summary["teacher_level_offset_dim"] = self.level_offset_dim
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        v13_mode = self.context_feature_mode
        inherited_mode = _v12d_inherited_context_mode(v13_mode)
        state["protocol_version"] = self.protocol_version
        state["context_feature_mode"] = inherited_mode
        state["v13_context_feature_mode"] = v13_mode
        state["v12d_inherited_context_feature_mode"] = inherited_mode
        state["fluctuation_signal_dim"] = self.fluctuation_signal_dim
        state["history_feature_dim"] = self.history_feature_dim
        state["level_offset_dim"] = self.level_offset_dim
        diagnostic_summary = dict(state.get("diagnostic_summary", {}))
        diagnostic_summary["teacher_context_feature_mode"] = v13_mode
        diagnostic_summary["teacher_v12d_inherited_context_feature_mode"] = inherited_mode
        state["diagnostic_summary"] = diagnostic_summary
        return state

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyFluctuationContextResidualTeacherV13":
        v13_mode = str(state.get("v13_context_feature_mode", state.get("context_feature_mode", self.context_feature_mode)))
        if v13_mode not in V13_CONTEXT_FEATURE_MODES:
            raise ValueError(f"loaded unsupported V13 context feature mode: {v13_mode}")
        base_state = dict(state)
        base_state["context_feature_mode"] = _v12d_inherited_context_mode(v13_mode)
        super().load_state_dict(base_state)
        self.context_feature_mode = v13_mode
        self.v12d_inherited_context_feature_mode = _v12d_inherited_context_mode(v13_mode)
        self.fluctuation_signal_dim = int(state.get("fluctuation_signal_dim", self.fluctuation_signal_dim))
        self.history_feature_dim = int(state.get("history_feature_dim", self.history_feature_dim))
        self.level_offset_dim = int(state.get("level_offset_dim", self.level_offset_dim))
        return self


def save_stage3_v13_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyFluctuationContextResidualTeacherV13,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V13 residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyFluctuationContextResidualTeacherV13.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyFluctuationContextResidualTeacherV13.__name__)
    signal_builder = getattr(teacher, "stage3_signal_builder_v13", None)
    if signal_builder is not None and hasattr(signal_builder, "state_dict"):
        payload_metadata.setdefault("stage3_signal_builder_v13_state", signal_builder.state_dict())
    history_builder = getattr(teacher, "stage3_history_builder_v13", None)
    if history_builder is not None and hasattr(history_builder, "state_dict"):
        payload_metadata.setdefault("stage3_history_builder_v13_state", history_builder.state_dict())
    torch.save(
        {
            "class_name": TrainOnlyFluctuationContextResidualTeacherV13.__name__,
            "protocol_version": TrainOnlyFluctuationContextResidualTeacherV13.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v13_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyFluctuationContextResidualTeacherV13, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    v13_mode = str(state.get("v13_context_feature_mode", state.get("context_feature_mode", "fluctuation_only")))
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        context_feature_mode=v13_mode,
        fluctuation_signal_dim=int(state.get("fluctuation_signal_dim", 0)),
        history_feature_dim=int(state.get("history_feature_dim", 0)),
        level_offset_dim=int(state.get("level_offset_dim", 0)),
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
        raise ValueError("loaded V13 residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V13 residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyFluctuationContextResidualTeacherV13.protocol_version)
    metadata.setdefault("class_name", TrainOnlyFluctuationContextResidualTeacherV13.__name__)
    return teacher, metadata

