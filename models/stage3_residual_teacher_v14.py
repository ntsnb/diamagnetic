from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.stage3_residual_teacher_v12d import TrainOnlyContextOOFResidualTeacherV12d


V14_CONTEXT_FEATURE_MODES = {
    "fluctuation_only",
    "history_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
}
V14_READOUT_MODES = {"linear", "blockwise_linear"}


@dataclass
class Stage3V14ContextFeatures:
    features: torch.Tensor
    mode: str
    fluctuation_dim: int
    history_dim: int
    level_offset_dim: int
    feature_dim: int
    protocol_version: str = "v14_stability_fluctuation_context_features"


def build_stage3_v14_context_features(
    *,
    fluctuation_signal: torch.Tensor,
    history_features: torch.Tensor,
    level_offset: torch.Tensor,
    mode: str,
) -> Stage3V14ContextFeatures:
    if mode not in V14_CONTEXT_FEATURE_MODES:
        raise ValueError(f"unsupported V14 context feature mode: {mode}")
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
        raise ValueError(f"unsupported V14 context feature mode: {mode}")
    return Stage3V14ContextFeatures(
        features=features.detach().float(),
        mode=mode,
        fluctuation_dim=int(fluctuation.shape[1]),
        history_dim=int(history.shape[1]),
        level_offset_dim=int(level.shape[1]),
        feature_dim=int(features.shape[1]),
    )


def _v12d_inherited_context_mode(v14_mode: str) -> str:
    if v14_mode == "history_only":
        return "history_only"
    if v14_mode in {"fluctuation_plus_history", "level_offset_plus_history"}:
        return "drift_plus_history"
    if v14_mode in {"fluctuation_only", "level_offset_only"}:
        return "drift_only"
    raise ValueError(f"unsupported V14 context feature mode: {v14_mode}")


class TrainOnlyStabilityFluctuationResidualTeacherV14(TrainOnlyContextOOFResidualTeacherV12d):
    protocol_version = "v14_stability_fluctuation_oof_residual_teacher"

    def __init__(
        self,
        *,
        context_feature_mode: str,
        fluctuation_signal_dim: int = 0,
        history_feature_dim: int = 0,
        level_offset_dim: int = 0,
        readout_mode: str = "linear",
        readout_scale: bool = True,
        readout_scale_quantile: float = 0.75,
        readout_eps: float = 1e-6,
        selected_policy: str = "strict",
        claim_policy: str = "strict",
        **kwargs: Any,
    ) -> None:
        if context_feature_mode not in V14_CONTEXT_FEATURE_MODES:
            raise ValueError(f"unsupported V14 context feature mode: {context_feature_mode}")
        if readout_mode not in V14_READOUT_MODES:
            raise ValueError(f"unsupported V14 readout mode: {readout_mode}")
        if not 0.0 < readout_scale_quantile <= 1.0:
            raise ValueError("readout_scale_quantile must be in (0, 1]")
        if readout_eps <= 0:
            raise ValueError("readout_eps must be positive")
        inherited_context_mode = _v12d_inherited_context_mode(context_feature_mode)
        super().__init__(
            context_feature_mode=inherited_context_mode,
            history_lags=kwargs.pop("history_lags", ()),
            feedback_delay=int(kwargs.pop("feedback_delay", 0)),
            history_feature_dim=int(history_feature_dim),
            drift_signal_dim=int(fluctuation_signal_dim),
            selected_policy=selected_policy,
            claim_policy=claim_policy,
            **kwargs,
        )
        self.context_feature_mode = context_feature_mode
        self.v12d_inherited_context_feature_mode = inherited_context_mode
        self.fluctuation_signal_dim = int(fluctuation_signal_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.level_offset_dim = int(level_offset_dim)
        self.readout_mode = readout_mode
        self.readout_scale = bool(readout_scale)
        self.readout_scale_quantile = float(readout_scale_quantile)
        self.readout_eps = float(readout_eps)
        self.readout_train_scale: torch.Tensor | None = None
        self.readout_scale_fit_split = ""
        self.readout_scale_fit_sample_count = 0

    @property
    def feature_dim(self) -> int:
        return self.drift_dim

    def readout_feature_names(self) -> list[str]:
        return [f"context_{idx}" for idx in range(self.drift_dim)]

    def fit_readout_scale(self, context_train: torch.Tensor, *, split: str = "train") -> "TrainOnlyStabilityFluctuationResidualTeacherV14":
        if split != "train":
            raise ValueError("V14 readout scale can only be fit on the train split")
        context = self._as_2d_drift(context_train)
        scale = torch.quantile(context.abs(), self.readout_scale_quantile, dim=0)
        fallback = context.abs().mean(dim=0)
        scale = torch.where(scale > self.readout_eps, scale, fallback)
        scale = torch.where(scale > self.readout_eps, scale, torch.ones_like(scale))
        self.readout_train_scale = scale.detach().float()
        self.readout_scale_fit_split = "train"
        self.readout_scale_fit_sample_count = int(context.shape[0])
        return self

    def _feature_map(self, drift: torch.Tensor) -> torch.Tensor:
        context = self._as_2d_drift(drift)
        if not self.readout_scale:
            return context
        if self.readout_train_scale is None:
            raise RuntimeError("V14 readout scale must be fit before feature mapping")
        scale = self.readout_train_scale.to(device=context.device, dtype=context.dtype).clamp_min(self.readout_eps)
        return context / scale.unsqueeze(0)

    def fit(
        self,
        *,
        drift_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
        sample_ids: torch.Tensor | None = None,
    ) -> "TrainOnlyStabilityFluctuationResidualTeacherV14":
        if split != "train":
            raise ValueError("TrainOnlyStabilityFluctuationResidualTeacherV14 can only be fit on the train split")
        self.fit_readout_scale(drift_train, split="train")
        return super().fit(
            drift_train=drift_train,
            y_base_train=y_base_train,
            y_train=y_train,
            split=split,
            sample_ids=sample_ids,
        )

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        summary = super().diagnostic_summary()
        summary["teacher_method"] = self.protocol_version
        summary["teacher_context_feature_mode"] = self.context_feature_mode
        summary["teacher_v12d_inherited_context_feature_mode"] = self.v12d_inherited_context_feature_mode
        summary["teacher_fluctuation_signal_dim"] = self.fluctuation_signal_dim
        summary["teacher_history_feature_dim"] = self.history_feature_dim
        summary["teacher_level_offset_dim"] = self.level_offset_dim
        summary["teacher_readout_mode"] = self.readout_mode
        summary["teacher_readout_scale"] = str(self.readout_scale)
        summary["teacher_readout_scale_fit_split"] = self.readout_scale_fit_split
        summary["teacher_readout_scale_fit_sample_count"] = self.readout_scale_fit_sample_count
        if self.readout_train_scale is not None:
            summary["teacher_readout_scale_abs_mean"] = float(self.readout_train_scale.abs().mean().item())
            summary["teacher_readout_scale_min"] = float(self.readout_train_scale.min().item())
            summary["teacher_readout_scale_max"] = float(self.readout_train_scale.max().item())
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["protocol_version"] = self.protocol_version
        state["context_feature_mode"] = self.v12d_inherited_context_feature_mode
        state["v14_context_feature_mode"] = self.context_feature_mode
        state["v12d_inherited_context_feature_mode"] = self.v12d_inherited_context_feature_mode
        state["fluctuation_signal_dim"] = self.fluctuation_signal_dim
        state["history_feature_dim"] = self.history_feature_dim
        state["level_offset_dim"] = self.level_offset_dim
        state["readout_mode"] = self.readout_mode
        state["readout_scale"] = self.readout_scale
        state["readout_scale_quantile"] = self.readout_scale_quantile
        state["readout_eps"] = self.readout_eps
        state["readout_train_scale"] = (
            self.readout_train_scale.detach().clone() if self.readout_train_scale is not None else torch.empty(0)
        )
        state["readout_scale_fit_split"] = self.readout_scale_fit_split
        state["readout_scale_fit_sample_count"] = self.readout_scale_fit_sample_count
        diagnostic_summary = dict(state.get("diagnostic_summary", {}))
        diagnostic_summary.update(self.diagnostic_summary())
        state["diagnostic_summary"] = diagnostic_summary
        return state

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyStabilityFluctuationResidualTeacherV14":
        protocol = str(state.get("protocol_version", ""))
        if protocol and protocol != self.protocol_version:
            raise ValueError(f"checkpoint is not a V14 stability teacher: {protocol}")
        v14_mode = str(state.get("v14_context_feature_mode", state.get("context_feature_mode", self.context_feature_mode)))
        if v14_mode not in V14_CONTEXT_FEATURE_MODES:
            raise ValueError(f"loaded unsupported V14 context feature mode: {v14_mode}")
        readout_mode = str(state.get("readout_mode", self.readout_mode))
        if readout_mode not in V14_READOUT_MODES:
            raise ValueError(f"loaded unsupported V14 readout mode: {readout_mode}")
        self.readout_mode = readout_mode
        self.readout_scale = bool(state.get("readout_scale", self.readout_scale))
        self.readout_scale_quantile = float(state.get("readout_scale_quantile", self.readout_scale_quantile))
        self.readout_eps = float(state.get("readout_eps", self.readout_eps))
        scale = torch.as_tensor(state.get("readout_train_scale", torch.empty(0))).detach().float()
        self.readout_train_scale = scale if scale.numel() else None
        self.readout_scale_fit_split = str(state.get("readout_scale_fit_split", "train" if scale.numel() else ""))
        self.readout_scale_fit_sample_count = int(state.get("readout_scale_fit_sample_count", 0))
        base_state = dict(state)
        base_state["context_feature_mode"] = _v12d_inherited_context_mode(v14_mode)
        super().load_state_dict(base_state)
        self.context_feature_mode = v14_mode
        self.v12d_inherited_context_feature_mode = _v12d_inherited_context_mode(v14_mode)
        self.fluctuation_signal_dim = int(state.get("fluctuation_signal_dim", self.fluctuation_signal_dim))
        self.history_feature_dim = int(state.get("history_feature_dim", self.history_feature_dim))
        self.level_offset_dim = int(state.get("level_offset_dim", self.level_offset_dim))
        return self


def save_stage3_v14_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyStabilityFluctuationResidualTeacherV14,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V14 residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyStabilityFluctuationResidualTeacherV14.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyStabilityFluctuationResidualTeacherV14.__name__)
    signal_builder = getattr(teacher, "stage3_signal_builder_v14", None)
    if signal_builder is not None and hasattr(signal_builder, "state_dict"):
        payload_metadata.setdefault("stage3_signal_builder_v14_state", signal_builder.state_dict())
    history_builder = getattr(teacher, "stage3_history_builder_v14", None)
    if history_builder is not None and hasattr(history_builder, "state_dict"):
        payload_metadata.setdefault("stage3_history_builder_v14_state", history_builder.state_dict())
    torch.save(
        {
            "class_name": TrainOnlyStabilityFluctuationResidualTeacherV14.__name__,
            "protocol_version": TrainOnlyStabilityFluctuationResidualTeacherV14.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v14_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyStabilityFluctuationResidualTeacherV14, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    protocol = str(checkpoint.get("protocol_version", ""))
    if protocol != TrainOnlyStabilityFluctuationResidualTeacherV14.protocol_version:
        raise ValueError(f"checkpoint is not a V14 stability teacher: {protocol}")
    state = checkpoint.get("state", checkpoint)
    v14_mode = str(state.get("v14_context_feature_mode", state.get("context_feature_mode", "fluctuation_only")))
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        context_feature_mode=v14_mode,
        fluctuation_signal_dim=int(state.get("fluctuation_signal_dim", 0)),
        history_feature_dim=int(state.get("history_feature_dim", 0)),
        level_offset_dim=int(state.get("level_offset_dim", 0)),
        readout_mode=str(state.get("readout_mode", "linear")),
        readout_scale=bool(state.get("readout_scale", True)),
        readout_scale_quantile=float(state.get("readout_scale_quantile", 0.75)),
        readout_eps=float(state.get("readout_eps", 1e-6)),
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
        allow_negative_alpha=bool(state.get("allow_negative_alpha", False)),
        reliability_gate_enabled=bool(state.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(state.get("reliability_gate_quantile", 0.95)),
        min_reliable_fraction=float(state.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(state.get("disable_if_unreliable", False)),
        selected_policy=str(state.get("selected_policy", "strict")),
        claim_policy=str(state.get("claim_policy", "strict")),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V14 residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V14 residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyStabilityFluctuationResidualTeacherV14.protocol_version)
    metadata.setdefault("class_name", TrainOnlyStabilityFluctuationResidualTeacherV14.__name__)
    return teacher, metadata
