from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import torch


class TrainOnlyResidualTeacherV9:
    """Train-only residual correction teacher for Stage III V9.

    The teacher maps the served generator-input drift to a forecast residual
    correction. It is intentionally fit only on the train split; validation and
    test are allowed to call ``predict_correction`` but not ``fit`` or
    ``update``. The feature map has no intercept so physical zero drift maps to
    exactly zero correction.
    """

    protocol_version = "v9_residual_teacher"

    def __init__(
        self,
        *,
        drift_dim: int,
        output_shape: tuple[int, ...],
        ridge: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if not output_shape or any(int(dim) <= 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.drift_dim = int(drift_dim)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge = float(ridge)
        self.eps = float(eps)

        self.correction_weight: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.fit_split = ""
        self.train_sample_ids = torch.empty(0, dtype=torch.long)
        self.teacher_residual_mse_before = float("nan")
        self.teacher_residual_mse_after = float("nan")
        self.teacher_residual_r2 = float("nan")

    @property
    def feature_dim(self) -> int:
        return self.drift_dim * 3 + 1

    def _as_2d_drift(self, drift: torch.Tensor) -> torch.Tensor:
        drift = drift.detach() if not drift.requires_grad else drift
        if drift.ndim == 1:
            drift = drift.unsqueeze(0)
        if drift.ndim != 2 or drift.shape[1] != self.drift_dim:
            raise ValueError(f"drift must have shape [N, {self.drift_dim}]")
        return drift.float()

    def _feature_map(self, drift: torch.Tensor) -> torch.Tensor:
        drift = self._as_2d_drift(drift)
        norm = torch.linalg.vector_norm(drift, dim=1, keepdim=True)
        return torch.cat([drift, drift.abs(), drift.pow(2), norm], dim=1)

    def fit(
        self,
        *,
        drift_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
        sample_ids: torch.Tensor | None = None,
    ) -> "TrainOnlyResidualTeacherV9":
        if split != "train":
            raise ValueError("TrainOnlyResidualTeacherV9 can only be fit on the train split")
        drift = self._as_2d_drift(drift_train)
        y_base = y_base_train.detach().float().reshape(drift.shape[0], -1)
        y = y_train.detach().float().reshape(drift.shape[0], -1)
        if y_base.shape != y.shape:
            raise ValueError("y_base_train and y_train must have the same flattened shape")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"teacher output dim {y.shape[1]} does not match expected {self.output_dim}")
        if drift.shape[0] != y.shape[0]:
            raise ValueError("drift_train and y_train must share sample count")

        features = self._feature_map(drift)
        residual = y - y_base
        eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        solution = torch.linalg.solve(
            features.T @ features + self.ridge * eye,
            features.T @ residual,
        )
        correction = features @ solution
        before_mse = torch.mean(residual.pow(2))
        after_mse = torch.mean((residual - correction).pow(2))
        total = torch.mean((residual - residual.mean(dim=0, keepdim=True)).pow(2)).clamp_min(self.eps)

        self.correction_weight = solution.detach().float()
        self.fit_sample_count = int(drift.shape[0])
        self.fit_split = "train"
        if sample_ids is None:
            self.train_sample_ids = torch.arange(self.fit_sample_count, dtype=torch.long)
        else:
            sample_ids = sample_ids.detach().cpu().long().flatten()
            if sample_ids.numel() != self.fit_sample_count:
                raise ValueError("sample_ids must match fit sample count")
            self.train_sample_ids = sample_ids
        self.teacher_residual_mse_before = float(before_mse.item())
        self.teacher_residual_mse_after = float(after_mse.item())
        self.teacher_residual_r2 = float(torch.clamp(1.0 - after_mse / total, min=-1.0, max=1.0).item())
        return self

    def update(
        self,
        *,
        drift: torch.Tensor,
        y_base: torch.Tensor,
        y: torch.Tensor,
        split: str,
    ) -> "TrainOnlyResidualTeacherV9":
        if split != "train":
            raise ValueError("TrainOnlyResidualTeacherV9 update is restricted to the train split")
        return self.fit(drift_train=drift, y_base_train=y_base, y_train=y, split=split)

    def predict_correction(self, drift: torch.Tensor) -> torch.Tensor:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV9 must be fit before predict_correction")
        features = self._feature_map(drift)
        weight = self.correction_weight.to(device=features.device, dtype=features.dtype)
        correction = features @ weight
        return correction.reshape(features.shape[0], *self.output_shape)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV9 must be fit before diagnostics")
        return {
            "teacher_method": self.protocol_version,
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
            "drift_dim": self.drift_dim,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "ridge": self.ridge,
            "teacher_residual_mse_before": self.teacher_residual_mse_before,
            "teacher_residual_mse_after": self.teacher_residual_mse_after,
            "teacher_residual_r2": self.teacher_residual_r2,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV9 must be fit before serialization")
        return {
            "class_name": self.__class__.__name__,
            "protocol_version": self.protocol_version,
            "drift_dim": self.drift_dim,
            "output_shape": self.output_shape,
            "ridge": self.ridge,
            "eps": self.eps,
            "correction_weight": self.correction_weight.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "fit_split": self.fit_split,
            "train_sample_ids": self.train_sample_ids.detach().clone(),
            "teacher_residual_mse_before": self.teacher_residual_mse_before,
            "teacher_residual_mse_after": self.teacher_residual_mse_after,
            "teacher_residual_r2": self.teacher_residual_r2,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyResidualTeacherV9":
        self.drift_dim = int(state["drift_dim"])
        self.output_shape = tuple(int(dim) for dim in state["output_shape"])
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        self.correction_weight = torch.as_tensor(state["correction_weight"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.fit_split = str(state.get("fit_split", "train"))
        self.train_sample_ids = torch.as_tensor(state.get("train_sample_ids", torch.arange(self.fit_sample_count))).long()
        self.teacher_residual_mse_before = float(state.get("teacher_residual_mse_before", float("nan")))
        self.teacher_residual_mse_after = float(state.get("teacher_residual_mse_after", float("nan")))
        self.teacher_residual_r2 = float(state.get("teacher_residual_r2", float("nan")))
        if self.correction_weight.shape != (self.feature_dim, self.output_dim):
            raise ValueError("loaded V9 residual teacher weight has incompatible shape")
        return self

    def state_hash(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_stage3_v9_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyResidualTeacherV9,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V9 residual teacher checkpoint: {path}")
    metadata = dict(metadata or {})
    metadata.setdefault("protocol_version", TrainOnlyResidualTeacherV9.protocol_version)
    metadata.setdefault("fit_sample_count", teacher.fit_sample_count)
    metadata.setdefault("fit_split", teacher.fit_split)
    torch.save(
        {
            "teacher_class": teacher.__class__.__name__,
            "teacher_state_dict": teacher.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_stage3_v9_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyResidualTeacherV9, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint["teacher_state_dict"]
    teacher = TrainOnlyResidualTeacherV9(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        ridge=float(state.get("ridge", 1e-3)),
        eps=float(state.get("eps", 1e-8)),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V9 residual teacher drift_dim does not match expected drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V9 residual teacher output_shape does not match expected output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyResidualTeacherV9.protocol_version)
    return teacher, metadata
