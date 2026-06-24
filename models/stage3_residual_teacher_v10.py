from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


class TrainOnlyResidualTeacherV10:
    """Train-only residual teacher with inner-train shrinkage calibration.

    V9 showed that a teacher can fit train residuals while becoming harmful on
    rolling validation. V10 therefore separates the residual basis fit from a
    train-only scalar calibration step. The calibration grid contains zero, so
    the teacher can abstain from applying a correction when the inner train
    holdout does not support it. The feature map has no intercept, preserving
    exact zero-drift identity.
    """

    protocol_version = "v10_shrunk_residual_teacher"

    def __init__(
        self,
        *,
        drift_dim: int,
        output_shape: tuple[int, ...],
        ridge: float = 1e-3,
        eps: float = 1e-8,
        inner_fit_fraction: float = 0.75,
        shrinkage_grid: Iterable[float] | None = None,
        min_calibration_gain: float = 0.0,
        fallback_alpha: float = 0.0,
        alpha_mode: str = "scalar_nonnegative",
        allow_negative_alpha: bool = False,
        calibration_mae_weight: float = 0.0,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if not output_shape or any(int(dim) <= 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.5 <= inner_fit_fraction <= 0.9:
            raise ValueError("inner_fit_fraction must be in [0.5, 0.9]")
        if alpha_mode not in {"scalar_nonnegative", "per_output"}:
            raise ValueError("alpha_mode must be scalar_nonnegative or per_output")
        if calibration_mae_weight < 0:
            raise ValueError("calibration_mae_weight must be non-negative")
        grid = [float(value) for value in (shrinkage_grid or (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0))]
        if allow_negative_alpha:
            grid.extend([-abs(value) for value in grid if value != 0.0])
        if not grid:
            raise ValueError("shrinkage_grid must contain at least one value")
        if 0.0 not in grid:
            grid.append(0.0)

        self.drift_dim = int(drift_dim)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.inner_fit_fraction = float(inner_fit_fraction)
        self.shrinkage_grid = torch.tensor(sorted(set(grid)), dtype=torch.float32)
        self.min_calibration_gain = float(min_calibration_gain)
        self.fallback_alpha = float(fallback_alpha)
        self.alpha_mode = alpha_mode
        self.allow_negative_alpha = bool(allow_negative_alpha)
        self.calibration_mae_weight = float(calibration_mae_weight)

        self.correction_weight: torch.Tensor | None = None
        self.shrinkage_alpha = 0.0
        self.shrinkage_alpha_by_output: torch.Tensor | None = None
        self.selected_alpha_source = "not_fit"
        self.fit_sample_count = 0
        self.inner_fit_sample_count = 0
        self.inner_calib_sample_count = 0
        self.fit_split = ""
        self.train_sample_ids = torch.empty(0, dtype=torch.long)
        self.teacher_residual_mse_before = float("nan")
        self.teacher_residual_mse_after = float("nan")
        self.teacher_residual_r2 = float("nan")
        self.calibration_base_mse = float("nan")
        self.calibration_best_mse = float("nan")
        self.calibration_best_gain = float("nan")
        self.calibration_best_alpha = 0.0
        self.calibration_selected_alpha = 0.0
        self.calibration_alpha_abs_mean = 0.0
        self.calibration_alpha_nonzero_fraction = 0.0
        self.calibration_min_output_gain = float("nan")
        self.calibration_mean_output_gain = float("nan")
        self.calibration_sign_agreement = float("nan")
        self.calibration_residual_correction_cosine = float("nan")

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

    def _solve_weight(self, features: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        return torch.linalg.solve(
            features.T @ features + self.ridge * eye,
            features.T @ residual,
        )

    def _inner_fit_stop(self, sample_count: int) -> int:
        if sample_count < 8:
            return sample_count
        fit_stop = int(round(sample_count * self.inner_fit_fraction))
        return min(max(fit_stop, 4), sample_count - 3)

    def _calibrate_alpha(
        self,
        *,
        features: torch.Tensor,
        residual: torch.Tensor,
        inner_weight: torch.Tensor,
    ) -> None:
        fit_stop = self._inner_fit_stop(features.shape[0])
        self.inner_fit_sample_count = int(fit_stop)
        self.inner_calib_sample_count = int(features.shape[0] - fit_stop)
        if self.inner_calib_sample_count <= 0:
            self.calibration_base_mse = float(torch.mean(residual.pow(2)).item())
            self.calibration_best_mse = self.calibration_base_mse
            self.calibration_best_gain = 0.0
            self.calibration_best_alpha = self.fallback_alpha
            self.calibration_selected_alpha = self.fallback_alpha
            self.shrinkage_alpha = self.fallback_alpha
            self.shrinkage_alpha_by_output = torch.full((self.output_dim,), self.fallback_alpha, dtype=torch.float32)
            self.selected_alpha_source = "fallback_no_inner_calibration"
            return

        calib_residual = residual[fit_stop:]
        calib_prediction = features[fit_stop:] @ inner_weight
        if self.alpha_mode == "per_output":
            self._calibrate_per_output_alpha(calib_residual, calib_prediction)
            return
        base_mse = torch.mean(calib_residual.pow(2))
        best_alpha = 0.0
        best_mse = None
        best_abs_alpha = float("inf")
        for alpha_value in self.shrinkage_grid.tolist():
            alpha = float(alpha_value)
            mse = torch.mean((calib_residual - alpha * calib_prediction).pow(2))
            mse_value = float(mse.item())
            if best_mse is None or mse_value < best_mse - self.eps or (
                abs(mse_value - best_mse) <= self.eps and abs(alpha) < best_abs_alpha
            ):
                best_mse = mse_value
                best_alpha = alpha
                best_abs_alpha = abs(alpha)

        assert best_mse is not None
        gain = 0.0 if float(base_mse.item()) == 0.0 else (float(base_mse.item()) - best_mse) / float(base_mse.item())
        calibration_passed = gain >= self.min_calibration_gain
        selected_alpha = best_alpha if calibration_passed else self.fallback_alpha
        self.calibration_base_mse = float(base_mse.item())
        self.calibration_best_mse = float(best_mse)
        self.calibration_best_gain = float(gain)
        self.calibration_best_alpha = float(best_alpha)
        self.calibration_selected_alpha = float(selected_alpha)
        self.shrinkage_alpha = float(selected_alpha)
        self.shrinkage_alpha_by_output = torch.full((self.output_dim,), float(selected_alpha), dtype=torch.float32)
        self.calibration_alpha_abs_mean = abs(float(selected_alpha))
        self.calibration_alpha_nonzero_fraction = float(abs(float(selected_alpha)) > self.eps)
        self.calibration_min_output_gain = float(gain)
        self.calibration_mean_output_gain = float(gain)
        self.selected_alpha_source = "inner_train_grid" if calibration_passed else "fallback_low_gain"

        selected_prediction = selected_alpha * calib_prediction
        self._store_calibration_direction_metrics(calib_residual, selected_prediction)

    def _calibrate_per_output_alpha(self, calib_residual: torch.Tensor, calib_prediction: torch.Tensor) -> None:
        grid = self.shrinkage_grid.to(device=calib_residual.device, dtype=calib_residual.dtype)
        selected = torch.full((self.output_dim,), self.fallback_alpha, device=calib_residual.device)
        best = torch.zeros(self.output_dim, device=calib_residual.device)
        gains = torch.zeros(self.output_dim, device=calib_residual.device)
        base_mse_by_output = torch.mean(calib_residual.pow(2), dim=0)
        base_mae_by_output = torch.mean(calib_residual.abs(), dim=0)
        base_loss_by_output = base_mse_by_output + self.calibration_mae_weight * base_mae_by_output
        best_loss = base_loss_by_output.clone()
        for output_idx in range(self.output_dim):
            pred_j = calib_prediction[:, output_idx]
            residual_j = calib_residual[:, output_idx]
            best_alpha = torch.tensor(0.0, device=calib_residual.device, dtype=calib_residual.dtype)
            best_j_loss = best_loss[output_idx]
            for alpha in grid:
                err = residual_j - alpha * pred_j
                loss = torch.mean(err.pow(2)) + self.calibration_mae_weight * torch.mean(err.abs())
                if float(loss.item()) < float(best_j_loss.item()) - self.eps or (
                    abs(float(loss.item()) - float(best_j_loss.item())) <= self.eps
                    and abs(float(alpha.item())) < abs(float(best_alpha.item()))
                ):
                    best_j_loss = loss
                    best_alpha = alpha
            gain = (base_loss_by_output[output_idx] - best_j_loss) / base_loss_by_output[output_idx].clamp_min(self.eps)
            best[output_idx] = best_alpha
            gains[output_idx] = gain
            if float(gain.item()) >= self.min_calibration_gain:
                selected[output_idx] = best_alpha

        selected_prediction = calib_prediction * selected.unsqueeze(0)
        selected_mse = torch.mean((calib_residual - selected_prediction).pow(2))
        base_mse = torch.mean(calib_residual.pow(2))
        best_prediction = calib_prediction * best.unsqueeze(0)
        best_mse = torch.mean((calib_residual - best_prediction).pow(2))
        best_gain = (base_mse - best_mse) / base_mse.clamp_min(self.eps)

        self.shrinkage_alpha_by_output = selected.detach().float().cpu()
        self.shrinkage_alpha = float(selected.mean().item())
        self.calibration_base_mse = float(base_mse.item())
        self.calibration_best_mse = float(best_mse.item())
        self.calibration_best_gain = float(best_gain.item())
        self.calibration_best_alpha = float(best.mean().item())
        self.calibration_selected_alpha = float(selected.mean().item())
        self.calibration_alpha_abs_mean = float(selected.abs().mean().item())
        self.calibration_alpha_nonzero_fraction = float((selected.abs() > self.eps).float().mean().item())
        self.calibration_min_output_gain = float(gains.min().item())
        self.calibration_mean_output_gain = float(gains.mean().item())
        self.selected_alpha_source = "per_output_grid" if bool((selected.abs() > self.eps).any()) else "fallback_low_gain"
        self._store_calibration_direction_metrics(calib_residual, selected_prediction)

    def _store_calibration_direction_metrics(
        self,
        calib_residual: torch.Tensor,
        selected_prediction: torch.Tensor,
    ) -> None:
        sign_mask = calib_residual.abs() > self.eps
        if sign_mask.any():
            sign_agreement = torch.eq(torch.sign(calib_residual[sign_mask]), torch.sign(selected_prediction[sign_mask]))
            self.calibration_sign_agreement = float(sign_agreement.float().mean().item())
        else:
            self.calibration_sign_agreement = float("nan")
        if calib_residual.numel() and selected_prediction.abs().max().item() > self.eps:
            self.calibration_residual_correction_cosine = float(
                F.cosine_similarity(
                    calib_residual.reshape(calib_residual.shape[0], -1),
                    selected_prediction.reshape(selected_prediction.shape[0], -1),
                    dim=1,
                    eps=self.eps,
                )
                .mean()
                .item()
            )
        else:
            self.calibration_residual_correction_cosine = float("nan")

    def fit(
        self,
        *,
        drift_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
        sample_ids: torch.Tensor | None = None,
    ) -> "TrainOnlyResidualTeacherV10":
        if split != "train":
            raise ValueError("TrainOnlyResidualTeacherV10 can only be fit on the train split")
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
        fit_stop = self._inner_fit_stop(features.shape[0])
        inner_weight = self._solve_weight(features[:fit_stop], residual[:fit_stop])
        self._calibrate_alpha(features=features, residual=residual, inner_weight=inner_weight)

        solution = self._solve_weight(features, residual)
        alpha = self._alpha_vector(device=features.device, dtype=features.dtype)
        correction = (features @ solution) * alpha.unsqueeze(0)
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
    ) -> "TrainOnlyResidualTeacherV10":
        if split != "train":
            raise ValueError("TrainOnlyResidualTeacherV10 update is restricted to the train split")
        return self.fit(drift_train=drift, y_base_train=y_base, y_train=y, split=split)

    def predict_correction(self, drift: torch.Tensor) -> torch.Tensor:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV10 must be fit before predict_correction")
        features = self._feature_map(drift)
        weight = self.correction_weight.to(device=features.device, dtype=features.dtype)
        alpha = self._alpha_vector(device=features.device, dtype=features.dtype)
        correction = (features @ weight) * alpha.unsqueeze(0)
        return correction.reshape(features.shape[0], *self.output_shape)

    def _alpha_vector(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.shrinkage_alpha_by_output is None:
            return torch.full((self.output_dim,), self.shrinkage_alpha, device=device, dtype=dtype)
        return self.shrinkage_alpha_by_output.to(device=device, dtype=dtype)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV10 must be fit before diagnostics")
        return {
            "teacher_method": self.protocol_version,
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
            "inner_fit_sample_count": self.inner_fit_sample_count,
            "inner_calib_sample_count": self.inner_calib_sample_count,
            "drift_dim": self.drift_dim,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "ridge": self.ridge,
            "teacher_residual_mse_before": self.teacher_residual_mse_before,
            "teacher_residual_mse_after": self.teacher_residual_mse_after,
            "teacher_residual_r2": self.teacher_residual_r2,
            "teacher_shrinkage_alpha": self.shrinkage_alpha,
            "teacher_alpha_source": self.selected_alpha_source,
            "teacher_calibration_base_mse": self.calibration_base_mse,
            "teacher_calibration_best_mse": self.calibration_best_mse,
            "teacher_calibration_best_gain": self.calibration_best_gain,
            "teacher_calibration_best_alpha": self.calibration_best_alpha,
            "teacher_calibration_selected_alpha": self.calibration_selected_alpha,
            "teacher_alpha_mode": self.alpha_mode,
            "teacher_allow_negative_alpha": str(self.allow_negative_alpha),
            "teacher_calibration_mae_weight": self.calibration_mae_weight,
            "teacher_alpha_abs_mean": self.calibration_alpha_abs_mean,
            "teacher_alpha_nonzero_fraction": self.calibration_alpha_nonzero_fraction,
            "teacher_calibration_min_output_gain": self.calibration_min_output_gain,
            "teacher_calibration_mean_output_gain": self.calibration_mean_output_gain,
            "teacher_calibration_sign_agreement": self.calibration_sign_agreement,
            "teacher_calibration_residual_correction_cosine": self.calibration_residual_correction_cosine,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyResidualTeacherV10 must be fit before serialization")
        return {
            "class_name": self.__class__.__name__,
            "protocol_version": self.protocol_version,
            "drift_dim": self.drift_dim,
            "output_shape": self.output_shape,
            "ridge": self.ridge,
            "eps": self.eps,
            "inner_fit_fraction": self.inner_fit_fraction,
            "shrinkage_grid": self.shrinkage_grid.detach().clone(),
            "min_calibration_gain": self.min_calibration_gain,
            "fallback_alpha": self.fallback_alpha,
            "alpha_mode": self.alpha_mode,
            "allow_negative_alpha": self.allow_negative_alpha,
            "calibration_mae_weight": self.calibration_mae_weight,
            "correction_weight": self.correction_weight.detach().clone(),
            "shrinkage_alpha": self.shrinkage_alpha,
            "shrinkage_alpha_by_output": self._alpha_vector(device=torch.device("cpu"), dtype=torch.float32),
            "selected_alpha_source": self.selected_alpha_source,
            "fit_sample_count": self.fit_sample_count,
            "inner_fit_sample_count": self.inner_fit_sample_count,
            "inner_calib_sample_count": self.inner_calib_sample_count,
            "fit_split": self.fit_split,
            "train_sample_ids": self.train_sample_ids.detach().clone(),
            "teacher_residual_mse_before": self.teacher_residual_mse_before,
            "teacher_residual_mse_after": self.teacher_residual_mse_after,
            "teacher_residual_r2": self.teacher_residual_r2,
            "calibration_base_mse": self.calibration_base_mse,
            "calibration_best_mse": self.calibration_best_mse,
            "calibration_best_gain": self.calibration_best_gain,
            "calibration_best_alpha": self.calibration_best_alpha,
            "calibration_selected_alpha": self.calibration_selected_alpha,
            "calibration_alpha_abs_mean": self.calibration_alpha_abs_mean,
            "calibration_alpha_nonzero_fraction": self.calibration_alpha_nonzero_fraction,
            "calibration_min_output_gain": self.calibration_min_output_gain,
            "calibration_mean_output_gain": self.calibration_mean_output_gain,
            "calibration_sign_agreement": self.calibration_sign_agreement,
            "calibration_residual_correction_cosine": self.calibration_residual_correction_cosine,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyResidualTeacherV10":
        self.drift_dim = int(state["drift_dim"])
        self.output_shape = tuple(int(dim) for dim in state["output_shape"])
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        self.inner_fit_fraction = float(state.get("inner_fit_fraction", self.inner_fit_fraction))
        self.shrinkage_grid = torch.as_tensor(state.get("shrinkage_grid", self.shrinkage_grid)).detach().float()
        self.min_calibration_gain = float(state.get("min_calibration_gain", self.min_calibration_gain))
        self.fallback_alpha = float(state.get("fallback_alpha", self.fallback_alpha))
        self.alpha_mode = str(state.get("alpha_mode", self.alpha_mode))
        self.allow_negative_alpha = bool(state.get("allow_negative_alpha", self.allow_negative_alpha))
        self.calibration_mae_weight = float(state.get("calibration_mae_weight", self.calibration_mae_weight))
        self.correction_weight = torch.as_tensor(state["correction_weight"]).detach().float()
        self.shrinkage_alpha = float(state.get("shrinkage_alpha", 0.0))
        self.shrinkage_alpha_by_output = torch.as_tensor(
            state.get("shrinkage_alpha_by_output", torch.full((self.output_dim,), self.shrinkage_alpha))
        ).detach().float()
        self.selected_alpha_source = str(state.get("selected_alpha_source", "loaded"))
        self.fit_sample_count = int(state["fit_sample_count"])
        self.inner_fit_sample_count = int(state.get("inner_fit_sample_count", 0))
        self.inner_calib_sample_count = int(state.get("inner_calib_sample_count", 0))
        self.fit_split = str(state.get("fit_split", "train"))
        self.train_sample_ids = torch.as_tensor(state.get("train_sample_ids", torch.arange(self.fit_sample_count))).long()
        self.teacher_residual_mse_before = float(state.get("teacher_residual_mse_before", float("nan")))
        self.teacher_residual_mse_after = float(state.get("teacher_residual_mse_after", float("nan")))
        self.teacher_residual_r2 = float(state.get("teacher_residual_r2", float("nan")))
        self.calibration_base_mse = float(state.get("calibration_base_mse", float("nan")))
        self.calibration_best_mse = float(state.get("calibration_best_mse", float("nan")))
        self.calibration_best_gain = float(state.get("calibration_best_gain", float("nan")))
        self.calibration_best_alpha = float(state.get("calibration_best_alpha", self.shrinkage_alpha))
        self.calibration_selected_alpha = float(state.get("calibration_selected_alpha", self.shrinkage_alpha))
        self.calibration_alpha_abs_mean = float(state.get("calibration_alpha_abs_mean", abs(self.shrinkage_alpha)))
        self.calibration_alpha_nonzero_fraction = float(
            state.get("calibration_alpha_nonzero_fraction", float(abs(self.shrinkage_alpha) > self.eps))
        )
        self.calibration_min_output_gain = float(state.get("calibration_min_output_gain", float("nan")))
        self.calibration_mean_output_gain = float(state.get("calibration_mean_output_gain", float("nan")))
        self.calibration_sign_agreement = float(state.get("calibration_sign_agreement", float("nan")))
        self.calibration_residual_correction_cosine = float(
            state.get("calibration_residual_correction_cosine", float("nan"))
        )
        if self.correction_weight.shape != (self.feature_dim, self.output_dim):
            raise ValueError("loaded V10 residual teacher weight has incompatible shape")
        if self.shrinkage_alpha_by_output.numel() != self.output_dim:
            raise ValueError("loaded V10 residual teacher alpha vector has incompatible shape")
        return self

    def state_hash(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_stage3_v10_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyResidualTeacherV10,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V10 residual teacher checkpoint: {path}")
    metadata = dict(metadata or {})
    metadata.setdefault("protocol_version", TrainOnlyResidualTeacherV10.protocol_version)
    metadata.setdefault("fit_sample_count", teacher.fit_sample_count)
    metadata.setdefault("fit_split", teacher.fit_split)
    metadata.setdefault("teacher_shrinkage_alpha", teacher.shrinkage_alpha)
    torch.save(
        {
            "teacher_class": teacher.__class__.__name__,
            "teacher_state_dict": teacher.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_stage3_v10_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyResidualTeacherV10, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint["teacher_state_dict"]
    teacher = TrainOnlyResidualTeacherV10(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        ridge=float(state.get("ridge", 1e-3)),
        eps=float(state.get("eps", 1e-8)),
        inner_fit_fraction=float(state.get("inner_fit_fraction", 0.75)),
        shrinkage_grid=torch.as_tensor(state.get("shrinkage_grid", [0.0])).flatten().tolist(),
        min_calibration_gain=float(state.get("min_calibration_gain", 0.0)),
        fallback_alpha=float(state.get("fallback_alpha", 0.0)),
        alpha_mode=str(state.get("alpha_mode", "scalar_nonnegative")),
        allow_negative_alpha=bool(state.get("allow_negative_alpha", False)),
        calibration_mae_weight=float(state.get("calibration_mae_weight", 0.0)),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V10 residual teacher drift_dim does not match expected drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V10 residual teacher output_shape does not match expected output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyResidualTeacherV10.protocol_version)
    return teacher, metadata
