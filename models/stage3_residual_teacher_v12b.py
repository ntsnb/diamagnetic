from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


class TrainOnlyChronologicalOOFResidualTeacherV12b:
    """Chronological out-of-fold residual teacher for Stage III V12b.

    V12b does not inherit the V11/V12 wrapper teacher. It selects shrinkage
    from purged forward-chaining out-of-fold corrections, supports
    per-output/per-horizon abstention, preserves exact zero-drift identity, and
    records pre-gate alpha/block diagnostics before any fail-closed decision.
    """

    protocol_version = "v12b_chronological_oof_residual_teacher"

    def __init__(
        self,
        *,
        drift_dim: int,
        output_shape: tuple[int, ...],
        ridge_grid: Iterable[float] | None = None,
        alpha_grid: Iterable[float] | None = None,
        eps: float = 1e-8,
        fold_count: int = 5,
        min_prefix_samples: int = 32,
        purge_gap: int = 0,
        min_block_mse_gain: float = 0.0,
        min_block_mae_gain: float = 0.0,
        min_block_sign_agreement: float = 0.5,
        min_block_pass_fraction: float = 0.6,
        min_output_gain: float = 0.0,
        calibration_mae_weight: float = 0.0,
        allow_negative_alpha: bool = True,
        reliability_gate_enabled: bool = True,
        reliability_gate_quantile: float = 0.95,
        min_reliable_fraction: float = 0.1,
        disable_if_unreliable: bool = False,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if not output_shape or any(int(dim) <= 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if fold_count < 1:
            raise ValueError("fold_count must be positive")
        if min_prefix_samples < 2:
            raise ValueError("min_prefix_samples must be at least 2")
        if purge_gap < 0:
            raise ValueError("purge_gap must be non-negative")
        if not 0.0 <= min_block_sign_agreement <= 1.0:
            raise ValueError("min_block_sign_agreement must be in [0, 1]")
        if not 0.0 <= min_block_pass_fraction <= 1.0:
            raise ValueError("min_block_pass_fraction must be in [0, 1]")
        if calibration_mae_weight < 0:
            raise ValueError("calibration_mae_weight must be non-negative")
        if not 0.0 < reliability_gate_quantile <= 1.0:
            raise ValueError("reliability_gate_quantile must be in (0, 1]")
        if not 0.0 <= min_reliable_fraction <= 1.0:
            raise ValueError("min_reliable_fraction must be in [0, 1]")

        ridges = [float(value) for value in (ridge_grid or (1e-5, 1e-4, 1e-3, 1e-2))]
        if not ridges or any(value < 0 for value in ridges):
            raise ValueError("ridge_grid must contain non-negative values")
        alphas = [float(value) for value in (alpha_grid or (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0))]
        if allow_negative_alpha:
            alphas.extend([-abs(value) for value in alphas if value != 0.0])
        if 0.0 not in alphas:
            alphas.append(0.0)

        self.drift_dim = int(drift_dim)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge_grid = torch.tensor(sorted(set(ridges)), dtype=torch.float32)
        self.alpha_grid = torch.tensor(sorted(set(alphas)), dtype=torch.float32)
        self.eps = float(eps)
        self.fold_count = int(fold_count)
        self.min_prefix_samples = int(min_prefix_samples)
        self.purge_gap = int(purge_gap)
        self.min_block_mse_gain = float(min_block_mse_gain)
        self.min_block_mae_gain = float(min_block_mae_gain)
        self.min_block_sign_agreement = float(min_block_sign_agreement)
        self.min_block_pass_fraction = float(min_block_pass_fraction)
        self.min_output_gain = float(min_output_gain)
        self.calibration_mae_weight = float(calibration_mae_weight)
        self.allow_negative_alpha = bool(allow_negative_alpha)
        self.reliability_gate_enabled = bool(reliability_gate_enabled)
        self.reliability_gate_quantile = float(reliability_gate_quantile)
        self.min_reliable_fraction = float(min_reliable_fraction)
        self.disable_if_unreliable = bool(disable_if_unreliable)

        self.correction_weight: torch.Tensor | None = None
        self.alpha_by_output = torch.zeros(self.output_dim, dtype=torch.float32)
        self.selected_ridge = float(self.ridge_grid[0].item())
        self.selected_alpha_source = "not_fit"
        self.fit_split = ""
        self.fit_sample_count = 0
        self.oof_covered_sample_count = 0
        self.oof_fold_count = 0
        self.train_sample_ids = torch.empty(0, dtype=torch.long)
        self.teacher_residual_mse_before = float("nan")
        self.teacher_residual_mse_after = float("nan")
        self.teacher_residual_mse_gain = float("nan")
        self.teacher_residual_r2 = float("nan")
        self.oof_base_mse = float("nan")
        self.oof_selected_mse = float("nan")
        self.oof_mse_gain = float("nan")
        self.oof_base_mae = float("nan")
        self.oof_selected_mae = float("nan")
        self.oof_mae_gain = float("nan")
        self.oof_sign_agreement = float("nan")
        self.oof_correction_cosine = float("nan")
        self.output_pass_count = 0
        self.output_count = self.output_dim
        self.best_alpha_abs_mean = 0.0
        self.selected_alpha_abs_mean = 0.0
        self.selected_alpha_nonzero_fraction = 0.0
        self.worst_block_mse_gain = float("nan")
        self.worst_block_mae_gain = float("nan")
        self.worst_block_sign_agreement = float("nan")
        self.block_pass_count = 0
        self.effective_block_count = 0
        self.reliability_gate_active = False
        self.reliability_gate_reason = "not_fit"
        self.reliable_fraction = 0.0
        self.reliable_drift_norm_min = 0.0
        self.reliable_drift_norm_max = float("inf")
        self.reliable_correction_norm_max = float("inf")
        self._alpha_block_rows: list[dict[str, int | float | str]] = []
        self._ridge_rows: list[dict[str, int | float | str]] = []

    @property
    def feature_dim(self) -> int:
        return self.drift_dim * 3 + 1

    @property
    def shrinkage_alpha(self) -> float:
        return float(self.alpha_by_output.mean().item()) if self.alpha_by_output.numel() else 0.0

    def _as_2d_drift(self, drift: torch.Tensor) -> torch.Tensor:
        if drift.ndim == 1:
            drift = drift.unsqueeze(0)
        if drift.ndim != 2 or drift.shape[1] != self.drift_dim:
            raise ValueError(f"drift must have shape [N, {self.drift_dim}]")
        return drift.detach().float()

    def _feature_map(self, drift: torch.Tensor) -> torch.Tensor:
        drift = self._as_2d_drift(drift)
        norm = torch.linalg.vector_norm(drift, dim=1, keepdim=True)
        return torch.cat([drift, drift.abs(), drift.pow(2), norm], dim=1)

    def _solve_weight(self, features: torch.Tensor, residual: torch.Tensor, ridge: float) -> torch.Tensor:
        if not torch.isfinite(features).all() or not torch.isfinite(residual).all():
            raise ValueError("V12b teacher received non-finite features or residuals")
        eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        system = features.T @ features + float(ridge) * eye
        rhs = features.T @ residual
        try:
            return torch.linalg.solve(system, rhs)
        except torch.linalg.LinAlgError:
            for jitter in (1e-6, 1e-4, 1e-2):
                try:
                    return torch.linalg.solve(system + float(jitter) * eye, rhs)
                except torch.linalg.LinAlgError:
                    continue
            return torch.linalg.pinv(system) @ rhs

    def _fold_slices(self, sample_count: int) -> list[tuple[int, slice, slice]]:
        start = min(max(self.min_prefix_samples, 2), sample_count)
        remaining = sample_count - start
        if remaining <= 0:
            return []
        block_count = min(self.fold_count, remaining)
        base = remaining // block_count
        rem = remaining % block_count
        folds: list[tuple[int, slice, slice]] = []
        cursor = start
        for fold_idx in range(block_count):
            size = base + (1 if fold_idx < rem else 0)
            stop = cursor + size
            train_stop = max(0, cursor - self.purge_gap)
            if train_stop >= 2 and stop > cursor:
                folds.append((fold_idx, slice(0, train_stop), slice(cursor, stop)))
            cursor = stop
        return folds

    def _oof_prediction_for_ridge(
        self,
        features: torch.Tensor,
        residual: torch.Tensor,
        ridge: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = torch.zeros_like(residual)
        covered = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        fold_ids = torch.full((features.shape[0],), -1, dtype=torch.long, device=features.device)
        folds = self._fold_slices(features.shape[0])
        for fold_idx, train_slice, calib_slice in folds:
            weight = self._solve_weight(features[train_slice], residual[train_slice], ridge)
            pred[calib_slice] = features[calib_slice] @ weight
            covered[calib_slice] = True
            fold_ids[calib_slice] = int(fold_idx)
        return pred, covered, fold_ids

    def _metric_values(self, residual: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float, float]:
        base_mse = torch.mean(residual.pow(2)).clamp_min(self.eps)
        corr_mse = torch.mean((residual - prediction).pow(2))
        base_mae = torch.mean(residual.abs()).clamp_min(self.eps)
        corr_mae = torch.mean((residual - prediction).abs())
        mse_gain = float(((base_mse - corr_mse) / base_mse).item())
        mae_gain = float(((base_mae - corr_mae) / base_mae).item())
        sign_mask = residual.abs() > self.eps
        if sign_mask.any() and prediction.abs().max().item() > self.eps:
            sign = torch.eq(torch.sign(residual[sign_mask]), torch.sign(prediction[sign_mask]))
            sign_score = float(sign.float().mean().item())
        else:
            sign_score = 0.0
        return mse_gain, mae_gain, sign_score

    def _block_metrics_by_fold(
        self,
        *,
        residual_j: torch.Tensor,
        pred_j: torch.Tensor,
        fold_ids: torch.Tensor,
        output_idx: int,
        alpha: float,
        ridge: float,
    ) -> tuple[float, float, float, int, int, list[dict[str, int | float | str]]]:
        rows: list[dict[str, int | float | str]] = []
        mse_values: list[float] = []
        mae_values: list[float] = []
        sign_values: list[float] = []
        pass_count = 0
        unique_folds = [int(v) for v in torch.unique(fold_ids).detach().cpu().tolist() if int(v) >= 0]
        for fold_idx in unique_folds:
            mask = fold_ids == fold_idx
            residual_b = residual_j[mask]
            prediction_b = pred_j[mask] * float(alpha)
            mse_gain, mae_gain, sign_score = self._metric_values(residual_b, prediction_b)
            passed = (
                mse_gain >= self.min_block_mse_gain
                and mae_gain >= self.min_block_mae_gain
                and sign_score >= self.min_block_sign_agreement
            )
            pass_count += int(passed)
            mse_values.append(mse_gain)
            mae_values.append(mae_gain)
            sign_values.append(sign_score)
            rows.append(
                {
                    "ridge": float(ridge),
                    "output_idx": int(output_idx),
                    "alpha": float(alpha),
                    "fold_idx": int(fold_idx),
                    "sample_count": int(mask.sum().item()),
                    "mse_gain": mse_gain,
                    "mae_gain": mae_gain,
                    "sign_agreement": sign_score,
                    "block_pass": str(bool(passed)),
                }
            )
        if not unique_folds:
            return 0.0, 0.0, 0.0, 0, 0, rows
        return min(mse_values), min(mae_values), min(sign_values), pass_count, len(unique_folds), rows

    def _select_alpha_for_oof(
        self,
        *,
        residual: torch.Tensor,
        prediction: torch.Tensor,
        fold_ids: torch.Tensor,
        ridge: float,
    ) -> tuple[torch.Tensor, dict[str, float | int | str], list[dict[str, int | float | str]]]:
        alpha_grid = self.alpha_grid.to(device=residual.device, dtype=residual.dtype)
        selected = torch.zeros(self.output_dim, device=residual.device, dtype=residual.dtype)
        best = torch.zeros_like(selected)
        output_gains = torch.zeros_like(selected)
        output_pass_count = 0
        all_rows: list[dict[str, int | float | str]] = []
        worst_mse_values: list[float] = []
        worst_mae_values: list[float] = []
        worst_sign_values: list[float] = []
        total_block_pass = 0
        total_blocks = 0

        for output_idx in range(self.output_dim):
            residual_j = residual[:, output_idx]
            pred_j = prediction[:, output_idx]
            best_score = -float("inf")
            best_alpha = 0.0
            best_metrics = (0.0, 0.0, 0.0, 0, 0)
            for alpha_t in alpha_grid:
                alpha = float(alpha_t.item())
                worst_mse, worst_mae, worst_sign, pass_count, block_count, rows = self._block_metrics_by_fold(
                    residual_j=residual_j,
                    pred_j=pred_j,
                    fold_ids=fold_ids,
                    output_idx=output_idx,
                    alpha=alpha,
                    ridge=ridge,
                )
                all_rows.extend(rows)
                pred_alpha = pred_j * alpha
                mean_mse_gain, mean_mae_gain, mean_sign = self._metric_values(
                    residual_j.unsqueeze(1),
                    pred_alpha.unsqueeze(1),
                )
                score = (
                    worst_mse
                    + self.calibration_mae_weight * worst_mae
                    + 0.01 * worst_sign
                    + 0.25 * mean_mse_gain
                    + 0.05 * mean_mae_gain
                    + 0.005 * mean_sign
                )
                if score > best_score + self.eps or (
                    abs(score - best_score) <= self.eps and abs(alpha) < abs(best_alpha)
                ):
                    best_score = score
                    best_alpha = alpha
                    best_metrics = (worst_mse, worst_mae, worst_sign, pass_count, block_count)

            best[output_idx] = float(best_alpha)
            best_prediction = pred_j * float(best_alpha)
            output_gain, _, _ = self._metric_values(residual_j.unsqueeze(1), best_prediction.unsqueeze(1))
            output_gains[output_idx] = float(output_gain)
            worst_mse, worst_mae, worst_sign, pass_count, block_count = best_metrics
            total_block_pass += pass_count
            total_blocks += block_count
            worst_mse_values.append(worst_mse)
            worst_mae_values.append(worst_mae)
            worst_sign_values.append(worst_sign)
            required_passes = int(torch.ceil(torch.tensor(self.min_block_pass_fraction * block_count)).item())
            output_passed = (
                block_count > 0
                and pass_count >= required_passes
                and output_gain >= self.min_output_gain
                and worst_mse >= self.min_block_mse_gain
                and worst_mae >= self.min_block_mae_gain
                and worst_sign >= self.min_block_sign_agreement
            )
            if output_passed:
                selected[output_idx] = float(best_alpha)
                output_pass_count += 1

        selected_prediction = prediction * selected.unsqueeze(0)
        best_prediction = prediction * best.unsqueeze(0)
        base_mse = torch.mean(residual.pow(2)).clamp_min(self.eps)
        selected_mse = torch.mean((residual - selected_prediction).pow(2))
        best_mse = torch.mean((residual - best_prediction).pow(2))
        base_mae = torch.mean(residual.abs()).clamp_min(self.eps)
        selected_mae = torch.mean((residual - selected_prediction).abs())
        sign_mask = residual.abs() > self.eps
        if sign_mask.any() and selected_prediction.abs().max().item() > self.eps:
            sign = torch.eq(torch.sign(residual[sign_mask]), torch.sign(selected_prediction[sign_mask]))
            sign_score = float(sign.float().mean().item())
        else:
            sign_score = 0.0
        summary: dict[str, float | int | str] = {
            "ridge": float(ridge),
            "oof_base_mse": float(base_mse.item()),
            "oof_selected_mse": float(selected_mse.item()),
            "oof_best_mse": float(best_mse.item()),
            "oof_mse_gain": float(((base_mse - selected_mse) / base_mse).item()),
            "oof_best_mse_gain": float(((base_mse - best_mse) / base_mse).item()),
            "oof_base_mae": float(base_mae.item()),
            "oof_selected_mae": float(selected_mae.item()),
            "oof_mae_gain": float(((base_mae - selected_mae) / base_mae).item()),
            "oof_sign_agreement": sign_score,
            "output_pass_count": int(output_pass_count),
            "output_count": int(self.output_dim),
            "selected_alpha_abs_mean": float(selected.abs().mean().item()),
            "selected_alpha_nonzero_fraction": float((selected.abs() > self.eps).float().mean().item()),
            "best_alpha_abs_mean": float(best.abs().mean().item()),
            "worst_block_mse_gain": float(min(worst_mse_values) if worst_mse_values else 0.0),
            "worst_block_mae_gain": float(min(worst_mae_values) if worst_mae_values else 0.0),
            "worst_block_sign_agreement": float(min(worst_sign_values) if worst_sign_values else 0.0),
            "block_pass_count": int(total_block_pass),
            "effective_block_count": int(total_blocks),
        }
        return selected, summary, all_rows

    def fit(
        self,
        *,
        drift_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
        sample_ids: torch.Tensor | None = None,
    ) -> "TrainOnlyChronologicalOOFResidualTeacherV12b":
        if split != "train":
            raise ValueError("TrainOnlyChronologicalOOFResidualTeacherV12b can only be fit on the train split")
        drift = self._as_2d_drift(drift_train)
        y_base = y_base_train.detach().float().reshape(drift.shape[0], -1)
        y = y_train.detach().float().reshape(drift.shape[0], -1)
        if y.shape != y_base.shape:
            raise ValueError("y_base_train and y_train must have the same flattened shape")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"teacher output dim {y.shape[1]} does not match expected {self.output_dim}")
        if drift.shape[0] != y.shape[0]:
            raise ValueError("drift_train and y_train must share sample count")

        features = self._feature_map(drift)
        residual = y - y_base
        best_summary: dict[str, float | int | str] | None = None
        best_alpha: torch.Tensor | None = None
        best_rows: list[dict[str, int | float | str]] = []
        best_score = -float("inf")
        best_covered = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        best_fold_ids = torch.full((features.shape[0],), -1, dtype=torch.long, device=features.device)
        best_oof_prediction = torch.zeros_like(residual)

        self._ridge_rows = []
        for ridge in self.ridge_grid.tolist():
            oof_pred, covered, fold_ids = self._oof_prediction_for_ridge(features, residual, float(ridge))
            if not covered.any():
                continue
            residual_oof = residual[covered]
            pred_oof = oof_pred[covered]
            fold_oof = fold_ids[covered]
            alpha, summary, rows = self._select_alpha_for_oof(
                residual=residual_oof,
                prediction=pred_oof,
                fold_ids=fold_oof,
                ridge=float(ridge),
            )
            self._ridge_rows.append(summary)
            score = (
                float(summary["oof_mse_gain"])
                + self.calibration_mae_weight * float(summary["oof_mae_gain"])
                + 0.01 * float(summary["oof_sign_agreement"])
                + 0.05 * float(summary["selected_alpha_nonzero_fraction"])
            )
            if score > best_score + self.eps:
                best_score = score
                best_summary = summary
                best_alpha = alpha
                best_rows = rows
                best_covered = covered
                best_fold_ids = fold_ids
                best_oof_prediction = oof_pred

        if best_summary is None or best_alpha is None:
            self.selected_alpha_source = "fallback_no_oof_folds"
            self.alpha_by_output = torch.zeros(self.output_dim, dtype=torch.float32)
            self.selected_ridge = float(self.ridge_grid[0].item())
            self._alpha_block_rows = []
            best_covered = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            best_fold_ids = torch.full((features.shape[0],), -1, dtype=torch.long, device=features.device)
        else:
            self.alpha_by_output = best_alpha.detach().float().cpu()
            self.selected_ridge = float(best_summary["ridge"])
            self.selected_alpha_source = (
                "oof_per_output_grid" if bool((self.alpha_by_output.abs() > self.eps).any()) else "fallback_oof_gate"
            )
            self._alpha_block_rows = best_rows

        solution = self._solve_weight(features, residual, self.selected_ridge)
        alpha_device = self.alpha_by_output.to(device=features.device, dtype=features.dtype)
        correction = (features @ solution) * alpha_device.unsqueeze(0)
        before_mse = torch.mean(residual.pow(2)).clamp_min(self.eps)
        after_mse = torch.mean((residual - correction).pow(2))
        centered_total = torch.mean((residual - residual.mean(dim=0, keepdim=True)).pow(2)).clamp_min(self.eps)

        self.correction_weight = solution.detach().float()
        self.fit_split = "train"
        self.fit_sample_count = int(drift.shape[0])
        self.oof_covered_sample_count = int(best_covered.sum().item())
        self.oof_fold_count = len([v for v in torch.unique(best_fold_ids).detach().cpu().tolist() if int(v) >= 0])
        if sample_ids is None:
            self.train_sample_ids = torch.arange(self.fit_sample_count, dtype=torch.long)
        else:
            sample_ids = sample_ids.detach().cpu().long().flatten()
            if sample_ids.numel() != self.fit_sample_count:
                raise ValueError("sample_ids must match fit sample count")
            self.train_sample_ids = sample_ids
        self.teacher_residual_mse_before = float(before_mse.item())
        self.teacher_residual_mse_after = float(after_mse.item())
        self.teacher_residual_mse_gain = float(((before_mse - after_mse) / before_mse).item())
        self.teacher_residual_r2 = float(torch.clamp(1.0 - after_mse / centered_total, min=-1.0, max=1.0).item())

        if best_summary is not None and best_covered.any():
            self.oof_base_mse = float(best_summary["oof_base_mse"])
            self.oof_selected_mse = float(best_summary["oof_selected_mse"])
            self.oof_mse_gain = float(best_summary["oof_mse_gain"])
            self.oof_base_mae = float(best_summary["oof_base_mae"])
            self.oof_selected_mae = float(best_summary["oof_selected_mae"])
            self.oof_mae_gain = float(best_summary["oof_mae_gain"])
            self.oof_sign_agreement = float(best_summary["oof_sign_agreement"])
            self.output_pass_count = int(best_summary["output_pass_count"])
            self.output_count = int(best_summary["output_count"])
            self.best_alpha_abs_mean = float(best_summary["best_alpha_abs_mean"])
            self.selected_alpha_abs_mean = float(best_summary["selected_alpha_abs_mean"])
            self.selected_alpha_nonzero_fraction = float(best_summary["selected_alpha_nonzero_fraction"])
            self.worst_block_mse_gain = float(best_summary["worst_block_mse_gain"])
            self.worst_block_mae_gain = float(best_summary["worst_block_mae_gain"])
            self.worst_block_sign_agreement = float(best_summary["worst_block_sign_agreement"])
            self.block_pass_count = int(best_summary["block_pass_count"])
            self.effective_block_count = int(best_summary["effective_block_count"])
            selected_oof = best_oof_prediction[best_covered] * alpha_device.unsqueeze(0)
            self.oof_correction_cosine = self._mean_cosine(residual[best_covered], selected_oof)
            self._calibrate_reliability_gate(drift=drift[best_covered], residual=residual[best_covered], correction=selected_oof)
        else:
            self.oof_base_mse = float("nan")
            self.oof_selected_mse = float("nan")
            self.oof_mse_gain = 0.0
            self.oof_base_mae = float("nan")
            self.oof_selected_mae = float("nan")
            self.oof_mae_gain = 0.0
            self.oof_sign_agreement = 0.0
            self.oof_correction_cosine = float("nan")
            self.output_pass_count = 0
            self.block_pass_count = 0
            self.effective_block_count = 0
            self.reliability_gate_active = False
            self.reliability_gate_reason = "no_oof_prediction"
        return self

    def update(
        self,
        *,
        drift: torch.Tensor,
        y_base: torch.Tensor,
        y: torch.Tensor,
        split: str,
    ) -> "TrainOnlyChronologicalOOFResidualTeacherV12b":
        if split != "train":
            raise ValueError("TrainOnlyChronologicalOOFResidualTeacherV12b update is restricted to train")
        return self.fit(drift_train=drift, y_base_train=y_base, y_train=y, split=split)

    def _mean_cosine(self, residual: torch.Tensor, correction: torch.Tensor) -> float:
        if residual.numel() == 0 or correction.abs().max().item() <= self.eps:
            return float("nan")
        return float(
            F.cosine_similarity(
                residual.reshape(residual.shape[0], -1),
                correction.reshape(correction.shape[0], -1),
                dim=1,
                eps=self.eps,
            )
            .mean()
            .item()
        )

    def _calibrate_reliability_gate(
        self,
        *,
        drift: torch.Tensor,
        residual: torch.Tensor,
        correction: torch.Tensor,
    ) -> None:
        if not self.reliability_gate_enabled or correction.abs().max().item() <= self.eps:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "disabled_or_zero_prediction"
            return
        drift_norm = torch.linalg.vector_norm(drift, dim=1)
        correction_norm = torch.linalg.vector_norm(correction.reshape(correction.shape[0], -1), dim=1)
        base_mse = torch.mean(residual.reshape(residual.shape[0], -1).pow(2), dim=1)
        corr_mse = torch.mean((residual - correction).reshape(residual.shape[0], -1).pow(2), dim=1)
        base_mae = torch.mean(residual.reshape(residual.shape[0], -1).abs(), dim=1)
        corr_mae = torch.mean((residual - correction).reshape(residual.shape[0], -1).abs(), dim=1)
        reliable = (corr_mse <= base_mse) & (corr_mae <= base_mae) & (correction_norm > self.eps)
        self.reliable_fraction = float(reliable.float().mean().item()) if reliable.numel() else 0.0
        if self.reliable_fraction < self.min_reliable_fraction:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "insufficient_reliable_oof_samples"
            if self.disable_if_unreliable:
                self.alpha_by_output = torch.zeros_like(self.alpha_by_output)
                self.selected_alpha_source = "fallback_unreliable_oof"
            return
        reliable_drift = drift_norm[reliable]
        reliable_correction = correction_norm[reliable]
        self.reliable_drift_norm_min = float(torch.quantile(reliable_drift, 1.0 - self.reliability_gate_quantile).item())
        self.reliable_drift_norm_max = float(torch.quantile(reliable_drift, self.reliability_gate_quantile).item())
        self.reliable_correction_norm_max = float(
            torch.quantile(reliable_correction, self.reliability_gate_quantile).item()
        )
        self.reliability_gate_active = True
        self.reliability_gate_reason = "oof_norm_quantile_gate"

    def _reliability_gate(self, drift: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        if not self.reliability_gate_active:
            return torch.ones(drift.shape[0], device=drift.device, dtype=drift.dtype)
        drift_norm = torch.linalg.vector_norm(drift, dim=1)
        correction_norm = torch.linalg.vector_norm(correction.reshape(correction.shape[0], -1), dim=1)
        gate = (
            (drift_norm >= self.reliable_drift_norm_min)
            & (drift_norm <= self.reliable_drift_norm_max)
            & (correction_norm <= self.reliable_correction_norm_max)
        )
        zero_mask = drift_norm <= self.eps
        gate = torch.where(zero_mask, torch.ones_like(gate), gate)
        return gate.to(dtype=drift.dtype)

    def predict_correction(self, drift: torch.Tensor) -> torch.Tensor:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyChronologicalOOFResidualTeacherV12b must be fit before prediction")
        drift_2d = self._as_2d_drift(drift)
        features = self._feature_map(drift_2d)
        weight = self.correction_weight.to(device=features.device, dtype=features.dtype)
        alpha = self.alpha_by_output.to(device=features.device, dtype=features.dtype)
        correction = (features @ weight) * alpha.unsqueeze(0)
        correction = correction * self._reliability_gate(drift_2d, correction).unsqueeze(1)
        return correction.reshape(features.shape[0], *self.output_shape)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyChronologicalOOFResidualTeacherV12b must be fit before diagnostics")
        return {
            "teacher_method": self.protocol_version,
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
            "oof_covered_sample_count": self.oof_covered_sample_count,
            "oof_fold_count": self.oof_fold_count,
            "drift_dim": self.drift_dim,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "selected_ridge": self.selected_ridge,
            "teacher_alpha_source": self.selected_alpha_source,
            "teacher_shrinkage_alpha": self.shrinkage_alpha,
            "teacher_alpha_abs_mean": self.selected_alpha_abs_mean,
            "teacher_alpha_nonzero_fraction": self.selected_alpha_nonzero_fraction,
            "teacher_best_alpha_abs_mean": self.best_alpha_abs_mean,
            "teacher_residual_mse_before": self.teacher_residual_mse_before,
            "teacher_residual_mse_after": self.teacher_residual_mse_after,
            "teacher_residual_mse_gain": self.teacher_residual_mse_gain,
            "teacher_residual_r2": self.teacher_residual_r2,
            "teacher_oof_base_mse": self.oof_base_mse,
            "teacher_oof_selected_mse": self.oof_selected_mse,
            "teacher_oof_mse_gain": self.oof_mse_gain,
            "teacher_oof_base_mae": self.oof_base_mae,
            "teacher_oof_selected_mae": self.oof_selected_mae,
            "teacher_oof_mae_gain": self.oof_mae_gain,
            "teacher_oof_sign_agreement": self.oof_sign_agreement,
            "teacher_oof_correction_cosine": self.oof_correction_cosine,
            "teacher_output_pass_count": self.output_pass_count,
            "teacher_output_count": self.output_count,
            "teacher_min_block_mse_gain": self.min_block_mse_gain,
            "teacher_min_block_mae_gain": self.min_block_mae_gain,
            "teacher_min_block_sign_agreement": self.min_block_sign_agreement,
            "teacher_min_block_pass_fraction": self.min_block_pass_fraction,
            "teacher_worst_block_mse_gain": self.worst_block_mse_gain,
            "teacher_worst_block_mae_gain": self.worst_block_mae_gain,
            "teacher_worst_block_sign_agreement": self.worst_block_sign_agreement,
            "teacher_block_pass_count": self.block_pass_count,
            "teacher_effective_block_count": self.effective_block_count,
            "teacher_reliability_gate_enabled": str(self.reliability_gate_enabled),
            "teacher_reliability_gate_active": str(self.reliability_gate_active),
            "teacher_reliability_gate_reason": self.reliability_gate_reason,
            "teacher_reliable_fraction": self.reliable_fraction,
            "teacher_reliable_drift_norm_min": self.reliable_drift_norm_min,
            "teacher_reliable_drift_norm_max": self.reliable_drift_norm_max,
            "teacher_reliable_correction_norm_max": self.reliable_correction_norm_max,
        }

    def alpha_block_diagnostic_rows(self) -> list[dict[str, int | float | str]]:
        return list(self._alpha_block_rows)

    def ridge_diagnostic_rows(self) -> list[dict[str, int | float | str]]:
        return list(self._ridge_rows)

    def state_dict(self) -> dict[str, Any]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyChronologicalOOFResidualTeacherV12b must be fit before serialization")
        return {
            "class_name": self.__class__.__name__,
            "protocol_version": self.protocol_version,
            "drift_dim": self.drift_dim,
            "output_shape": self.output_shape,
            "ridge_grid": self.ridge_grid.detach().clone(),
            "alpha_grid": self.alpha_grid.detach().clone(),
            "eps": self.eps,
            "fold_count": self.fold_count,
            "min_prefix_samples": self.min_prefix_samples,
            "purge_gap": self.purge_gap,
            "min_block_mse_gain": self.min_block_mse_gain,
            "min_block_mae_gain": self.min_block_mae_gain,
            "min_block_sign_agreement": self.min_block_sign_agreement,
            "min_block_pass_fraction": self.min_block_pass_fraction,
            "min_output_gain": self.min_output_gain,
            "calibration_mae_weight": self.calibration_mae_weight,
            "allow_negative_alpha": self.allow_negative_alpha,
            "reliability_gate_enabled": self.reliability_gate_enabled,
            "reliability_gate_quantile": self.reliability_gate_quantile,
            "min_reliable_fraction": self.min_reliable_fraction,
            "disable_if_unreliable": self.disable_if_unreliable,
            "correction_weight": self.correction_weight.detach().clone(),
            "alpha_by_output": self.alpha_by_output.detach().clone(),
            "selected_ridge": self.selected_ridge,
            "selected_alpha_source": self.selected_alpha_source,
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
            "oof_covered_sample_count": self.oof_covered_sample_count,
            "oof_fold_count": self.oof_fold_count,
            "train_sample_ids": self.train_sample_ids.detach().clone(),
            "diagnostic_summary": self.diagnostic_summary(),
            "alpha_block_rows": list(self._alpha_block_rows),
            "ridge_rows": list(self._ridge_rows),
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyChronologicalOOFResidualTeacherV12b":
        self.drift_dim = int(state["drift_dim"])
        self.output_shape = tuple(int(dim) for dim in state["output_shape"])
        self.output_dim = int(torch.tensor(self.output_shape).prod().item())
        self.ridge_grid = torch.as_tensor(state.get("ridge_grid", self.ridge_grid)).detach().float()
        self.alpha_grid = torch.as_tensor(state.get("alpha_grid", self.alpha_grid)).detach().float()
        self.eps = float(state.get("eps", self.eps))
        self.fold_count = int(state.get("fold_count", self.fold_count))
        self.min_prefix_samples = int(state.get("min_prefix_samples", self.min_prefix_samples))
        self.purge_gap = int(state.get("purge_gap", self.purge_gap))
        self.min_block_mse_gain = float(state.get("min_block_mse_gain", self.min_block_mse_gain))
        self.min_block_mae_gain = float(state.get("min_block_mae_gain", self.min_block_mae_gain))
        self.min_block_sign_agreement = float(
            state.get("min_block_sign_agreement", self.min_block_sign_agreement)
        )
        self.min_block_pass_fraction = float(state.get("min_block_pass_fraction", self.min_block_pass_fraction))
        self.min_output_gain = float(state.get("min_output_gain", self.min_output_gain))
        self.calibration_mae_weight = float(state.get("calibration_mae_weight", self.calibration_mae_weight))
        self.allow_negative_alpha = bool(state.get("allow_negative_alpha", self.allow_negative_alpha))
        self.reliability_gate_enabled = bool(state.get("reliability_gate_enabled", self.reliability_gate_enabled))
        self.reliability_gate_quantile = float(
            state.get("reliability_gate_quantile", self.reliability_gate_quantile)
        )
        self.min_reliable_fraction = float(state.get("min_reliable_fraction", self.min_reliable_fraction))
        self.disable_if_unreliable = bool(state.get("disable_if_unreliable", self.disable_if_unreliable))
        self.correction_weight = torch.as_tensor(state["correction_weight"]).detach().float()
        self.alpha_by_output = torch.as_tensor(state["alpha_by_output"]).detach().float()
        self.selected_ridge = float(state.get("selected_ridge", self.selected_ridge))
        self.selected_alpha_source = str(state.get("selected_alpha_source", "loaded"))
        self.fit_split = str(state.get("fit_split", "train"))
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        self.oof_covered_sample_count = int(state.get("oof_covered_sample_count", 0))
        self.oof_fold_count = int(state.get("oof_fold_count", 0))
        self.train_sample_ids = torch.as_tensor(state.get("train_sample_ids", torch.arange(self.fit_sample_count))).long()
        summary = dict(state.get("diagnostic_summary", {}))
        self.teacher_residual_mse_before = float(summary.get("teacher_residual_mse_before", float("nan")))
        self.teacher_residual_mse_after = float(summary.get("teacher_residual_mse_after", float("nan")))
        self.teacher_residual_mse_gain = float(summary.get("teacher_residual_mse_gain", float("nan")))
        self.teacher_residual_r2 = float(summary.get("teacher_residual_r2", float("nan")))
        self.oof_base_mse = float(summary.get("teacher_oof_base_mse", float("nan")))
        self.oof_selected_mse = float(summary.get("teacher_oof_selected_mse", float("nan")))
        self.oof_mse_gain = float(summary.get("teacher_oof_mse_gain", float("nan")))
        self.oof_base_mae = float(summary.get("teacher_oof_base_mae", float("nan")))
        self.oof_selected_mae = float(summary.get("teacher_oof_selected_mae", float("nan")))
        self.oof_mae_gain = float(summary.get("teacher_oof_mae_gain", float("nan")))
        self.oof_sign_agreement = float(summary.get("teacher_oof_sign_agreement", float("nan")))
        self.oof_correction_cosine = float(summary.get("teacher_oof_correction_cosine", float("nan")))
        self.output_pass_count = int(summary.get("teacher_output_pass_count", 0))
        self.output_count = int(summary.get("teacher_output_count", self.output_dim))
        self.best_alpha_abs_mean = float(summary.get("teacher_best_alpha_abs_mean", 0.0))
        self.selected_alpha_abs_mean = float(summary.get("teacher_alpha_abs_mean", 0.0))
        self.selected_alpha_nonzero_fraction = float(summary.get("teacher_alpha_nonzero_fraction", 0.0))
        self.worst_block_mse_gain = float(summary.get("teacher_worst_block_mse_gain", float("nan")))
        self.worst_block_mae_gain = float(summary.get("teacher_worst_block_mae_gain", float("nan")))
        self.worst_block_sign_agreement = float(summary.get("teacher_worst_block_sign_agreement", float("nan")))
        self.block_pass_count = int(summary.get("teacher_block_pass_count", 0))
        self.effective_block_count = int(summary.get("teacher_effective_block_count", 0))
        self.reliability_gate_active = str(summary.get("teacher_reliability_gate_active", "False")) == "True"
        self.reliability_gate_reason = str(summary.get("teacher_reliability_gate_reason", "loaded"))
        self.reliable_fraction = float(summary.get("teacher_reliable_fraction", 0.0))
        self.reliable_drift_norm_min = float(summary.get("teacher_reliable_drift_norm_min", 0.0))
        self.reliable_drift_norm_max = float(summary.get("teacher_reliable_drift_norm_max", float("inf")))
        self.reliable_correction_norm_max = float(summary.get("teacher_reliable_correction_norm_max", float("inf")))
        self._alpha_block_rows = list(state.get("alpha_block_rows", []))
        self._ridge_rows = list(state.get("ridge_rows", []))
        if self.correction_weight.shape != (self.feature_dim, self.output_dim):
            raise ValueError("loaded V12b residual teacher weight has incompatible shape")
        if self.alpha_by_output.numel() != self.output_dim:
            raise ValueError("loaded V12b residual teacher alpha vector has incompatible shape")
        return self

    def state_hash(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_stage3_v12b_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyChronologicalOOFResidualTeacherV12b,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V12b residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyChronologicalOOFResidualTeacherV12b.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyChronologicalOOFResidualTeacherV12b.__name__)
    torch.save(
        {
            "class_name": TrainOnlyChronologicalOOFResidualTeacherV12b.__name__,
            "protocol_version": TrainOnlyChronologicalOOFResidualTeacherV12b.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v12b_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyChronologicalOOFResidualTeacherV12b, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12b(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
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
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V12b residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V12b residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyChronologicalOOFResidualTeacherV12b.protocol_version)
    metadata.setdefault("class_name", TrainOnlyChronologicalOOFResidualTeacherV12b.__name__)
    return teacher, metadata
