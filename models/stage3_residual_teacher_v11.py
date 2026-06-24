from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


class TrainOnlyRobustBlockResidualTeacherV11:
    """Train-only residual teacher with worst-block shrinkage calibration.

    V10 improved the harmful V9 teacher, but the public-3 matrix showed that
    aggregate calibration can still hide late-window or dataset-specific harm.
    V11 selects shrinkage only when every chronological calibration block
    satisfies MSE, MAE, and sign gates. It also adds a train-calibrated
    reliability gate based on drift/correction norms. The feature map has no
    intercept, preserving exact zero-drift identity.
    """

    protocol_version = "v11_block_calibrated_residual_teacher"

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
        calibration_block_count: int = 5,
        min_block_mse_gain: float = 0.0,
        min_block_mae_gain: float = 0.0,
        min_block_sign_agreement: float = 0.5,
        reliability_gate_enabled: bool = True,
        reliability_gate_quantile: float = 0.9,
        min_reliable_fraction: float = 0.1,
        disable_if_unreliable: bool = True,
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
        if alpha_mode not in {"scalar_nonnegative", "per_output", "block_scalar", "block_horizon"}:
            raise ValueError("alpha_mode must be scalar_nonnegative, per_output, block_scalar, or block_horizon")
        if calibration_mae_weight < 0:
            raise ValueError("calibration_mae_weight must be non-negative")
        if calibration_block_count < 1:
            raise ValueError("calibration_block_count must be positive")
        if not 0.0 <= min_block_sign_agreement <= 1.0:
            raise ValueError("min_block_sign_agreement must be in [0, 1]")
        if not 0.0 < reliability_gate_quantile <= 1.0:
            raise ValueError("reliability_gate_quantile must be in (0, 1]")
        if not 0.0 <= min_reliable_fraction <= 1.0:
            raise ValueError("min_reliable_fraction must be in [0, 1]")
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
        self.calibration_block_count = int(calibration_block_count)
        self.min_block_mse_gain = float(min_block_mse_gain)
        self.min_block_mae_gain = float(min_block_mae_gain)
        self.min_block_sign_agreement = float(min_block_sign_agreement)
        self.reliability_gate_enabled = bool(reliability_gate_enabled)
        self.reliability_gate_quantile = float(reliability_gate_quantile)
        self.min_reliable_fraction = float(min_reliable_fraction)
        self.disable_if_unreliable = bool(disable_if_unreliable)

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
        self.teacher_residual_mse_gain = float("nan")
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
        self.calibration_worst_block_mse_gain = float("nan")
        self.calibration_worst_block_mae_gain = float("nan")
        self.calibration_worst_block_sign_agreement = float("nan")
        self.calibration_block_pass_count = 0
        self.calibration_effective_block_count = 0
        self.reliability_gate_active = False
        self.reliability_gate_reason = "not_fit"
        self.reliable_fraction = 0.0
        self.reliable_drift_norm_min = 0.0
        self.reliable_drift_norm_max = float("inf")
        self.reliable_correction_norm_max = float("inf")

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
        if not torch.isfinite(features).all() or not torch.isfinite(residual).all():
            raise ValueError("V11 residual teacher received non-finite features or residuals")
        eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        system = features.T @ features + self.ridge * eye
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

        if self.alpha_mode in {"block_scalar", "block_horizon"}:
            self._calibrate_blockwise_alpha(
                features=features,
                residual=residual,
                inner_weight=inner_weight,
                fit_stop=fit_stop,
            )
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
        self._calibrate_reliability_gate(
            drift=self._last_fit_drift[fit_stop:] if hasattr(self, "_last_fit_drift") else features.new_zeros((calib_residual.shape[0], self.drift_dim)),
            residual=calib_residual,
            selected_prediction=selected_prediction,
        )

    def _block_slices(self, sample_count: int) -> list[slice]:
        block_count = min(self.calibration_block_count, max(1, sample_count))
        base = sample_count // block_count
        remainder = sample_count % block_count
        slices: list[slice] = []
        start = 0
        for block_idx in range(block_count):
            size = base + (1 if block_idx < remainder else 0)
            stop = start + size
            if stop > start:
                slices.append(slice(start, stop))
            start = stop
        return slices

    def _block_metrics(
        self,
        residual: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[float, float, float, int, int]:
        mse_gains: list[float] = []
        mae_gains: list[float] = []
        sign_scores: list[float] = []
        pass_count = 0
        blocks = self._block_slices(residual.shape[0])
        for block in blocks:
            residual_b = residual[block]
            pred_b = prediction[block]
            base_mse = torch.mean(residual_b.pow(2)).clamp_min(self.eps)
            corr_mse = torch.mean((residual_b - pred_b).pow(2))
            base_mae = torch.mean(residual_b.abs()).clamp_min(self.eps)
            corr_mae = torch.mean((residual_b - pred_b).abs())
            mse_gain = float(((base_mse - corr_mse) / base_mse).item())
            mae_gain = float(((base_mae - corr_mae) / base_mae).item())
            sign_mask = residual_b.abs() > self.eps
            if sign_mask.any() and pred_b.abs().max().item() > self.eps:
                sign_agree = torch.eq(torch.sign(residual_b[sign_mask]), torch.sign(pred_b[sign_mask]))
                sign_score = float(sign_agree.float().mean().item())
            else:
                sign_score = 0.0
            mse_gains.append(mse_gain)
            mae_gains.append(mae_gain)
            sign_scores.append(sign_score)
            pass_count += int(
                mse_gain >= self.min_block_mse_gain
                and mae_gain >= self.min_block_mae_gain
                and sign_score >= self.min_block_sign_agreement
            )
        if not blocks:
            return 0.0, 0.0, 0.0, 0, 0
        return min(mse_gains), min(mae_gains), min(sign_scores), pass_count, len(blocks)

    def _calibrate_blockwise_alpha(
        self,
        *,
        features: torch.Tensor,
        residual: torch.Tensor,
        inner_weight: torch.Tensor,
        fit_stop: int,
    ) -> None:
        calib_drift = self._last_fit_drift[fit_stop:] if hasattr(self, "_last_fit_drift") else None
        calib_residual = residual[fit_stop:]
        calib_prediction = features[fit_stop:] @ inner_weight
        base_mse = torch.mean(calib_residual.pow(2)).clamp_min(self.eps)
        grid = self.shrinkage_grid.to(device=calib_residual.device, dtype=calib_residual.dtype)

        if self.alpha_mode == "block_horizon":
            selected = torch.full((self.output_dim,), self.fallback_alpha, device=calib_residual.device)
            best = torch.zeros(self.output_dim, device=calib_residual.device)
            block_mse_values: list[float] = []
            block_mae_values: list[float] = []
            block_sign_values: list[float] = []
            selected_count = 0
            for output_idx in range(self.output_dim):
                residual_j = calib_residual[:, output_idx : output_idx + 1]
                pred_j = calib_prediction[:, output_idx : output_idx + 1]
                best_score = -float("inf")
                best_alpha = torch.tensor(0.0, device=calib_residual.device, dtype=calib_residual.dtype)
                best_metrics = (0.0, 0.0, 0.0, 0, 0)
                for alpha in grid:
                    prediction = pred_j * alpha
                    worst_mse, worst_mae, worst_sign, pass_count, block_count = self._block_metrics(residual_j, prediction)
                    score = worst_mse + self.calibration_mae_weight * worst_mae + 0.01 * worst_sign
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_metrics = (worst_mse, worst_mae, worst_sign, pass_count, block_count)
                best[output_idx] = best_alpha
                worst_mse, worst_mae, worst_sign, pass_count, block_count = best_metrics
                block_mse_values.append(worst_mse)
                block_mae_values.append(worst_mae)
                block_sign_values.append(worst_sign)
                if (
                    pass_count == block_count
                    and worst_mse >= self.min_block_mse_gain
                    and worst_mae >= self.min_block_mae_gain
                    and worst_sign >= self.min_block_sign_agreement
                ):
                    selected[output_idx] = best_alpha
                    selected_count += 1
            selected_prediction = calib_prediction * selected.unsqueeze(0)
            best_prediction = calib_prediction * best.unsqueeze(0)
            best_mse = torch.mean((calib_residual - best_prediction).pow(2))
            selected_mse = torch.mean((calib_residual - selected_prediction).pow(2))
            del selected_mse
            self.shrinkage_alpha_by_output = selected.detach().float().cpu()
            self.shrinkage_alpha = float(selected.mean().item())
            self.calibration_best_alpha = float(best.mean().item())
            self.calibration_selected_alpha = float(selected.mean().item())
            self.calibration_alpha_abs_mean = float(selected.abs().mean().item())
            self.calibration_alpha_nonzero_fraction = float((selected.abs() > self.eps).float().mean().item())
            self.calibration_worst_block_mse_gain = float(min(block_mse_values) if block_mse_values else 0.0)
            self.calibration_worst_block_mae_gain = float(min(block_mae_values) if block_mae_values else 0.0)
            self.calibration_worst_block_sign_agreement = float(min(block_sign_values) if block_sign_values else 0.0)
            self.calibration_block_pass_count = int(selected_count)
            self.calibration_effective_block_count = int(self.output_dim)
            self.selected_alpha_source = "block_horizon_grid" if selected_count else "fallback_block_gate"
        else:
            best_score = -float("inf")
            best_alpha = torch.tensor(0.0, device=calib_residual.device, dtype=calib_residual.dtype)
            best_metrics = (0.0, 0.0, 0.0, 0, 0)
            best_mse = base_mse
            for alpha in grid:
                prediction = calib_prediction * alpha
                worst_mse, worst_mae, worst_sign, pass_count, block_count = self._block_metrics(
                    calib_residual,
                    prediction,
                )
                mse = torch.mean((calib_residual - prediction).pow(2))
                score = worst_mse + self.calibration_mae_weight * worst_mae + 0.01 * worst_sign
                if score > best_score or (
                    abs(score - best_score) <= self.eps and abs(float(alpha.item())) < abs(float(best_alpha.item()))
                ):
                    best_score = score
                    best_alpha = alpha
                    best_metrics = (worst_mse, worst_mae, worst_sign, pass_count, block_count)
                    best_mse = mse
            worst_mse, worst_mae, worst_sign, pass_count, block_count = best_metrics
            calibration_passed = (
                pass_count == block_count
                and worst_mse >= self.min_block_mse_gain
                and worst_mae >= self.min_block_mae_gain
                and worst_sign >= self.min_block_sign_agreement
            )
            selected_alpha = best_alpha if calibration_passed else torch.tensor(
                self.fallback_alpha,
                device=calib_residual.device,
                dtype=calib_residual.dtype,
            )
            selected_prediction = selected_alpha * calib_prediction
            self.shrinkage_alpha = float(selected_alpha.item())
            self.shrinkage_alpha_by_output = torch.full((self.output_dim,), self.shrinkage_alpha, dtype=torch.float32)
            self.calibration_best_alpha = float(best_alpha.item())
            self.calibration_selected_alpha = float(selected_alpha.item())
            self.calibration_alpha_abs_mean = abs(float(selected_alpha.item()))
            self.calibration_alpha_nonzero_fraction = float(abs(float(selected_alpha.item())) > self.eps)
            self.calibration_worst_block_mse_gain = float(worst_mse)
            self.calibration_worst_block_mae_gain = float(worst_mae)
            self.calibration_worst_block_sign_agreement = float(worst_sign)
            self.calibration_block_pass_count = int(pass_count)
            self.calibration_effective_block_count = int(block_count)
            self.selected_alpha_source = "block_scalar_grid" if calibration_passed else "fallback_block_gate"

        self.calibration_base_mse = float(base_mse.item())
        self.calibration_best_mse = float(best_mse.item())
        self.calibration_best_gain = float(((base_mse - best_mse) / base_mse).item())
        self.calibration_min_output_gain = self.calibration_worst_block_mse_gain
        self.calibration_mean_output_gain = self.calibration_best_gain
        self._store_calibration_direction_metrics(calib_residual, selected_prediction)
        self._calibrate_reliability_gate(
            drift=calib_drift if calib_drift is not None else torch.zeros(calib_residual.shape[0], self.drift_dim),
            residual=calib_residual,
            selected_prediction=selected_prediction,
        )

    def _calibrate_reliability_gate(
        self,
        *,
        drift: torch.Tensor,
        residual: torch.Tensor,
        selected_prediction: torch.Tensor,
    ) -> None:
        if not self.reliability_gate_enabled or selected_prediction.abs().max().item() <= self.eps:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "disabled_or_zero_prediction"
            return
        drift = self._as_2d_drift(drift)
        drift_norm = torch.linalg.vector_norm(drift, dim=1)
        correction_norm = torch.linalg.vector_norm(selected_prediction.reshape(selected_prediction.shape[0], -1), dim=1)
        base_mse = torch.mean(residual.reshape(residual.shape[0], -1).pow(2), dim=1)
        corr_mse = torch.mean((residual - selected_prediction).reshape(residual.shape[0], -1).pow(2), dim=1)
        base_mae = torch.mean(residual.reshape(residual.shape[0], -1).abs(), dim=1)
        corr_mae = torch.mean((residual - selected_prediction).reshape(residual.shape[0], -1).abs(), dim=1)
        reliable = (corr_mse <= base_mse) & (corr_mae <= base_mae) & (correction_norm > self.eps)
        self.reliable_fraction = float(reliable.float().mean().item()) if reliable.numel() else 0.0
        if self.reliable_fraction < self.min_reliable_fraction:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "insufficient_reliable_calibration_samples"
            if self.disable_if_unreliable:
                self.shrinkage_alpha = 0.0
                self.shrinkage_alpha_by_output = torch.zeros(self.output_dim, dtype=torch.float32)
            return
        reliable_drift = drift_norm[reliable]
        reliable_correction = correction_norm[reliable]
        self.reliable_drift_norm_min = float(torch.quantile(reliable_drift, 1.0 - self.reliability_gate_quantile).item())
        self.reliable_drift_norm_max = float(torch.quantile(reliable_drift, self.reliability_gate_quantile).item())
        self.reliable_correction_norm_max = float(
            torch.quantile(reliable_correction, self.reliability_gate_quantile).item()
        )
        self.reliability_gate_active = True
        self.reliability_gate_reason = "norm_quantile_gate"

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
    ) -> "TrainOnlyRobustBlockResidualTeacherV11":
        if split != "train":
            raise ValueError("TrainOnlyRobustBlockResidualTeacherV11 can only be fit on the train split")
        drift = self._as_2d_drift(drift_train)
        self._last_fit_drift = drift.detach().float()
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
        self.teacher_residual_mse_gain = float(((before_mse - after_mse) / before_mse.clamp_min(self.eps)).item())
        self.teacher_residual_r2 = float(torch.clamp(1.0 - after_mse / total, min=-1.0, max=1.0).item())
        return self

    def update(
        self,
        *,
        drift: torch.Tensor,
        y_base: torch.Tensor,
        y: torch.Tensor,
        split: str,
    ) -> "TrainOnlyRobustBlockResidualTeacherV11":
        if split != "train":
            raise ValueError("TrainOnlyRobustBlockResidualTeacherV11 update is restricted to the train split")
        return self.fit(drift_train=drift, y_base_train=y_base, y_train=y, split=split)

    def predict_correction(self, drift: torch.Tensor) -> torch.Tensor:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyRobustBlockResidualTeacherV11 must be fit before predict_correction")
        drift_2d = self._as_2d_drift(drift)
        features = self._feature_map(drift_2d)
        weight = self.correction_weight.to(device=features.device, dtype=features.dtype)
        alpha = self._alpha_vector(device=features.device, dtype=features.dtype)
        correction = (features @ weight) * alpha.unsqueeze(0)
        correction = correction * self._reliability_gate(drift_2d, correction).unsqueeze(1)
        return correction.reshape(features.shape[0], *self.output_shape)

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

    def _alpha_vector(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.shrinkage_alpha_by_output is None:
            return torch.full((self.output_dim,), self.shrinkage_alpha, device=device, dtype=dtype)
        return self.shrinkage_alpha_by_output.to(device=device, dtype=dtype)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyRobustBlockResidualTeacherV11 must be fit before diagnostics")
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
            "teacher_residual_mse_gain": self.teacher_residual_mse_gain,
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
            "teacher_calibration_block_count": self.calibration_block_count,
            "teacher_min_block_mse_gain": self.min_block_mse_gain,
            "teacher_min_block_mae_gain": self.min_block_mae_gain,
            "teacher_min_block_sign_agreement": self.min_block_sign_agreement,
            "teacher_worst_block_mse_gain": self.calibration_worst_block_mse_gain,
            "teacher_worst_block_mae_gain": self.calibration_worst_block_mae_gain,
            "teacher_worst_block_sign_agreement": self.calibration_worst_block_sign_agreement,
            "teacher_block_pass_count": self.calibration_block_pass_count,
            "teacher_effective_block_count": self.calibration_effective_block_count,
            "teacher_reliability_gate_enabled": str(self.reliability_gate_enabled),
            "teacher_reliability_gate_active": str(self.reliability_gate_active),
            "teacher_reliability_gate_reason": self.reliability_gate_reason,
            "teacher_reliable_fraction": self.reliable_fraction,
            "teacher_reliable_drift_norm_min": self.reliable_drift_norm_min,
            "teacher_reliable_drift_norm_max": self.reliable_drift_norm_max,
            "teacher_reliable_correction_norm_max": self.reliable_correction_norm_max,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyRobustBlockResidualTeacherV11 must be fit before serialization")
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
            "calibration_block_count": self.calibration_block_count,
            "min_block_mse_gain": self.min_block_mse_gain,
            "min_block_mae_gain": self.min_block_mae_gain,
            "min_block_sign_agreement": self.min_block_sign_agreement,
            "reliability_gate_enabled": self.reliability_gate_enabled,
            "reliability_gate_quantile": self.reliability_gate_quantile,
            "min_reliable_fraction": self.min_reliable_fraction,
            "disable_if_unreliable": self.disable_if_unreliable,
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
            "teacher_residual_mse_gain": self.teacher_residual_mse_gain,
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
            "calibration_worst_block_mse_gain": self.calibration_worst_block_mse_gain,
            "calibration_worst_block_mae_gain": self.calibration_worst_block_mae_gain,
            "calibration_worst_block_sign_agreement": self.calibration_worst_block_sign_agreement,
            "calibration_block_pass_count": self.calibration_block_pass_count,
            "calibration_effective_block_count": self.calibration_effective_block_count,
            "reliability_gate_active": self.reliability_gate_active,
            "reliability_gate_reason": self.reliability_gate_reason,
            "reliable_fraction": self.reliable_fraction,
            "reliable_drift_norm_min": self.reliable_drift_norm_min,
            "reliable_drift_norm_max": self.reliable_drift_norm_max,
            "reliable_correction_norm_max": self.reliable_correction_norm_max,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyRobustBlockResidualTeacherV11":
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
        self.calibration_block_count = int(state.get("calibration_block_count", self.calibration_block_count))
        self.min_block_mse_gain = float(state.get("min_block_mse_gain", self.min_block_mse_gain))
        self.min_block_mae_gain = float(state.get("min_block_mae_gain", self.min_block_mae_gain))
        self.min_block_sign_agreement = float(
            state.get("min_block_sign_agreement", self.min_block_sign_agreement)
        )
        self.reliability_gate_enabled = bool(state.get("reliability_gate_enabled", self.reliability_gate_enabled))
        self.reliability_gate_quantile = float(
            state.get("reliability_gate_quantile", self.reliability_gate_quantile)
        )
        self.min_reliable_fraction = float(state.get("min_reliable_fraction", self.min_reliable_fraction))
        self.disable_if_unreliable = bool(state.get("disable_if_unreliable", self.disable_if_unreliable))
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
        if "teacher_residual_mse_gain" in state:
            self.teacher_residual_mse_gain = float(state["teacher_residual_mse_gain"])
        elif self.teacher_residual_mse_before == self.teacher_residual_mse_before:
            denominator = max(self.teacher_residual_mse_before, self.eps)
            self.teacher_residual_mse_gain = (
                self.teacher_residual_mse_before - self.teacher_residual_mse_after
            ) / denominator
        else:
            self.teacher_residual_mse_gain = float("nan")
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
        self.calibration_worst_block_mse_gain = float(
            state.get("calibration_worst_block_mse_gain", float("nan"))
        )
        self.calibration_worst_block_mae_gain = float(
            state.get("calibration_worst_block_mae_gain", float("nan"))
        )
        self.calibration_worst_block_sign_agreement = float(
            state.get("calibration_worst_block_sign_agreement", float("nan"))
        )
        self.calibration_block_pass_count = int(state.get("calibration_block_pass_count", 0))
        self.calibration_effective_block_count = int(state.get("calibration_effective_block_count", 0))
        self.reliability_gate_active = bool(state.get("reliability_gate_active", False))
        self.reliability_gate_reason = str(state.get("reliability_gate_reason", "loaded"))
        self.reliable_fraction = float(state.get("reliable_fraction", 0.0))
        self.reliable_drift_norm_min = float(state.get("reliable_drift_norm_min", 0.0))
        self.reliable_drift_norm_max = float(state.get("reliable_drift_norm_max", float("inf")))
        self.reliable_correction_norm_max = float(state.get("reliable_correction_norm_max", float("inf")))
        if self.correction_weight.shape != (self.feature_dim, self.output_dim):
            raise ValueError("loaded V11 residual teacher weight has incompatible shape")
        if self.shrinkage_alpha_by_output.numel() != self.output_dim:
            raise ValueError("loaded V11 residual teacher alpha vector has incompatible shape")
        return self

    def state_hash(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_stage3_v11_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyRobustBlockResidualTeacherV11,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V11 residual teacher checkpoint: {path}")
    metadata = dict(metadata or {})
    metadata.setdefault("protocol_version", TrainOnlyRobustBlockResidualTeacherV11.protocol_version)
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


def load_stage3_v11_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyRobustBlockResidualTeacherV11, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint["teacher_state_dict"]
    teacher = TrainOnlyRobustBlockResidualTeacherV11(
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
        raise ValueError("loaded V11 residual teacher drift_dim does not match expected drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V11 residual teacher output_shape does not match expected output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyRobustBlockResidualTeacherV11.protocol_version)
    return teacher, metadata
