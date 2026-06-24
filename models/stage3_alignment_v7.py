from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _CCAFit:
    projection: torch.Tensor
    residual_projection: torch.Tensor
    sign: torch.Tensor
    canonical_correlations: torch.Tensor
    x_transition_mean: torch.Tensor
    y_transition_mean: torch.Tensor


class TrainSetChronologicalResidualizedReducedRankCCAAligner:
    """Train-only residualized reduced-rank CCA aligner for Stage III V7.

    V6 aligns the full ``phi_v`` distribution to ``phi_i`` before CCA. V7 keeps
    the residual interpretation explicit: it first removes the component of
    ``phi_v - phi_i`` explainable by ``phi_i`` on the train split, then keeps
    only CCA coordinates that are stable on an inner chronological calibration
    slice of the train split. Validation/test never influence residualization,
    projection, sign selection, or active-rank selection.
    """

    def __init__(
        self,
        *,
        ridge: float = 1e-3,
        eps: float = 1e-6,
        inner_fit_fraction: float = 0.75,
        max_rank: int | None = None,
        min_fit_correlation: float = 0.05,
        min_calibration_correlation: float = 0.05,
        min_calibration_r2: float = 0.0025,
        bootstrap_samples: int = 64,
        min_bootstrap_sign_stability: float = 0.6,
        bootstrap_seed: int = 1729,
        scale_by_correlation: bool = False,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.5 <= inner_fit_fraction <= 0.9:
            raise ValueError("inner_fit_fraction must be in [0.5, 0.9]")
        if max_rank is not None and max_rank <= 0:
            raise ValueError("max_rank must be positive when provided")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")
        self.ridge = ridge
        self.eps = eps
        self.inner_fit_fraction = inner_fit_fraction
        self.max_rank = max_rank
        self.min_fit_correlation = min_fit_correlation
        self.min_calibration_correlation = min_calibration_correlation
        self.min_calibration_r2 = min_calibration_r2
        self.bootstrap_samples = bootstrap_samples
        self.min_bootstrap_sign_stability = min_bootstrap_sign_stability
        self.bootstrap_seed = bootstrap_seed
        self.scale_by_correlation = scale_by_correlation

        self.residualizer_weight: torch.Tensor | None = None
        self.residualizer_bias: torch.Tensor | None = None
        self.residualizer_r2: torch.Tensor | None = None
        self.residual_mean: torch.Tensor | None = None
        self.projection: torch.Tensor | None = None
        self.residual_projection: torch.Tensor | None = None
        self.sign: torch.Tensor | None = None
        self.canonical_correlations: torch.Tensor | None = None
        self.calibration_correlations: torch.Tensor | None = None
        self.calibration_r2: torch.Tensor | None = None
        self.bootstrap_sign_stability: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.inner_fit_sample_count = 0
        self.inner_calib_sample_count = 0
        self.transition_sample_count = 0
        self.active_rank = 0
        self.alignment_gate_pass = False
        self.alignment_gate_failure_reasons = "not_fit"

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_inv_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetChronologicalResidualizedReducedRankCCAAligner":
        self._validate_fit_tensors(phi_i_train, phi_v_train, y_inv_train, y_train)
        phi_i = phi_i_train.detach().float()
        phi_v = phi_v_train.detach().float()
        y_inv = y_inv_train.detach().float().reshape(phi_i.shape[0], -1)
        y = y_train.detach().float().reshape(phi_i.shape[0], -1)
        raw_delta = phi_v - phi_i
        residual = y - y_inv

        residualized_delta = self._fit_residualizer(phi_i, raw_delta)
        fit_stop = self._inner_fit_stop(phi_i.shape[0])
        fit_delta = residualized_delta[:fit_stop]
        fit_residual = residual[:fit_stop]
        calib_delta = residualized_delta[fit_stop:]
        calib_residual = residual[fit_stop:]

        cca_fit = self._fit_cca(fit_delta, fit_residual)
        (
            self.projection,
            self.residual_projection,
            self.sign,
            self.canonical_correlations,
        ) = (
            cca_fit.projection,
            cca_fit.residual_projection,
            cca_fit.sign,
            cca_fit.canonical_correlations,
        )
        self.x_transition_mean = cca_fit.x_transition_mean
        self.y_transition_mean = cca_fit.y_transition_mean
        self.residual_mean = residualized_delta.mean(dim=0).detach().float()
        self.fit_sample_count = int(phi_i.shape[0])
        self.inner_fit_sample_count = int(fit_delta.shape[0])
        self.inner_calib_sample_count = int(calib_delta.shape[0])
        self.transition_sample_count = int(max(0, fit_delta.shape[0] - 1))
        self._select_active_dimensions(calib_delta, calib_residual)
        return self

    def _validate_fit_tensors(
        self,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_inv_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> None:
        if phi_i_train.ndim != 2 or phi_v_train.ndim != 2:
            raise ValueError("phi_i_train and phi_v_train must be 2-D")
        if phi_i_train.shape != phi_v_train.shape:
            raise ValueError("phi_i_train and phi_v_train must have the same shape")
        if phi_i_train.shape[0] < 8:
            raise ValueError("at least eight ordered train samples are required for V7 alignment")
        if y_inv_train.shape[0] != phi_i_train.shape[0] or y_train.shape[0] != phi_i_train.shape[0]:
            raise ValueError("phi_i_train, phi_v_train, y_inv_train, and y_train must share sample count")

    def _inner_fit_stop(self, sample_count: int) -> int:
        fit_stop = int(round(sample_count * self.inner_fit_fraction))
        return min(max(fit_stop, 4), sample_count - 3)

    def _fit_residualizer(self, phi_i: torch.Tensor, raw_delta: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(phi_i.shape[0], 1, device=phi_i.device, dtype=phi_i.dtype)
        design = torch.cat([phi_i, ones], dim=1)
        eye = torch.eye(design.shape[1], device=phi_i.device, dtype=phi_i.dtype)
        eye[-1, -1] = 0.0
        solution = torch.linalg.solve(
            design.T @ design + self.ridge * eye,
            design.T @ raw_delta,
        )
        predicted = design @ solution
        residualized = raw_delta - predicted
        centered = raw_delta - raw_delta.mean(dim=0, keepdim=True)
        ss_total = torch.sum(centered.pow(2), dim=0).clamp_min(self.eps)
        ss_residual = torch.sum(residualized.pow(2), dim=0)
        self.residualizer_weight = solution[:-1].detach().float()
        self.residualizer_bias = solution[-1].detach().float()
        self.residualizer_r2 = torch.clamp(1.0 - ss_residual / ss_total, min=-1.0, max=1.0).detach().float()
        return residualized.detach().float()

    def _fit_cca(self, residualized_delta: torch.Tensor, residual: torch.Tensor) -> _CCAFit:
        x_transition = residualized_delta[1:] - residualized_delta[:-1]
        y_transition = residual[1:] - residual[:-1]
        if x_transition.shape[0] < 3:
            raise ValueError("at least three transition samples are required for V7 CCA")
        x_mean = x_transition.mean(dim=0, keepdim=True)
        y_mean = y_transition.mean(dim=0, keepdim=True)
        x_centered = x_transition - x_mean
        y_centered = y_transition - y_mean
        n = max(1, x_centered.shape[0] - 1)
        x_dim = x_centered.shape[1]
        y_dim = y_centered.shape[1]
        sxx = x_centered.T @ x_centered / float(n)
        syy = y_centered.T @ y_centered / float(n)
        sxx = sxx + self.ridge * torch.eye(x_dim, device=sxx.device, dtype=sxx.dtype)
        syy = syy + self.ridge * torch.eye(y_dim, device=syy.device, dtype=syy.dtype)
        x_invsqrt = _symmetric_invsqrt(sxx, self.eps)
        y_invsqrt = _symmetric_invsqrt(syy, self.eps)
        cross = x_centered.T @ y_centered / float(n)
        whitened_cross = x_invsqrt @ cross @ y_invsqrt
        u, singular_values, vh = torch.linalg.svd(whitened_cross, full_matrices=True)
        projection = x_invsqrt @ u
        residual_projection = y_invsqrt @ vh.T

        sign = torch.ones(x_dim, dtype=torch.float32, device=residualized_delta.device)
        usable = min(singular_values.numel(), residual_projection.shape[1], projection.shape[1])
        if usable > 0:
            x_scores = x_centered @ projection[:, :usable]
            y_scores = y_centered @ residual_projection[:, :usable]
            score_sign = torch.sign(torch.sum(x_scores * y_scores, dim=0))
            score_sign = torch.where(score_sign == 0.0, torch.ones_like(score_sign), score_sign)
            sign[:usable] = score_sign

        correlations = torch.zeros(x_dim, dtype=torch.float32, device=residualized_delta.device)
        correlations[: singular_values.numel()] = singular_values.detach().float().clamp(min=0.0, max=1.0)
        if self.scale_by_correlation:
            projection = projection * torch.clamp(correlations, min=self.eps).to(projection.dtype).unsqueeze(0)
        return _CCAFit(
            projection=projection.detach().float(),
            residual_projection=residual_projection.detach().float(),
            sign=sign.detach().float(),
            canonical_correlations=correlations.detach().float(),
            x_transition_mean=x_mean.squeeze(0).detach().float(),
            y_transition_mean=y_mean.squeeze(0).detach().float(),
        )

    def _select_active_dimensions(self, calib_delta: torch.Tensor, calib_residual: torch.Tensor) -> None:
        self._ensure_projection_ready()
        dim = int(self.projection.shape[1])
        self.calibration_correlations = torch.zeros(dim, dtype=torch.float32)
        self.calibration_r2 = torch.zeros(dim, dtype=torch.float32)
        self.bootstrap_sign_stability = torch.zeros(dim, dtype=torch.float32)
        active_mask = torch.zeros(dim, dtype=torch.bool)

        usable = min(
            dim,
            self.residual_projection.shape[1],
            int(self.canonical_correlations.numel()),
            self.max_rank if self.max_rank is not None else dim,
        )
        if calib_delta.shape[0] >= 3 and usable > 0:
            x_transition = calib_delta[1:] - calib_delta[:-1]
            y_transition = calib_residual[1:] - calib_residual[:-1]
            x_centered = x_transition - self.x_transition_mean.to(x_transition.device, x_transition.dtype)
            y_centered = y_transition - self.y_transition_mean.to(y_transition.device, y_transition.dtype)
            x_scores = (x_centered @ self.projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            y_scores = (y_centered @ self.residual_projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            for idx in range(usable):
                corr = _pearson_corr(x_scores[:, idx], y_scores[:, idx], self.eps)
                self.calibration_correlations[idx] = corr.detach().float()
                self.calibration_r2[idx] = torch.clamp(corr.pow(2), max=1.0).detach().float()
                self.bootstrap_sign_stability[idx] = self._bootstrap_sign_stability(
                    x_scores[:, idx],
                    y_scores[:, idx],
                    corr,
                )
                active_mask[idx] = bool(
                    abs(float(self.canonical_correlations[idx].item())) >= self.min_fit_correlation
                    and abs(float(self.calibration_correlations[idx].item())) >= self.min_calibration_correlation
                    and float(self.calibration_r2[idx].item()) >= self.min_calibration_r2
                    and float(self.bootstrap_sign_stability[idx].item()) >= self.min_bootstrap_sign_stability
                )

        self.active_mask = active_mask
        self.active_rank = int(active_mask.sum().item())
        failures: list[str] = []
        if self.active_rank <= 0:
            failures.append("active_rank")
        self.alignment_gate_pass = not failures
        self.alignment_gate_failure_reasons = ";".join(failures)

    def _bootstrap_sign_stability(self, x: torch.Tensor, y: torch.Tensor, reference_corr: torch.Tensor) -> torch.Tensor:
        if self.bootstrap_samples == 0:
            return torch.tensor(1.0 if abs(float(reference_corr.item())) > self.eps else 0.0)
        if x.numel() < 3 or abs(float(reference_corr.item())) <= self.eps:
            return torch.tensor(0.0)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.bootstrap_seed)
        reference_sign = 1.0 if float(reference_corr.item()) >= 0.0 else -1.0
        agree = 0
        for _ in range(self.bootstrap_samples):
            indices = torch.randint(0, x.numel(), (x.numel(),), generator=generator, device=x.device)
            corr = _pearson_corr(x[indices], y[indices], self.eps)
            if abs(float(corr.item())) <= self.eps:
                continue
            corr_sign = 1.0 if float(corr.item()) >= 0.0 else -1.0
            agree += int(corr_sign == reference_sign)
        return torch.tensor(float(agree) / float(self.bootstrap_samples))

    def residualize_delta(self, phi_i: torch.Tensor, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.residualizer_weight is None or self.residualizer_bias is None:
            raise RuntimeError("V7 aligner must be fit before residualize_delta")
        weight = self.residualizer_weight.to(device=phi_i.device, dtype=phi_i.dtype)
        bias = self.residualizer_bias.to(device=phi_i.device, dtype=phi_i.dtype)
        return delta_phi_v - (phi_i @ weight + bias)

    def transform_phi_pair(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        if phi_i.shape != phi_v.shape:
            raise ValueError("phi_i and phi_v must have the same shape")
        raw_delta = phi_v - phi_i
        residualized_delta = self.residualize_delta(phi_i, raw_delta)
        return self.transform_delta(residualized_delta)

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        self._ensure_projection_ready()
        if self.residual_mean is None or self.active_mask is None:
            raise RuntimeError("V7 aligner must be fit before transform_delta")
        residual_mean = self.residual_mean.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        projection = self.projection.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        sign = self.sign.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        active_mask = self.active_mask.to(device=delta_phi_v.device)
        transformed = ((delta_phi_v - residual_mean) @ projection) * sign.unsqueeze(0)
        transformed[:, ~active_mask] = 0.0
        return transformed

    def _ensure_projection_ready(self) -> None:
        if self.projection is None or self.residual_projection is None or self.sign is None or self.canonical_correlations is None:
            raise RuntimeError("V7 aligner must be fit before use")

    def diagnostic_summary(self) -> dict[str, int | float | bool | str]:
        if self.active_mask is None or self.residualizer_r2 is None:
            raise RuntimeError("V7 aligner must be fit before diagnostics")
        active_corr = self.calibration_correlations[self.active_mask] if self.active_rank else torch.empty(0)
        active_stability = self.bootstrap_sign_stability[self.active_mask] if self.active_rank else torch.empty(0)
        return {
            "alignment_method": "residualized_rrcca_v7",
            "alignment_gate_pass": bool(self.alignment_gate_pass),
            "alignment_gate_failure_reasons": self.alignment_gate_failure_reasons,
            "fit_sample_count": self.fit_sample_count,
            "inner_fit_count": self.inner_fit_sample_count,
            "inner_calib_count": self.inner_calib_sample_count,
            "transition_sample_count": self.transition_sample_count,
            "active_rank": self.active_rank,
            "max_rank": int(self.max_rank or int(self.active_mask.numel())),
            "active_dim_fraction": float(self.active_rank / max(1, int(self.active_mask.numel()))),
            "inactive_dim_zeroed": int((~self.active_mask).sum().item()),
            "residualizer_ridge": float(self.ridge),
            "residualizer_r2_mean": float(self.residualizer_r2.mean().item()),
            "residualizer_r2_max": float(self.residualizer_r2.max().item()),
            "calib_top_corr": float(self.calibration_correlations.abs().max().item()) if self.calibration_correlations is not None else 0.0,
            "calib_mean_active_corr": float(active_corr.abs().mean().item()) if active_corr.numel() else 0.0,
            "calib_transition_r2": float(self.calibration_r2.max().item()) if self.calibration_r2 is not None else 0.0,
            "bootstrap_sign_stability_min": float(active_stability.min().item()) if active_stability.numel() else 0.0,
            "bootstrap_sign_stability_mean": float(active_stability.mean().item()) if active_stability.numel() else 0.0,
            "projection_condition_number": _condition_number(self.projection),
        }

    def state_dict(self) -> dict[str, torch.Tensor | int | float | bool | str | None]:
        if (
            self.residualizer_weight is None
            or self.residualizer_bias is None
            or self.residualizer_r2 is None
            or self.residual_mean is None
            or self.projection is None
            or self.residual_projection is None
            or self.sign is None
            or self.canonical_correlations is None
            or self.calibration_correlations is None
            or self.calibration_r2 is None
            or self.bootstrap_sign_stability is None
            or self.active_mask is None
        ):
            raise RuntimeError("V7 aligner must be fit before serialization")
        return {
            "class_name": "TrainSetChronologicalResidualizedReducedRankCCAAligner",
            "method": "residualized_rrcca_v7",
            "residualizer_weight": self.residualizer_weight.detach().clone(),
            "residualizer_bias": self.residualizer_bias.detach().clone(),
            "residualizer_r2": self.residualizer_r2.detach().clone(),
            "residual_mean": self.residual_mean.detach().clone(),
            "x_transition_mean": self.x_transition_mean.detach().clone(),
            "y_transition_mean": self.y_transition_mean.detach().clone(),
            "projection": self.projection.detach().clone(),
            "residual_projection": self.residual_projection.detach().clone(),
            "sign": self.sign.detach().clone(),
            "canonical_correlations": self.canonical_correlations.detach().clone(),
            "calibration_correlations": self.calibration_correlations.detach().clone(),
            "calibration_r2": self.calibration_r2.detach().clone(),
            "bootstrap_sign_stability": self.bootstrap_sign_stability.detach().clone(),
            "active_mask": self.active_mask.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "inner_fit_sample_count": self.inner_fit_sample_count,
            "inner_calib_sample_count": self.inner_calib_sample_count,
            "transition_sample_count": self.transition_sample_count,
            "active_rank": self.active_rank,
            "alignment_gate_pass": self.alignment_gate_pass,
            "alignment_gate_failure_reasons": self.alignment_gate_failure_reasons,
            "ridge": self.ridge,
            "eps": self.eps,
            "inner_fit_fraction": self.inner_fit_fraction,
            "max_rank": self.max_rank,
            "min_fit_correlation": self.min_fit_correlation,
            "min_calibration_correlation": self.min_calibration_correlation,
            "min_calibration_r2": self.min_calibration_r2,
            "bootstrap_samples": self.bootstrap_samples,
            "min_bootstrap_sign_stability": self.min_bootstrap_sign_stability,
            "bootstrap_seed": self.bootstrap_seed,
            "scale_by_correlation": self.scale_by_correlation,
        }

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor | int | float | bool | str | None],
    ) -> "TrainSetChronologicalResidualizedReducedRankCCAAligner":
        self.residualizer_weight = torch.as_tensor(state["residualizer_weight"]).detach().float()
        self.residualizer_bias = torch.as_tensor(state["residualizer_bias"]).detach().float()
        self.residualizer_r2 = torch.as_tensor(state["residualizer_r2"]).detach().float()
        self.residual_mean = torch.as_tensor(state["residual_mean"]).detach().float()
        self.x_transition_mean = torch.as_tensor(state["x_transition_mean"]).detach().float()
        self.y_transition_mean = torch.as_tensor(state["y_transition_mean"]).detach().float()
        self.projection = torch.as_tensor(state["projection"]).detach().float()
        self.residual_projection = torch.as_tensor(state["residual_projection"]).detach().float()
        self.sign = torch.as_tensor(state["sign"]).detach().float()
        self.canonical_correlations = torch.as_tensor(state["canonical_correlations"]).detach().float()
        self.calibration_correlations = torch.as_tensor(state["calibration_correlations"]).detach().float()
        self.calibration_r2 = torch.as_tensor(state["calibration_r2"]).detach().float()
        self.bootstrap_sign_stability = torch.as_tensor(state["bootstrap_sign_stability"]).detach().float()
        self.active_mask = torch.as_tensor(state["active_mask"]).detach().bool()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.inner_fit_sample_count = int(state["inner_fit_sample_count"])
        self.inner_calib_sample_count = int(state["inner_calib_sample_count"])
        self.transition_sample_count = int(state["transition_sample_count"])
        self.active_rank = int(state["active_rank"])
        self.alignment_gate_pass = bool(state["alignment_gate_pass"])
        self.alignment_gate_failure_reasons = str(state["alignment_gate_failure_reasons"])
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        self.inner_fit_fraction = float(state.get("inner_fit_fraction", self.inner_fit_fraction))
        loaded_max_rank = state.get("max_rank", self.max_rank)
        self.max_rank = None if loaded_max_rank is None else int(loaded_max_rank)
        self.min_fit_correlation = float(state.get("min_fit_correlation", self.min_fit_correlation))
        self.min_calibration_correlation = float(state.get("min_calibration_correlation", self.min_calibration_correlation))
        self.min_calibration_r2 = float(state.get("min_calibration_r2", self.min_calibration_r2))
        self.bootstrap_samples = int(state.get("bootstrap_samples", self.bootstrap_samples))
        self.min_bootstrap_sign_stability = float(
            state.get("min_bootstrap_sign_stability", self.min_bootstrap_sign_stability)
        )
        self.bootstrap_seed = int(state.get("bootstrap_seed", self.bootstrap_seed))
        self.scale_by_correlation = bool(state.get("scale_by_correlation", self.scale_by_correlation))
        return self


def _symmetric_invsqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denom.item()) <= eps:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return torch.clamp(torch.sum(x_centered * y_centered) / denom, min=-1.0, max=1.0)


def _condition_number(matrix: torch.Tensor | None) -> float:
    if matrix is None or matrix.numel() == 0:
        return 0.0
    singular_values = torch.linalg.svdvals(matrix.detach().float())
    if singular_values.numel() == 0:
        return 0.0
    min_value = torch.clamp(singular_values.min(), min=1e-12)
    return float((singular_values.max() / min_value).item())
