from __future__ import annotations

from dataclasses import dataclass

import torch

from models.stage3_alignment_v7 import (
    _CCAFit,
    _pearson_corr,
    TrainSetChronologicalResidualizedReducedRankCCAAligner,
)


@dataclass(frozen=True)
class _LeakageRankCandidate:
    rank: int
    score: float
    teacher_residual_r2: float
    invariant_predictability_r2: float
    retained_variance_fraction: float
    removed_variance_fraction: float
    eligible: bool


class TrainSetCrossFittedResidualTransferPLSAlignerV11(
    TrainSetChronologicalResidualizedReducedRankCCAAligner
):
    """Zero-preserving residual-transfer PLS aligner for Stage III V11.

    V9 removed invariant-explainable drift with an additive residualizer
    ``delta - f(phi_i)``. That can mix stable phi_i structure into the served
    drift and only preserves physical zero drift through masking. V10 estimated
    a leakage subspace in delta space. V11 keeps that zero-preserving leakage
    projection, but replaces transition CCA with direct residual-transfer PLS:
    the active drift coordinates are selected by chronological calibration
    blocks according to whether they transfer to forecast residual directions.
    """

    def __init__(
        self,
        *,
        leakage_fold_count: int = 5,
        max_leakage_rank: int = 16,
        min_prefix_samples: int = 16,
        min_retained_variance_fraction: float = 0.2,
        max_invariant_predictability_r2: float = 0.98,
        min_teacher_residual_r2: float = -1.0,
        leakage_penalty: float = 0.1,
        leakage_removal_reward: float = 0.01,
        collapse_penalty: float = 0.5,
        rank_penalty: float = 0.001,
        direct_block_count: int = 5,
        min_direct_calibration_correlation: float = 0.03,
        min_block_transfer_r2: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if leakage_fold_count <= 1:
            raise ValueError("leakage_fold_count must be greater than one")
        if max_leakage_rank < 0:
            raise ValueError("max_leakage_rank must be non-negative")
        if min_prefix_samples < 2:
            raise ValueError("min_prefix_samples must be at least two")
        if not 0.0 <= min_retained_variance_fraction <= 1.0:
            raise ValueError("min_retained_variance_fraction must be in [0, 1]")
        if direct_block_count < 1:
            raise ValueError("direct_block_count must be positive")
        self.leakage_fold_count = int(leakage_fold_count)
        self.max_leakage_rank = int(max_leakage_rank)
        self.min_prefix_samples = int(min_prefix_samples)
        self.min_retained_variance_fraction = float(min_retained_variance_fraction)
        self.max_invariant_predictability_r2 = float(max_invariant_predictability_r2)
        self.min_teacher_residual_r2 = float(min_teacher_residual_r2)
        self.leakage_penalty = float(leakage_penalty)
        self.leakage_removal_reward = float(leakage_removal_reward)
        self.collapse_penalty = float(collapse_penalty)
        self.rank_penalty = float(rank_penalty)
        self.direct_block_count = int(direct_block_count)
        self.min_direct_calibration_correlation = float(min_direct_calibration_correlation)
        self.min_block_transfer_r2 = float(min_block_transfer_r2)

        self.leakage_basis: torch.Tensor | None = None
        self.selected_leakage_rank = 0
        self.leakage_rank_candidates: list[_LeakageRankCandidate] = []
        self.oof_projection_sample_count = 0
        self.oof_projection_fold_count = 0
        self.selected_teacher_residual_r2 = 0.0
        self.selected_invariant_predictability_r2 = 0.0
        self.selected_retained_variance_fraction = 1.0
        self.selected_removed_variance_fraction = 0.0
        self.teacher_residual_r2 = 0.0
        self.direct_block_min_r2: torch.Tensor | None = None
        self.direct_block_sign_stability: torch.Tensor | None = None

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetCrossFittedResidualTransferPLSAlignerV11":
        self._validate_fit_tensors(phi_i_train, phi_v_train, y_base_train, y_train)
        phi_i = phi_i_train.detach().float()
        phi_v = phi_v_train.detach().float()
        y_base = y_base_train.detach().float().reshape(phi_i.shape[0], -1)
        y = y_train.detach().float().reshape(phi_i.shape[0], -1)
        raw_delta = phi_v - phi_i
        teacher_residual = y - y_base

        selected = self._select_leakage_rank(phi_i, raw_delta, teacher_residual)
        self.selected_leakage_rank = selected.rank
        self.selected_teacher_residual_r2 = selected.teacher_residual_r2
        self.selected_invariant_predictability_r2 = selected.invariant_predictability_r2
        self.selected_retained_variance_fraction = selected.retained_variance_fraction
        self.selected_removed_variance_fraction = selected.removed_variance_fraction
        self.teacher_residual_r2 = selected.teacher_residual_r2

        residualized_delta = self._cross_fitted_project(phi_i, raw_delta, selected.rank)
        self.leakage_basis = self._fit_leakage_basis(phi_i, raw_delta, selected.rank).detach().float()

        fit_stop = self._inner_fit_stop(phi_i.shape[0])
        fit_delta = residualized_delta[:fit_stop]
        fit_residual = teacher_residual[:fit_stop]
        calib_delta = residualized_delta[fit_stop:]
        calib_residual = teacher_residual[fit_stop:]

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
        self.residual_mean = torch.zeros(raw_delta.shape[1], dtype=torch.float32)
        self.fit_sample_count = int(phi_i.shape[0])
        self.inner_fit_sample_count = int(fit_delta.shape[0])
        self.inner_calib_sample_count = int(calib_delta.shape[0])
        self.transition_sample_count = int(max(0, fit_delta.shape[0] - 1))
        self._select_active_dimensions(calib_delta, calib_residual)

        self.residualizer_weight = torch.zeros(phi_i.shape[1], raw_delta.shape[1], dtype=torch.float32)
        self.residualizer_bias = torch.zeros(raw_delta.shape[1], dtype=torch.float32)
        self.residualizer_r2 = torch.full(
            (raw_delta.shape[1],),
            float(self.selected_removed_variance_fraction),
            dtype=torch.float32,
        )
        return self

    def _fit_cca(self, residualized_delta: torch.Tensor, residual: torch.Tensor) -> _CCAFit:
        if residualized_delta.shape[0] < 3:
            raise ValueError("at least three samples are required for V11 residual-transfer PLS")
        x = residualized_delta.detach().float()
        y = residual.detach().float().reshape(x.shape[0], -1)
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        cross = x_centered.T @ y_centered / float(max(1, x.shape[0] - 1))
        u, singular_values, vh = torch.linalg.svd(cross, full_matrices=True)
        x_dim = x.shape[1]
        projection = u[:, :x_dim]
        residual_projection = vh.T
        usable = min(singular_values.numel(), projection.shape[1], residual_projection.shape[1])
        sign = torch.ones(x_dim, dtype=torch.float32, device=x.device)
        if usable > 0:
            x_scores = x_centered @ projection[:, :usable]
            y_scores = y_centered @ residual_projection[:, :usable]
            score_sign = torch.sign(torch.sum(x_scores * y_scores, dim=0))
            score_sign = torch.where(score_sign == 0.0, torch.ones_like(score_sign), score_sign)
            sign[:usable] = score_sign
        correlations = torch.zeros(x_dim, dtype=torch.float32, device=x.device)
        if singular_values.numel():
            scale = singular_values.max().clamp_min(self.eps)
            correlations[: singular_values.numel()] = (singular_values / scale).detach().float().clamp(0.0, 1.0)
        if self.scale_by_correlation:
            projection = projection * torch.clamp(correlations, min=self.eps).to(projection.dtype).unsqueeze(0)
        return _CCAFit(
            projection=projection.detach().float(),
            residual_projection=residual_projection.detach().float(),
            sign=sign.detach().float(),
            canonical_correlations=correlations.detach().float(),
            x_transition_mean=torch.zeros(x_dim, dtype=torch.float32, device=x.device),
            y_transition_mean=torch.zeros(y.shape[1], dtype=torch.float32, device=x.device),
        )

    def _direct_block_slices(self, sample_count: int) -> list[slice]:
        block_count = min(self.direct_block_count, max(1, sample_count))
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

    def _select_active_dimensions(self, calib_delta: torch.Tensor, calib_residual: torch.Tensor) -> None:
        self._ensure_projection_ready()
        dim = int(self.projection.shape[1])
        self.calibration_correlations = torch.zeros(dim, dtype=torch.float32)
        self.calibration_r2 = torch.zeros(dim, dtype=torch.float32)
        self.bootstrap_sign_stability = torch.zeros(dim, dtype=torch.float32)
        self.direct_block_min_r2 = torch.zeros(dim, dtype=torch.float32)
        self.direct_block_sign_stability = torch.zeros(dim, dtype=torch.float32)
        active_mask = torch.zeros(dim, dtype=torch.bool)
        usable = min(
            dim,
            self.residual_projection.shape[1],
            int(self.canonical_correlations.numel()),
            self.max_rank if self.max_rank is not None else dim,
        )
        if calib_delta.shape[0] >= 3 and usable > 0:
            x_centered = calib_delta - calib_delta.mean(dim=0, keepdim=True)
            y = calib_residual.reshape(calib_delta.shape[0], -1)
            y_centered = y - y.mean(dim=0, keepdim=True)
            x_scores = (x_centered @ self.projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            y_scores = (y_centered @ self.residual_projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            blocks = self._direct_block_slices(calib_delta.shape[0])
            for idx in range(usable):
                corr = _pearson_corr(x_scores[:, idx], y_scores[:, idx], self.eps)
                self.calibration_correlations[idx] = corr.detach().float()
                self.calibration_r2[idx] = torch.clamp(corr.pow(2), max=1.0).detach().float()
                self.bootstrap_sign_stability[idx] = self._bootstrap_sign_stability(
                    x_scores[:, idx],
                    y_scores[:, idx],
                    corr,
                )
                block_r2_values: list[float] = []
                block_sign_values: list[float] = []
                reference_sign = 1.0 if float(corr.item()) >= 0.0 else -1.0
                for block in blocks:
                    if (block.stop or 0) - (block.start or 0) < 3:
                        continue
                    block_corr = _pearson_corr(x_scores[block, idx], y_scores[block, idx], self.eps)
                    block_r2_values.append(float(torch.clamp(block_corr.pow(2), max=1.0).item()))
                    if abs(float(block_corr.item())) > self.eps:
                        block_sign = 1.0 if float(block_corr.item()) >= 0.0 else -1.0
                        block_sign_values.append(float(block_sign == reference_sign))
                min_block_r2 = min(block_r2_values) if block_r2_values else 0.0
                sign_stability = sum(block_sign_values) / len(block_sign_values) if block_sign_values else 0.0
                self.direct_block_min_r2[idx] = torch.tensor(min_block_r2)
                self.direct_block_sign_stability[idx] = torch.tensor(sign_stability)
                active_mask[idx] = bool(
                    abs(float(self.canonical_correlations[idx].item())) >= self.min_fit_correlation
                    and abs(float(self.calibration_correlations[idx].item()))
                    >= self.min_direct_calibration_correlation
                    and float(self.calibration_r2[idx].item()) >= self.min_calibration_r2
                    and min_block_r2 >= self.min_block_transfer_r2
                    and float(self.bootstrap_sign_stability[idx].item()) >= self.min_bootstrap_sign_stability
                )
        self.active_mask = active_mask
        self.active_rank = int(active_mask.sum().item())
        failures: list[str] = []
        if self.active_rank <= 0:
            failures.append("active_rank")
        self.alignment_gate_pass = not failures
        self.alignment_gate_failure_reasons = ";".join(failures)

    def _candidate_ranks(self, phi_i: torch.Tensor, raw_delta: torch.Tensor) -> list[int]:
        max_by_shape = min(phi_i.shape[1], raw_delta.shape[1], max(0, phi_i.shape[0] - 2))
        max_rank = min(self.max_leakage_rank, max_by_shape)
        return list(range(max_rank + 1))

    def _select_leakage_rank(
        self,
        phi_i: torch.Tensor,
        raw_delta: torch.Tensor,
        teacher_residual: torch.Tensor,
    ) -> _LeakageRankCandidate:
        candidates: list[_LeakageRankCandidate] = []
        for rank in self._candidate_ranks(phi_i, raw_delta):
            cleaned = self._cross_fitted_project(phi_i, raw_delta, rank)
            retained = _variance_fraction(cleaned, raw_delta, self.eps)
            removed = max(0.0, 1.0 - retained)
            teacher_r2 = _ridge_predictability_r2(cleaned, teacher_residual, self.ridge, self.eps)
            invariant_r2 = _ridge_predictability_r2(phi_i, cleaned, self.ridge, self.eps)
            eligible = bool(
                retained >= self.min_retained_variance_fraction
                and invariant_r2 <= self.max_invariant_predictability_r2
                and teacher_r2 >= self.min_teacher_residual_r2
            )
            score = (
                teacher_r2
                + self.leakage_removal_reward * removed
                - self.leakage_penalty * max(0.0, invariant_r2 - self.max_invariant_predictability_r2)
                - self.collapse_penalty * max(0.0, self.min_retained_variance_fraction - retained)
                - self.rank_penalty * rank
            )
            candidates.append(
                _LeakageRankCandidate(
                    rank=rank,
                    score=float(score),
                    teacher_residual_r2=float(teacher_r2),
                    invariant_predictability_r2=float(invariant_r2),
                    retained_variance_fraction=float(retained),
                    removed_variance_fraction=float(removed),
                    eligible=eligible,
                )
            )
        self.leakage_rank_candidates = candidates
        eligible_candidates = [candidate for candidate in candidates if candidate.eligible]
        pool = eligible_candidates or candidates
        return max(pool, key=lambda item: (item.score, -item.rank))

    def _chronological_folds(self, sample_count: int) -> list[tuple[int, int]]:
        fold_count = min(self.leakage_fold_count, sample_count)
        base = sample_count // fold_count
        remainder = sample_count % fold_count
        folds = []
        start = 0
        for fold_id in range(fold_count):
            size = base + (1 if fold_id < remainder else 0)
            stop = start + size
            folds.append((start, stop))
            start = stop
        return folds

    def _cross_fitted_project(self, phi_i: torch.Tensor, raw_delta: torch.Tensor, rank: int) -> torch.Tensor:
        if rank <= 0:
            self.oof_projection_sample_count = int(raw_delta.shape[0])
            self.oof_projection_fold_count = int(min(self.leakage_fold_count, raw_delta.shape[0]))
            return raw_delta.detach().float()
        cleaned = torch.empty_like(raw_delta)
        folds = self._chronological_folds(raw_delta.shape[0])
        for start, stop in folds:
            if start < self.min_prefix_samples:
                cleaned[start:stop] = raw_delta[start:stop]
                continue
            basis = self._fit_leakage_basis(phi_i[:start], raw_delta[:start], rank)
            cleaned[start:stop] = self._project_with_basis(raw_delta[start:stop], basis)
        self.oof_projection_sample_count = int(raw_delta.shape[0])
        self.oof_projection_fold_count = int(len(folds))
        return cleaned.detach().float()

    def _fit_leakage_basis(self, phi_i: torch.Tensor, raw_delta: torch.Tensor, rank: int) -> torch.Tensor:
        rank = int(rank)
        drift_dim = int(raw_delta.shape[1])
        if rank <= 0:
            return torch.empty(drift_dim, 0, device=raw_delta.device, dtype=raw_delta.dtype)
        x = phi_i - phi_i.mean(dim=0, keepdim=True)
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        weight = torch.linalg.solve(
            x.T @ x + self.ridge * eye,
            x.T @ raw_delta,
        )
        _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        usable = min(rank, vh.shape[0], singular_values.numel())
        if usable <= 0:
            return torch.empty(drift_dim, 0, device=raw_delta.device, dtype=raw_delta.dtype)
        return vh[:usable].T.detach()

    @staticmethod
    def _project_with_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.numel() == 0:
            return delta
        return delta - (delta @ basis) @ basis.T

    def residualize_delta(self, phi_i: torch.Tensor, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.leakage_basis is None:
            raise RuntimeError("V11 aligner must be fit before residualize_delta")
        basis = self.leakage_basis.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        cleaned = self._project_with_basis(delta_phi_v, basis)
        zero_mask = torch.linalg.vector_norm(delta_phi_v, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(cleaned), cleaned)

    def transform_phi_pair(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        if phi_i.shape != phi_v.shape:
            raise ValueError("phi_i and phi_v must have the same shape")
        raw_delta = phi_v - phi_i
        cleaned = self.residualize_delta(phi_i, raw_delta)
        transformed = self.transform_delta(cleaned)
        zero_mask = torch.linalg.vector_norm(raw_delta, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(transformed), transformed)

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        self._ensure_projection_ready()
        if self.active_mask is None:
            raise RuntimeError("V11 aligner must be fit before transform_delta")
        projection = self.projection.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        sign = self.sign.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        active_mask = self.active_mask.to(device=delta_phi_v.device)
        transformed = (delta_phi_v @ projection) * sign.unsqueeze(0)
        transformed[:, ~active_mask] = 0.0
        zero_mask = torch.linalg.vector_norm(delta_phi_v, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(transformed), transformed)

    def diagnostic_summary(self) -> dict[str, int | float | bool | str]:
        summary = super().diagnostic_summary()
        summary["alignment_method"] = "cross_fitted_residual_transfer_pls_v11"
        summary["selected_leakage_rank"] = int(self.selected_leakage_rank)
        summary["max_leakage_rank"] = int(self.max_leakage_rank)
        summary["oof_projection_sample_count"] = int(self.oof_projection_sample_count)
        summary["oof_projection_fold_count"] = int(self.oof_projection_fold_count)
        summary["teacher_residual_r2"] = float(self.teacher_residual_r2)
        summary["selected_invariant_predictability_r2"] = float(self.selected_invariant_predictability_r2)
        summary["selected_retained_variance_fraction"] = float(self.selected_retained_variance_fraction)
        summary["selected_removed_variance_fraction"] = float(self.selected_removed_variance_fraction)
        summary["direct_block_count"] = int(self.direct_block_count)
        summary["min_direct_calibration_correlation"] = float(self.min_direct_calibration_correlation)
        summary["min_block_transfer_r2"] = float(self.min_block_transfer_r2)
        if self.direct_block_min_r2 is not None and self.active_mask is not None and self.active_rank:
            summary["active_direct_block_min_r2"] = float(self.direct_block_min_r2[self.active_mask].min().item())
        else:
            summary["active_direct_block_min_r2"] = 0.0
        if self.direct_block_sign_stability is not None and self.active_mask is not None and self.active_rank:
            summary["active_direct_block_sign_stability"] = float(
                self.direct_block_sign_stability[self.active_mask].min().item()
            )
        else:
            summary["active_direct_block_sign_stability"] = 0.0
        if self.leakage_rank_candidates:
            summary["leakage_rank_candidate_count"] = int(len(self.leakage_rank_candidates))
            summary["leakage_rank_best_score"] = float(
                max(candidate.score for candidate in self.leakage_rank_candidates)
            )
        return summary

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self.leakage_basis is None:
            raise RuntimeError("V11 aligner must be fit before serialization")
        state["class_name"] = "TrainSetCrossFittedResidualTransferPLSAlignerV11"
        state["method"] = "cross_fitted_residual_transfer_pls_v11"
        state["leakage_basis"] = self.leakage_basis.detach().clone()
        state["selected_leakage_rank"] = self.selected_leakage_rank
        state["leakage_fold_count"] = self.leakage_fold_count
        state["max_leakage_rank"] = self.max_leakage_rank
        state["min_prefix_samples"] = self.min_prefix_samples
        state["min_retained_variance_fraction"] = self.min_retained_variance_fraction
        state["max_invariant_predictability_r2"] = self.max_invariant_predictability_r2
        state["min_teacher_residual_r2"] = self.min_teacher_residual_r2
        state["leakage_penalty"] = self.leakage_penalty
        state["leakage_removal_reward"] = self.leakage_removal_reward
        state["collapse_penalty"] = self.collapse_penalty
        state["rank_penalty"] = self.rank_penalty
        state["direct_block_count"] = self.direct_block_count
        state["min_direct_calibration_correlation"] = self.min_direct_calibration_correlation
        state["min_block_transfer_r2"] = self.min_block_transfer_r2
        state["oof_projection_sample_count"] = self.oof_projection_sample_count
        state["oof_projection_fold_count"] = self.oof_projection_fold_count
        state["selected_teacher_residual_r2"] = self.selected_teacher_residual_r2
        state["selected_invariant_predictability_r2"] = self.selected_invariant_predictability_r2
        state["selected_retained_variance_fraction"] = self.selected_retained_variance_fraction
        state["selected_removed_variance_fraction"] = self.selected_removed_variance_fraction
        state["teacher_residual_r2"] = self.teacher_residual_r2
        state["direct_block_min_r2"] = (
            self.direct_block_min_r2.detach().clone() if self.direct_block_min_r2 is not None else None
        )
        state["direct_block_sign_stability"] = (
            self.direct_block_sign_stability.detach().clone()
            if self.direct_block_sign_stability is not None
            else None
        )
        state["leakage_rank_candidates"] = [
            {
                "rank": candidate.rank,
                "score": candidate.score,
                "teacher_residual_r2": candidate.teacher_residual_r2,
                "invariant_predictability_r2": candidate.invariant_predictability_r2,
                "retained_variance_fraction": candidate.retained_variance_fraction,
                "removed_variance_fraction": candidate.removed_variance_fraction,
                "eligible": candidate.eligible,
            }
            for candidate in self.leakage_rank_candidates
        ]
        return state

    def load_state_dict(self, state: dict) -> "TrainSetCrossFittedResidualTransferPLSAlignerV11":
        super().load_state_dict(state)
        self.leakage_basis = torch.as_tensor(state["leakage_basis"]).detach().float()
        self.selected_leakage_rank = int(state.get("selected_leakage_rank", self.leakage_basis.shape[1]))
        self.leakage_fold_count = int(state.get("leakage_fold_count", self.leakage_fold_count))
        self.max_leakage_rank = int(state.get("max_leakage_rank", self.max_leakage_rank))
        self.min_prefix_samples = int(state.get("min_prefix_samples", self.min_prefix_samples))
        self.min_retained_variance_fraction = float(
            state.get("min_retained_variance_fraction", self.min_retained_variance_fraction)
        )
        self.max_invariant_predictability_r2 = float(
            state.get("max_invariant_predictability_r2", self.max_invariant_predictability_r2)
        )
        self.min_teacher_residual_r2 = float(state.get("min_teacher_residual_r2", self.min_teacher_residual_r2))
        self.leakage_penalty = float(state.get("leakage_penalty", self.leakage_penalty))
        self.leakage_removal_reward = float(state.get("leakage_removal_reward", self.leakage_removal_reward))
        self.collapse_penalty = float(state.get("collapse_penalty", self.collapse_penalty))
        self.rank_penalty = float(state.get("rank_penalty", self.rank_penalty))
        self.direct_block_count = int(state.get("direct_block_count", self.direct_block_count))
        self.min_direct_calibration_correlation = float(
            state.get("min_direct_calibration_correlation", self.min_direct_calibration_correlation)
        )
        self.min_block_transfer_r2 = float(state.get("min_block_transfer_r2", self.min_block_transfer_r2))
        self.oof_projection_sample_count = int(state.get("oof_projection_sample_count", 0))
        self.oof_projection_fold_count = int(state.get("oof_projection_fold_count", 0))
        self.selected_teacher_residual_r2 = float(state.get("selected_teacher_residual_r2", 0.0))
        self.selected_invariant_predictability_r2 = float(state.get("selected_invariant_predictability_r2", 0.0))
        self.selected_retained_variance_fraction = float(state.get("selected_retained_variance_fraction", 1.0))
        self.selected_removed_variance_fraction = float(state.get("selected_removed_variance_fraction", 0.0))
        self.teacher_residual_r2 = float(state.get("teacher_residual_r2", self.selected_teacher_residual_r2))
        direct_block_min_r2 = state.get("direct_block_min_r2")
        self.direct_block_min_r2 = (
            torch.as_tensor(direct_block_min_r2).detach().float() if direct_block_min_r2 is not None else None
        )
        direct_block_sign_stability = state.get("direct_block_sign_stability")
        self.direct_block_sign_stability = (
            torch.as_tensor(direct_block_sign_stability).detach().float()
            if direct_block_sign_stability is not None
            else None
        )
        self.leakage_rank_candidates = [
            _LeakageRankCandidate(
                rank=int(candidate["rank"]),
                score=float(candidate["score"]),
                teacher_residual_r2=float(candidate["teacher_residual_r2"]),
                invariant_predictability_r2=float(candidate["invariant_predictability_r2"]),
                retained_variance_fraction=float(candidate["retained_variance_fraction"]),
                removed_variance_fraction=float(candidate["removed_variance_fraction"]),
                eligible=bool(candidate["eligible"]),
            )
            for candidate in state.get("leakage_rank_candidates", [])
        ]
        return self


def _ridge_predictability_r2(x: torch.Tensor, y: torch.Tensor, ridge: float, eps: float) -> float:
    x = x.detach().float()
    y = y.detach().float().reshape(x.shape[0], -1)
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    eye = torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
    eye[-1, -1] = 0.0
    solution = torch.linalg.solve(
        design.T @ design + ridge * eye,
        design.T @ y,
    )
    prediction = design @ solution
    residual_error = torch.mean((y - prediction).pow(2))
    total = torch.mean((y - y.mean(dim=0, keepdim=True)).pow(2)).clamp_min(eps)
    return float(torch.clamp(1.0 - residual_error / total, min=-1.0, max=1.0).item())


def _variance_fraction(cleaned: torch.Tensor, raw: torch.Tensor, eps: float) -> float:
    raw_centered = raw - raw.mean(dim=0, keepdim=True)
    cleaned_centered = cleaned - cleaned.mean(dim=0, keepdim=True)
    raw_var = torch.mean(raw_centered.pow(2)).clamp_min(eps)
    cleaned_var = torch.mean(cleaned_centered.pow(2))
    return float(torch.clamp(cleaned_var / raw_var, min=0.0, max=10.0).item())
