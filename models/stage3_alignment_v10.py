from __future__ import annotations

from dataclasses import dataclass

import torch

from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner


@dataclass(frozen=True)
class _LeakageRankCandidate:
    rank: int
    score: float
    teacher_residual_r2: float
    invariant_predictability_r2: float
    retained_variance_fraction: float
    removed_variance_fraction: float
    eligible: bool


class TrainSetCrossFittedTeacherResidualProjectionCCAAligner(
    TrainSetChronologicalResidualizedReducedRankCCAAligner
):
    """Zero-preserving cross-fitted projection aligner for Stage III V10.

    V9 removed invariant-explainable drift with an additive residualizer
    ``delta - f(phi_i)``. That can mix stable phi_i structure into the served
    drift and only preserves physical zero drift through masking. V10 instead
    estimates a leakage subspace in delta space and serves a linear projection
    ``delta @ (I - B B^T)``. The transform is algebraically zero-preserving:
    if ``delta == 0``, the cleaned drift and final CCA coordinates are zero.
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

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetCrossFittedTeacherResidualProjectionCCAAligner":
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
            raise RuntimeError("V10 aligner must be fit before residualize_delta")
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
            raise RuntimeError("V10 aligner must be fit before transform_delta")
        projection = self.projection.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        sign = self.sign.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        active_mask = self.active_mask.to(device=delta_phi_v.device)
        transformed = (delta_phi_v @ projection) * sign.unsqueeze(0)
        transformed[:, ~active_mask] = 0.0
        zero_mask = torch.linalg.vector_norm(delta_phi_v, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(transformed), transformed)

    def diagnostic_summary(self) -> dict[str, int | float | bool | str]:
        summary = super().diagnostic_summary()
        summary["alignment_method"] = "cross_fitted_teacher_projection_rrcca_v10"
        summary["selected_leakage_rank"] = int(self.selected_leakage_rank)
        summary["max_leakage_rank"] = int(self.max_leakage_rank)
        summary["oof_projection_sample_count"] = int(self.oof_projection_sample_count)
        summary["oof_projection_fold_count"] = int(self.oof_projection_fold_count)
        summary["teacher_residual_r2"] = float(self.teacher_residual_r2)
        summary["selected_invariant_predictability_r2"] = float(self.selected_invariant_predictability_r2)
        summary["selected_retained_variance_fraction"] = float(self.selected_retained_variance_fraction)
        summary["selected_removed_variance_fraction"] = float(self.selected_removed_variance_fraction)
        if self.leakage_rank_candidates:
            summary["leakage_rank_candidate_count"] = int(len(self.leakage_rank_candidates))
            summary["leakage_rank_best_score"] = float(
                max(candidate.score for candidate in self.leakage_rank_candidates)
            )
        return summary

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self.leakage_basis is None:
            raise RuntimeError("V10 aligner must be fit before serialization")
        state["class_name"] = "TrainSetCrossFittedTeacherResidualProjectionCCAAligner"
        state["method"] = "cross_fitted_teacher_projection_rrcca_v10"
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
        state["oof_projection_sample_count"] = self.oof_projection_sample_count
        state["oof_projection_fold_count"] = self.oof_projection_fold_count
        state["selected_teacher_residual_r2"] = self.selected_teacher_residual_r2
        state["selected_invariant_predictability_r2"] = self.selected_invariant_predictability_r2
        state["selected_retained_variance_fraction"] = self.selected_retained_variance_fraction
        state["selected_removed_variance_fraction"] = self.selected_removed_variance_fraction
        state["teacher_residual_r2"] = self.teacher_residual_r2
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

    def load_state_dict(self, state: dict) -> "TrainSetCrossFittedTeacherResidualProjectionCCAAligner":
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
        self.oof_projection_sample_count = int(state.get("oof_projection_sample_count", 0))
        self.oof_projection_fold_count = int(state.get("oof_projection_fold_count", 0))
        self.selected_teacher_residual_r2 = float(state.get("selected_teacher_residual_r2", 0.0))
        self.selected_invariant_predictability_r2 = float(state.get("selected_invariant_predictability_r2", 0.0))
        self.selected_retained_variance_fraction = float(state.get("selected_retained_variance_fraction", 1.0))
        self.selected_removed_variance_fraction = float(state.get("selected_removed_variance_fraction", 0.0))
        self.teacher_residual_r2 = float(state.get("teacher_residual_r2", self.selected_teacher_residual_r2))
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
