from __future__ import annotations

import torch

from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner


class TrainSetTeacherResidualizedReducedRankCCAAligner(
    TrainSetChronologicalResidualizedReducedRankCCAAligner
):
    """Teacher-aware residualized reduced-rank CCA aligner for Stage III V9.

    V7 aligns residual drift transitions to forecast-residual transitions. V9
    keeps the same train-only chronological discipline, but aligns the cleaned
    drift to teacher residual corrections. The invariant residualizer uses a
    slightly richer train-only feature map ``[phi_i, phi_i^2, 1]`` while
    preserving physical zero drift during transform.
    """

    def __init__(
        self,
        *,
        max_invariant_residualizer_rank: int = 16,
        use_quadratic_residualizer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_invariant_residualizer_rank <= 0:
            raise ValueError("max_invariant_residualizer_rank must be positive")
        self.max_invariant_residualizer_rank = int(max_invariant_residualizer_rank)
        self.use_quadratic_residualizer = bool(use_quadratic_residualizer)
        self.residualizer_feature_weight: torch.Tensor | None = None
        self.invariant_mean: torch.Tensor | None = None
        self.invariant_basis: torch.Tensor | None = None
        self.residualizer_include_quadratic = False
        self.teacher_residual_r2 = 0.0

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetTeacherResidualizedReducedRankCCAAligner":
        self._validate_fit_tensors(phi_i_train, phi_v_train, y_base_train, y_train)
        phi_i = phi_i_train.detach().float()
        phi_v = phi_v_train.detach().float()
        y_base = y_base_train.detach().float().reshape(phi_i.shape[0], -1)
        y = y_train.detach().float().reshape(phi_i.shape[0], -1)
        raw_delta = phi_v - phi_i
        teacher_residual = y - y_base
        residualized_delta = self._fit_residualizer(phi_i, raw_delta)
        self.teacher_residual_r2 = self._fit_residual_predictability_r2(residualized_delta, teacher_residual)

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
        self.residual_mean = residualized_delta.mean(dim=0).detach().float()
        self.fit_sample_count = int(phi_i.shape[0])
        self.inner_fit_sample_count = int(fit_delta.shape[0])
        self.inner_calib_sample_count = int(calib_delta.shape[0])
        self.transition_sample_count = int(max(0, fit_delta.shape[0] - 1))
        self._select_active_dimensions(calib_delta, calib_residual)
        return self

    def _fit_residual_predictability_r2(self, residualized_delta: torch.Tensor, teacher_residual: torch.Tensor) -> float:
        ones = torch.ones(residualized_delta.shape[0], 1, device=residualized_delta.device, dtype=residualized_delta.dtype)
        design = torch.cat([residualized_delta, ones], dim=1)
        eye = torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
        eye[-1, -1] = 0.0
        solution = torch.linalg.solve(
            design.T @ design + self.ridge * eye,
            design.T @ teacher_residual,
        )
        predicted = design @ solution
        residual_error = torch.mean((teacher_residual - predicted).pow(2))
        centered = teacher_residual - teacher_residual.mean(dim=0, keepdim=True)
        total = torch.mean(centered.pow(2)).clamp_min(self.eps)
        return float(torch.clamp(1.0 - residual_error / total, min=-1.0, max=1.0).item())

    def _residualizer_features(self, phi_i: torch.Tensor) -> torch.Tensor:
        if self.invariant_mean is None or self.invariant_basis is None:
            raise RuntimeError("V9 aligner invariant residualizer basis is not initialized")
        mean = self.invariant_mean.to(device=phi_i.device, dtype=phi_i.dtype)
        basis = self.invariant_basis.to(device=phi_i.device, dtype=phi_i.dtype)
        scores = (phi_i - mean) @ basis
        parts = [scores]
        if self.residualizer_include_quadratic:
            parts.append(scores.pow(2))
        ones = torch.ones(phi_i.shape[0], 1, device=phi_i.device, dtype=phi_i.dtype)
        parts.append(ones)
        return torch.cat(parts, dim=1)

    def _fit_residualizer(self, phi_i: torch.Tensor, raw_delta: torch.Tensor) -> torch.Tensor:
        centered_phi = phi_i - phi_i.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered_phi, full_matrices=False)
        max_rank_by_sample = max(1, phi_i.shape[0] // 4)
        rank = min(self.max_invariant_residualizer_rank, max_rank_by_sample, phi_i.shape[1], vh.shape[0])
        self.invariant_mean = phi_i.mean(dim=0).detach().float()
        self.invariant_basis = vh[:rank].T.detach().float()
        self.residualizer_include_quadratic = bool(
            self.use_quadratic_residualizer and phi_i.shape[0] >= 4 * (2 * rank + 1)
        )
        design = self._residualizer_features(phi_i)
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
        self.residualizer_feature_weight = solution.detach().float()
        self.residualizer_weight = solution[:-1].detach().float()
        self.residualizer_bias = solution[-1].detach().float()
        self.residualizer_r2 = torch.clamp(1.0 - ss_residual / ss_total, min=-1.0, max=1.0).detach().float()
        return residualized.detach().float()

    def residualize_delta(self, phi_i: torch.Tensor, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.residualizer_feature_weight is None:
            raise RuntimeError("V9 aligner must be fit before residualize_delta")
        weight = self.residualizer_feature_weight.to(device=phi_i.device, dtype=phi_i.dtype)
        design = self._residualizer_features(phi_i)
        residualized = delta_phi_v - design @ weight
        zero_mask = torch.linalg.vector_norm(delta_phi_v, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(residualized), residualized)

    def transform_phi_pair(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        if phi_i.shape != phi_v.shape:
            raise ValueError("phi_i and phi_v must have the same shape")
        raw_delta = phi_v - phi_i
        transformed = super().transform_phi_pair(phi_i, phi_v)
        zero_mask = torch.linalg.vector_norm(raw_delta, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(transformed), transformed)

    def diagnostic_summary(self) -> dict[str, int | float | bool | str]:
        summary = super().diagnostic_summary()
        summary["alignment_method"] = "teacher_residual_rrcca_v9"
        summary["teacher_residual_r2"] = float(self.teacher_residual_r2)
        return summary

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self.residualizer_feature_weight is None:
            raise RuntimeError("V9 aligner must be fit before serialization")
        state["class_name"] = "TrainSetTeacherResidualizedReducedRankCCAAligner"
        state["method"] = "teacher_residual_rrcca_v9"
        state["residualizer_feature_weight"] = self.residualizer_feature_weight.detach().clone()
        state["invariant_mean"] = self.invariant_mean.detach().clone()
        state["invariant_basis"] = self.invariant_basis.detach().clone()
        state["max_invariant_residualizer_rank"] = self.max_invariant_residualizer_rank
        state["use_quadratic_residualizer"] = self.use_quadratic_residualizer
        state["residualizer_include_quadratic"] = self.residualizer_include_quadratic
        state["teacher_residual_r2"] = float(self.teacher_residual_r2)
        return state

    def load_state_dict(self, state: dict) -> "TrainSetTeacherResidualizedReducedRankCCAAligner":
        super().load_state_dict(state)
        self.residualizer_feature_weight = torch.as_tensor(state["residualizer_feature_weight"]).detach().float()
        self.invariant_mean = torch.as_tensor(state["invariant_mean"]).detach().float()
        self.invariant_basis = torch.as_tensor(state["invariant_basis"]).detach().float()
        self.max_invariant_residualizer_rank = int(
            state.get("max_invariant_residualizer_rank", self.max_invariant_residualizer_rank)
        )
        self.use_quadratic_residualizer = bool(state.get("use_quadratic_residualizer", self.use_quadratic_residualizer))
        self.residualizer_include_quadratic = bool(state.get("residualizer_include_quadratic", False))
        self.teacher_residual_r2 = float(state.get("teacher_residual_r2", 0.0))
        return self
