from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.stage3_alignment_v7 import _pearson_corr


@dataclass(frozen=True)
class _LeakageCandidateV12:
    rank: int
    score: float
    retained_variance_fraction: float
    invariant_predictability_r2: float
    residual_predictability_r2: float
    oof_transfer_mse_gain: float
    eligible: bool


class TrainOnlyOutcomeAlignedTransportAlignerV12:
    """Train-only outcome-aligned transport for Stage III drift signals.

    V12 addresses the raw ``phi_v - phi_i`` coordinate mismatch by first
    learning a shared train-only latent anchor from both embeddings. It then
    removes invariant-predictable leakage in that shared space and selects
    transport dimensions only if they predict forecast residual directions on
    chronological calibration blocks. The served transform is zero-preserving:
    identical ``phi_i`` and ``phi_v`` always return an all-zero drift vector.
    """

    protocol_version = "v12_outcome_aligned_transport"

    def __init__(
        self,
        *,
        anchor_rank: int | None = None,
        max_rank: int | None = None,
        max_leakage_rank: int = 16,
        leakage_fold_count: int = 5,
        min_prefix_samples: int = 16,
        inner_fit_fraction: float = 0.75,
        ridge: float = 1e-3,
        eps: float = 1e-8,
        min_retained_variance_fraction: float = 0.05,
        max_invariant_predictability_r2: float = 0.98,
        min_residual_predictability_r2: float = -1.0,
        min_fit_correlation: float = 0.0,
        min_calibration_r2: float = 0.0025,
        direct_block_count: int = 5,
        min_block_transfer_r2: float = 0.0,
        min_block_sign_agreement: float = 0.55,
        bootstrap_samples: int = 64,
        bootstrap_seed: int = 1729,
        min_bootstrap_sign_stability: float = 0.0,
        leakage_penalty: float = 0.1,
        leakage_removal_reward: float = 0.01,
        transfer_gain_reward: float = 1.0,
        rank_penalty: float = 0.001,
        reliability_gate_enabled: bool = True,
        reliability_quantile: float = 0.95,
        min_reliable_fraction: float = 0.1,
    ) -> None:
        if anchor_rank is not None and anchor_rank <= 0:
            raise ValueError("anchor_rank must be positive when provided")
        if max_rank is not None and max_rank <= 0:
            raise ValueError("max_rank must be positive when provided")
        if max_leakage_rank < 0:
            raise ValueError("max_leakage_rank must be non-negative")
        if leakage_fold_count <= 1:
            raise ValueError("leakage_fold_count must be greater than one")
        if min_prefix_samples < 2:
            raise ValueError("min_prefix_samples must be at least two")
        if not 0.5 <= inner_fit_fraction <= 0.9:
            raise ValueError("inner_fit_fraction must be in [0.5, 0.9]")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if direct_block_count < 1:
            raise ValueError("direct_block_count must be positive")
        if not 0.0 <= min_block_sign_agreement <= 1.0:
            raise ValueError("min_block_sign_agreement must be in [0, 1]")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")
        if not 0.0 < reliability_quantile <= 1.0:
            raise ValueError("reliability_quantile must be in (0, 1]")
        if not 0.0 <= min_reliable_fraction <= 1.0:
            raise ValueError("min_reliable_fraction must be in [0, 1]")

        self.anchor_rank_request = anchor_rank
        self.max_rank = max_rank
        self.max_leakage_rank = int(max_leakage_rank)
        self.leakage_fold_count = int(leakage_fold_count)
        self.min_prefix_samples = int(min_prefix_samples)
        self.inner_fit_fraction = float(inner_fit_fraction)
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.min_retained_variance_fraction = float(min_retained_variance_fraction)
        self.max_invariant_predictability_r2 = float(max_invariant_predictability_r2)
        self.min_residual_predictability_r2 = float(min_residual_predictability_r2)
        self.min_fit_correlation = float(min_fit_correlation)
        self.min_calibration_r2 = float(min_calibration_r2)
        self.direct_block_count = int(direct_block_count)
        self.min_block_transfer_r2 = float(min_block_transfer_r2)
        self.min_block_sign_agreement = float(min_block_sign_agreement)
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.min_bootstrap_sign_stability = float(min_bootstrap_sign_stability)
        self.leakage_penalty = float(leakage_penalty)
        self.leakage_removal_reward = float(leakage_removal_reward)
        self.transfer_gain_reward = float(transfer_gain_reward)
        self.rank_penalty = float(rank_penalty)
        self.reliability_gate_enabled = bool(reliability_gate_enabled)
        self.reliability_quantile = float(reliability_quantile)
        self.min_reliable_fraction = float(min_reliable_fraction)

        self.fit_split = ""
        self.fit_sample_count = 0
        self.inner_fit_sample_count = 0
        self.inner_calib_sample_count = 0
        self.anchor_rank = 0
        self.selected_leakage_rank = 0
        self.active_rank = 0
        self.alignment_gate_pass = False
        self.alignment_gate_failure_reasons = "not_fit"
        self.oof_transfer_mse_gain = 0.0
        self.oof_transfer_mae_gain = 0.0
        self.selected_retained_variance_fraction = 0.0
        self.selected_invariant_predictability_r2 = 0.0
        self.selected_residual_predictability_r2 = 0.0
        self.reliability_gate_active = False
        self.reliability_gate_reason = "not_fit"
        self.reliable_fraction = 0.0
        self.reliable_norm_min = 0.0
        self.reliable_norm_max = float("inf")
        self.leakage_candidates: list[_LeakageCandidateV12] = []

        self.anchor_basis: torch.Tensor | None = None
        self.leakage_basis: torch.Tensor | None = None
        self.transport_projection: torch.Tensor | None = None
        self.residual_projection: torch.Tensor | None = None
        self.sign: torch.Tensor | None = None
        self.fit_correlations: torch.Tensor | None = None
        self.calibration_r2: torch.Tensor | None = None
        self.direct_block_min_r2: torch.Tensor | None = None
        self.direct_block_sign_agreement: torch.Tensor | None = None
        self.bootstrap_sign_stability: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
    ) -> "TrainOnlyOutcomeAlignedTransportAlignerV12":
        if split != "train":
            raise ValueError("V12 outcome-aligned transport must be fit on the train split")
        self._validate_fit_tensors(phi_i_train, phi_v_train, y_base_train, y_train)
        phi_i = phi_i_train.detach().float()
        phi_v = phi_v_train.detach().float()
        residual = (y_train.detach().float() - y_base_train.detach().float()).reshape(phi_i.shape[0], -1)

        self.anchor_basis = self._fit_shared_anchor(phi_i, phi_v)
        phi_i_anchor = phi_i @ self.anchor_basis
        raw_delta_anchor = (phi_v - phi_i) @ self.anchor_basis
        self.anchor_rank = int(self.anchor_basis.shape[1])

        selected = self._select_leakage_rank(phi_i_anchor, raw_delta_anchor, residual)
        self.selected_leakage_rank = selected.rank
        self.selected_retained_variance_fraction = selected.retained_variance_fraction
        self.selected_invariant_predictability_r2 = selected.invariant_predictability_r2
        self.selected_residual_predictability_r2 = selected.residual_predictability_r2
        self.oof_transfer_mse_gain = selected.oof_transfer_mse_gain
        self.leakage_basis = self._fit_leakage_basis(phi_i_anchor, raw_delta_anchor, selected.rank)
        cleaned_oof = self._cross_fitted_clean(phi_i_anchor, raw_delta_anchor, selected.rank)

        fit_stop = self._inner_fit_stop(phi_i.shape[0])
        fit_delta = cleaned_oof[:fit_stop]
        fit_residual = residual[:fit_stop]
        calib_delta = cleaned_oof[fit_stop:]
        calib_residual = residual[fit_stop:]
        self.fit_sample_count = int(phi_i.shape[0])
        self.inner_fit_sample_count = int(fit_stop)
        self.inner_calib_sample_count = int(phi_i.shape[0] - fit_stop)

        self._fit_transport(fit_delta, fit_residual)
        self._select_active_dimensions(calib_delta, calib_residual)
        self._calibrate_reliability_gate(calib_delta, calib_residual)
        return self

    def transform_phi_pair(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        if phi_i.shape != phi_v.shape:
            raise ValueError("phi_i and phi_v must have the same shape")
        raw_delta = phi_v.detach().float() - phi_i.detach().float()
        return self.transform_delta(raw_delta)

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        self._ensure_fit()
        delta = delta_phi_v.detach().float()
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        anchor_basis = self.anchor_basis.to(device=delta.device, dtype=delta.dtype)
        leakage_basis = self.leakage_basis.to(device=delta.device, dtype=delta.dtype)
        projection = self.transport_projection.to(device=delta.device, dtype=delta.dtype)
        sign = self.sign.to(device=delta.device, dtype=delta.dtype)
        active_mask = self.active_mask.to(device=delta.device)

        delta_anchor = delta @ anchor_basis
        cleaned = self._project_with_basis(delta_anchor, leakage_basis)
        transformed = (cleaned @ projection) * sign.unsqueeze(0)
        transformed[:, ~active_mask] = 0.0
        if self.reliability_gate_enabled and self.reliability_gate_active:
            norm = torch.linalg.vector_norm(transformed, dim=1)
            reliable = (norm >= self.reliable_norm_min) & (norm <= self.reliable_norm_max)
            transformed = torch.where(reliable.unsqueeze(1), transformed, torch.zeros_like(transformed))
        zero_mask = torch.linalg.vector_norm(delta, dim=1, keepdim=True) <= self.eps
        return torch.where(zero_mask, torch.zeros_like(transformed), transformed)

    def diagnostic_summary(self) -> dict[str, int | float | bool | str]:
        return {
            "alignment_method": "outcome_aligned_transport_v12",
            "protocol_version": self.protocol_version,
            "fit_split": self.fit_split,
            "fit_sample_count": int(self.fit_sample_count),
            "inner_fit_sample_count": int(self.inner_fit_sample_count),
            "inner_calib_sample_count": int(self.inner_calib_sample_count),
            "anchor_rank": int(self.anchor_rank),
            "selected_leakage_rank": int(self.selected_leakage_rank),
            "active_rank": int(self.active_rank),
            "alignment_gate_pass": bool(self.alignment_gate_pass),
            "alignment_gate_failure_reasons": self.alignment_gate_failure_reasons,
            "selected_retained_variance_fraction": float(self.selected_retained_variance_fraction),
            "selected_invariant_predictability_r2": float(self.selected_invariant_predictability_r2),
            "selected_residual_predictability_r2": float(self.selected_residual_predictability_r2),
            "oof_transfer_mse_gain": float(self.oof_transfer_mse_gain),
            "oof_transfer_mae_gain": float(self.oof_transfer_mae_gain),
            "active_min_calibration_r2": self._active_min(self.calibration_r2),
            "active_min_block_transfer_r2": self._active_min(self.direct_block_min_r2),
            "active_min_block_sign_agreement": self._active_min(self.direct_block_sign_agreement),
            "active_min_bootstrap_sign_stability": self._active_min(self.bootstrap_sign_stability),
            "reliability_gate_enabled": bool(self.reliability_gate_enabled),
            "reliability_gate_active": bool(self.reliability_gate_active),
            "reliability_gate_reason": self.reliability_gate_reason,
            "reliable_fraction": float(self.reliable_fraction),
            "leakage_candidate_count": int(len(self.leakage_candidates)),
        }

    def state_dict(self) -> dict[str, Any]:
        self._ensure_fit()
        return {
            "class_name": self.__class__.__name__,
            "protocol_version": self.protocol_version,
            "config": {
                "anchor_rank": self.anchor_rank_request,
                "max_rank": self.max_rank,
                "max_leakage_rank": self.max_leakage_rank,
                "leakage_fold_count": self.leakage_fold_count,
                "min_prefix_samples": self.min_prefix_samples,
                "inner_fit_fraction": self.inner_fit_fraction,
                "ridge": self.ridge,
                "eps": self.eps,
                "min_retained_variance_fraction": self.min_retained_variance_fraction,
                "max_invariant_predictability_r2": self.max_invariant_predictability_r2,
                "min_residual_predictability_r2": self.min_residual_predictability_r2,
                "min_fit_correlation": self.min_fit_correlation,
                "min_calibration_r2": self.min_calibration_r2,
                "direct_block_count": self.direct_block_count,
                "min_block_transfer_r2": self.min_block_transfer_r2,
                "min_block_sign_agreement": self.min_block_sign_agreement,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
                "min_bootstrap_sign_stability": self.min_bootstrap_sign_stability,
                "leakage_penalty": self.leakage_penalty,
                "leakage_removal_reward": self.leakage_removal_reward,
                "transfer_gain_reward": self.transfer_gain_reward,
                "rank_penalty": self.rank_penalty,
                "reliability_gate_enabled": self.reliability_gate_enabled,
                "reliability_quantile": self.reliability_quantile,
                "min_reliable_fraction": self.min_reliable_fraction,
            },
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_sample_count,
            "inner_fit_sample_count": self.inner_fit_sample_count,
            "inner_calib_sample_count": self.inner_calib_sample_count,
            "anchor_rank": self.anchor_rank,
            "selected_leakage_rank": self.selected_leakage_rank,
            "active_rank": self.active_rank,
            "alignment_gate_pass": self.alignment_gate_pass,
            "alignment_gate_failure_reasons": self.alignment_gate_failure_reasons,
            "oof_transfer_mse_gain": self.oof_transfer_mse_gain,
            "oof_transfer_mae_gain": self.oof_transfer_mae_gain,
            "selected_retained_variance_fraction": self.selected_retained_variance_fraction,
            "selected_invariant_predictability_r2": self.selected_invariant_predictability_r2,
            "selected_residual_predictability_r2": self.selected_residual_predictability_r2,
            "reliability_gate_active": self.reliability_gate_active,
            "reliability_gate_reason": self.reliability_gate_reason,
            "reliable_fraction": self.reliable_fraction,
            "reliable_norm_min": self.reliable_norm_min,
            "reliable_norm_max": self.reliable_norm_max,
            "anchor_basis": self.anchor_basis.detach().clone(),
            "leakage_basis": self.leakage_basis.detach().clone(),
            "transport_projection": self.transport_projection.detach().clone(),
            "residual_projection": self.residual_projection.detach().clone(),
            "sign": self.sign.detach().clone(),
            "fit_correlations": self.fit_correlations.detach().clone(),
            "calibration_r2": self.calibration_r2.detach().clone(),
            "direct_block_min_r2": self.direct_block_min_r2.detach().clone(),
            "direct_block_sign_agreement": self.direct_block_sign_agreement.detach().clone(),
            "bootstrap_sign_stability": self.bootstrap_sign_stability.detach().clone(),
            "active_mask": self.active_mask.detach().clone(),
            "leakage_candidates": [candidate.__dict__ for candidate in self.leakage_candidates],
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyOutcomeAlignedTransportAlignerV12":
        config = dict(state.get("config", {}))
        self.__init__(**config)
        self.fit_split = str(state.get("fit_split", "train"))
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        self.inner_fit_sample_count = int(state.get("inner_fit_sample_count", 0))
        self.inner_calib_sample_count = int(state.get("inner_calib_sample_count", 0))
        self.anchor_rank = int(state.get("anchor_rank", 0))
        self.selected_leakage_rank = int(state.get("selected_leakage_rank", 0))
        self.active_rank = int(state.get("active_rank", 0))
        self.alignment_gate_pass = bool(state.get("alignment_gate_pass", False))
        self.alignment_gate_failure_reasons = str(state.get("alignment_gate_failure_reasons", "loaded"))
        self.oof_transfer_mse_gain = float(state.get("oof_transfer_mse_gain", 0.0))
        self.oof_transfer_mae_gain = float(state.get("oof_transfer_mae_gain", 0.0))
        self.selected_retained_variance_fraction = float(state.get("selected_retained_variance_fraction", 0.0))
        self.selected_invariant_predictability_r2 = float(state.get("selected_invariant_predictability_r2", 0.0))
        self.selected_residual_predictability_r2 = float(state.get("selected_residual_predictability_r2", 0.0))
        self.reliability_gate_active = bool(state.get("reliability_gate_active", False))
        self.reliability_gate_reason = str(state.get("reliability_gate_reason", "loaded"))
        self.reliable_fraction = float(state.get("reliable_fraction", 0.0))
        self.reliable_norm_min = float(state.get("reliable_norm_min", 0.0))
        self.reliable_norm_max = float(state.get("reliable_norm_max", float("inf")))
        self.anchor_basis = torch.as_tensor(state["anchor_basis"]).detach().float()
        self.leakage_basis = torch.as_tensor(state["leakage_basis"]).detach().float()
        self.transport_projection = torch.as_tensor(state["transport_projection"]).detach().float()
        self.residual_projection = torch.as_tensor(state["residual_projection"]).detach().float()
        self.sign = torch.as_tensor(state["sign"]).detach().float()
        self.fit_correlations = torch.as_tensor(state["fit_correlations"]).detach().float()
        self.calibration_r2 = torch.as_tensor(state["calibration_r2"]).detach().float()
        self.direct_block_min_r2 = torch.as_tensor(state["direct_block_min_r2"]).detach().float()
        self.direct_block_sign_agreement = torch.as_tensor(state["direct_block_sign_agreement"]).detach().float()
        self.bootstrap_sign_stability = torch.as_tensor(state["bootstrap_sign_stability"]).detach().float()
        self.active_mask = torch.as_tensor(state["active_mask"]).detach().bool()
        self.leakage_candidates = [
            _LeakageCandidateV12(
                rank=int(candidate["rank"]),
                score=float(candidate["score"]),
                retained_variance_fraction=float(candidate["retained_variance_fraction"]),
                invariant_predictability_r2=float(candidate["invariant_predictability_r2"]),
                residual_predictability_r2=float(candidate["residual_predictability_r2"]),
                oof_transfer_mse_gain=float(candidate["oof_transfer_mse_gain"]),
                eligible=bool(candidate["eligible"]),
            )
            for candidate in state.get("leakage_candidates", [])
        ]
        return self

    def _validate_fit_tensors(
        self,
        phi_i: torch.Tensor,
        phi_v: torch.Tensor,
        y_base: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if phi_i.ndim != 2 or phi_v.ndim != 2 or phi_i.shape != phi_v.shape:
            raise ValueError("phi_i_train and phi_v_train must have the same [N, D] shape")
        if phi_i.shape[0] < 8:
            raise ValueError("V12 alignment requires at least eight train samples")
        if y_base.shape[0] != phi_i.shape[0] or y.shape[0] != phi_i.shape[0]:
            raise ValueError("y_base_train and y_train must share the train sample count")
        tensors = (phi_i, phi_v, y_base, y)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("V12 alignment received non-finite inputs")

    def _fit_shared_anchor(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([phi_i, phi_v], dim=0)
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        requested = self.anchor_rank_request or phi_i.shape[1]
        rank = min(int(requested), phi_i.shape[1], vh.shape[0])
        if rank <= 0:
            raise ValueError("V12 shared anchor rank collapsed to zero")
        return vh[:rank].T.detach().float()

    def _inner_fit_stop(self, sample_count: int) -> int:
        fit_stop = int(round(sample_count * self.inner_fit_fraction))
        return min(max(fit_stop, 4), sample_count - 3)

    def _candidate_leakage_ranks(self, phi_i_anchor: torch.Tensor, raw_delta_anchor: torch.Tensor) -> list[int]:
        max_by_shape = min(phi_i_anchor.shape[1], raw_delta_anchor.shape[1], max(0, phi_i_anchor.shape[0] - 2))
        return list(range(min(self.max_leakage_rank, max_by_shape) + 1))

    def _select_leakage_rank(
        self,
        phi_i_anchor: torch.Tensor,
        raw_delta_anchor: torch.Tensor,
        residual: torch.Tensor,
    ) -> _LeakageCandidateV12:
        candidates: list[_LeakageCandidateV12] = []
        for rank in self._candidate_leakage_ranks(phi_i_anchor, raw_delta_anchor):
            cleaned = self._cross_fitted_clean(phi_i_anchor, raw_delta_anchor, rank)
            retained = _variance_fraction(cleaned, raw_delta_anchor, self.eps)
            invariant_r2 = _ridge_predictability_r2(phi_i_anchor, cleaned, self.ridge, self.eps)
            residual_r2 = _ridge_predictability_r2(cleaned, residual, self.ridge, self.eps)
            transfer_gain = self._oof_transfer_gain(cleaned, residual)[0]
            eligible = bool(
                retained >= self.min_retained_variance_fraction
                and invariant_r2 <= self.max_invariant_predictability_r2
                and residual_r2 >= self.min_residual_predictability_r2
            )
            score = (
                residual_r2
                + self.transfer_gain_reward * transfer_gain
                + self.leakage_removal_reward * max(0.0, 1.0 - retained)
                - self.leakage_penalty * max(0.0, invariant_r2 - self.max_invariant_predictability_r2)
                - self.rank_penalty * rank
            )
            candidates.append(
                _LeakageCandidateV12(
                    rank=rank,
                    score=float(score),
                    retained_variance_fraction=float(retained),
                    invariant_predictability_r2=float(invariant_r2),
                    residual_predictability_r2=float(residual_r2),
                    oof_transfer_mse_gain=float(transfer_gain),
                    eligible=eligible,
                )
            )
        self.leakage_candidates = candidates
        pool = [candidate for candidate in candidates if candidate.eligible] or candidates
        return max(pool, key=lambda candidate: (candidate.score, -candidate.rank))

    def _chronological_folds(self, sample_count: int) -> list[tuple[int, int]]:
        fold_count = min(self.leakage_fold_count, sample_count)
        base = sample_count // fold_count
        remainder = sample_count % fold_count
        folds: list[tuple[int, int]] = []
        start = 0
        for fold_id in range(fold_count):
            size = base + (1 if fold_id < remainder else 0)
            stop = start + size
            folds.append((start, stop))
            start = stop
        return folds

    def _cross_fitted_clean(
        self,
        phi_i_anchor: torch.Tensor,
        raw_delta_anchor: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        if rank <= 0:
            return raw_delta_anchor.detach().float()
        cleaned = torch.empty_like(raw_delta_anchor)
        for start, stop in self._chronological_folds(raw_delta_anchor.shape[0]):
            if start < self.min_prefix_samples:
                cleaned[start:stop] = raw_delta_anchor[start:stop]
                continue
            basis = self._fit_leakage_basis(phi_i_anchor[:start], raw_delta_anchor[:start], rank)
            cleaned[start:stop] = self._project_with_basis(raw_delta_anchor[start:stop], basis)
        return cleaned.detach().float()

    def _fit_leakage_basis(self, phi_i_anchor: torch.Tensor, raw_delta_anchor: torch.Tensor, rank: int) -> torch.Tensor:
        if rank <= 0:
            return torch.empty(raw_delta_anchor.shape[1], 0, device=raw_delta_anchor.device, dtype=raw_delta_anchor.dtype)
        x = phi_i_anchor - phi_i_anchor.mean(dim=0, keepdim=True)
        weight = _solve_ridge(x, raw_delta_anchor, self.ridge)
        _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        usable = min(int(rank), vh.shape[0], singular_values.numel())
        if usable <= 0:
            return torch.empty(raw_delta_anchor.shape[1], 0, device=raw_delta_anchor.device, dtype=raw_delta_anchor.dtype)
        return vh[:usable].T.detach().float()

    @staticmethod
    def _project_with_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.numel() == 0:
            return delta
        return delta - (delta @ basis) @ basis.T

    def _fit_transport(self, fit_delta: torch.Tensor, fit_residual: torch.Tensor) -> None:
        x = fit_delta.detach().float()
        y = fit_residual.detach().float().reshape(x.shape[0], -1)
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        cross = x_centered.T @ y_centered / float(max(1, x.shape[0] - 1))
        u, singular_values, vh = torch.linalg.svd(cross, full_matrices=True)
        dim = x.shape[1]
        projection = u[:, :dim]
        residual_projection = vh.T
        usable = min(singular_values.numel(), projection.shape[1], residual_projection.shape[1])
        sign = torch.ones(dim, dtype=torch.float32, device=x.device)
        correlations = torch.zeros(dim, dtype=torch.float32, device=x.device)
        if usable > 0:
            x_scores = x_centered @ projection[:, :usable]
            y_scores = y_centered @ residual_projection[:, :usable]
            score_sign = torch.sign(torch.sum(x_scores * y_scores, dim=0))
            sign[:usable] = torch.where(score_sign == 0.0, torch.ones_like(score_sign), score_sign)
            scale = singular_values.max().clamp_min(self.eps)
            correlations[: singular_values.numel()] = torch.clamp(singular_values / scale, 0.0, 1.0)
        self.transport_projection = projection.detach().float()
        self.residual_projection = residual_projection.detach().float()
        self.sign = sign.detach().float()
        self.fit_correlations = correlations.detach().float()

    def _select_active_dimensions(self, calib_delta: torch.Tensor, calib_residual: torch.Tensor) -> None:
        self._ensure_transport()
        dim = int(self.transport_projection.shape[1])
        usable = min(
            dim,
            self.residual_projection.shape[1],
            self.max_rank if self.max_rank is not None else dim,
        )
        active_mask = torch.zeros(dim, dtype=torch.bool)
        calibration_r2 = torch.zeros(dim, dtype=torch.float32)
        block_r2 = torch.zeros(dim, dtype=torch.float32)
        block_sign = torch.zeros(dim, dtype=torch.float32)
        bootstrap_sign = torch.zeros(dim, dtype=torch.float32)
        if calib_delta.shape[0] >= 3 and usable > 0:
            x = calib_delta - calib_delta.mean(dim=0, keepdim=True)
            y = calib_residual.reshape(calib_delta.shape[0], -1)
            y = y - y.mean(dim=0, keepdim=True)
            x_scores = (x @ self.transport_projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            y_scores = (y @ self.residual_projection[:, :usable]) * self.sign[:usable].unsqueeze(0)
            for idx in range(usable):
                corr = _pearson_corr(x_scores[:, idx], y_scores[:, idx], self.eps)
                calibration_r2[idx] = torch.clamp(corr.pow(2), max=1.0).detach().float()
                min_block_r2, min_block_sign = self._block_transfer_metrics(x_scores[:, idx], y_scores[:, idx], corr)
                block_r2[idx] = torch.tensor(min_block_r2, dtype=torch.float32)
                block_sign[idx] = torch.tensor(min_block_sign, dtype=torch.float32)
                bootstrap_sign[idx] = self._bootstrap_sign_stability(x_scores[:, idx], y_scores[:, idx], corr)
                active_mask[idx] = bool(
                    abs(float(self.fit_correlations[idx].item())) >= self.min_fit_correlation
                    and float(calibration_r2[idx].item()) >= self.min_calibration_r2
                    and min_block_r2 >= self.min_block_transfer_r2
                    and min_block_sign >= self.min_block_sign_agreement
                    and float(bootstrap_sign[idx].item()) >= self.min_bootstrap_sign_stability
                )
        self.active_mask = active_mask
        self.active_rank = int(active_mask.sum().item())
        self.calibration_r2 = calibration_r2
        self.direct_block_min_r2 = block_r2
        self.direct_block_sign_agreement = block_sign
        self.bootstrap_sign_stability = bootstrap_sign
        failures: list[str] = []
        if self.active_rank <= 0:
            failures.append("active_rank")
        self.alignment_gate_pass = not failures
        self.alignment_gate_failure_reasons = ";".join(failures)
        z_fit = self._apply_transport(calib_delta)
        self.oof_transfer_mse_gain, self.oof_transfer_mae_gain = self._oof_transfer_gain(z_fit, calib_residual)

    def _apply_transport(self, cleaned_delta: torch.Tensor) -> torch.Tensor:
        self._ensure_transport()
        z = (cleaned_delta @ self.transport_projection) * self.sign.unsqueeze(0)
        if self.active_mask is not None:
            z[:, ~self.active_mask] = 0.0
        return z

    def _block_transfer_metrics(self, x_score: torch.Tensor, y_score: torch.Tensor, reference_corr: torch.Tensor) -> tuple[float, float]:
        r2_values: list[float] = []
        sign_values: list[float] = []
        reference_sign = 1.0 if float(reference_corr.item()) >= 0.0 else -1.0
        for block in self._block_slices(x_score.shape[0]):
            if (block.stop or 0) - (block.start or 0) < 3:
                continue
            corr = _pearson_corr(x_score[block], y_score[block], self.eps)
            r2_values.append(float(torch.clamp(corr.pow(2), max=1.0).item()))
            if abs(float(corr.item())) > self.eps:
                sign_values.append(float((1.0 if float(corr.item()) >= 0.0 else -1.0) == reference_sign))
        return (min(r2_values) if r2_values else 0.0, sum(sign_values) / len(sign_values) if sign_values else 0.0)

    def _block_slices(self, sample_count: int) -> list[slice]:
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

    def _bootstrap_sign_stability(self, x: torch.Tensor, y: torch.Tensor, reference_corr: torch.Tensor) -> torch.Tensor:
        if self.bootstrap_samples <= 0 or x.numel() < 4:
            return torch.tensor(1.0, dtype=torch.float32)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.bootstrap_seed)
        reference_sign = 1.0 if float(reference_corr.item()) >= 0.0 else -1.0
        agree = 0
        for _ in range(self.bootstrap_samples):
            idx = torch.randint(0, x.numel(), (x.numel(),), generator=generator, device=x.device)
            corr = _pearson_corr(x[idx], y[idx], self.eps)
            if abs(float(corr.item())) <= self.eps:
                continue
            agree += int((1.0 if float(corr.item()) >= 0.0 else -1.0) == reference_sign)
        return torch.tensor(float(agree / max(1, self.bootstrap_samples)), dtype=torch.float32)

    def _oof_transfer_gain(self, features: torch.Tensor, residual: torch.Tensor) -> tuple[float, float]:
        if features.shape[0] < 8 or torch.linalg.vector_norm(features).item() <= self.eps:
            return 0.0, 0.0
        fit_stop = self._inner_fit_stop(features.shape[0])
        train_x = features[:fit_stop]
        train_y = residual[:fit_stop]
        calib_x = features[fit_stop:]
        calib_y = residual[fit_stop:]
        if calib_x.numel() == 0:
            return 0.0, 0.0
        weight = _solve_ridge(train_x, train_y, self.ridge)
        pred = calib_x @ weight
        base_mse = torch.mean(calib_y.pow(2)).clamp_min(self.eps)
        corr_mse = torch.mean((calib_y - pred).pow(2))
        base_mae = torch.mean(calib_y.abs()).clamp_min(self.eps)
        corr_mae = torch.mean((calib_y - pred).abs())
        return (
            float(((base_mse - corr_mse) / base_mse).item()),
            float(((base_mae - corr_mae) / base_mae).item()),
        )

    def _calibrate_reliability_gate(self, calib_delta: torch.Tensor, calib_residual: torch.Tensor) -> None:
        del calib_residual
        if not self.reliability_gate_enabled or self.active_rank <= 0:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "disabled_or_no_active_dimensions"
            return
        z = self._apply_transport(calib_delta)
        norm = torch.linalg.vector_norm(z, dim=1)
        reliable = norm > self.eps
        self.reliable_fraction = float(reliable.float().mean().item()) if reliable.numel() else 0.0
        if self.reliable_fraction < self.min_reliable_fraction:
            self.reliability_gate_active = False
            self.reliability_gate_reason = "insufficient_reliable_calibration_samples"
            return
        reliable_norm = norm[reliable]
        self.reliable_norm_min = float(torch.quantile(reliable_norm, 1.0 - self.reliability_quantile).item())
        self.reliable_norm_max = float(torch.quantile(reliable_norm, self.reliability_quantile).item())
        self.reliability_gate_active = True
        self.reliability_gate_reason = "transport_norm_quantile_gate"

    def _active_min(self, values: torch.Tensor | None) -> float:
        if values is None or self.active_mask is None or self.active_rank <= 0:
            return 0.0
        return float(values[self.active_mask].min().item())

    def _ensure_fit(self) -> None:
        if (
            self.anchor_basis is None
            or self.leakage_basis is None
            or self.transport_projection is None
            or self.sign is None
            or self.active_mask is None
        ):
            raise RuntimeError("V12 outcome-aligned transport must be fit before use")

    def _ensure_transport(self) -> None:
        if self.transport_projection is None or self.residual_projection is None or self.sign is None:
            raise RuntimeError("V12 transport projection is not fit")


def _solve_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    system = x.T @ x + float(ridge) * eye
    rhs = x.T @ y
    try:
        return torch.linalg.solve(system, rhs)
    except torch.linalg.LinAlgError:
        for jitter in (1e-6, 1e-4, 1e-2):
            try:
                return torch.linalg.solve(system + float(jitter) * eye, rhs)
            except torch.linalg.LinAlgError:
                continue
        return torch.linalg.pinv(system) @ rhs


def _ridge_predictability_r2(x: torch.Tensor, y: torch.Tensor, ridge: float, eps: float) -> float:
    if x.shape[0] < 3 or torch.linalg.vector_norm(x).item() <= eps:
        return 0.0
    weight = _solve_ridge(x.detach().float(), y.detach().float().reshape(x.shape[0], -1), ridge)
    pred = x.detach().float() @ weight
    target = y.detach().float().reshape(x.shape[0], -1)
    sse = torch.sum((target - pred).pow(2))
    centered = target - target.mean(dim=0, keepdim=True)
    sst = torch.sum(centered.pow(2)).clamp_min(eps)
    return float((1.0 - sse / sst).item())


def _variance_fraction(cleaned: torch.Tensor, raw: torch.Tensor, eps: float) -> float:
    return float((torch.sum(cleaned.pow(2)) / torch.sum(raw.pow(2)).clamp_min(eps)).item())
