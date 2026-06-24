from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlignedRepresentationOutput:
    phi_i: torch.Tensor
    delta_phi_v: torch.Tensor
    phi_v: torch.Tensor


class SharedResidualFactorEncoder(nn.Module):
    """Shared-trunk invariant/variant representation for Stage III V2.

    The variant embedding is anchored to the invariant embedding:

    ``phi_v = phi_i + delta_phi_v``.

    This makes ``phi_v - phi_i`` a learned residual in the same coordinate
    system, rather than a subtraction between two independent encoders.
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        zero_init_variant: bool = False,
        use_projected_residual: bool = False,
        stop_gradient_projection_anchor: bool = True,
        projection_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.use_projected_residual = use_projected_residual
        self.stop_gradient_projection_anchor = stop_gradient_projection_anchor
        self.projection_eps = projection_eps
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.invariant_head = nn.Linear(hidden_dim, repr_dim)
        self.variant_head = nn.Linear(hidden_dim, repr_dim)
        if zero_init_variant:
            nn.init.zeros_(self.variant_head.weight)
            nn.init.zeros_(self.variant_head.bias)

    def forward(self, x: torch.Tensor) -> AlignedRepresentationOutput:
        h = self.trunk(x.reshape(x.shape[0], -1))
        phi_i = self.invariant_head(h)
        delta_raw = self.variant_head(h)
        if self.use_projected_residual:
            anchor = phi_i.detach() if self.stop_gradient_projection_anchor else phi_i
            denom = anchor.pow(2).sum(dim=1, keepdim=True).clamp_min(self.projection_eps)
            parallel = (delta_raw * anchor).sum(dim=1, keepdim=True) / denom * anchor
            delta_phi_v = delta_raw - parallel
        else:
            delta_phi_v = delta_raw
        phi_v = phi_i + delta_phi_v
        return AlignedRepresentationOutput(
            phi_i=phi_i,
            delta_phi_v=delta_phi_v,
            phi_v=phi_v,
        )


class TrainSetAffineAligner:
    """Train-only ridge affine alignment from legacy ``phi_v`` to ``phi_i``."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.fit_sample_count = 0

    def fit(self, *, phi_i_train: torch.Tensor, phi_v_train: torch.Tensor) -> "TrainSetAffineAligner":
        if phi_i_train.ndim != 2 or phi_v_train.ndim != 2:
            raise ValueError("phi_i_train and phi_v_train must be 2-D")
        if phi_i_train.shape[0] != phi_v_train.shape[0]:
            raise ValueError("phi_i_train and phi_v_train must have the same sample count")
        if phi_i_train.shape[0] < 2:
            raise ValueError("at least two samples are required to fit aligner")

        x = phi_v_train.detach().float()
        y = phi_i_train.detach().float()
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        design = torch.cat([x, ones], dim=1)
        eye = torch.eye(design.shape[1], device=x.device, dtype=x.dtype)
        eye[-1, -1] = 0.0
        solution = torch.linalg.solve(
            design.T @ design + self.ridge * eye,
            design.T @ y,
        )
        self.weight = solution[:-1].detach()
        self.bias = solution[-1].detach()
        self.fit_sample_count = int(x.shape[0])
        return self

    def transform_phi_v(self, phi_v: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("TrainSetAffineAligner must be fit before transform")
        weight = self.weight.to(device=phi_v.device, dtype=phi_v.dtype)
        bias = self.bias.to(device=phi_v.device, dtype=phi_v.dtype)
        return phi_v @ weight + bias


class TrainSetWhitenedProcrustesAligner:
    """Train-only CORAL/whitening plus orthogonal Procrustes phi-space alignment.

    This deliberately constrains the map from ``phi_v`` to ``phi_i`` to a
    covariance match plus rotation. It is less flexible than a free affine map
    and therefore less prone to split-specific shear overfitting.
    """

    def __init__(self, ridge: float = 1e-4, eps: float = 1e-6) -> None:
        self.ridge = ridge
        self.eps = eps
        self.phi_i_mean: torch.Tensor | None = None
        self.phi_v_mean: torch.Tensor | None = None
        self.phi_v_whitener: torch.Tensor | None = None
        self.phi_i_whitener: torch.Tensor | None = None
        self.phi_i_recolor: torch.Tensor | None = None
        self.rotation: torch.Tensor | None = None
        self.fit_sample_count = 0

    def fit(self, *, phi_i_train: torch.Tensor, phi_v_train: torch.Tensor) -> "TrainSetWhitenedProcrustesAligner":
        if phi_i_train.ndim != 2 or phi_v_train.ndim != 2:
            raise ValueError("phi_i_train and phi_v_train must be 2-D")
        if phi_i_train.shape != phi_v_train.shape:
            raise ValueError("phi_i_train and phi_v_train must have the same shape")
        if phi_i_train.shape[0] < 3:
            raise ValueError("at least three samples are required to fit whitened Procrustes aligner")

        phi_i = phi_i_train.detach().float()
        phi_v = phi_v_train.detach().float()
        self.phi_i_mean = phi_i.mean(dim=0)
        self.phi_v_mean = phi_v.mean(dim=0)
        centered_i = phi_i - self.phi_i_mean
        centered_v = phi_v - self.phi_v_mean
        denom = float(max(1, phi_i.shape[0] - 1))
        dim = phi_i.shape[1]
        eye = torch.eye(dim, device=phi_i.device, dtype=phi_i.dtype)
        cov_i = centered_i.T @ centered_i / denom + self.ridge * eye
        cov_v = centered_v.T @ centered_v / denom + self.ridge * eye
        phi_v_whitener = _symmetric_invsqrt(cov_v, self.eps)
        phi_i_whitener = _symmetric_invsqrt(cov_i, self.eps)
        phi_i_recolor = _symmetric_sqrt(cov_i, self.eps)
        v_white = centered_v @ phi_v_whitener
        i_white = centered_i @ phi_i_whitener
        u, _, vh = torch.linalg.svd(v_white.T @ i_white, full_matrices=False)
        self.rotation = (u @ vh).detach().float()
        self.phi_v_whitener = phi_v_whitener.detach().float()
        self.phi_i_whitener = phi_i_whitener.detach().float()
        self.phi_i_recolor = phi_i_recolor.detach().float()
        self.fit_sample_count = int(phi_i.shape[0])
        return self

    def transform_phi_v(self, phi_v: torch.Tensor) -> torch.Tensor:
        if (
            self.phi_i_mean is None
            or self.phi_v_mean is None
            or self.phi_v_whitener is None
            or self.phi_i_recolor is None
            or self.rotation is None
        ):
            raise RuntimeError("TrainSetWhitenedProcrustesAligner must be fit before transform")
        phi_i_mean = self.phi_i_mean.to(device=phi_v.device, dtype=phi_v.dtype)
        phi_v_mean = self.phi_v_mean.to(device=phi_v.device, dtype=phi_v.dtype)
        phi_v_whitener = self.phi_v_whitener.to(device=phi_v.device, dtype=phi_v.dtype)
        phi_i_recolor = self.phi_i_recolor.to(device=phi_v.device, dtype=phi_v.dtype)
        rotation = self.rotation.to(device=phi_v.device, dtype=phi_v.dtype)
        return ((phi_v - phi_v_mean) @ phi_v_whitener @ rotation @ phi_i_recolor) + phi_i_mean

    def state_dict(self) -> dict[str, torch.Tensor | int | float | str]:
        if (
            self.phi_i_mean is None
            or self.phi_v_mean is None
            or self.phi_v_whitener is None
            or self.phi_i_whitener is None
            or self.phi_i_recolor is None
            or self.rotation is None
        ):
            raise RuntimeError("TrainSetWhitenedProcrustesAligner must be fit before serialization")
        return {
            "class_name": "TrainSetWhitenedProcrustesAligner",
            "phi_i_mean": self.phi_i_mean.detach().clone(),
            "phi_v_mean": self.phi_v_mean.detach().clone(),
            "phi_v_whitener": self.phi_v_whitener.detach().clone(),
            "phi_i_whitener": self.phi_i_whitener.detach().clone(),
            "phi_i_recolor": self.phi_i_recolor.detach().clone(),
            "rotation": self.rotation.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "ridge": self.ridge,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float | str]) -> "TrainSetWhitenedProcrustesAligner":
        self.phi_i_mean = torch.as_tensor(state["phi_i_mean"]).detach().float()
        self.phi_v_mean = torch.as_tensor(state["phi_v_mean"]).detach().float()
        self.phi_v_whitener = torch.as_tensor(state["phi_v_whitener"]).detach().float()
        self.phi_i_whitener = torch.as_tensor(state["phi_i_whitener"]).detach().float()
        self.phi_i_recolor = torch.as_tensor(state["phi_i_recolor"]).detach().float()
        self.rotation = torch.as_tensor(state["rotation"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        return self


def _project_delta_orthogonal_to_anchor(
    delta: torch.Tensor,
    anchor: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    denom = anchor.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    parallel = (delta * anchor).sum(dim=1, keepdim=True) / denom * anchor
    return delta - parallel


class TrainSetDeltaStandardizer:
    """Train-only standardization for aligned residual drift vectors."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.fit_sample_count = 0

    def fit(self, delta_phi_v_train: torch.Tensor) -> "TrainSetDeltaStandardizer":
        if delta_phi_v_train.ndim != 2:
            raise ValueError("delta_phi_v_train must be 2-D")
        if delta_phi_v_train.shape[0] < 2:
            raise ValueError("at least two samples are required to fit delta standardizer")
        x = delta_phi_v_train.detach().float()
        self.mean = x.mean(dim=0)
        self.std = torch.clamp(x.std(dim=0, unbiased=False), min=self.eps)
        self.fit_sample_count = int(x.shape[0])
        return self

    def transform(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("TrainSetDeltaStandardizer must be fit before transform")
        mean = self.mean.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        std = self.std.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        return (delta_phi_v - mean) / std

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        if self.mean is None or self.std is None:
            raise RuntimeError("TrainSetDeltaStandardizer must be fit before serialization")
        return {
            "mean": self.mean.detach().clone(),
            "std": self.std.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> "TrainSetDeltaStandardizer":
        self.mean = torch.as_tensor(state["mean"]).detach().float()
        self.std = torch.as_tensor(state["std"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.eps = float(state.get("eps", self.eps))
        return self


class TrainSetResidualDriftCanonicalizer:
    """Train-only canonical coordinates for residual drift transitions.

    The fit step uses only ordered training samples. It builds a canonical
    basis from the cross-covariance between residual drift transitions
    ``delta_phi_v[t] - delta_phi_v[t-1]`` and forecast-residual transitions.
    Validation/test splits only call ``transform_delta`` with frozen state.
    """

    def __init__(self, ridge: float = 1e-3, eps: float = 1e-6) -> None:
        self.ridge = ridge
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.sign: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.transition_sample_count = 0

    def fit(
        self,
        *,
        delta_phi_v_train: torch.Tensor,
        y_inv_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetResidualDriftCanonicalizer":
        if delta_phi_v_train.ndim != 2:
            raise ValueError("delta_phi_v_train must be 2-D")
        if y_inv_train.shape[0] != delta_phi_v_train.shape[0] or y_train.shape[0] != delta_phi_v_train.shape[0]:
            raise ValueError("delta_phi_v_train, y_inv_train, and y_train must have the same sample count")
        if delta_phi_v_train.shape[0] < 3:
            raise ValueError("at least three ordered samples are required to fit residual drift canonicalizer")

        delta = delta_phi_v_train.detach().float()
        residual = (y_train.detach().float() - y_inv_train.detach().float()).reshape(delta.shape[0], -1)
        transition = delta[1:] - delta[:-1]
        residual_transition = residual[1:] - residual[:-1]
        self.mean = delta.mean(dim=0)
        self.std = torch.clamp(transition.std(dim=0, unbiased=False), min=self.eps)
        transition_z = (transition - transition.mean(dim=0, keepdim=True)) / self.std
        residual_z = residual_transition - residual_transition.mean(dim=0, keepdim=True)
        residual_std = torch.clamp(residual_z.std(dim=0, unbiased=False, keepdim=True), min=self.eps)
        residual_z = residual_z / residual_std
        cross = transition_z.T @ residual_z / float(max(1, transition_z.shape[0] - 1))
        if float(torch.linalg.vector_norm(cross).item()) <= self.eps:
            self.basis = torch.eye(delta.shape[1], dtype=torch.float32)
            self.sign = torch.ones(delta.shape[1], dtype=torch.float32)
        else:
            ridge_eye = self.ridge * torch.eye(cross.shape[0], dtype=cross.dtype, device=cross.device)
            left = cross @ cross.T + ridge_eye
            eigvals, eigvecs = torch.linalg.eigh(left)
            order = torch.argsort(eigvals, descending=True)
            basis = eigvecs[:, order]
            sign_scores = torch.sum(cross.T @ basis, dim=0)
            sign = torch.where(sign_scores < 0.0, -torch.ones_like(sign_scores), torch.ones_like(sign_scores))
            self.basis = basis.detach().float()
            self.sign = sign.detach().float()
        self.fit_sample_count = int(delta.shape[0])
        self.transition_sample_count = int(transition.shape[0])
        return self

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or self.basis is None or self.sign is None:
            raise RuntimeError("TrainSetResidualDriftCanonicalizer must be fit before transform")
        mean = self.mean.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        std = self.std.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        basis = self.basis.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        sign = self.sign.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        return ((delta_phi_v - mean) / std) @ (basis * sign.unsqueeze(0))

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        if self.mean is None or self.std is None or self.basis is None or self.sign is None:
            raise RuntimeError("TrainSetResidualDriftCanonicalizer must be fit before serialization")
        return {
            "mean": self.mean.detach().clone(),
            "std": self.std.detach().clone(),
            "basis": self.basis.detach().clone(),
            "sign": self.sign.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "transition_sample_count": self.transition_sample_count,
            "ridge": self.ridge,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> "TrainSetResidualDriftCanonicalizer":
        self.mean = torch.as_tensor(state["mean"]).detach().float()
        self.std = torch.as_tensor(state["std"]).detach().float()
        self.basis = torch.as_tensor(state["basis"]).detach().float()
        self.sign = torch.as_tensor(state["sign"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.transition_sample_count = int(state["transition_sample_count"])
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        return self


def _symmetric_invsqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T


def _symmetric_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T


class TrainSetTemporalResidualCCAAligner:
    """Train-only CCA coordinates for deployment-time residual drift.

    The older residual canonicalizer uses a PLS-like cross-covariance basis.
    This aligner is stricter: it ridge-whitens both residual-drift transitions
    and forecast-residual transitions, then uses CCA directions to define a
    frozen drift coordinate system. Validation and test only call
    ``transform_delta`` with the train-fitted projection.
    """

    def __init__(
        self,
        ridge: float = 1e-3,
        eps: float = 1e-6,
        scale_by_correlation: bool = False,
    ) -> None:
        self.ridge = ridge
        self.eps = eps
        self.scale_by_correlation = scale_by_correlation
        self.mean: torch.Tensor | None = None
        self.transition_mean: torch.Tensor | None = None
        self.projection: torch.Tensor | None = None
        self.residual_projection: torch.Tensor | None = None
        self.sign: torch.Tensor | None = None
        self.canonical_correlations: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.transition_sample_count = 0

    def fit(
        self,
        *,
        delta_phi_v_train: torch.Tensor,
        y_inv_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetTemporalResidualCCAAligner":
        if delta_phi_v_train.ndim != 2:
            raise ValueError("delta_phi_v_train must be 2-D")
        if y_inv_train.shape[0] != delta_phi_v_train.shape[0] or y_train.shape[0] != delta_phi_v_train.shape[0]:
            raise ValueError("delta_phi_v_train, y_inv_train, and y_train must have the same sample count")
        if delta_phi_v_train.shape[0] < 3:
            raise ValueError("at least three ordered samples are required to fit temporal residual CCA aligner")

        delta = delta_phi_v_train.detach().float()
        residual = (y_train.detach().float() - y_inv_train.detach().float()).reshape(delta.shape[0], -1)
        x_transition = delta[1:] - delta[:-1]
        y_transition = residual[1:] - residual[:-1]
        x_centered = x_transition - x_transition.mean(dim=0, keepdim=True)
        y_centered = y_transition - y_transition.mean(dim=0, keepdim=True)
        n = max(1, x_centered.shape[0] - 1)
        x_dim = x_centered.shape[1]
        y_dim = y_centered.shape[1]

        sxx = x_centered.T @ x_centered / float(n)
        syy = y_centered.T @ y_centered / float(n)
        sxx = sxx + self.ridge * torch.eye(x_dim, device=sxx.device, dtype=sxx.dtype)
        syy = syy + self.ridge * torch.eye(y_dim, device=syy.device, dtype=syy.dtype)
        x_invsqrt = _symmetric_invsqrt(sxx, self.eps)
        y_invsqrt = _symmetric_invsqrt(syy, self.eps)
        whitened_cross = x_invsqrt @ (x_centered.T @ y_centered / float(n)) @ y_invsqrt
        u, singular_values, vh = torch.linalg.svd(whitened_cross, full_matrices=True)
        projection = x_invsqrt @ u
        residual_projection = y_invsqrt @ vh.T

        usable = min(singular_values.numel(), residual_projection.shape[1], projection.shape[1])
        sign = torch.ones(projection.shape[1], dtype=torch.float32, device=projection.device)
        if usable > 0:
            x_scores = x_centered @ projection[:, :usable]
            y_scores = y_centered @ residual_projection[:, :usable]
            score_sign = torch.sign(torch.sum(x_scores * y_scores, dim=0))
            score_sign = torch.where(score_sign == 0.0, torch.ones_like(score_sign), score_sign)
            sign[:usable] = score_sign

        correlations = torch.zeros(x_dim, dtype=torch.float32, device=delta.device)
        correlations[: singular_values.numel()] = singular_values.detach().float().clamp(min=0.0, max=1.0)
        if self.scale_by_correlation:
            scale = torch.clamp(correlations, min=self.eps)
            projection = projection * scale.unsqueeze(0).to(projection.dtype)

        self.mean = delta.mean(dim=0).detach().float()
        self.transition_mean = x_transition.mean(dim=0).detach().float()
        self.projection = projection.detach().float()
        self.residual_projection = residual_projection.detach().float()
        self.sign = sign.detach().float()
        self.canonical_correlations = correlations.detach().float()
        self.fit_sample_count = int(delta.shape[0])
        self.transition_sample_count = int(x_transition.shape[0])
        return self

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.projection is None or self.sign is None:
            raise RuntimeError("TrainSetTemporalResidualCCAAligner must be fit before transform")
        mean = self.mean.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        projection = self.projection.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        sign = self.sign.to(device=delta_phi_v.device, dtype=delta_phi_v.dtype)
        return ((delta_phi_v - mean) @ projection) * sign.unsqueeze(0)

    def state_dict(self) -> dict[str, torch.Tensor | int | float | bool | str]:
        if (
            self.mean is None
            or self.transition_mean is None
            or self.projection is None
            or self.residual_projection is None
            or self.sign is None
            or self.canonical_correlations is None
        ):
            raise RuntimeError("TrainSetTemporalResidualCCAAligner must be fit before serialization")
        return {
            "class_name": "TrainSetTemporalResidualCCAAligner",
            "method": "temporal_residual_cca_v3",
            "mean": self.mean.detach().clone(),
            "transition_mean": self.transition_mean.detach().clone(),
            "projection": self.projection.detach().clone(),
            "residual_projection": self.residual_projection.detach().clone(),
            "sign": self.sign.detach().clone(),
            "canonical_correlations": self.canonical_correlations.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "transition_sample_count": self.transition_sample_count,
            "ridge": self.ridge,
            "eps": self.eps,
            "scale_by_correlation": self.scale_by_correlation,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float | bool | str]) -> "TrainSetTemporalResidualCCAAligner":
        self.mean = torch.as_tensor(state["mean"]).detach().float()
        self.transition_mean = torch.as_tensor(state["transition_mean"]).detach().float()
        self.projection = torch.as_tensor(state["projection"]).detach().float()
        self.residual_projection = torch.as_tensor(state["residual_projection"]).detach().float()
        self.sign = torch.as_tensor(state["sign"]).detach().float()
        self.canonical_correlations = torch.as_tensor(state["canonical_correlations"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.transition_sample_count = int(state["transition_sample_count"])
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        self.scale_by_correlation = bool(state.get("scale_by_correlation", self.scale_by_correlation))
        return self


class TrainSetWhitenedProcrustesTemporalResidualCCAAligner:
    """Train-only phi-space alignment followed by temporal residual CCA.

    The transform is intentionally state-free at validation/test time: all
    whitening, rotation, and CCA bases are fitted on the ordered train split.
    Serving calls ``transform_phi_pair(phi_i, phi_v)`` so the residual drift is
    computed after aligning ``phi_v`` into the invariant embedding geometry.
    """

    def __init__(
        self,
        ridge: float = 1e-3,
        eps: float = 1e-6,
        scale_by_correlation: bool = False,
        orthogonalize_delta: bool = True,
    ) -> None:
        self.ridge = ridge
        self.eps = eps
        self.scale_by_correlation = scale_by_correlation
        self.orthogonalize_delta = orthogonalize_delta
        self.phi_aligner = TrainSetWhitenedProcrustesAligner(ridge=ridge, eps=eps)
        self.temporal_aligner = TrainSetTemporalResidualCCAAligner(
            ridge=ridge,
            eps=eps,
            scale_by_correlation=scale_by_correlation,
        )
        self.fit_sample_count = 0
        self.transition_sample_count = 0

    @property
    def canonical_correlations(self) -> torch.Tensor | None:
        return self.temporal_aligner.canonical_correlations

    def fit(
        self,
        *,
        phi_i_train: torch.Tensor,
        phi_v_train: torch.Tensor,
        y_inv_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "TrainSetWhitenedProcrustesTemporalResidualCCAAligner":
        if phi_i_train.ndim != 2 or phi_v_train.ndim != 2:
            raise ValueError("phi_i_train and phi_v_train must be 2-D")
        if phi_i_train.shape != phi_v_train.shape:
            raise ValueError("phi_i_train and phi_v_train must have the same shape")
        if y_inv_train.shape[0] != phi_i_train.shape[0] or y_train.shape[0] != phi_i_train.shape[0]:
            raise ValueError("phi_i_train, phi_v_train, y_inv_train, and y_train must share sample count")
        self.phi_aligner = TrainSetWhitenedProcrustesAligner(ridge=self.ridge, eps=self.eps).fit(
            phi_i_train=phi_i_train,
            phi_v_train=phi_v_train,
        )
        delta = self._aligned_delta(phi_i_train.detach().float(), phi_v_train.detach().float())
        self.temporal_aligner = TrainSetTemporalResidualCCAAligner(
            ridge=self.ridge,
            eps=self.eps,
            scale_by_correlation=self.scale_by_correlation,
        ).fit(
            delta_phi_v_train=delta,
            y_inv_train=y_inv_train,
            y_train=y_train,
        )
        self.fit_sample_count = self.temporal_aligner.fit_sample_count
        self.transition_sample_count = self.temporal_aligner.transition_sample_count
        return self

    def _aligned_delta(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        aligned_phi_v = self.phi_aligner.transform_phi_v(phi_v)
        delta = aligned_phi_v - phi_i
        if self.orthogonalize_delta:
            delta = _project_delta_orthogonal_to_anchor(delta, phi_i, self.eps)
        return delta

    def transform_phi_pair(self, phi_i: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        delta = self._aligned_delta(phi_i, phi_v)
        return self.temporal_aligner.transform_delta(delta)

    def transform_delta(self, delta_phi_v: torch.Tensor) -> torch.Tensor:
        return self.temporal_aligner.transform_delta(delta_phi_v)

    def state_dict(self) -> dict[str, torch.Tensor | int | float | bool | str | dict]:
        return {
            "class_name": "TrainSetWhitenedProcrustesTemporalResidualCCAAligner",
            "method": "whitened_procrustes_temporal_residual_cca_v6",
            "phi_aligner": self.phi_aligner.state_dict(),
            "temporal_aligner": self.temporal_aligner.state_dict(),
            "fit_sample_count": self.fit_sample_count,
            "transition_sample_count": self.transition_sample_count,
            "ridge": self.ridge,
            "eps": self.eps,
            "scale_by_correlation": self.scale_by_correlation,
            "orthogonalize_delta": self.orthogonalize_delta,
        }

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor | int | float | bool | str | dict],
    ) -> "TrainSetWhitenedProcrustesTemporalResidualCCAAligner":
        self.ridge = float(state.get("ridge", self.ridge))
        self.eps = float(state.get("eps", self.eps))
        self.scale_by_correlation = bool(state.get("scale_by_correlation", self.scale_by_correlation))
        self.orthogonalize_delta = bool(state.get("orthogonalize_delta", self.orthogonalize_delta))
        self.phi_aligner = TrainSetWhitenedProcrustesAligner(ridge=self.ridge, eps=self.eps).load_state_dict(
            state["phi_aligner"]
        )
        self.temporal_aligner = TrainSetTemporalResidualCCAAligner(
            ridge=self.ridge,
            eps=self.eps,
            scale_by_correlation=self.scale_by_correlation,
        ).load_state_dict(state["temporal_aligner"])
        self.fit_sample_count = int(state["fit_sample_count"])
        self.transition_sample_count = int(state["transition_sample_count"])
        return self


def _effective_rank(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / float(max(1, centered.shape[0] - 1))
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if float(total.item()) <= eps:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    probs = eigvals / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return torch.exp(entropy)


def _abs_corr_matrix(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros(x.shape[1], y.shape[1], device=x.device, dtype=x.dtype)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    x_std = torch.sqrt(x.var(dim=0, unbiased=False, keepdim=True) + eps)
    y_std = torch.sqrt(y.var(dim=0, unbiased=False, keepdim=True) + eps)
    x = x / x_std
    y = y / y_std
    return torch.abs(x.T @ y / float(x.shape[0]))


def _norm_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x_norm = torch.linalg.vector_norm(x, dim=1)
    y_norm = torch.linalg.vector_norm(y, dim=1)
    x_centered = x_norm - x_norm.mean()
    y_centered = y_norm - y_norm.mean()
    denom = torch.sqrt(torch.mean(x_centered.pow(2)) * torch.mean(y_centered.pow(2)) + eps)
    return torch.mean(x_centered * y_centered) / denom


def temporal_residual_alignment_diagnostics(
    *,
    delta_phi_v: torch.Tensor,
    y_inv: torch.Tensor,
    y: torch.Tensor,
    aligner: TrainSetTemporalResidualCCAAligner | TrainSetResidualDriftCanonicalizer | None = None,
    prefix: str = "",
    collapse_std_threshold: float = 1e-4,
) -> dict[str, float]:
    if delta_phi_v.ndim != 2:
        raise ValueError("delta_phi_v must be 2-D")
    if delta_phi_v.shape[0] != y_inv.shape[0] or delta_phi_v.shape[0] != y.shape[0]:
        raise ValueError("delta_phi_v, y_inv, and y must have the same sample count")
    key_prefix = f"{prefix}_" if prefix else ""
    delta = delta_phi_v.detach().float()
    if aligner is not None:
        delta = aligner.transform_delta(delta).detach().float()
    residual = (y.detach().float() - y_inv.detach().float()).reshape(delta.shape[0], -1)
    if delta.shape[0] < 2:
        return {
            f"{key_prefix}fit_sample_count": float(delta.shape[0]),
            f"{key_prefix}transition_sample_count": 0.0,
            f"{key_prefix}top_abs_transition_residual_corr": 0.0,
            f"{key_prefix}mean_best_abs_transition_residual_corr": 0.0,
            f"{key_prefix}transition_norm_corr": 0.0,
            f"{key_prefix}delta_transition_effective_rank": 0.0,
            f"{key_prefix}collapse_dim_fraction": 1.0,
        }
    delta_transition = delta[1:] - delta[:-1]
    residual_transition = residual[1:] - residual[:-1]
    abs_corr = _abs_corr_matrix(delta_transition, residual_transition)
    per_residual_best = abs_corr.max(dim=0).values if abs_corr.numel() else torch.zeros(1)
    delta_std = delta_transition.std(dim=0, unbiased=False)
    collapse_fraction = torch.mean((delta_std < collapse_std_threshold).float())
    return {
        f"{key_prefix}fit_sample_count": float(delta.shape[0]),
        f"{key_prefix}transition_sample_count": float(delta_transition.shape[0]),
        f"{key_prefix}top_abs_transition_residual_corr": float(abs_corr.max().cpu().item()) if abs_corr.numel() else 0.0,
        f"{key_prefix}mean_best_abs_transition_residual_corr": float(per_residual_best.mean().cpu().item()),
        f"{key_prefix}transition_norm_corr": float(_norm_corr(delta_transition, residual_transition).cpu().item()),
        f"{key_prefix}delta_transition_norm_mean": float(torch.linalg.vector_norm(delta_transition, dim=1).mean().cpu().item()),
        f"{key_prefix}residual_transition_norm_mean": float(torch.linalg.vector_norm(residual_transition, dim=1).mean().cpu().item()),
        f"{key_prefix}delta_transition_effective_rank": float(_effective_rank(delta_transition).cpu().item()),
        f"{key_prefix}collapse_dim_fraction": float(collapse_fraction.cpu().item()),
    }


@dataclass
class AlignedRepresentationLossConfig:
    lambda_forecast: float = 1.0
    lambda_residual: float = 1.0
    lambda_orth: float = 0.1
    lambda_smooth: float = 0.1
    lambda_norm: float = 0.01
    lambda_hsic: float = 0.0
    lambda_inv_forecast: float = 0.0
    lambda_barlow: float = 0.0
    lambda_temporal_ranking: float = 0.0
    temporal_ranking_margin: float = 0.01
    lambda_deployed_temporal_ranking: float = 0.0
    deployed_temporal_ranking_margin: float = 0.01
    lambda_temporal_contrastive: float = 0.0
    temporal_contrastive_temperature: float = 1.0
    lambda_vicreg_variance: float = 0.0
    lambda_vicreg_covariance: float = 0.0
    lambda_phi_vicreg_variance: float = 0.0
    lambda_phi_vicreg_covariance: float = 0.0
    vicreg_variance_target: float = 1.0


def _correlation_penalty(phi_i: torch.Tensor, delta_phi_v: torch.Tensor) -> torch.Tensor:
    phi_i_centered = phi_i - phi_i.mean(dim=0, keepdim=True)
    delta_centered = delta_phi_v - delta_phi_v.mean(dim=0, keepdim=True)
    phi_i_norm = F.normalize(phi_i_centered, dim=0)
    delta_norm = F.normalize(delta_centered, dim=0)
    corr = phi_i_norm.T @ delta_norm / max(1, phi_i.shape[0] - 1)
    return torch.mean(corr.pow(2))


def _centered_gram(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T
    return gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()


def _linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    k = _centered_gram(x)
    l = _centered_gram(y)
    return torch.sum(k * l) / float((x.shape[0] - 1) ** 2)


def _barlow_cross_correlation_penalty(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    x = x / torch.sqrt(x.var(dim=0, unbiased=False, keepdim=True) + eps)
    y = y / torch.sqrt(y.var(dim=0, unbiased=False, keepdim=True) + eps)
    cross_corr = (x.T @ y) / float(x.shape[0])
    return cross_corr.pow(2).mean()


def _parallel_ratio(phi_i: torch.Tensor, delta_phi_v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    phi_norm = torch.linalg.vector_norm(phi_i, dim=1)
    delta_norm = torch.linalg.vector_norm(delta_phi_v, dim=1)
    denom = (phi_norm * delta_norm).clamp_min(eps)
    return torch.mean(torch.abs(torch.sum(phi_i * delta_phi_v, dim=1)) / denom)


def _temporal_ranking_loss(
    *,
    delta_phi_v: torch.Tensor,
    y_inv: torch.Tensor,
    y: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if delta_phi_v.shape[0] < 2:
        return torch.zeros((), device=delta_phi_v.device, dtype=delta_phi_v.dtype)
    drift_score = torch.linalg.vector_norm(delta_phi_v, dim=1)
    residual_score = torch.mean(torch.abs(y - y_inv.detach()), dim=1)
    target_diff = residual_score[1:] - residual_score[:-1]
    valid = torch.abs(target_diff) > 1e-8
    if not bool(valid.any()):
        return torch.zeros((), device=delta_phi_v.device, dtype=delta_phi_v.dtype)
    score_diff = drift_score[1:] - drift_score[:-1]
    direction = torch.sign(target_diff[valid])
    return torch.mean(F.relu(float(margin) - direction * score_diff[valid]))


def _zero_like_loss(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _deployed_temporal_ranking_loss(
    *,
    delta_phi_v: torch.Tensor,
    y_inv: torch.Tensor,
    y: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Rank transition drift by transition residual need.

    Stage III online serving consumes the change of the residual drift signal
    across consecutive stream samples. This loss therefore scores
    ``delta_phi_v[t] - delta_phi_v[t-1]`` rather than the absolute residual
    embedding level.
    """

    if delta_phi_v.shape[0] < 3:
        return _zero_like_loss(delta_phi_v)
    drift_transition = delta_phi_v[1:] - delta_phi_v[:-1]
    drift_score = torch.linalg.vector_norm(drift_transition, dim=1)
    residual = (y - y_inv.detach()).reshape(y.shape[0], -1)
    residual_transition_score = torch.mean(torch.abs(residual[1:] - residual[:-1]), dim=1)
    target_diff = residual_transition_score[:, None] - residual_transition_score[None, :]
    valid = torch.abs(target_diff) > 1e-8
    if not bool(valid.any()):
        return _zero_like_loss(delta_phi_v)
    score_diff = drift_score[:, None] - drift_score[None, :]
    direction = torch.sign(target_diff[valid])
    return torch.mean(F.relu(float(margin) - direction * score_diff[valid]))


def _temporal_contrastive_loss(
    *,
    delta_phi_v: torch.Tensor,
    y_inv: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Calibrate temporal drift-vector similarity against residual-change similarity."""

    if delta_phi_v.shape[0] < 3:
        return _zero_like_loss(delta_phi_v)
    drift_transition = delta_phi_v[1:] - delta_phi_v[:-1]
    if drift_transition.shape[0] < 2:
        return _zero_like_loss(delta_phi_v)
    z = F.normalize(drift_transition, dim=1)
    pred_similarity = z @ z.T
    residual = (y - y_inv.detach()).reshape(y.shape[0], -1)
    residual_transition = residual[1:] - residual[:-1]
    residual_distance = torch.cdist(residual_transition.detach(), residual_transition.detach(), p=1)
    scale = residual_distance[residual_distance > 0].mean() if bool((residual_distance > 0).any()) else None
    if scale is None:
        return _zero_like_loss(delta_phi_v)
    tau = torch.as_tensor(max(float(temperature), 1e-6), device=delta_phi_v.device, dtype=delta_phi_v.dtype)
    target_similarity = torch.exp(-residual_distance / scale.clamp_min(1e-6) / tau)
    eye = torch.eye(pred_similarity.shape[0], device=delta_phi_v.device, dtype=torch.bool)
    return F.mse_loss(pred_similarity[~eye], target_similarity.to(pred_similarity.dtype)[~eye])


def _vicreg_variance_penalty(x: torch.Tensor, target_std: float, eps: float = 1e-4) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    target = torch.as_tensor(target_std, device=x.device, dtype=x.dtype)
    return torch.mean(F.relu(target - std))


def _vicreg_covariance_penalty(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / float(x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / x.shape[1]


def aligned_representation_losses(
    *,
    phi_i: torch.Tensor,
    delta_phi_v: torch.Tensor,
    y_inv: torch.Tensor,
    y_var: torch.Tensor,
    y: torch.Tensor,
    cfg: AlignedRepresentationLossConfig,
) -> dict[str, torch.Tensor]:
    forecast = F.mse_loss(y_inv + y_var, y)
    inv_forecast = F.mse_loss(y_inv, y)
    residual_target = (y - y_inv).detach()
    residual = F.mse_loss(y_var, residual_target)
    orthogonal = _correlation_penalty(phi_i, delta_phi_v)
    if delta_phi_v.shape[0] > 1:
        smooth = torch.mean((delta_phi_v[1:] - delta_phi_v[:-1]).pow(2))
    else:
        smooth = torch.zeros((), device=delta_phi_v.device, dtype=delta_phi_v.dtype)
    norm = torch.mean(delta_phi_v.pow(2))
    hsic = _linear_hsic(phi_i, delta_phi_v)
    barlow = _barlow_cross_correlation_penalty(phi_i, delta_phi_v)
    temporal_ranking = _temporal_ranking_loss(
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y=y,
        margin=cfg.temporal_ranking_margin,
    )
    deployed_temporal_ranking = _deployed_temporal_ranking_loss(
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y=y,
        margin=cfg.deployed_temporal_ranking_margin,
    )
    temporal_contrastive = _temporal_contrastive_loss(
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y=y,
        temperature=cfg.temporal_contrastive_temperature,
    )
    vicreg_variance = _vicreg_variance_penalty(delta_phi_v, cfg.vicreg_variance_target)
    vicreg_covariance = _vicreg_covariance_penalty(delta_phi_v)
    phi_vicreg_variance = _vicreg_variance_penalty(phi_i, cfg.vicreg_variance_target)
    phi_vicreg_covariance = _vicreg_covariance_penalty(phi_i)
    total = (
        cfg.lambda_forecast * forecast
        + cfg.lambda_inv_forecast * inv_forecast
        + cfg.lambda_residual * residual
        + cfg.lambda_orth * orthogonal
        + cfg.lambda_smooth * smooth
        + cfg.lambda_norm * norm
        + cfg.lambda_hsic * hsic
        + cfg.lambda_barlow * barlow
        + cfg.lambda_temporal_ranking * temporal_ranking
        + cfg.lambda_deployed_temporal_ranking * deployed_temporal_ranking
        + cfg.lambda_temporal_contrastive * temporal_contrastive
        + cfg.lambda_vicreg_variance * vicreg_variance
        + cfg.lambda_vicreg_covariance * vicreg_covariance
        + cfg.lambda_phi_vicreg_variance * phi_vicreg_variance
        + cfg.lambda_phi_vicreg_covariance * phi_vicreg_covariance
    )
    return {
        "total": total,
        "forecast": forecast,
        "inv_forecast": inv_forecast,
        "residual": residual,
        "orthogonal": orthogonal,
        "smooth": smooth,
        "norm": norm,
        "hsic": hsic,
        "barlow": barlow,
        "temporal_ranking": temporal_ranking,
        "deployed_temporal_ranking": deployed_temporal_ranking,
        "temporal_contrastive": temporal_contrastive,
        "parallel_ratio": _parallel_ratio(phi_i, delta_phi_v),
        "vicreg_variance": vicreg_variance,
        "vicreg_covariance": vicreg_covariance,
        "phi_vicreg_variance": phi_vicreg_variance,
        "phi_vicreg_covariance": phi_vicreg_covariance,
    }
