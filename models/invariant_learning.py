from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvariantModule(nn.Module):
    """Two-branch invariant/variant encoders.

    Produces:
        phi_I: invariant embedding
        phi_V: variant embedding
    """

    def __init__(self, encoder_i: nn.Module, encoder_v: nn.Module) -> None:
        super().__init__()
        self.encoder_i = encoder_i
        self.encoder_v = encoder_v

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_i(x), self.encoder_v(x)


@dataclass
class EnvInferenceConfig:
    num_envs: int
    smooth_radius: int = 0


class EnvInference(nn.Module):
    """Environment inference module (M_EI).

    Implements dynamic soft assignment based on regressor losses,
    with optional temporal majority-vote smoothing.
    """

    def __init__(self, regressors: nn.ModuleList, cfg: EnvInferenceConfig) -> None:
        super().__init__()
        self.regressors = regressors
        self.cfg = cfg

    @torch.no_grad()
    def assign(self, phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Assign each sample to the nearest regressor by loss (Eq. 6-7)."""

        losses = []
        for reg in self.regressors:
            y_hat = reg(phi)
            loss = F.mse_loss(y_hat, y, reduction="none")
            # reduce over output dims
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
            losses.append(loss)
        loss_mat = torch.stack(losses, dim=1)  # [B, E]
        labels = torch.argmin(loss_mat, dim=1)
        if self.cfg.smooth_radius > 0:
            labels = self._smooth_labels(labels)
        return labels

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        radius = self.cfg.smooth_radius
        if radius <= 0:
            return labels
        out = labels.clone()
        for i in range(labels.numel()):
            start = max(0, i - radius)
            end = min(labels.numel(), i + radius + 1)
            window = labels[start:end]
            counts = torch.bincount(window, minlength=self.cfg.num_envs)
            out[i] = torch.argmax(counts)
        return out

    def fit_regressors(
        self, phi: torch.Tensor, y: torch.Tensor, labels: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Compute per-env losses; caller handles optimizer steps."""

        per_env_loss: Dict[int, torch.Tensor] = {}
        for e, reg in enumerate(self.regressors):
            idx = labels == e
            if idx.sum() == 0:
                continue
            y_hat = reg(phi[idx])
            per_env_loss[e] = F.mse_loss(y_hat, y[idx])
        return per_env_loss


# -------------------------------
# FOIL-style MTIL + MTEI modules
# -------------------------------


@dataclass
class MTILConfig:
    lambda_erm: float = 1.0
    lambda_var: float = 1.0


class FoilBackbone(nn.Module):
    """Wraps a TSF backbone to expose representation phi(X)."""

    def __init__(self, backbone: nn.Module, repr_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FoilRegressor(nn.Module):
    """Single regressor head ρ(·) mapping phi -> Y."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)


class MTEI(nn.Module):
    """Time-Series Environment Inference Module (Eq. 6-8).

    Multi-head regressors as environment-specific centers.
    """

    def __init__(self, in_dim: int, out_dim: int, cfg: EnvInferenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.regressors = nn.ModuleList(
            [FoilRegressor(in_dim, out_dim) for _ in range(cfg.num_envs)]
        )

    def m_step(self, phi: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, loss_fn):
        """Optimize environment-specific regressors (Eq. 6).

        Returns per-env losses for logging.
        """

        per_env_loss: Dict[int, torch.Tensor] = {}
        for e, reg in enumerate(self.regressors):
            idx = labels == e
            if idx.sum() == 0:
                continue
            y_hat = reg(phi[idx])
            per_env_loss[e] = loss_fn(y_hat, y[idx])
        return per_env_loss

    @torch.no_grad()
    def e_step(self, phi: torch.Tensor, y: torch.Tensor, loss_fn):
        """Reallocate env labels (Eq. 7) + label propagation (Eq. 8)."""

        losses = []
        for reg in self.regressors:
            y_hat = reg(phi)
            loss = loss_fn(y_hat, y, reduction="none")
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
            losses.append(loss)
        loss_mat = torch.stack(losses, dim=1)  # [B, E]
        labels = torch.argmin(loss_mat, dim=1)
        if self.cfg.smooth_radius > 0:
            labels = self._label_propagation(labels)
        return labels

    def _label_propagation(self, labels: torch.Tensor) -> torch.Tensor:
        radius = self.cfg.smooth_radius
        if radius <= 0:
            return labels
        out = labels.clone()
        for i in range(labels.numel()):
            start = max(0, i - radius)
            end = min(labels.numel(), i + radius + 1)
            window = labels[start:end]
            counts = torch.bincount(window, minlength=self.cfg.num_envs)
            out[i] = torch.argmax(counts)
        return out


class MTIL(nn.Module):
    """Time-Series Invariant Learning Module (Eq. 10)."""

    def __init__(self, backbone: FoilBackbone, regressor: FoilRegressor) -> None:
        super().__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.backbone(x)
        y_hat = self.regressor(phi)
        return phi, y_hat

    @staticmethod
    def loss(
        per_env_loss: Dict[int, torch.Tensor],
        y_hat: torch.Tensor,
        y: torch.Tensor,
        cfg: MTILConfig,
        erm_loss_fn,
    ) -> torch.Tensor:
        """Eq. 10: mean + var of per-env suf loss + lambda_erm * ERM."""

        if not per_env_loss:
            raise ValueError("per_env_loss is empty")
        losses = torch.stack(list(per_env_loss.values()))
        mean = losses.mean()
        var = losses.var(unbiased=False)
        erm = erm_loss_fn(y_hat, y)
        return mean + cfg.lambda_erm * erm + cfg.lambda_var * var
