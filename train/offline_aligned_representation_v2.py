from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aligned_representation_v2 import (
    AlignedRepresentationLossConfig,
    AlignedRepresentationOutput,
    SharedResidualFactorEncoder,
    aligned_representation_losses,
)


@dataclass
class AlignedRepresentationTrainConfig:
    lambda_forecast: float = 1.0
    lambda_residual: float = 0.5
    lambda_orth: float = 0.1
    lambda_smooth: float = 0.2
    lambda_norm: float = 0.05
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


@dataclass
class AlignedRepresentationModelOutput:
    representation: AlignedRepresentationOutput
    y_inv: torch.Tensor
    y_var: torch.Tensor
    y_hat: torch.Tensor


class AlignedRepresentationModel(nn.Module):
    def __init__(
        self,
        encoder: SharedResidualFactorEncoder,
        invariant_regressor: nn.Module,
        variant_regressor: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.invariant_regressor = invariant_regressor
        self.variant_regressor = variant_regressor

    def forward(self, x: torch.Tensor) -> AlignedRepresentationModelOutput:
        representation = self.encoder(x)
        y_inv = self.invariant_regressor(representation.phi_i)
        y_var = self.variant_regressor(representation.delta_phi_v)
        return AlignedRepresentationModelOutput(
            representation=representation,
            y_inv=y_inv,
            y_var=y_var,
            y_hat=y_inv + y_var,
        )


def build_aligned_representation_model(
    *,
    input_dim: int,
    repr_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    zero_init_variant: bool = False,
    zero_init_variant_regressor: bool = False,
    use_projected_residual: bool = False,
    stop_gradient_projection_anchor: bool = True,
) -> AlignedRepresentationModel:
    encoder = SharedResidualFactorEncoder(
        input_dim=input_dim,
        repr_dim=repr_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        zero_init_variant=zero_init_variant,
        use_projected_residual=use_projected_residual,
        stop_gradient_projection_anchor=stop_gradient_projection_anchor,
    )
    invariant_regressor = nn.Linear(repr_dim, out_dim)
    variant_regressor = nn.Sequential(
        nn.Linear(repr_dim, repr_dim),
        nn.GELU(),
        nn.Linear(repr_dim, out_dim),
    )
    if zero_init_variant_regressor:
        last = variant_regressor[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
    return AlignedRepresentationModel(encoder, invariant_regressor, variant_regressor)


def _loss_config(cfg: AlignedRepresentationTrainConfig) -> AlignedRepresentationLossConfig:
    return AlignedRepresentationLossConfig(
        lambda_forecast=cfg.lambda_forecast,
        lambda_residual=cfg.lambda_residual,
        lambda_orth=cfg.lambda_orth,
        lambda_smooth=cfg.lambda_smooth,
        lambda_norm=cfg.lambda_norm,
        lambda_hsic=cfg.lambda_hsic,
        lambda_inv_forecast=cfg.lambda_inv_forecast,
        lambda_barlow=cfg.lambda_barlow,
        lambda_temporal_ranking=cfg.lambda_temporal_ranking,
        temporal_ranking_margin=cfg.temporal_ranking_margin,
        lambda_deployed_temporal_ranking=cfg.lambda_deployed_temporal_ranking,
        deployed_temporal_ranking_margin=cfg.deployed_temporal_ranking_margin,
        lambda_temporal_contrastive=cfg.lambda_temporal_contrastive,
        temporal_contrastive_temperature=cfg.temporal_contrastive_temperature,
        lambda_vicreg_variance=cfg.lambda_vicreg_variance,
        lambda_vicreg_covariance=cfg.lambda_vicreg_covariance,
        lambda_phi_vicreg_variance=cfg.lambda_phi_vicreg_variance,
        lambda_phi_vicreg_covariance=cfg.lambda_phi_vicreg_covariance,
        vicreg_variance_target=cfg.vicreg_variance_target,
    )


def train_aligned_representation_epoch(
    dataloader: Iterable,
    model: AlignedRepresentationModel,
    optimizer: torch.optim.Optimizer,
    cfg: AlignedRepresentationTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "total": 0.0,
        "forecast": 0.0,
        "residual": 0.0,
        "orthogonal": 0.0,
        "smooth": 0.0,
        "norm": 0.0,
        "hsic": 0.0,
        "inv_forecast": 0.0,
        "barlow": 0.0,
        "temporal_ranking": 0.0,
        "deployed_temporal_ranking": 0.0,
        "temporal_contrastive": 0.0,
        "parallel_ratio": 0.0,
        "vicreg_variance": 0.0,
        "vicreg_covariance": 0.0,
        "phi_vicreg_variance": 0.0,
        "phi_vicreg_covariance": 0.0,
    }
    count = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        losses = aligned_representation_losses(
            phi_i=out.representation.phi_i,
            delta_phi_v=out.representation.delta_phi_v,
            y_inv=out.y_inv,
            y_var=out.y_var,
            y=y,
            cfg=_loss_config(cfg),
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        batch_size = x.shape[0]
        count += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu().item()) * batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def evaluate_aligned_representation(
    dataloader: Iterable,
    model: AlignedRepresentationModel,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    inv_mse = 0.0
    var_residual_mse = 0.0
    parallel_ratio = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_mse += F.mse_loss(out.y_hat, y).item() * x.shape[0]
            total_mae += F.l1_loss(out.y_hat, y).item() * x.shape[0]
            inv_mse += F.mse_loss(out.y_inv, y).item() * x.shape[0]
            var_residual_mse += F.mse_loss(out.y_var, (y - out.y_inv)).item() * x.shape[0]
            dot = torch.sum(out.representation.phi_i * out.representation.delta_phi_v, dim=1)
            denom = (
                torch.linalg.vector_norm(out.representation.phi_i, dim=1)
                * torch.linalg.vector_norm(out.representation.delta_phi_v, dim=1)
            ).clamp_min(1e-8)
            parallel_ratio += torch.mean(torch.abs(dot) / denom).item() * x.shape[0]
            count += x.shape[0]
    return {
        "forecast_mse": total_mse / max(1, count),
        "forecast_mae": total_mae / max(1, count),
        "inv_forecast_mse": inv_mse / max(1, count),
        "variant_residual_mse": var_residual_mse / max(1, count),
        "parallel_ratio": parallel_ratio / max(1, count),
    }
