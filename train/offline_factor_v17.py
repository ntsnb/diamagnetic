from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from losses.vae_loss import vae_loss
from models.aligned_representation_v2 import AlignedRepresentationLossConfig, aligned_representation_losses
from utils.progress import get_progress


@dataclass
class FactorMechanismV17LossConfig:
    beta_vae: float = 0.1
    lambda_load_balance: float = 0.05
    lambda_env_balance: float = 0.01
    lambda_expert_diversity: float = 0.01
    lambda_local_inv_forecast: float = 0.2
    lambda_local_residual: float = 0.2
    lambda_env_temporal: float = 0.05
    lambda_router_stability: float = 0.1
    representation: AlignedRepresentationLossConfig = field(default_factory=AlignedRepresentationLossConfig)


def _load_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(alpha.mean(dim=0), 1.0 / alpha.shape[1])
    return torch.mean((alpha.mean(dim=0) - target).pow(2))


def _env_balance_loss(env_posteriors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    marginal = env_posteriors.mean(dim=(0, 1, 2))
    target = torch.full_like(marginal, 1.0 / marginal.shape[0])
    balance = torch.mean((marginal - target).pow(2))
    entropy = -torch.sum(env_posteriors.clamp_min(eps) * torch.log(env_posteriors.clamp_min(eps)), dim=-1).mean()
    return balance - 0.1 * entropy


def _expert_diversity_loss(expert_delta_phi_v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if expert_delta_phi_v.shape[1] < 2:
        return torch.zeros((), device=expert_delta_phi_v.device, dtype=expert_delta_phi_v.dtype)
    x = expert_delta_phi_v - expert_delta_phi_v.mean(dim=0, keepdim=True)
    x = F.normalize(x, dim=-1, eps=eps)
    sim = torch.einsum("bkd,bjd->bkj", x, x)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_select(~eye).pow(2).mean()


def _local_inv_forecast_loss(expert_y_inv: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    mse = torch.mean((expert_y_inv - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _local_residual_loss(expert_y_var: torch.Tensor, expert_y_inv: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    target = y.unsqueeze(1) - expert_y_inv.detach()
    mse = torch.mean((expert_y_var - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _env_temporal_loss(
    env_prior_posteriors: torch.Tensor,
    env_posteriors: torch.Tensor,
    env_shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if env_posteriors.shape[2] <= 1:
        return torch.zeros((), device=env_posteriors.device, dtype=env_posteriors.dtype)
    prior = env_prior_posteriors[:, :, 1:, :].clamp_min(eps)
    post = env_posteriors[:, :, 1:, :].clamp_min(eps)
    kl = torch.sum(post * (torch.log(post) - torch.log(prior)), dim=-1).mean()
    shift_penalty = env_shift[:, :, 1:, :].pow(2).mean()
    return kl + 0.1 * shift_penalty


def _router_stability_loss(alpha: torch.Tensor, expert_y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_hat)
    expert_error = torch.mean((expert_y_hat - target).pow(2), dim=-1)
    target_alpha = torch.softmax(-expert_error.detach(), dim=-1)
    return torch.mean((alpha - target_alpha).pow(2))


def evaluate_factor_v17(dataloader: Iterable, model, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_mse += torch.mean((out.y_hat - y) ** 2).item() * x.size(0)
            total_mae += torch.mean(torch.abs(out.y_hat - y)).item() * x.size(0)
            n += x.size(0)
    return total_mse / max(n, 1), total_mae / max(n, 1)


def train_offline_factor_v17(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    loss_cfg: FactorMechanismV17LossConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.train()
    rows: list[dict[str, float]] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor_v17")
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        out = model(x)

        rep_losses = aligned_representation_losses(
            phi_i=out.pooled_phi_i,
            delta_phi_v=out.pooled_delta_phi_v,
            y_inv=out.y_inv,
            y_var=out.y_var,
            y=y,
            cfg=loss_cfg.representation,
        )
        if getattr(model, "use_vae", True):
            l_vae = vae_loss(out.x_recon, x, out.mu_z, out.log_var_z, loss_cfg.beta_vae)
        else:
            l_vae = torch.zeros((), device=device, dtype=out.y_hat.dtype)
        load_balance = _load_balance_loss(out.alpha)
        env_balance = _env_balance_loss(out.env_posteriors)
        expert_diversity = _expert_diversity_loss(out.expert_delta_phi_v)
        local_inv_forecast = _local_inv_forecast_loss(out.expert_y_inv, y, out.alpha)
        local_residual = _local_residual_loss(out.expert_y_var, out.expert_y_inv, y, out.alpha)
        env_temporal = _env_temporal_loss(out.env_prior_posteriors, out.env_posteriors, out.env_shift)
        router_stability = _router_stability_loss(out.alpha, out.expert_y_hat, y)
        total_loss = (
            rep_losses["total"]
            + l_vae
            + loss_cfg.lambda_load_balance * load_balance
            + loss_cfg.lambda_env_balance * env_balance
            + loss_cfg.lambda_expert_diversity * expert_diversity
            + loss_cfg.lambda_local_inv_forecast * local_inv_forecast
            + loss_cfg.lambda_local_residual * local_residual
            + loss_cfg.lambda_env_temporal * env_temporal
            + loss_cfg.lambda_router_stability * router_stability
        )

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        row = {
            "batch_index": float(batch_idx),
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_forecast": float(rep_losses["forecast"].detach().cpu().item()),
            "loss_inv_forecast": float(rep_losses["inv_forecast"].detach().cpu().item()),
            "loss_residual": float(rep_losses["residual"].detach().cpu().item()),
            "loss_orthogonal": float(rep_losses["orthogonal"].detach().cpu().item()),
            "loss_smooth": float(rep_losses["smooth"].detach().cpu().item()),
            "loss_norm": float(rep_losses["norm"].detach().cpu().item()),
            "loss_hsic": float(rep_losses["hsic"].detach().cpu().item()),
            "loss_barlow": float(rep_losses["barlow"].detach().cpu().item()),
            "loss_temporal_ranking": float(rep_losses["temporal_ranking"].detach().cpu().item()),
            "loss_deployed_temporal_ranking": float(rep_losses["deployed_temporal_ranking"].detach().cpu().item()),
            "loss_temporal_contrastive": float(rep_losses["temporal_contrastive"].detach().cpu().item()),
            "loss_vicreg_variance": float(rep_losses["vicreg_variance"].detach().cpu().item()),
            "loss_vicreg_covariance": float(rep_losses["vicreg_covariance"].detach().cpu().item()),
            "loss_phi_vicreg_variance": float(rep_losses["phi_vicreg_variance"].detach().cpu().item()),
            "loss_phi_vicreg_covariance": float(rep_losses["phi_vicreg_covariance"].detach().cpu().item()),
            "loss_parallel_ratio": float(rep_losses["parallel_ratio"].detach().cpu().item()),
            "loss_vae": float(l_vae.detach().cpu().item()),
            "loss_load_balance": float(load_balance.detach().cpu().item()),
            "loss_env_balance": float(env_balance.detach().cpu().item()),
            "loss_expert_diversity": float(expert_diversity.detach().cpu().item()),
            "loss_local_inv_forecast": float(local_inv_forecast.detach().cpu().item()),
            "loss_local_residual": float(local_residual.detach().cpu().item()),
            "loss_env_temporal": float(env_temporal.detach().cpu().item()),
            "loss_router_stability": float(router_stability.detach().cpu().item()),
        }
        rows.append(row)
        if pbar is not None:
            pbar.update(x.size(0))
    if pbar is not None:
        pbar.close()
    return rows
