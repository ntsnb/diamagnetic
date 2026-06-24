from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from losses.vae_loss import vae_loss
from models.aligned_representation_v2 import AlignedRepresentationLossConfig, aligned_representation_losses
from utils.progress import get_progress


@dataclass
class FactorMechanismV16LossConfig:
    beta_vae: float = 0.1
    lambda_load_balance: float = 0.05
    lambda_env_balance: float = 0.01
    lambda_expert_diversity: float = 0.01
    representation: AlignedRepresentationLossConfig = field(default_factory=AlignedRepresentationLossConfig)


def _load_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [B, K]")
    target = torch.full_like(alpha.mean(dim=0), 1.0 / alpha.shape[1])
    return torch.mean((alpha.mean(dim=0) - target).pow(2))


def _env_balance_loss(env_posteriors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if env_posteriors.ndim != 3:
        raise ValueError("env_posteriors must have shape [B, K, E]")
    marginal = env_posteriors.mean(dim=(0, 1))
    target = torch.full_like(marginal, 1.0 / marginal.shape[0])
    balance = torch.mean((marginal - target).pow(2))
    entropy = -torch.sum(env_posteriors.clamp_min(eps) * torch.log(env_posteriors.clamp_min(eps)), dim=-1).mean()
    return balance - 0.1 * entropy


def _expert_diversity_loss(expert_delta_phi_v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if expert_delta_phi_v.ndim != 3:
        raise ValueError("expert_delta_phi_v must have shape [B, K, D]")
    if expert_delta_phi_v.shape[1] < 2:
        return torch.zeros((), device=expert_delta_phi_v.device, dtype=expert_delta_phi_v.dtype)
    x = expert_delta_phi_v - expert_delta_phi_v.mean(dim=0, keepdim=True)
    x = F.normalize(x, dim=-1, eps=eps)
    sim = torch.einsum("bkd,bjd->bkj", x, x)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_select(~eye).pow(2).mean()


def evaluate_factor_v16(
    dataloader: Iterable,
    model,
    device: torch.device,
) -> Tuple[float, float]:
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


def train_offline_factor_v16(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    loss_cfg: FactorMechanismV16LossConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Offline training for V16 base model with integrated representation losses."""

    model.train()
    rows: list[dict[str, float]] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor_v16")
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
        total_loss = (
            rep_losses["total"]
            + l_vae
            + loss_cfg.lambda_load_balance * load_balance
            + loss_cfg.lambda_env_balance * env_balance
            + loss_cfg.lambda_expert_diversity * expert_diversity
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
        }
        rows.append(row)
        if pbar is not None:
            pbar.update(x.size(0))
    if pbar is not None:
        pbar.close()
    return rows
