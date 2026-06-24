from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import torch

from losses.vae_loss import vae_loss
from models.aligned_representation_v2 import AlignedRepresentationLossConfig, aligned_representation_losses
from train.offline_factor_v18 import (
    _anchor_preservation_loss,
    _env_balance_loss,
    _env_temporal_loss,
    _load_balance_loss,
    _pairwise_similarity_penalty,
    _release_confidence_alignment_loss,
    _release_temporal_loss,
    _router_stability_loss,
    _suppression_sparsity_loss,
)
from train.offline_factor_v19 import (
    _conditional_advantage_loss,
    _counterfactual_shuffle_loss,
    _local_inv_forecast_loss,
    _local_release_increment_loss,
    _release_null_loss,
)
from utils.progress import get_progress


@dataclass
class FactorMechanismV20LossConfig:
    beta_vae: float = 0.1
    lambda_load_balance: float = 0.05
    lambda_env_balance: float = 0.01
    lambda_expert_diversity: float = 0.01
    lambda_local_inv_forecast: float = 0.2
    lambda_local_history_proxy: float = 0.12
    lambda_local_release_increment: float = 0.35
    lambda_env_temporal: float = 0.05
    lambda_router_stability: float = 0.1
    lambda_anchor_preservation: float = 0.1
    lambda_env_correction_sparsity: float = 0.02
    lambda_release_confidence: float = 0.1
    lambda_anchor_diversity: float = 0.02
    lambda_release_temporal: float = 0.05
    lambda_conditional_advantage: float = 0.2
    conditional_advantage_margin: float = 0.001
    lambda_release_null: float = 0.05
    lambda_counterfactual_shuffle: float = 0.1
    counterfactual_shuffle_margin: float = 0.001
    lambda_budget_alignment: float = 0.15
    lambda_history_budget_cap: float = 0.08
    representation: AlignedRepresentationLossConfig = field(default_factory=AlignedRepresentationLossConfig)


def _budgeted_history_proxy_loss(
    expert_y_hist: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_history_budget: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = expert_history_budget.detach() * (y.unsqueeze(1) - expert_y_inv.detach())
    mse = torch.mean((expert_y_hist - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _budget_alignment_loss(
    expert_release_budget: torch.Tensor,
    expert_y_hat: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_hat)
    history_only = expert_y_inv + expert_y_hist
    mse_full = torch.mean((expert_y_hat - target).pow(2), dim=-1)
    mse_hist = torch.mean((history_only - target).pow(2), dim=-1)
    desired_release_budget = torch.sigmoid(6.0 * (mse_hist - mse_full).detach())
    observed_release_budget = expert_release_budget.squeeze(-1)
    return torch.mean(alpha.detach() * (observed_release_budget - desired_release_budget).pow(2))


def _history_budget_cap_loss(
    expert_history_budget: torch.Tensor,
    expert_release_confidence: torch.Tensor,
    env_shift: torch.Tensor,
) -> torch.Tensor:
    shift_mag = torch.sqrt(env_shift.pow(2).mean(dim=(2, 3)).clamp_min(1e-8)).clamp(0.0, 1.0)
    confidence = expert_release_confidence.squeeze(-1).clamp(0.0, 1.0)
    max_history_budget = (1.0 - 0.5 * shift_mag - 0.35 * confidence).clamp(0.1, 1.0)
    observed_history_budget = expert_history_budget.squeeze(-1)
    return torch.relu(observed_history_budget - max_history_budget).mean()


def evaluate_factor_v20(dataloader: Iterable, model, device: torch.device) -> Tuple[float, float]:
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


def train_offline_factor_v20(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    loss_cfg: FactorMechanismV20LossConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.train()
    rows: list[dict[str, float]] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor_v20")
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
        expert_diversity = _pairwise_similarity_penalty(out.expert_delta_phi_v)
        local_inv_forecast = _local_inv_forecast_loss(out.expert_y_inv, y, out.alpha)
        local_history_proxy = _budgeted_history_proxy_loss(
            out.expert_y_hist,
            out.expert_y_inv,
            out.expert_history_budget,
            y,
            out.alpha,
        )
        local_release_increment = _local_release_increment_loss(
            out.expert_y_var,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        env_temporal = _env_temporal_loss(out.env_prior_posteriors, out.env_posteriors, out.env_shift)
        router_stability = _router_stability_loss(out.alpha, out.expert_y_hat, y)
        anchor_preservation = _anchor_preservation_loss(out.expert_base + out.expert_invariant_adapter, out.expert_phi_i)
        env_correction_sparsity = _suppression_sparsity_loss(out.expert_env_correction)
        release_confidence = _release_confidence_alignment_loss(
            out.expert_release_confidence,
            out.expert_y_inv + out.expert_y_hist,
            out.expert_y_hat,
            y,
        )
        anchor_diversity = _pairwise_similarity_penalty(out.expert_base)
        release_temporal = _release_temporal_loss(torch.stack(out.drift_expert_embs, dim=1), out.env_shift)
        conditional_advantage = _conditional_advantage_loss(
            out.expert_y_hat,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
            margin=loss_cfg.conditional_advantage_margin,
        )
        release_null = _release_null_loss(out.expert_y_var, out.expert_release_confidence, out.env_shift)
        counterfactual_shuffle = _counterfactual_shuffle_loss(
            out.expert_y_var,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
            margin=loss_cfg.counterfactual_shuffle_margin,
        )
        budget_alignment = _budget_alignment_loss(
            out.expert_release_budget,
            out.expert_y_hat,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        history_budget_cap = _history_budget_cap_loss(
            out.expert_history_budget,
            out.expert_release_confidence,
            out.env_shift,
        )
        total_loss = (
            rep_losses["total"]
            + l_vae
            + loss_cfg.lambda_load_balance * load_balance
            + loss_cfg.lambda_env_balance * env_balance
            + loss_cfg.lambda_expert_diversity * expert_diversity
            + loss_cfg.lambda_local_inv_forecast * local_inv_forecast
            + loss_cfg.lambda_local_history_proxy * local_history_proxy
            + loss_cfg.lambda_local_release_increment * local_release_increment
            + loss_cfg.lambda_env_temporal * env_temporal
            + loss_cfg.lambda_router_stability * router_stability
            + loss_cfg.lambda_anchor_preservation * anchor_preservation
            + loss_cfg.lambda_env_correction_sparsity * env_correction_sparsity
            + loss_cfg.lambda_release_confidence * release_confidence
            + loss_cfg.lambda_anchor_diversity * anchor_diversity
            + loss_cfg.lambda_release_temporal * release_temporal
            + loss_cfg.lambda_conditional_advantage * conditional_advantage
            + loss_cfg.lambda_release_null * release_null
            + loss_cfg.lambda_counterfactual_shuffle * counterfactual_shuffle
            + loss_cfg.lambda_budget_alignment * budget_alignment
            + loss_cfg.lambda_history_budget_cap * history_budget_cap
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
            "loss_local_history_proxy": float(local_history_proxy.detach().cpu().item()),
            "loss_local_release_increment": float(local_release_increment.detach().cpu().item()),
            "loss_env_temporal": float(env_temporal.detach().cpu().item()),
            "loss_router_stability": float(router_stability.detach().cpu().item()),
            "loss_anchor_preservation": float(anchor_preservation.detach().cpu().item()),
            "loss_env_correction_sparsity": float(env_correction_sparsity.detach().cpu().item()),
            "loss_release_confidence": float(release_confidence.detach().cpu().item()),
            "loss_anchor_diversity": float(anchor_diversity.detach().cpu().item()),
            "loss_release_temporal": float(release_temporal.detach().cpu().item()),
            "loss_conditional_advantage": float(conditional_advantage.detach().cpu().item()),
            "loss_release_null": float(release_null.detach().cpu().item()),
            "loss_counterfactual_shuffle": float(counterfactual_shuffle.detach().cpu().item()),
            "loss_budget_alignment": float(budget_alignment.detach().cpu().item()),
            "loss_history_budget_cap": float(history_budget_cap.detach().cpu().item()),
        }
        rows.append(row)
        if pbar is not None:
            pbar.update(x.size(0))
    if pbar is not None:
        pbar.close()
    return rows

