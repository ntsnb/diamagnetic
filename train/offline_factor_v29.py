from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from losses.vae_loss import vae_loss
from models.aligned_representation_v2 import aligned_representation_losses
from train.offline_factor_v26 import (
    _anchor_preservation_loss,
    _budget_alignment_loss,
    _change_readout_loss,
    _completion_after_release_loss,
    _counterfactual_shuffle_loss,
    _delayed_change_loss,
    _env_balance_loss,
    _env_temporal_loss,
    _history_after_release_loss,
    _history_budget_cap_loss,
    _load_balance_loss,
    _local_inv_forecast_loss,
    _multiscale_change_alignment_loss,
    _observable_change_alignment_loss,
    _pairwise_similarity_penalty,
    _release_confidence_alignment_loss,
    _release_first_increment_loss,
    _release_null_loss,
    _release_priority_advantage_loss,
    _release_state_advantage_loss,
    _release_temporal_loss,
    _router_stability_loss,
    _suppression_sparsity_loss,
)
from train.offline_factor_v28 import (
    FactorMechanismV28LossConfig,
    _support_alignment_loss,
    _support_conditioned_advantage_loss,
    _suppression_alignment_loss,
    _suppression_gate_sparsity_loss,
)
from utils.progress import get_progress


@dataclass
class FactorMechanismV29LossConfig(FactorMechanismV28LossConfig):
    lambda_local_env_consistency: float = 0.08
    lambda_invariant_correction_budget: float = 0.04
    lambda_residual_release_alignment: float = 0.12
    lambda_residual_release_advantage: float = 0.12
    residual_release_advantage_margin: float = 0.001
    lambda_expert_release_orthogonal: float = 0.04
    lambda_residual_state_noncollapse: float = 0.03
    residual_state_std_floor: float = 0.05


def _local_env_consistency_loss(
    expert_local_env_posteriors: torch.Tensor,
    env_posteriors: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = env_posteriors.detach().mean(dim=2)
    return torch.mean(alpha.detach().unsqueeze(-1) * (expert_local_env_posteriors - target).pow(2))


def _invariant_correction_budget_loss(
    expert_invariant_correction: torch.Tensor,
    expert_invariant_correction_gate: torch.Tensor,
    expert_release_budget: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    correction_mag = (expert_invariant_correction * expert_invariant_correction_gate).pow(2).mean(dim=-1, keepdim=True)
    low_release_pressure = (1.0 - expert_release_budget.detach()).clamp(0.0, 1.0)
    return torch.mean(alpha.detach().unsqueeze(-1) * low_release_pressure * correction_mag)


def _residual_release_alignment_loss(
    expert_residual_release_state: torch.Tensor,
    expert_change_memory: torch.Tensor,
    expert_observable_change: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = (expert_change_memory + expert_observable_change).detach()
    cosine = F.cosine_similarity(expert_residual_release_state, target, dim=-1).unsqueeze(-1)
    return torch.mean(alpha.detach().unsqueeze(-1) * (1.0 - cosine).clamp(0.0, 2.0))


def _residual_release_advantage_loss(
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    expert_y_change_raw: torch.Tensor,
    expert_y_var: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    raw_pred = expert_y_inv + expert_y_hist + expert_y_change_raw
    corrected_pred = expert_y_inv + expert_y_hist + expert_y_var
    mse_raw = torch.mean((raw_pred - target).pow(2), dim=-1)
    mse_corrected = torch.mean((corrected_pred - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * torch.relu(float(margin) + mse_corrected - mse_raw))


def _expert_release_orthogonal_loss(
    expert_phi_i: torch.Tensor,
    expert_residual_release_state: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    dot = torch.sum(expert_phi_i * expert_residual_release_state, dim=-1)
    denom = torch.linalg.vector_norm(expert_phi_i, dim=-1).clamp_min(1e-6) * torch.linalg.vector_norm(
        expert_residual_release_state,
        dim=-1,
    ).clamp_min(1e-6)
    squared_cosine = (dot / denom).pow(2)
    return torch.mean(alpha.detach() * squared_cosine)


def _residual_state_noncollapse_loss(
    pooled_residual_release_state: torch.Tensor,
    *,
    std_floor: float,
) -> torch.Tensor:
    if pooled_residual_release_state.shape[0] <= 1:
        return torch.zeros((), device=pooled_residual_release_state.device, dtype=pooled_residual_release_state.dtype)
    std = torch.sqrt(pooled_residual_release_state.var(dim=0, unbiased=False).clamp_min(1e-8))
    return torch.mean(torch.relu(float(std_floor) - std))


def evaluate_factor_v29(dataloader: Iterable, model, device: torch.device) -> Tuple[float, float]:
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


def train_offline_factor_v29(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    loss_cfg: FactorMechanismV29LossConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.train()
    rows: list[dict[str, float]] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor_v29")
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
        history_after_release = _history_after_release_loss(
            out.expert_y_hist,
            out.expert_y_inv,
            out.expert_y_var,
            out.expert_history_budget,
            y,
            out.alpha,
        )
        release_first_increment = _release_first_increment_loss(out.expert_y_var, out.expert_y_inv, y, out.alpha)
        env_temporal = _env_temporal_loss(out.env_prior_posteriors, out.env_posteriors, out.env_shift)
        router_stability = _router_stability_loss(out.alpha, out.expert_y_hat, y)
        anchor_preservation = _anchor_preservation_loss(out.expert_shared_anchor, out.expert_phi_i)
        env_correction_sparsity = _suppression_sparsity_loss(out.expert_invariant_correction)
        release_confidence = _release_confidence_alignment_loss(
            out.expert_release_confidence,
            out.expert_y_inv,
            out.expert_y_inv + out.expert_y_var,
            y,
        )
        anchor_diversity = _pairwise_similarity_penalty(out.expert_shared_anchor)
        release_temporal = _release_temporal_loss(torch.stack(out.drift_expert_embs, dim=1), out.env_shift)
        observable_change_alignment = _observable_change_alignment_loss(
            out.expert_observable_change,
            out.env_shift,
            out.alpha,
        )
        change_readout = _change_readout_loss(
            out.expert_y_change_raw,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        delayed_change = _delayed_change_loss(
            out.expert_change_delta,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        multiscale_change_alignment = _multiscale_change_alignment_loss(
            out.expert_multiscale_change_score,
            out.expert_y_change_raw,
            out.expert_change_delta,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        release_priority_advantage = _release_priority_advantage_loss(
            out.expert_y_inv,
            out.expert_y_hist,
            out.expert_y_var,
            y,
            out.alpha,
            margin=loss_cfg.release_priority_margin,
        )
        completion_after_release = _completion_after_release_loss(
            out.expert_y_hat,
            out.expert_y_inv,
            out.expert_y_var,
            y,
            out.alpha,
            margin=loss_cfg.completion_after_release_margin,
        )
        release_state_advantage = _release_state_advantage_loss(
            out.expert_y_inv,
            out.expert_y_var,
            y,
            out.alpha,
            margin=loss_cfg.release_state_margin,
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
            out.expert_y_inv + out.expert_y_var,
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
        support_alignment = _support_alignment_loss(
            out.expert_support_score,
            out.expert_y_change_raw,
            out.expert_change_delta,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        suppression_alignment = _suppression_alignment_loss(
            out.expert_suppression_score,
            out.expert_y_change_raw,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        support_conditioned_advantage = _support_conditioned_advantage_loss(
            out.expert_y_inv,
            out.expert_y_hist,
            out.expert_y_change_raw,
            out.expert_y_var,
            y,
            out.alpha,
            margin=loss_cfg.support_advantage_margin,
        )
        suppression_gate_sparsity = _suppression_gate_sparsity_loss(
            out.expert_suppression_score,
            out.expert_release_budget,
            out.alpha,
        )
        local_env_consistency = _local_env_consistency_loss(
            out.expert_local_env_posteriors,
            out.env_posteriors,
            out.alpha,
        )
        invariant_correction_budget = _invariant_correction_budget_loss(
            out.expert_invariant_correction,
            out.expert_invariant_correction_gate,
            out.expert_release_budget,
            out.alpha,
        )
        residual_release_alignment = _residual_release_alignment_loss(
            out.expert_residual_release_state,
            out.expert_change_memory,
            out.expert_observable_change,
            out.alpha,
        )
        residual_release_advantage = _residual_release_advantage_loss(
            out.expert_y_inv,
            out.expert_y_hist,
            out.expert_y_change_raw,
            out.expert_y_var,
            y,
            out.alpha,
            margin=loss_cfg.residual_release_advantage_margin,
        )
        expert_release_orthogonal = _expert_release_orthogonal_loss(
            out.expert_phi_i,
            out.expert_residual_release_state,
            out.alpha,
        )
        residual_state_noncollapse = _residual_state_noncollapse_loss(
            out.pooled_residual_release_state,
            std_floor=loss_cfg.residual_state_std_floor,
        )

        total_loss = (
            rep_losses["total"]
            + l_vae
            + loss_cfg.lambda_load_balance * load_balance
            + loss_cfg.lambda_env_balance * env_balance
            + loss_cfg.lambda_expert_diversity * expert_diversity
            + loss_cfg.lambda_local_inv_forecast * local_inv_forecast
            + loss_cfg.lambda_history_after_release * history_after_release
            + loss_cfg.lambda_release_first_increment * release_first_increment
            + loss_cfg.lambda_env_temporal * env_temporal
            + loss_cfg.lambda_router_stability * router_stability
            + loss_cfg.lambda_anchor_preservation * anchor_preservation
            + loss_cfg.lambda_env_correction_sparsity * env_correction_sparsity
            + loss_cfg.lambda_release_confidence * release_confidence
            + loss_cfg.lambda_anchor_diversity * anchor_diversity
            + loss_cfg.lambda_release_temporal * release_temporal
            + loss_cfg.lambda_observable_change_alignment * observable_change_alignment
            + loss_cfg.lambda_change_readout * change_readout
            + loss_cfg.lambda_delayed_change * delayed_change
            + loss_cfg.lambda_multiscale_change_alignment * multiscale_change_alignment
            + loss_cfg.lambda_release_priority_advantage * release_priority_advantage
            + loss_cfg.lambda_release_state_advantage * release_state_advantage
            + loss_cfg.lambda_completion_after_release * completion_after_release
            + loss_cfg.lambda_release_null * release_null
            + loss_cfg.lambda_counterfactual_shuffle * counterfactual_shuffle
            + loss_cfg.lambda_budget_alignment * budget_alignment
            + loss_cfg.lambda_history_budget_cap * history_budget_cap
            + loss_cfg.lambda_support_alignment * support_alignment
            + loss_cfg.lambda_suppression_alignment * suppression_alignment
            + loss_cfg.lambda_support_conditioned_advantage * support_conditioned_advantage
            + loss_cfg.lambda_suppression_gate_sparsity * suppression_gate_sparsity
            + loss_cfg.lambda_local_env_consistency * local_env_consistency
            + loss_cfg.lambda_invariant_correction_budget * invariant_correction_budget
            + loss_cfg.lambda_residual_release_alignment * residual_release_alignment
            + loss_cfg.lambda_residual_release_advantage * residual_release_advantage
            + loss_cfg.lambda_expert_release_orthogonal * expert_release_orthogonal
            + loss_cfg.lambda_residual_state_noncollapse * residual_state_noncollapse
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
            "loss_history_after_release": float(history_after_release.detach().cpu().item()),
            "loss_release_first_increment": float(release_first_increment.detach().cpu().item()),
            "loss_env_temporal": float(env_temporal.detach().cpu().item()),
            "loss_router_stability": float(router_stability.detach().cpu().item()),
            "loss_anchor_preservation": float(anchor_preservation.detach().cpu().item()),
            "loss_env_correction_sparsity": float(env_correction_sparsity.detach().cpu().item()),
            "loss_release_confidence": float(release_confidence.detach().cpu().item()),
            "loss_anchor_diversity": float(anchor_diversity.detach().cpu().item()),
            "loss_release_temporal": float(release_temporal.detach().cpu().item()),
            "loss_observable_change_alignment": float(observable_change_alignment.detach().cpu().item()),
            "loss_change_readout": float(change_readout.detach().cpu().item()),
            "loss_delayed_change": float(delayed_change.detach().cpu().item()),
            "loss_multiscale_change_alignment": float(multiscale_change_alignment.detach().cpu().item()),
            "loss_release_priority_advantage": float(release_priority_advantage.detach().cpu().item()),
            "loss_release_state_advantage": float(release_state_advantage.detach().cpu().item()),
            "loss_completion_after_release": float(completion_after_release.detach().cpu().item()),
            "loss_release_null": float(release_null.detach().cpu().item()),
            "loss_counterfactual_shuffle": float(counterfactual_shuffle.detach().cpu().item()),
            "loss_budget_alignment": float(budget_alignment.detach().cpu().item()),
            "loss_history_budget_cap": float(history_budget_cap.detach().cpu().item()),
            "loss_support_alignment": float(support_alignment.detach().cpu().item()),
            "loss_suppression_alignment": float(suppression_alignment.detach().cpu().item()),
            "loss_support_conditioned_advantage": float(support_conditioned_advantage.detach().cpu().item()),
            "loss_suppression_gate_sparsity": float(suppression_gate_sparsity.detach().cpu().item()),
            "loss_local_env_consistency": float(local_env_consistency.detach().cpu().item()),
            "loss_invariant_correction_budget": float(invariant_correction_budget.detach().cpu().item()),
            "loss_residual_release_alignment": float(residual_release_alignment.detach().cpu().item()),
            "loss_residual_release_advantage": float(residual_release_advantage.detach().cpu().item()),
            "loss_expert_release_orthogonal": float(expert_release_orthogonal.detach().cpu().item()),
            "loss_residual_state_noncollapse": float(residual_state_noncollapse.detach().cpu().item()),
        }
        rows.append(row)
        pbar.update(x.size(0))
    pbar.close()
    return rows
