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
from train.offline_factor_v19 import _counterfactual_shuffle_loss, _release_null_loss
from train.offline_factor_v20 import _budget_alignment_loss, _history_budget_cap_loss, _budgeted_history_proxy_loss
from utils.progress import get_progress


@dataclass
class FactorMechanismV25LossConfig:
    beta_vae: float = 0.1
    lambda_load_balance: float = 0.05
    lambda_env_balance: float = 0.01
    lambda_expert_diversity: float = 0.01
    lambda_local_inv_forecast: float = 0.2
    lambda_history_after_release: float = 0.1
    lambda_release_first_increment: float = 0.45
    lambda_env_temporal: float = 0.05
    lambda_router_stability: float = 0.1
    lambda_anchor_preservation: float = 0.1
    lambda_env_correction_sparsity: float = 0.02
    lambda_release_confidence: float = 0.08
    lambda_anchor_diversity: float = 0.02
    lambda_release_temporal: float = 0.05
    lambda_observable_change_alignment: float = 0.08
    lambda_change_readout: float = 0.35
    lambda_delayed_change: float = 0.22
    lambda_change_quality_alignment: float = 0.15
    lambda_release_priority_advantage: float = 0.25
    release_priority_margin: float = 0.001
    lambda_release_state_advantage: float = 0.15
    release_state_margin: float = 0.001
    lambda_completion_after_release: float = 0.08
    completion_after_release_margin: float = 0.001
    lambda_release_null: float = 0.04
    lambda_counterfactual_shuffle: float = 0.08
    counterfactual_shuffle_margin: float = 0.001
    lambda_budget_alignment: float = 0.25
    lambda_history_budget_cap: float = 0.03
    representation: AlignedRepresentationLossConfig = field(default_factory=AlignedRepresentationLossConfig)


def _local_inv_forecast_loss(expert_y_inv: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    mse = torch.mean((expert_y_inv - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _release_first_increment_loss(
    expert_y_var: torch.Tensor,
    expert_y_inv: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = y.unsqueeze(1) - expert_y_inv.detach()
    mse = torch.mean((expert_y_var - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _history_after_release_loss(
    expert_y_hist: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_var: torch.Tensor,
    expert_history_budget: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = expert_history_budget.detach() * (y.unsqueeze(1) - expert_y_inv.detach() - expert_y_var.detach())
    mse = torch.mean((expert_y_hist - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _release_priority_advantage_loss(
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    expert_y_var: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    release_first = expert_y_inv + expert_y_var
    history_first = expert_y_inv + expert_y_hist
    mse_release = torch.mean((release_first - target).pow(2), dim=-1)
    mse_history = torch.mean((history_first - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * torch.relu(float(margin) + mse_release - mse_history))


def _completion_after_release_loss(
    expert_y_hat: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_var: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_hat)
    release_first = expert_y_inv + expert_y_var
    mse_full = torch.mean((expert_y_hat - target).pow(2), dim=-1)
    mse_release = torch.mean((release_first - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * torch.relu(float(margin) + mse_full - mse_release))


def _release_state_advantage_loss(
    expert_y_inv: torch.Tensor,
    expert_y_var: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    mse_inv = torch.mean((expert_y_inv - target).pow(2), dim=-1)
    mse_release = torch.mean((expert_y_inv + expert_y_var - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * torch.relu(float(margin) + mse_release - mse_inv))


def _observable_change_alignment_loss(
    expert_observable_change: torch.Tensor,
    env_shift_seq: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    env_shift_delta = torch.zeros_like(env_shift_seq)
    if env_shift_seq.shape[2] > 1:
        env_shift_delta[:, :, 1:, :] = env_shift_seq[:, :, 1:, :] - env_shift_seq[:, :, :-1, :]
    target_mag = torch.linalg.vector_norm(env_shift_delta, dim=-1).mean(dim=-1)
    pred_mag = torch.linalg.vector_norm(expert_observable_change, dim=-1)
    return torch.mean(alpha.detach() * (pred_mag - target_mag.detach()).pow(2))


def _vector_difference(x: torch.Tensor) -> torch.Tensor:
    diff = torch.zeros_like(x)
    if x.shape[-1] > 1:
        diff[..., 1:] = x[..., 1:] - x[..., :-1]
    return diff


def _change_residual_target(
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return y.unsqueeze(1) - expert_y_inv.detach() - expert_y_hist.detach()


def _change_readout_loss(
    expert_y_change_raw: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = _change_residual_target(expert_y_inv, expert_y_hist, y)
    mse = torch.mean((expert_y_change_raw - target).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _delayed_change_loss(
    expert_y_change_raw: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = _change_residual_target(expert_y_inv, expert_y_hist, y)
    pred_delta = _vector_difference(expert_y_change_raw)
    target_delta = _vector_difference(target)
    mse = torch.mean((pred_delta - target_delta).pow(2), dim=-1)
    return torch.mean(alpha.detach() * mse)


def _change_quality_alignment_loss(
    expert_change_quality_score: torch.Tensor,
    expert_y_change_raw: torch.Tensor,
    expert_y_inv: torch.Tensor,
    expert_y_hist: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    target = y.unsqueeze(1).expand_as(expert_y_inv)
    hist_only = expert_y_inv.detach() + expert_y_hist.detach()
    with_change = hist_only + expert_y_change_raw.detach()
    mse_hist = torch.mean((hist_only - target).pow(2), dim=-1, keepdim=True)
    mse_change = torch.mean((with_change - target).pow(2), dim=-1, keepdim=True)
    improvement = torch.clamp((mse_hist - mse_change) / mse_hist.clamp_min(1e-6), min=0.0, max=1.0)
    return torch.mean(alpha.detach().unsqueeze(-1) * (expert_change_quality_score - improvement).pow(2))


def evaluate_factor_v25(dataloader: Iterable, model, device: torch.device) -> Tuple[float, float]:
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


def train_offline_factor_v25(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    loss_cfg: FactorMechanismV25LossConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.train()
    rows: list[dict[str, float]] = []
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor_v25")
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
        release_first_increment = _release_first_increment_loss(
            out.expert_y_var,
            out.expert_y_inv,
            y,
            out.alpha,
        )
        env_temporal = _env_temporal_loss(out.env_prior_posteriors, out.env_posteriors, out.env_shift)
        router_stability = _router_stability_loss(out.alpha, out.expert_y_hat, y)
        anchor_preservation = _anchor_preservation_loss(out.expert_base + out.expert_invariant_adapter, out.expert_phi_i)
        env_correction_sparsity = _suppression_sparsity_loss(out.expert_env_correction)
        release_confidence = _release_confidence_alignment_loss(
            out.expert_release_confidence,
            out.expert_y_inv,
            out.expert_y_inv + out.expert_y_var,
            y,
        )
        anchor_diversity = _pairwise_similarity_penalty(out.expert_base)
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
            out.expert_y_change_raw,
            out.expert_y_inv,
            out.expert_y_hist,
            y,
            out.alpha,
        )
        change_quality_alignment = _change_quality_alignment_loss(
            out.expert_change_quality_score,
            out.expert_y_change_raw,
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
            + loss_cfg.lambda_change_quality_alignment * change_quality_alignment
            + loss_cfg.lambda_release_priority_advantage * release_priority_advantage
            + loss_cfg.lambda_release_state_advantage * release_state_advantage
            + loss_cfg.lambda_completion_after_release * completion_after_release
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
            "loss_change_quality_alignment": float(change_quality_alignment.detach().cpu().item()),
            "loss_release_priority_advantage": float(release_priority_advantage.detach().cpu().item()),
            "loss_release_state_advantage": float(release_state_advantage.detach().cpu().item()),
            "loss_completion_after_release": float(completion_after_release.detach().cpu().item()),
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


