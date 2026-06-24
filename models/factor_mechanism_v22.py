from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models.attention_blocks import TransformerEncoderConfig, VariableAxisExpert, build_temporal_encoder
from models.factor_mechanism import RevIN
from models.factor_mechanism_v18 import CausalEnvironmentFilter, _project_delta_orthogonal_to_anchor
from models.vae import VAEConfig, VAEDecoder, VAEEncoder


@dataclass
class ExpertLocalReleaseFirstBudgetOutput:
    shared: torch.Tensor
    base_seq: torch.Tensor
    invariant_adapter_seq: torch.Tensor
    env_correction_seq: torch.Tensor
    phi_i_seq: torch.Tensor
    release_proposal_seq: torch.Tensor
    release_confidence_seq: torch.Tensor
    release_floor_seq: torch.Tensor
    release_budget_seq: torch.Tensor
    budgeted_release_seq: torch.Tensor
    release_state_seq: torch.Tensor
    history_proposal_seq: torch.Tensor
    history_budget_seq: torch.Tensor
    budgeted_history_seq: torch.Tensor
    delta_phi_v_seq: torch.Tensor
    phi_v_seq: torch.Tensor
    env_prior_logits_seq: torch.Tensor
    env_prior_posteriors_seq: torch.Tensor
    env_logits_seq: torch.Tensor
    env_posterior_seq: torch.Tensor
    env_shift_seq: torch.Tensor
    pooled_shared: torch.Tensor
    pooled_base: torch.Tensor
    pooled_invariant_adapter: torch.Tensor
    pooled_env_correction: torch.Tensor
    pooled_phi_i: torch.Tensor
    pooled_release_proposal: torch.Tensor
    pooled_release_confidence: torch.Tensor
    pooled_release_floor: torch.Tensor
    pooled_release_budget: torch.Tensor
    pooled_budgeted_release: torch.Tensor
    pooled_release_state: torch.Tensor
    pooled_history_proposal: torch.Tensor
    pooled_history_budget: torch.Tensor
    pooled_budgeted_history: torch.Tensor
    pooled_delta_phi_v: torch.Tensor
    pooled_phi_v: torch.Tensor
    pooled_env_prior_logits: torch.Tensor
    pooled_env_prior_posteriors: torch.Tensor
    pooled_env_logits: torch.Tensor
    pooled_env_posterior: torch.Tensor
    pooled_env_shift: torch.Tensor


@dataclass
class FactorOutputsV22:
    y_hat: torch.Tensor
    y_inv: torch.Tensor
    y_hist: torch.Tensor
    y_var: torch.Tensor
    z: torch.Tensor
    mu_z: torch.Tensor
    log_var_z: torch.Tensor
    x_recon: torch.Tensor
    alpha: torch.Tensor
    alpha_logits: torch.Tensor
    env_router_bias: torch.Tensor
    expert_embs: List[torch.Tensor]
    invariant_expert_embs: List[torch.Tensor]
    drift_expert_embs: List[torch.Tensor]
    expert_base: torch.Tensor
    expert_invariant_adapter: torch.Tensor
    expert_env_correction: torch.Tensor
    expert_phi_i: torch.Tensor
    expert_release_proposal: torch.Tensor
    expert_release_confidence: torch.Tensor
    expert_release_floor: torch.Tensor
    expert_release_budget: torch.Tensor
    expert_raw_release: torch.Tensor
    expert_release_state: torch.Tensor
    expert_history_proposal: torch.Tensor
    expert_history_budget: torch.Tensor
    expert_budgeted_history: torch.Tensor
    expert_delta_phi_v: torch.Tensor
    expert_phi_v: torch.Tensor
    expert_y_inv: torch.Tensor
    expert_y_hist: torch.Tensor
    expert_y_var: torch.Tensor
    expert_y_hat: torch.Tensor
    env_prior_logits: torch.Tensor
    env_prior_posteriors: torch.Tensor
    env_logits: torch.Tensor
    env_posteriors: torch.Tensor
    env_shift: torch.Tensor
    pooled_shared: torch.Tensor
    pooled_base: torch.Tensor
    pooled_invariant_adapter: torch.Tensor
    pooled_env_correction: torch.Tensor
    pooled_phi_i: torch.Tensor
    pooled_release_proposal: torch.Tensor
    pooled_release_confidence: torch.Tensor
    pooled_release_floor: torch.Tensor
    pooled_release_budget: torch.Tensor
    pooled_budgeted_release: torch.Tensor
    pooled_release_state: torch.Tensor
    pooled_history_proposal: torch.Tensor
    pooled_history_budget: torch.Tensor
    pooled_budgeted_history: torch.Tensor
    pooled_raw_release: torch.Tensor
    pooled_delta_phi_v: torch.Tensor
    pooled_phi_v: torch.Tensor
    x_used: torch.Tensor


class ExpertLocalReleaseFirstBudgetHead(nn.Module):
    """Release-driven expert-local correction with explicit increment-to-state integration."""

    def __init__(
        self,
        *,
        input_dim: int,
        repr_dim: int,
        hidden_dim: int,
        history_hidden_dim: int,
        release_hidden_dim: int,
        env_count: int,
        dropout: float = 0.0,
        zero_init_variant: bool = True,
        use_projected_residual: bool = True,
        projection_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.use_projected_residual = bool(use_projected_residual)
        self.projection_eps = float(projection_eps)
        self.shared_post = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.env_filter = CausalEnvironmentFilter(hidden_dim=hidden_dim, env_count=env_count)
        self.base_head = nn.Linear(hidden_dim, repr_dim)
        self.invariant_adapter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )
        self.correction_gate = nn.Sequential(nn.Linear(hidden_dim, repr_dim), nn.Sigmoid())
        self.env_basis_correction = nn.Parameter(torch.empty(env_count, repr_dim))
        nn.init.xavier_uniform_(self.env_basis_correction)

        self.release_hidden_init = nn.Parameter(torch.zeros(release_hidden_dim))
        self.release_update = nn.GRUCell(hidden_dim + repr_dim + repr_dim + env_count, release_hidden_dim)
        self.release_head = nn.Linear(release_hidden_dim, repr_dim)
        self.transition_gate = nn.Sequential(nn.Linear(hidden_dim, repr_dim), nn.Tanh())
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim, repr_dim), nn.Tanh())
        self.env_basis_transition = nn.Parameter(torch.empty(env_count, repr_dim))
        nn.init.xavier_uniform_(self.env_basis_transition)
        self.release_confidence_head = nn.Linear(release_hidden_dim + hidden_dim + env_count, 1)
        self.release_floor_head = nn.Linear(hidden_dim + env_count + env_count, 1)
        self.release_budget_head = nn.Linear(release_hidden_dim + hidden_dim + env_count + env_count, 1)

        self.history_hidden_init = nn.Parameter(torch.zeros(history_hidden_dim))
        self.history_update = nn.GRUCell(hidden_dim + repr_dim + repr_dim, history_hidden_dim)
        self.history_head = nn.Linear(history_hidden_dim, repr_dim)
        self.history_context_gate = nn.Sequential(nn.Linear(hidden_dim, repr_dim), nn.Tanh())
        if zero_init_variant:
            nn.init.zeros_(self.release_head.weight)
            nn.init.zeros_(self.release_head.bias)

    def forward(self, expert_state: torch.Tensor) -> ExpertLocalReleaseFirstBudgetOutput:
        shared = self.shared_post(expert_state)
        env_prior_logits, env_prior_posteriors, env_logits, env_posteriors, env_shift = self.env_filter(shared)

        base = self.base_head(shared)
        invariant_adapter = self.invariant_adapter_head(shared)
        env_correction_basis = torch.einsum("ble,er->blr", env_posteriors, self.env_basis_correction)
        env_correction = self.correction_gate(shared) * env_correction_basis
        phi_i = base + invariant_adapter - env_correction

        batch, steps, _ = shared.shape
        release_hidden = self.release_hidden_init.unsqueeze(0).expand(batch, -1)
        history_hidden = self.history_hidden_init.unsqueeze(0).expand(batch, -1)
        release_state = torch.zeros_like(phi_i[:, 0, :])
        release_proposal_parts: list[torch.Tensor] = []
        release_confidence_parts: list[torch.Tensor] = []
        release_floor_parts: list[torch.Tensor] = []
        release_budget_parts: list[torch.Tensor] = []
        budgeted_release_parts: list[torch.Tensor] = []
        release_state_parts: list[torch.Tensor] = []
        history_proposal_parts: list[torch.Tensor] = []
        history_budget_parts: list[torch.Tensor] = []
        budgeted_history_parts: list[torch.Tensor] = []

        for step in range(steps):
            shared_t = shared[:, step, :]
            phi_i_t = phi_i[:, step, :]
            env_shift_t = env_shift[:, step, :]
            env_post_t = env_posteriors[:, step, :]

            release_hidden = self.release_update(
                torch.cat([shared_t, phi_i_t, release_state, env_shift_t], dim=-1),
                release_hidden,
            )
            release_base = self.release_head(release_hidden)
            transition = self.transition_gate(shared_t) * torch.matmul(env_shift_t, self.env_basis_transition)
            context = self.context_gate(shared_t) * phi_i_t
            release_proposal = release_base + transition + context
            if self.use_projected_residual:
                release_proposal = _project_delta_orthogonal_to_anchor(release_proposal, phi_i_t, self.projection_eps)

            release_confidence = torch.sigmoid(
                self.release_confidence_head(torch.cat([release_hidden, shared_t, env_post_t], dim=-1))
            )
            shift_mag = torch.sqrt(env_shift_t.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
            release_floor = torch.sigmoid(
                self.release_floor_head(torch.cat([shared_t, env_shift_t, env_post_t], dim=-1))
            ) * shift_mag
            raw_release_budget = torch.sigmoid(
                self.release_budget_head(torch.cat([release_hidden, shared_t, env_post_t, env_shift_t], dim=-1))
            )
            release_budget = release_floor + (1.0 - release_floor) * raw_release_budget
            history_budget = 1.0 - release_budget
            budgeted_release = release_budget * release_confidence * release_proposal
            release_state = release_state + budgeted_release

            release_adjusted_remainder = phi_i_t - release_state
            history_hidden = self.history_update(
                torch.cat([shared_t, release_adjusted_remainder, release_state], dim=-1),
                history_hidden,
            )
            history_proposal = self.history_head(history_hidden) + self.history_context_gate(shared_t) * release_adjusted_remainder
            budgeted_history = history_budget * history_proposal

            release_proposal_parts.append(release_proposal)
            release_confidence_parts.append(release_confidence)
            release_floor_parts.append(release_floor)
            release_budget_parts.append(release_budget)
            budgeted_release_parts.append(budgeted_release)
            release_state_parts.append(release_state)
            history_proposal_parts.append(history_proposal)
            history_budget_parts.append(history_budget)
            budgeted_history_parts.append(budgeted_history)

        release_proposal_seq = torch.stack(release_proposal_parts, dim=1)
        release_confidence_seq = torch.stack(release_confidence_parts, dim=1)
        release_floor_seq = torch.stack(release_floor_parts, dim=1)
        release_budget_seq = torch.stack(release_budget_parts, dim=1)
        budgeted_release_seq = torch.stack(budgeted_release_parts, dim=1)
        release_state_seq = torch.stack(release_state_parts, dim=1)
        history_proposal_seq = torch.stack(history_proposal_parts, dim=1)
        history_budget_seq = torch.stack(history_budget_parts, dim=1)
        budgeted_history_seq = torch.stack(budgeted_history_parts, dim=1)
        delta_phi_v = release_state_seq
        phi_v = phi_i + delta_phi_v

        return ExpertLocalReleaseFirstBudgetOutput(
            shared=shared,
            base_seq=base,
            invariant_adapter_seq=invariant_adapter,
            env_correction_seq=env_correction,
            phi_i_seq=phi_i,
            release_proposal_seq=release_proposal_seq,
            release_confidence_seq=release_confidence_seq,
            release_floor_seq=release_floor_seq,
            release_budget_seq=release_budget_seq,
            budgeted_release_seq=budgeted_release_seq,
            release_state_seq=release_state_seq,
            history_proposal_seq=history_proposal_seq,
            history_budget_seq=history_budget_seq,
            budgeted_history_seq=budgeted_history_seq,
            delta_phi_v_seq=delta_phi_v,
            phi_v_seq=phi_v,
            env_prior_logits_seq=env_prior_logits,
            env_prior_posteriors_seq=env_prior_posteriors,
            env_logits_seq=env_logits,
            env_posterior_seq=env_posteriors,
            env_shift_seq=env_shift,
            pooled_shared=shared.mean(dim=1),
            pooled_base=base.mean(dim=1),
            pooled_invariant_adapter=invariant_adapter.mean(dim=1),
            pooled_env_correction=env_correction.mean(dim=1),
            pooled_phi_i=phi_i.mean(dim=1),
            pooled_release_proposal=release_proposal_seq.mean(dim=1),
            pooled_release_confidence=release_confidence_seq.mean(dim=1),
            pooled_release_floor=release_floor_seq.mean(dim=1),
            pooled_release_budget=release_budget_seq.mean(dim=1),
            pooled_budgeted_release=budgeted_release_seq.mean(dim=1),
            pooled_release_state=release_state_seq.mean(dim=1),
            pooled_history_proposal=history_proposal_seq.mean(dim=1),
            pooled_history_budget=history_budget_seq.mean(dim=1),
            pooled_budgeted_history=budgeted_history_seq.mean(dim=1),
            pooled_delta_phi_v=delta_phi_v.mean(dim=1),
            pooled_phi_v=phi_v.mean(dim=1),
            pooled_env_prior_logits=env_prior_logits.mean(dim=1),
            pooled_env_prior_posteriors=env_prior_posteriors.mean(dim=1),
            pooled_env_logits=env_logits.mean(dim=1),
            pooled_env_posterior=env_posteriors.mean(dim=1),
            pooled_env_shift=env_shift.mean(dim=1),
        )


class FactorMechanismV22(nn.Module):
    """V22 base model with release-driven state accumulation."""

    def __init__(
        self,
        *,
        temporal_encoder: nn.Module,
        vae_encoder: nn.Module | None,
        vae_decoder: nn.Module | None,
        alpha_head: nn.Module,
        router_context_head: nn.Module,
        experts: nn.ModuleList,
        dual_stream_heads: nn.ModuleList,
        invariant_output_heads: nn.ModuleList,
        history_output_heads: nn.ModuleList,
        variant_output_heads: nn.ModuleList,
        gate_proj: nn.Module | None = None,
        revin: RevIN | None = None,
        target_index: int = 0,
        use_vae: bool = True,
        vae_eval_mode: str = "mean",
    ) -> None:
        super().__init__()
        if vae_eval_mode not in {"mean", "sample"}:
            raise ValueError(f"Unsupported vae_eval_mode: {vae_eval_mode}")
        self.temporal_encoder = temporal_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.alpha_head = alpha_head
        self.router_context_head = router_context_head
        self.experts = experts
        self.dual_stream_heads = dual_stream_heads
        self.invariant_output_heads = invariant_output_heads
        self.history_output_heads = history_output_heads
        self.variant_output_heads = variant_output_heads
        self.gate_proj = gate_proj
        self.revin = revin
        self.target_index = int(target_index)
        self.use_vae = bool(use_vae)
        self.vae_eval_mode = vae_eval_mode

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training or self.vae_eval_mode == "sample":
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _target_scale(self, mean: torch.Tensor, std: torch.Tensor, out_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_mean = mean[:, :, self.target_index].expand(-1, out_dim)
        target_std = std[:, :, self.target_index].expand(-1, out_dim)
        return target_mean, target_std

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV22:
        if self.revin is not None:
            x_model, mean, std = self.revin.normalize(x)
        else:
            x_model = x
            mean = std = None

        h = self.temporal_encoder(x_model)
        if self.use_vae:
            if self.vae_encoder is None or self.vae_decoder is None:
                raise RuntimeError("VAE components must exist when use_vae=True")
            mu_z, log_var_z = self.vae_encoder(h)
            z = self._reparameterize(mu_z, log_var_z)
            x_recon = self.vae_decoder(z).view(x.shape[0], x.shape[1], x.shape[2])
            if self.revin is not None and mean is not None and std is not None:
                x_recon = self.revin.denormalize_all(x_recon, mean, std)
        else:
            if self.gate_proj is None:
                raise RuntimeError("gate_proj must be provided when use_vae=False")
            z = self.gate_proj(h)
            mu_z = z
            log_var_z = torch.zeros_like(z)
            x_recon = x

        expert_embs: List[torch.Tensor] = []
        invariant_expert_embs: List[torch.Tensor] = []
        drift_expert_embs: List[torch.Tensor] = []
        local_invariant_models: list[torch.Tensor] = []
        local_history_models: list[torch.Tensor] = []
        local_variant_models: list[torch.Tensor] = []
        pooled_shared_parts: list[torch.Tensor] = []
        pooled_base_parts: list[torch.Tensor] = []
        pooled_invariant_adapter_parts: list[torch.Tensor] = []
        pooled_env_correction_parts: list[torch.Tensor] = []
        pooled_phi_i_parts: list[torch.Tensor] = []
        pooled_release_proposal_parts: list[torch.Tensor] = []
        pooled_release_confidence_parts: list[torch.Tensor] = []
        pooled_release_floor_parts: list[torch.Tensor] = []
        pooled_release_budget_parts: list[torch.Tensor] = []
        pooled_budgeted_release_parts: list[torch.Tensor] = []
        pooled_release_state_parts: list[torch.Tensor] = []
        pooled_history_proposal_parts: list[torch.Tensor] = []
        pooled_history_budget_parts: list[torch.Tensor] = []
        pooled_budgeted_history_parts: list[torch.Tensor] = []
        pooled_delta_parts: list[torch.Tensor] = []
        pooled_phi_v_parts: list[torch.Tensor] = []
        pooled_env_prior_logits_parts: list[torch.Tensor] = []
        pooled_env_prior_post_parts: list[torch.Tensor] = []
        pooled_env_logits_parts: list[torch.Tensor] = []
        pooled_env_post_parts: list[torch.Tensor] = []
        pooled_env_shift_parts: list[torch.Tensor] = []
        env_prior_logits_parts: list[torch.Tensor] = []
        env_prior_post_parts: list[torch.Tensor] = []
        env_logits_parts: list[torch.Tensor] = []
        env_post_parts: list[torch.Tensor] = []
        env_shift_parts: list[torch.Tensor] = []

        for expert, head_module, inv_head, hist_head, var_head in zip(
            self.experts,
            self.dual_stream_heads,
            self.invariant_output_heads,
            self.history_output_heads,
            self.variant_output_heads,
        ):
            expert_state = expert(x_model)
            dual = head_module(expert_state)
            expert_embs.append(expert_state)
            invariant_expert_embs.append(dual.phi_i_seq)
            drift_expert_embs.append(dual.delta_phi_v_seq)
            pooled_shared_parts.append(dual.pooled_shared)
            pooled_base_parts.append(dual.pooled_base)
            pooled_invariant_adapter_parts.append(dual.pooled_invariant_adapter)
            pooled_env_correction_parts.append(dual.pooled_env_correction)
            pooled_phi_i_parts.append(dual.pooled_phi_i)
            pooled_release_proposal_parts.append(dual.pooled_release_proposal)
            pooled_release_confidence_parts.append(dual.pooled_release_confidence)
            pooled_release_floor_parts.append(dual.pooled_release_floor)
            pooled_release_budget_parts.append(dual.pooled_release_budget)
            pooled_budgeted_release_parts.append(dual.pooled_budgeted_release)
            pooled_release_state_parts.append(dual.pooled_release_state)
            pooled_history_proposal_parts.append(dual.pooled_history_proposal)
            pooled_history_budget_parts.append(dual.pooled_history_budget)
            pooled_budgeted_history_parts.append(dual.pooled_budgeted_history)
            pooled_delta_parts.append(dual.pooled_delta_phi_v)
            pooled_phi_v_parts.append(dual.pooled_phi_v)
            pooled_env_prior_logits_parts.append(dual.pooled_env_prior_logits)
            pooled_env_prior_post_parts.append(dual.pooled_env_prior_posteriors)
            pooled_env_logits_parts.append(dual.pooled_env_logits)
            pooled_env_post_parts.append(dual.pooled_env_posterior)
            pooled_env_shift_parts.append(dual.pooled_env_shift)
            env_prior_logits_parts.append(dual.env_prior_logits_seq)
            env_prior_post_parts.append(dual.env_prior_posteriors_seq)
            env_logits_parts.append(dual.env_logits_seq)
            env_post_parts.append(dual.env_posterior_seq)
            env_shift_parts.append(dual.env_shift_seq)
            local_invariant_models.append(inv_head(dual.pooled_phi_i))
            hist_input = torch.cat(
                [dual.pooled_phi_i - dual.pooled_release_state, dual.pooled_budgeted_history, dual.pooled_history_budget],
                dim=-1,
            )
            local_history_models.append(hist_head(hist_input))
            var_input = torch.cat(
                [
                    dual.pooled_release_state,
                    dual.pooled_budgeted_release,
                    dual.pooled_release_confidence,
                    dual.pooled_release_budget,
                    dual.pooled_env_shift,
                ],
                dim=-1,
            )
            local_variant_models.append(var_head(var_input))

        pooled_env_shift = torch.stack(pooled_env_shift_parts, dim=1)
        pooled_env_correction = torch.stack(pooled_env_correction_parts, dim=1)
        pooled_release_budget = torch.stack(pooled_release_budget_parts, dim=1)
        pooled_env_correction_norm = torch.linalg.vector_norm(pooled_env_correction, dim=-1, keepdim=True)
        router_features = torch.cat([pooled_env_shift, pooled_env_correction_norm, pooled_release_budget], dim=-1)
        env_router_bias = self.router_context_head(router_features.reshape(router_features.shape[0], -1))
        alpha_logits = self.alpha_head(z) + env_router_bias
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)

        pooled_shared = torch.stack(pooled_shared_parts, dim=1)
        expert_base = torch.stack(pooled_base_parts, dim=1)
        expert_invariant_adapter = torch.stack(pooled_invariant_adapter_parts, dim=1)
        expert_env_correction = torch.stack(pooled_env_correction_parts, dim=1)
        expert_phi_i = torch.stack(pooled_phi_i_parts, dim=1)
        expert_release_proposal = torch.stack(pooled_release_proposal_parts, dim=1)
        expert_release_confidence = torch.stack(pooled_release_confidence_parts, dim=1)
        expert_release_floor = torch.stack(pooled_release_floor_parts, dim=1)
        expert_release_budget = torch.stack(pooled_release_budget_parts, dim=1)
        expert_raw_release = torch.stack(pooled_budgeted_release_parts, dim=1)
        expert_release_state = torch.stack(pooled_release_state_parts, dim=1)
        expert_history_proposal = torch.stack(pooled_history_proposal_parts, dim=1)
        expert_history_budget = torch.stack(pooled_history_budget_parts, dim=1)
        expert_budgeted_history = torch.stack(pooled_budgeted_history_parts, dim=1)
        expert_delta_phi_v = torch.stack(pooled_delta_parts, dim=1)
        expert_phi_v = torch.stack(pooled_phi_v_parts, dim=1)
        env_prior_logits = torch.stack(env_prior_logits_parts, dim=1)
        env_prior_posteriors = torch.stack(env_prior_post_parts, dim=1)
        env_logits = torch.stack(env_logits_parts, dim=1)
        env_posteriors = torch.stack(env_post_parts, dim=1)
        env_shift = torch.stack(env_shift_parts, dim=1)

        expert_y_inv_model = torch.stack(local_invariant_models, dim=1)
        expert_y_hist_model = torch.stack(local_history_models, dim=1)
        expert_y_var_model = torch.stack(local_variant_models, dim=1)
        if self.revin is not None and mean is not None and std is not None:
            target_mean, target_std = self._target_scale(mean, std, expert_y_inv_model.shape[-1])
            expert_y_inv = expert_y_inv_model * target_std.unsqueeze(1) + target_mean.unsqueeze(1)
            expert_y_hist = expert_y_hist_model * target_std.unsqueeze(1)
            expert_y_var = expert_y_var_model * target_std.unsqueeze(1)
        else:
            expert_y_inv = expert_y_inv_model
            expert_y_hist = expert_y_hist_model
            expert_y_var = expert_y_var_model
        expert_y_hat = expert_y_inv + expert_y_var + expert_y_hist

        alpha_expanded = alpha.unsqueeze(-1)
        pooled_shared_fused = torch.sum(alpha_expanded * pooled_shared, dim=1)
        pooled_base = torch.sum(alpha_expanded * expert_base, dim=1)
        pooled_invariant_adapter = torch.sum(alpha_expanded * expert_invariant_adapter, dim=1)
        pooled_env_correction = torch.sum(alpha_expanded * expert_env_correction, dim=1)
        pooled_phi_i = torch.sum(alpha_expanded * expert_phi_i, dim=1)
        pooled_release_proposal = torch.sum(alpha_expanded * expert_release_proposal, dim=1)
        pooled_release_confidence = torch.sum(alpha_expanded * expert_release_confidence, dim=1)
        pooled_release_floor = torch.sum(alpha_expanded * expert_release_floor, dim=1)
        pooled_release_budget = torch.sum(alpha_expanded * expert_release_budget, dim=1)
        pooled_budgeted_release = torch.sum(alpha_expanded * expert_raw_release, dim=1)
        pooled_release_state = torch.sum(alpha_expanded * expert_release_state, dim=1)
        pooled_history_proposal = torch.sum(alpha_expanded * expert_history_proposal, dim=1)
        pooled_history_budget = torch.sum(alpha_expanded * expert_history_budget, dim=1)
        pooled_budgeted_history = torch.sum(alpha_expanded * expert_budgeted_history, dim=1)
        pooled_delta_phi_v = torch.sum(alpha_expanded * expert_delta_phi_v, dim=1)
        pooled_phi_v = pooled_phi_i + pooled_delta_phi_v
        y_inv = torch.sum(alpha_expanded * expert_y_inv, dim=1)
        y_hist = torch.sum(alpha_expanded * expert_y_hist, dim=1)
        y_var = torch.sum(alpha_expanded * expert_y_var, dim=1)
        y_hat = y_inv + y_var + y_hist

        return FactorOutputsV22(
            y_hat=y_hat,
            y_inv=y_inv,
            y_hist=y_hist,
            y_var=y_var,
            z=z,
            mu_z=mu_z,
            log_var_z=log_var_z,
            x_recon=x_recon,
            alpha=alpha,
            alpha_logits=alpha_logits,
            env_router_bias=env_router_bias,
            expert_embs=expert_embs,
            invariant_expert_embs=invariant_expert_embs,
            drift_expert_embs=drift_expert_embs,
            expert_base=expert_base,
            expert_invariant_adapter=expert_invariant_adapter,
            expert_env_correction=expert_env_correction,
            expert_phi_i=expert_phi_i,
            expert_release_proposal=expert_release_proposal,
            expert_release_confidence=expert_release_confidence,
            expert_release_floor=expert_release_floor,
            expert_release_budget=expert_release_budget,
            expert_raw_release=expert_raw_release,
            expert_release_state=expert_release_state,
            expert_history_proposal=expert_history_proposal,
            expert_history_budget=expert_history_budget,
            expert_budgeted_history=expert_budgeted_history,
            expert_delta_phi_v=expert_delta_phi_v,
            expert_phi_v=expert_phi_v,
            expert_y_inv=expert_y_inv,
            expert_y_hist=expert_y_hist,
            expert_y_var=expert_y_var,
            expert_y_hat=expert_y_hat,
            env_prior_logits=env_prior_logits,
            env_prior_posteriors=env_prior_posteriors,
            env_logits=env_logits,
            env_posteriors=env_posteriors,
            env_shift=env_shift,
            pooled_shared=pooled_shared_fused,
            pooled_base=pooled_base,
            pooled_invariant_adapter=pooled_invariant_adapter,
            pooled_env_correction=pooled_env_correction,
            pooled_phi_i=pooled_phi_i,
            pooled_release_proposal=pooled_release_proposal,
            pooled_release_confidence=pooled_release_confidence,
            pooled_release_floor=pooled_release_floor,
            pooled_release_budget=pooled_release_budget,
            pooled_budgeted_release=pooled_budgeted_release,
            pooled_release_state=pooled_release_state,
            pooled_history_proposal=pooled_history_proposal,
            pooled_history_budget=pooled_history_budget,
            pooled_budgeted_history=pooled_budgeted_history,
            pooled_raw_release=pooled_delta_phi_v,
            pooled_delta_phi_v=pooled_delta_phi_v,
            pooled_phi_v=pooled_phi_v,
            x_used=x,
        )


@dataclass
class FactorMechanismV22Config:
    input_dim: int
    seq_len: int
    out_dim: int
    d_model: int = 128
    n_experts: int = 4
    latent_dim: int = 64
    repr_dim: int = 64
    expert_hidden_dim: int = 128
    history_hidden_dim: int = 96
    release_hidden_dim: int = 96
    env_count: int = 4
    router_env_hidden_dim: int = 64
    temporal_encoder: str = "transformer"
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_revin: bool = False
    target_index: int = 0
    use_vae: bool = True
    vae_eval_mode: str = "mean"
    zero_init_variant: bool = True
    use_projected_residual: bool = True
    projection_eps: float = 1e-8


def build_factor_mechanism_v22(cfg: FactorMechanismV22Config) -> FactorMechanismV22:
    t_cfg = TransformerEncoderConfig(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=4 * cfg.d_model,
        dropout=cfg.dropout,
        pooling="mean",
        seq_len=cfg.seq_len,
    )
    temporal_encoder = build_temporal_encoder(cfg.temporal_encoder, cfg.input_dim, t_cfg)

    gate_proj = None
    if cfg.use_vae:
        v_cfg = VAEConfig(in_dim=cfg.d_model, latent_dim=cfg.latent_dim, hidden_dim=cfg.d_model)
        vae_encoder: nn.Module | None = VAEEncoder(v_cfg)
        vae_decoder: nn.Module | None = VAEDecoder(
            cfg.latent_dim,
            cfg.seq_len * cfg.input_dim,
            hidden_dim=cfg.d_model,
        )
    else:
        vae_encoder = None
        vae_decoder = None
        gate_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )

    alpha_head = nn.Sequential(
        nn.Linear(cfg.latent_dim, cfg.d_model),
        nn.ReLU(),
        nn.Linear(cfg.d_model, cfg.n_experts),
    )
    router_context_head = nn.Sequential(
        nn.Linear(cfg.n_experts * (cfg.env_count + 2), cfg.router_env_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.router_env_hidden_dim, cfg.n_experts),
    )
    experts = nn.ModuleList(
        [
            VariableAxisExpert(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_experts)
        ]
    )
    dual_stream_heads = nn.ModuleList(
        [
            ExpertLocalReleaseFirstBudgetHead(
                input_dim=cfg.d_model,
                repr_dim=cfg.repr_dim,
                hidden_dim=cfg.expert_hidden_dim,
                history_hidden_dim=cfg.history_hidden_dim,
                release_hidden_dim=cfg.release_hidden_dim,
                env_count=cfg.env_count,
                dropout=cfg.dropout,
                zero_init_variant=cfg.zero_init_variant,
                use_projected_residual=cfg.use_projected_residual,
                projection_eps=cfg.projection_eps,
            )
            for _ in range(cfg.n_experts)
        ]
    )
    invariant_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(cfg.repr_dim, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    history_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(2 * cfg.repr_dim + 1, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    variant_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(2 * cfg.repr_dim + 2 + cfg.env_count, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    revin = RevIN() if cfg.use_revin else None
    return FactorMechanismV22(
        temporal_encoder=temporal_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        alpha_head=alpha_head,
        router_context_head=router_context_head,
        experts=experts,
        dual_stream_heads=dual_stream_heads,
        invariant_output_heads=invariant_output_heads,
        history_output_heads=history_output_heads,
        variant_output_heads=variant_output_heads,
        gate_proj=gate_proj,
        revin=revin,
        target_index=cfg.target_index,
        use_vae=cfg.use_vae,
        vae_eval_mode=cfg.vae_eval_mode,
    )
