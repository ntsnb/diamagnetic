from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.factor_mechanism_v26 import FactorMechanismV26Config, build_factor_mechanism_v26


class FactorOutputsV29:
    """Forwarding output container for the V29 expert-local invariant-release line."""

    def __init__(self, base, **kwargs) -> None:
        self.base = base
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


@dataclass
class FactorMechanismV29Config:
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
    change_memory_hidden_dim: int = 96
    env_count: int = 4
    router_env_hidden_dim: int = 64
    invariant_release_hidden_dim: int = 128
    router_invariant_hidden_dim: int = 64
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


class ExpertLocalInvariantReleaseCorrector(nn.Module):
    """Per-expert correction head that keeps invariant and release states co-sourced.

    V29 treats `phi_i` as an environment-conditioned correction of an expert
    anchor and treats the served drift state as a learned residual-release
    state. Compatibility aliases can still expose `delta_phi_v`, but the native
    Stage III signal is the release state package.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        repr_dim: int,
        out_dim: int,
        env_count: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.env_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_count),
        )
        invariant_input_dim = feature_dim + env_count
        self.invariant_residual_head = nn.Sequential(
            nn.Linear(invariant_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )
        self.invariant_correction_head = nn.Sequential(
            nn.Linear(invariant_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )
        self.invariant_correction_gate = nn.Sequential(
            nn.Linear(invariant_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.Sigmoid(),
        )
        release_input_dim = feature_dim + env_count + repr_dim
        self.residual_release_state_head = nn.Sequential(
            nn.Linear(release_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, repr_dim),
        )
        release_joint_dim = release_input_dim + repr_dim
        self.release_support_head = nn.Sequential(
            nn.Linear(release_joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.release_suppression_head = nn.Sequential(
            nn.Linear(release_joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.release_delta_y_head = nn.Sequential(
            nn.Linear(release_joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.release_suppression_y_head = nn.Sequential(
            nn.Linear(release_joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        *,
        shared_anchor: torch.Tensor,
        base_phi_i: torch.Tensor,
        env_correction: torch.Tensor,
        raw_release: torch.Tensor,
        release_state: torch.Tensor,
        change_memory: torch.Tensor,
        observable_change: torch.Tensor,
        release_budget: torch.Tensor,
        release_confidence: torch.Tensor,
        history_budget: torch.Tensor,
        raw_change_y: torch.Tensor,
        change_delta_y: torch.Tensor,
        env_shift: torch.Tensor,
        env_posterior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature = torch.cat(
            [
                shared_anchor,
                base_phi_i,
                env_correction,
                raw_release,
                release_state,
                change_memory,
                observable_change,
                release_budget,
                release_confidence,
                history_budget,
                raw_change_y,
                change_delta_y,
                env_shift,
                env_posterior,
            ],
            dim=-1,
        )
        local_env_logits = self.env_head(feature)
        local_env_posterior = torch.softmax(local_env_logits, dim=-1)
        invariant_input = torch.cat([feature, local_env_posterior], dim=-1)
        invariant_residual = self.invariant_residual_head(invariant_input)
        invariant_correction = self.invariant_correction_head(invariant_input)
        correction_gate = self.invariant_correction_gate(invariant_input)
        phi_i = shared_anchor + invariant_residual - correction_gate * invariant_correction

        release_input = torch.cat([feature, local_env_posterior, phi_i], dim=-1)
        residual_release_state = self.residual_release_state_head(release_input)
        release_joint = torch.cat([release_input, residual_release_state], dim=-1)
        support_score = torch.sigmoid(self.release_support_head(release_joint))
        suppression_score = torch.sigmoid(self.release_suppression_head(release_joint))
        release_delta_y = change_delta_y + self.release_delta_y_head(release_joint)
        release_suppression_y = self.release_suppression_y_head(release_joint)
        corrected_change = raw_change_y + support_score * release_delta_y - suppression_score * release_suppression_y

        servable_change = torch.cat(
            [
                phi_i,
                residual_release_state,
                invariant_correction,
                invariant_residual,
                local_env_posterior,
                env_shift,
                support_score,
                suppression_score,
                release_budget,
                release_confidence,
                raw_release,
                change_memory,
                observable_change,
                release_delta_y,
                release_suppression_y,
                corrected_change,
                raw_change_y,
                change_delta_y,
            ],
            dim=-1,
        )
        return {
            "local_env_logits": local_env_logits,
            "local_env_posterior": local_env_posterior,
            "invariant_residual": invariant_residual,
            "invariant_correction": invariant_correction,
            "invariant_correction_gate": correction_gate,
            "phi_i": phi_i,
            "residual_release_state": residual_release_state,
            "support_score": support_score,
            "suppression_score": suppression_score,
            "release_delta_y": release_delta_y,
            "release_suppression_y": release_suppression_y,
            "corrected_change": corrected_change,
            "servable_change": servable_change,
        }


class FactorMechanismV29(nn.Module):
    """V29 expert-local invariant correction with directly served release state."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        corrector_heads: nn.ModuleList,
        router_invariant_head: nn.Module,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.corrector_heads = corrector_heads
        self.router_invariant_head = router_invariant_head
        self.use_vae = getattr(base_model, "use_vae", True)
        self.vae_eval_mode = getattr(base_model, "vae_eval_mode", "mean")

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV29:
        base = self.base_model(x)
        pooled_env_shift = base.env_shift.mean(dim=2)
        pooled_env_posteriors = base.env_posteriors.mean(dim=2)
        shared_anchor = base.expert_base + base.expert_invariant_adapter

        local_env_logits_parts: list[torch.Tensor] = []
        local_env_post_parts: list[torch.Tensor] = []
        invariant_residual_parts: list[torch.Tensor] = []
        invariant_correction_parts: list[torch.Tensor] = []
        invariant_gate_parts: list[torch.Tensor] = []
        phi_i_parts: list[torch.Tensor] = []
        residual_release_parts: list[torch.Tensor] = []
        support_score_parts: list[torch.Tensor] = []
        suppression_score_parts: list[torch.Tensor] = []
        release_delta_y_parts: list[torch.Tensor] = []
        release_suppression_y_parts: list[torch.Tensor] = []
        corrected_change_parts: list[torch.Tensor] = []
        servable_change_parts: list[torch.Tensor] = []

        for expert_idx, corrector in enumerate(self.corrector_heads):
            corrected = corrector(
                shared_anchor=shared_anchor[:, expert_idx, :],
                base_phi_i=base.expert_phi_i[:, expert_idx, :],
                env_correction=base.expert_env_correction[:, expert_idx, :],
                raw_release=base.expert_raw_release[:, expert_idx, :],
                release_state=base.expert_release_state[:, expert_idx, :],
                change_memory=base.expert_change_memory[:, expert_idx, :],
                observable_change=base.expert_observable_change[:, expert_idx, :],
                release_budget=base.expert_release_budget[:, expert_idx, :],
                release_confidence=base.expert_release_confidence[:, expert_idx, :],
                history_budget=base.expert_history_budget[:, expert_idx, :],
                raw_change_y=base.expert_y_change_raw[:, expert_idx, :],
                change_delta_y=base.expert_change_delta[:, expert_idx, :],
                env_shift=pooled_env_shift[:, expert_idx, :],
                env_posterior=pooled_env_posteriors[:, expert_idx, :],
            )
            local_env_logits_parts.append(corrected["local_env_logits"])
            local_env_post_parts.append(corrected["local_env_posterior"])
            invariant_residual_parts.append(corrected["invariant_residual"])
            invariant_correction_parts.append(corrected["invariant_correction"])
            invariant_gate_parts.append(corrected["invariant_correction_gate"])
            phi_i_parts.append(corrected["phi_i"])
            residual_release_parts.append(corrected["residual_release_state"])
            support_score_parts.append(corrected["support_score"])
            suppression_score_parts.append(corrected["suppression_score"])
            release_delta_y_parts.append(corrected["release_delta_y"])
            release_suppression_y_parts.append(corrected["release_suppression_y"])
            corrected_change_parts.append(corrected["corrected_change"])
            servable_change_parts.append(corrected["servable_change"])

        expert_local_env_logits = torch.stack(local_env_logits_parts, dim=1)
        expert_local_env_posteriors = torch.stack(local_env_post_parts, dim=1)
        expert_invariant_residual = torch.stack(invariant_residual_parts, dim=1)
        expert_invariant_correction = torch.stack(invariant_correction_parts, dim=1)
        expert_invariant_correction_gate = torch.stack(invariant_gate_parts, dim=1)
        expert_phi_i = torch.stack(phi_i_parts, dim=1)
        expert_residual_release_state = torch.stack(residual_release_parts, dim=1)
        expert_release_support_score = torch.stack(support_score_parts, dim=1)
        expert_release_suppression_score = torch.stack(suppression_score_parts, dim=1)
        expert_release_delta_y = torch.stack(release_delta_y_parts, dim=1)
        expert_release_suppression_y = torch.stack(release_suppression_y_parts, dim=1)
        expert_corrected_change = torch.stack(corrected_change_parts, dim=1)
        expert_servable_change = torch.stack(servable_change_parts, dim=1)

        anchor_norm = torch.linalg.vector_norm(shared_anchor, dim=-1, keepdim=True)
        correction_norm = torch.linalg.vector_norm(expert_invariant_correction, dim=-1, keepdim=True)
        release_norm = torch.linalg.vector_norm(expert_residual_release_state, dim=-1, keepdim=True)
        router_features = torch.cat(
            [
                expert_local_env_posteriors,
                pooled_env_shift,
                anchor_norm,
                correction_norm,
                release_norm,
                expert_release_support_score,
                expert_release_suppression_score,
                base.expert_release_budget,
                base.expert_release_confidence,
            ],
            dim=-1,
        )
        invariant_router_bias = self.router_invariant_head(router_features.reshape(router_features.shape[0], -1))
        alpha_logits = base.alpha_logits + invariant_router_bias
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)
        alpha_expanded = alpha.unsqueeze(-1)

        y_inv = torch.sum(alpha_expanded * base.expert_y_inv, dim=1)
        y_hist = torch.sum(alpha_expanded * base.expert_y_hist, dim=1)
        y_var = torch.sum(alpha_expanded * expert_corrected_change, dim=1)
        y_hat = y_inv + y_hist + y_var
        expert_y_hat = base.expert_y_inv + base.expert_y_hist + expert_corrected_change

        pooled_shared_anchor = torch.sum(alpha_expanded * shared_anchor, dim=1)
        pooled_local_env_logits = torch.sum(alpha_expanded * expert_local_env_logits, dim=1)
        pooled_local_env_posteriors = torch.sum(alpha_expanded * expert_local_env_posteriors, dim=1)
        pooled_invariant_residual = torch.sum(alpha_expanded * expert_invariant_residual, dim=1)
        pooled_invariant_correction = torch.sum(alpha_expanded * expert_invariant_correction, dim=1)
        pooled_invariant_correction_gate = torch.sum(alpha_expanded * expert_invariant_correction_gate, dim=1)
        pooled_phi_i = torch.sum(alpha_expanded * expert_phi_i, dim=1)
        pooled_residual_release_state = torch.sum(alpha_expanded * expert_residual_release_state, dim=1)
        pooled_release_support_score = torch.sum(alpha_expanded * expert_release_support_score, dim=1)
        pooled_release_suppression_score = torch.sum(alpha_expanded * expert_release_suppression_score, dim=1)
        pooled_release_delta_y = torch.sum(alpha_expanded * expert_release_delta_y, dim=1)
        pooled_release_suppression_y = torch.sum(alpha_expanded * expert_release_suppression_y, dim=1)
        pooled_corrected_change = torch.sum(alpha_expanded * expert_corrected_change, dim=1)
        pooled_servable_change = torch.sum(alpha_expanded * expert_servable_change, dim=1)
        expert_phi_v = expert_phi_i + expert_residual_release_state
        pooled_phi_v = pooled_phi_i + pooled_residual_release_state

        return FactorOutputsV29(
            base,
            y_hat=y_hat,
            y_inv=y_inv,
            y_hist=y_hist,
            y_var=y_var,
            alpha=alpha,
            alpha_logits=alpha_logits,
            env_router_bias=base.env_router_bias + invariant_router_bias,
            invariant_router_bias=invariant_router_bias,
            expert_y_var=expert_corrected_change,
            expert_y_hat=expert_y_hat,
            expert_shared_anchor=shared_anchor,
            expert_local_env_logits=expert_local_env_logits,
            expert_local_env_posteriors=expert_local_env_posteriors,
            expert_invariant_residual=expert_invariant_residual,
            expert_invariant_correction=expert_invariant_correction,
            expert_invariant_correction_gate=expert_invariant_correction_gate,
            expert_phi_i=expert_phi_i,
            expert_residual_release_state=expert_residual_release_state,
            expert_release_support_score=expert_release_support_score,
            expert_release_suppression_score=expert_release_suppression_score,
            expert_release_delta_y=expert_release_delta_y,
            expert_release_suppression_y=expert_release_suppression_y,
            expert_corrected_change=expert_corrected_change,
            expert_servable_change=expert_servable_change,
            pooled_shared_anchor=pooled_shared_anchor,
            pooled_local_env_logits=pooled_local_env_logits,
            pooled_local_env_posteriors=pooled_local_env_posteriors,
            pooled_invariant_residual=pooled_invariant_residual,
            pooled_invariant_correction=pooled_invariant_correction,
            pooled_invariant_correction_gate=pooled_invariant_correction_gate,
            pooled_phi_i=pooled_phi_i,
            pooled_residual_release_state=pooled_residual_release_state,
            pooled_release_support_score=pooled_release_support_score,
            pooled_release_suppression_score=pooled_release_suppression_score,
            pooled_release_delta_y=pooled_release_delta_y,
            pooled_release_suppression_y=pooled_release_suppression_y,
            pooled_corrected_change=pooled_corrected_change,
            pooled_servable_change=pooled_servable_change,
            # Compatibility aliases for inherited losses and generator code.
            expert_delta_phi_v=expert_residual_release_state,
            pooled_delta_phi_v=pooled_residual_release_state,
            expert_phi_v=expert_phi_v,
            pooled_phi_v=pooled_phi_v,
            expert_support_score=expert_release_support_score,
            pooled_support_score=pooled_release_support_score,
            expert_suppression_score=expert_release_suppression_score,
            pooled_suppression_score=pooled_release_suppression_score,
            expert_supported_delta=expert_release_delta_y,
            pooled_supported_delta=pooled_release_delta_y,
            expert_suppression_vector=expert_release_suppression_y,
            pooled_suppression_vector=pooled_release_suppression_y,
            expert_multiscale_change_score=expert_release_support_score,
            pooled_multiscale_change_score=pooled_release_support_score,
            pooled_multiscale_change=pooled_residual_release_state,
            pooled_raw_release=pooled_servable_change,
        )


def build_factor_mechanism_v29(cfg: FactorMechanismV29Config) -> FactorMechanismV29:
    base_model = build_factor_mechanism_v26(
        FactorMechanismV26Config(
            input_dim=cfg.input_dim,
            seq_len=cfg.seq_len,
            out_dim=cfg.out_dim,
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            latent_dim=cfg.latent_dim,
            repr_dim=cfg.repr_dim,
            expert_hidden_dim=cfg.expert_hidden_dim,
            history_hidden_dim=cfg.history_hidden_dim,
            release_hidden_dim=cfg.release_hidden_dim,
            change_memory_hidden_dim=cfg.change_memory_hidden_dim,
            env_count=cfg.env_count,
            router_env_hidden_dim=cfg.router_env_hidden_dim,
            temporal_encoder=cfg.temporal_encoder,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_revin=cfg.use_revin,
            target_index=cfg.target_index,
            use_vae=cfg.use_vae,
            vae_eval_mode=cfg.vae_eval_mode,
            zero_init_variant=cfg.zero_init_variant,
            use_projected_residual=cfg.use_projected_residual,
            projection_eps=cfg.projection_eps,
        )
    )
    corrector_feature_dim = 7 * cfg.repr_dim + 3 + 2 * cfg.out_dim + 2 * cfg.env_count
    corrector_heads = nn.ModuleList(
        [
            ExpertLocalInvariantReleaseCorrector(
                feature_dim=corrector_feature_dim,
                repr_dim=cfg.repr_dim,
                out_dim=cfg.out_dim,
                env_count=cfg.env_count,
                hidden_dim=cfg.invariant_release_hidden_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_experts)
        ]
    )
    router_invariant_head = nn.Sequential(
        nn.Linear(cfg.n_experts * (2 * cfg.env_count + 7), cfg.router_invariant_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.router_invariant_hidden_dim, cfg.n_experts),
    )
    return FactorMechanismV29(
        base_model=base_model,
        corrector_heads=corrector_heads,
        router_invariant_head=router_invariant_head,
    )
