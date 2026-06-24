from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models.attention_blocks import TransformerEncoderConfig, VariableAxisExpert, build_temporal_encoder
from models.factor_mechanism import RevIN
from models.factor_mechanism_v18 import ExpertLocalStructuredEnvReleaseHead
from models.vae import VAEConfig, VAEDecoder, VAEEncoder


@dataclass
class FactorOutputsV19:
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
    expert_anchor: torch.Tensor
    expert_suppression: torch.Tensor
    expert_phi_i: torch.Tensor
    expert_raw_release: torch.Tensor
    expert_release_confidence: torch.Tensor
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
    pooled_anchor: torch.Tensor
    pooled_suppression: torch.Tensor
    pooled_phi_i: torch.Tensor
    pooled_raw_release: torch.Tensor
    pooled_release_confidence: torch.Tensor
    pooled_delta_phi_v: torch.Tensor
    pooled_phi_v: torch.Tensor
    x_used: torch.Tensor


class FactorMechanismV19(nn.Module):
    """V19 base model with explicit history-proxy and release-increment readouts."""

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

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV19:
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
        pooled_anchor_parts: list[torch.Tensor] = []
        pooled_suppression_parts: list[torch.Tensor] = []
        pooled_phi_i_parts: list[torch.Tensor] = []
        pooled_raw_release_parts: list[torch.Tensor] = []
        pooled_release_confidence_parts: list[torch.Tensor] = []
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
            drift_expert_embs.append(dual.raw_release_seq)
            pooled_shared_parts.append(dual.pooled_shared)
            pooled_anchor_parts.append(dual.pooled_anchor)
            pooled_suppression_parts.append(dual.pooled_suppression)
            pooled_phi_i_parts.append(dual.pooled_phi_i)
            pooled_raw_release_parts.append(dual.pooled_raw_release)
            pooled_release_confidence_parts.append(dual.pooled_release_confidence)
            pooled_delta_parts.append(dual.pooled_raw_release)
            pooled_phi_v_parts.append(dual.pooled_phi_i + dual.pooled_raw_release)
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
            hist_input = torch.cat([dual.pooled_anchor, dual.pooled_shared], dim=-1)
            local_history_models.append(hist_head(hist_input))
            var_input = torch.cat(
                [dual.pooled_raw_release, dual.pooled_release_confidence, dual.pooled_env_shift],
                dim=-1,
            )
            local_variant_models.append(var_head(var_input))

        pooled_env_shift = torch.stack(pooled_env_shift_parts, dim=1)
        pooled_release_confidence = torch.stack(pooled_release_confidence_parts, dim=1)
        pooled_suppression = torch.stack(pooled_suppression_parts, dim=1)
        pooled_suppression_norm = torch.linalg.vector_norm(pooled_suppression, dim=-1, keepdim=True)
        router_features = torch.cat([pooled_env_shift, pooled_release_confidence, pooled_suppression_norm], dim=-1)
        env_router_bias = self.router_context_head(router_features.reshape(router_features.shape[0], -1))
        alpha_logits = self.alpha_head(z) + env_router_bias
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)

        pooled_shared = torch.stack(pooled_shared_parts, dim=1)
        expert_anchor = torch.stack(pooled_anchor_parts, dim=1)
        expert_suppression = torch.stack(pooled_suppression_parts, dim=1)
        expert_phi_i = torch.stack(pooled_phi_i_parts, dim=1)
        expert_raw_release = torch.stack(pooled_raw_release_parts, dim=1)
        expert_release_confidence = pooled_release_confidence
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
        expert_y_hat = expert_y_inv + expert_y_hist + expert_y_var

        alpha_expanded = alpha.unsqueeze(-1)
        pooled_shared_fused = torch.sum(alpha_expanded * pooled_shared, dim=1)
        pooled_anchor = torch.sum(alpha_expanded * expert_anchor, dim=1)
        pooled_suppression = torch.sum(alpha_expanded * expert_suppression, dim=1)
        pooled_phi_i = torch.sum(alpha_expanded * expert_phi_i, dim=1)
        pooled_raw_release = torch.sum(alpha_expanded * expert_raw_release, dim=1)
        pooled_release_confidence = torch.sum(alpha_expanded * expert_release_confidence, dim=1)
        pooled_delta_phi_v = torch.sum(alpha_expanded * expert_delta_phi_v, dim=1)
        pooled_phi_v = pooled_phi_i + pooled_delta_phi_v
        y_inv = torch.sum(alpha_expanded * expert_y_inv, dim=1)
        y_hist = torch.sum(alpha_expanded * expert_y_hist, dim=1)
        y_var = torch.sum(alpha_expanded * expert_y_var, dim=1)
        y_hat = torch.sum(alpha_expanded * expert_y_hat, dim=1)

        return FactorOutputsV19(
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
            expert_anchor=expert_anchor,
            expert_suppression=expert_suppression,
            expert_phi_i=expert_phi_i,
            expert_raw_release=expert_raw_release,
            expert_release_confidence=expert_release_confidence,
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
            pooled_anchor=pooled_anchor,
            pooled_suppression=pooled_suppression,
            pooled_phi_i=pooled_phi_i,
            pooled_raw_release=pooled_raw_release,
            pooled_release_confidence=pooled_release_confidence,
            pooled_delta_phi_v=pooled_delta_phi_v,
            pooled_phi_v=pooled_phi_v,
            x_used=x,
        )


@dataclass
class FactorMechanismV19Config:
    input_dim: int
    seq_len: int
    out_dim: int
    d_model: int = 128
    n_experts: int = 4
    latent_dim: int = 64
    repr_dim: int = 64
    expert_hidden_dim: int = 128
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


def build_factor_mechanism_v19(cfg: FactorMechanismV19Config) -> FactorMechanismV19:
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
            ExpertLocalStructuredEnvReleaseHead(
                input_dim=cfg.d_model,
                repr_dim=cfg.repr_dim,
                hidden_dim=cfg.expert_hidden_dim,
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
                nn.Linear(cfg.repr_dim + cfg.expert_hidden_dim, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    variant_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(cfg.repr_dim + 1 + cfg.env_count, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    revin = RevIN() if cfg.use_revin else None
    return FactorMechanismV19(
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
