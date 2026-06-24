from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention_blocks import (
    TransformerEncoderConfig,
    VariableAxisExpert,
    build_temporal_encoder,
)
from models.factor_mechanism import RevIN
from models.vae import VAEConfig, VAEDecoder, VAEEncoder


@dataclass
class ExpertLocalDualStreamOutput:
    shared: torch.Tensor
    phi_i_seq: torch.Tensor
    delta_phi_v_seq: torch.Tensor
    phi_v_seq: torch.Tensor
    env_logits_seq: torch.Tensor
    env_posterior_seq: torch.Tensor
    pooled_shared: torch.Tensor
    pooled_phi_i: torch.Tensor
    pooled_delta_phi_v: torch.Tensor
    pooled_phi_v: torch.Tensor
    pooled_env_logits: torch.Tensor
    pooled_env_posterior: torch.Tensor


@dataclass
class FactorOutputsV16:
    y_hat: torch.Tensor
    y_inv: torch.Tensor
    y_var: torch.Tensor
    z: torch.Tensor
    mu_z: torch.Tensor
    log_var_z: torch.Tensor
    x_recon: torch.Tensor
    alpha: torch.Tensor
    alpha_logits: torch.Tensor
    expert_embs: List[torch.Tensor]
    invariant_expert_embs: List[torch.Tensor]
    drift_expert_embs: List[torch.Tensor]
    expert_phi_i: torch.Tensor
    expert_delta_phi_v: torch.Tensor
    expert_phi_v: torch.Tensor
    env_logits: torch.Tensor
    env_posteriors: torch.Tensor
    pooled_shared: torch.Tensor
    pooled_phi_i: torch.Tensor
    pooled_delta_phi_v: torch.Tensor
    pooled_phi_v: torch.Tensor
    x_used: torch.Tensor


def _project_delta_orthogonal_to_anchor(
    delta: torch.Tensor,
    anchor: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    delta_2d = delta.reshape(-1, delta.shape[-1])
    anchor_2d = anchor.reshape(-1, anchor.shape[-1])
    denom = anchor_2d.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    parallel = (delta_2d * anchor_2d).sum(dim=1, keepdim=True) / denom * anchor_2d
    projected = delta_2d - parallel
    return projected.reshape_as(delta)


class ExpertLocalDualStreamHead(nn.Module):
    """Late-branch expert-local invariant/drift decomposition.

    The head first refines the expert state with a shared post-trunk, then
    derives:

    - `phi_i` through an invariant anchor minus an environment-conditioned
      correction term;
    - `delta_phi_v` through a residual drift branch plus environment
      modulation;
    - `phi_v = phi_i + delta_phi_v`.

    This keeps invariant/variant states in one expert-local coordinate system.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        repr_dim: int,
        hidden_dim: int,
        env_count: int,
        dropout: float = 0.0,
        zero_init_variant: bool = True,
        use_projected_residual: bool = True,
        projection_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.env_count = int(env_count)
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
        self.invariant_anchor = nn.Linear(hidden_dim, repr_dim)
        self.invariant_gate = nn.Sequential(
            nn.Linear(hidden_dim, repr_dim),
            nn.Sigmoid(),
        )
        self.env_router = nn.Linear(hidden_dim, env_count)
        self.drift_head = nn.Linear(hidden_dim, repr_dim)
        self.drift_gate = nn.Sequential(
            nn.Linear(hidden_dim, repr_dim),
            nn.Tanh(),
        )
        self.env_basis_invariant = nn.Parameter(torch.empty(env_count, repr_dim))
        self.env_basis_drift = nn.Parameter(torch.empty(env_count, repr_dim))
        nn.init.xavier_uniform_(self.env_basis_invariant)
        nn.init.xavier_uniform_(self.env_basis_drift)
        if zero_init_variant:
            nn.init.zeros_(self.drift_head.weight)
            nn.init.zeros_(self.drift_head.bias)

    def forward(self, expert_state: torch.Tensor) -> ExpertLocalDualStreamOutput:
        shared = self.shared_post(expert_state)
        env_logits = self.env_router(shared)
        env_posterior = torch.softmax(env_logits, dim=-1)

        invariant_anchor = self.invariant_anchor(shared)
        invariant_env = env_posterior @ self.env_basis_invariant
        invariant_gate = self.invariant_gate(shared)
        phi_i = invariant_anchor - invariant_gate * invariant_env

        drift_base = self.drift_head(shared)
        drift_env = self.drift_gate(shared) * (env_posterior @ self.env_basis_drift)
        delta_phi_v = drift_base + drift_env
        if self.use_projected_residual:
            delta_phi_v = _project_delta_orthogonal_to_anchor(
                delta_phi_v,
                phi_i,
                self.projection_eps,
            )
        phi_v = phi_i + delta_phi_v

        pooled_shared = shared.mean(dim=1)
        pooled_phi_i = phi_i.mean(dim=1)
        pooled_delta_phi_v = delta_phi_v.mean(dim=1)
        pooled_phi_v = phi_v.mean(dim=1)
        pooled_env_logits = env_logits.mean(dim=1)
        pooled_env_posterior = env_posterior.mean(dim=1)
        return ExpertLocalDualStreamOutput(
            shared=shared,
            phi_i_seq=phi_i,
            delta_phi_v_seq=delta_phi_v,
            phi_v_seq=phi_v,
            env_logits_seq=env_logits,
            env_posterior_seq=env_posterior,
            pooled_shared=pooled_shared,
            pooled_phi_i=pooled_phi_i,
            pooled_delta_phi_v=pooled_delta_phi_v,
            pooled_phi_v=pooled_phi_v,
            pooled_env_logits=pooled_env_logits,
            pooled_env_posterior=pooled_env_posterior,
        )


class FactorMechanismV16(nn.Module):
    """V16 base model with expert-local dual-stream invariant learning."""

    def __init__(
        self,
        *,
        temporal_encoder: nn.Module,
        vae_encoder: nn.Module | None,
        vae_decoder: nn.Module | None,
        alpha_head: nn.Module,
        experts: nn.ModuleList,
        dual_stream_heads: nn.ModuleList,
        invariant_output_head: nn.Module,
        variant_output_head: nn.Module,
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
        self.experts = experts
        self.dual_stream_heads = dual_stream_heads
        self.invariant_output_head = invariant_output_head
        self.variant_output_head = variant_output_head
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

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV16:
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

        alpha_logits = self.alpha_head(z)
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)

        expert_embs: List[torch.Tensor] = []
        invariant_expert_embs: List[torch.Tensor] = []
        drift_expert_embs: List[torch.Tensor] = []
        pooled_shared_parts: list[torch.Tensor] = []
        pooled_phi_i_parts: list[torch.Tensor] = []
        pooled_delta_parts: list[torch.Tensor] = []
        pooled_phi_v_parts: list[torch.Tensor] = []
        pooled_env_logits_parts: list[torch.Tensor] = []
        pooled_env_post_parts: list[torch.Tensor] = []
        for expert, head_module in zip(self.experts, self.dual_stream_heads):
            expert_state = expert(x_model)
            dual = head_module(expert_state)
            expert_embs.append(expert_state)
            invariant_expert_embs.append(dual.phi_i_seq)
            drift_expert_embs.append(dual.delta_phi_v_seq)
            pooled_shared_parts.append(dual.pooled_shared)
            pooled_phi_i_parts.append(dual.pooled_phi_i)
            pooled_delta_parts.append(dual.pooled_delta_phi_v)
            pooled_phi_v_parts.append(dual.pooled_phi_v)
            pooled_env_logits_parts.append(dual.pooled_env_logits)
            pooled_env_post_parts.append(dual.pooled_env_posterior)

        pooled_shared = torch.stack(pooled_shared_parts, dim=1)
        expert_phi_i = torch.stack(pooled_phi_i_parts, dim=1)
        expert_delta_phi_v = torch.stack(pooled_delta_parts, dim=1)
        expert_phi_v = torch.stack(pooled_phi_v_parts, dim=1)
        env_logits = torch.stack(pooled_env_logits_parts, dim=1)
        env_posteriors = torch.stack(pooled_env_post_parts, dim=1)

        alpha_expanded = alpha.unsqueeze(-1)
        pooled_shared_fused = torch.sum(alpha_expanded * pooled_shared, dim=1)
        pooled_phi_i = torch.sum(alpha_expanded * expert_phi_i, dim=1)
        pooled_delta_phi_v = torch.sum(alpha_expanded * expert_delta_phi_v, dim=1)
        pooled_phi_v = pooled_phi_i + pooled_delta_phi_v

        y_inv_model = self.invariant_output_head(pooled_phi_i)
        y_var_model = self.variant_output_head(pooled_delta_phi_v)
        if self.revin is not None and mean is not None and std is not None:
            target_mean, target_std = self._target_scale(mean, std, y_inv_model.shape[1])
            y_inv = y_inv_model * target_std + target_mean
            y_var = y_var_model * target_std
            y_hat = y_inv + y_var
        else:
            y_inv = y_inv_model
            y_var = y_var_model
            y_hat = y_inv + y_var

        return FactorOutputsV16(
            y_hat=y_hat,
            y_inv=y_inv,
            y_var=y_var,
            z=z,
            mu_z=mu_z,
            log_var_z=log_var_z,
            x_recon=x_recon,
            alpha=alpha,
            alpha_logits=alpha_logits,
            expert_embs=expert_embs,
            invariant_expert_embs=invariant_expert_embs,
            drift_expert_embs=drift_expert_embs,
            expert_phi_i=expert_phi_i,
            expert_delta_phi_v=expert_delta_phi_v,
            expert_phi_v=expert_phi_v,
            env_logits=env_logits,
            env_posteriors=env_posteriors,
            pooled_shared=pooled_shared_fused,
            pooled_phi_i=pooled_phi_i,
            pooled_delta_phi_v=pooled_delta_phi_v,
            pooled_phi_v=pooled_phi_v,
            x_used=x,
        )


@dataclass
class FactorMechanismV16Config:
    input_dim: int
    seq_len: int
    out_dim: int
    d_model: int = 128
    n_experts: int = 4
    latent_dim: int = 64
    repr_dim: int = 64
    expert_hidden_dim: int = 128
    env_count: int = 4
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


def build_factor_mechanism_v16(cfg: FactorMechanismV16Config) -> FactorMechanismV16:
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
            ExpertLocalDualStreamHead(
                input_dim=cfg.d_model,
                repr_dim=cfg.repr_dim,
                hidden_dim=cfg.expert_hidden_dim,
                env_count=cfg.env_count,
                dropout=cfg.dropout,
                zero_init_variant=cfg.zero_init_variant,
                use_projected_residual=cfg.use_projected_residual,
                projection_eps=cfg.projection_eps,
            )
            for _ in range(cfg.n_experts)
        ]
    )
    invariant_output_head = nn.Sequential(
        nn.Linear(cfg.repr_dim, cfg.d_model),
        nn.ReLU(),
        nn.Linear(cfg.d_model, cfg.out_dim),
    )
    variant_output_head = nn.Sequential(
        nn.Linear(cfg.repr_dim, cfg.d_model),
        nn.ReLU(),
        nn.Linear(cfg.d_model, cfg.out_dim),
    )
    revin = RevIN() if cfg.use_revin else None
    return FactorMechanismV16(
        temporal_encoder=temporal_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        alpha_head=alpha_head,
        experts=experts,
        dual_stream_heads=dual_stream_heads,
        invariant_output_head=invariant_output_head,
        variant_output_head=variant_output_head,
        gate_proj=gate_proj,
        revin=revin,
        target_index=cfg.target_index,
        use_vae=cfg.use_vae,
        vae_eval_mode=cfg.vae_eval_mode,
    )
