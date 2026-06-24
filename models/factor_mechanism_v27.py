from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models.factor_mechanism_v26 import FactorMechanismV26Config, build_factor_mechanism_v26


class FactorOutputsV27:
    """Forwarding output container for the V27 quality-conditioned line."""

    def __init__(self, base, **kwargs) -> None:
        self.base = base
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


@dataclass
class FactorMechanismV27Config:
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
    quality_hidden_dim: int = 128
    router_quality_hidden_dim: int = 64
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


class FactorMechanismV27(nn.Module):
    """V27 quality-conditioned dual-stream line built on the V26 correction backbone."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        quality_adapter_heads: nn.ModuleList,
        quality_output_heads: nn.ModuleList,
        router_quality_head: nn.Module,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.quality_adapter_heads = quality_adapter_heads
        self.quality_output_heads = quality_output_heads
        self.router_quality_head = router_quality_head
        self.use_vae = getattr(base_model, "use_vae", True)
        self.vae_eval_mode = getattr(base_model, "vae_eval_mode", "mean")

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV27:
        base = self.base_model(x)
        pooled_env_shift = getattr(base, "pooled_env_shift", base.env_shift.mean(dim=2))

        quality_states: list[torch.Tensor] = []
        quality_scores: list[torch.Tensor] = []
        quality_weighted_changes: list[torch.Tensor] = []
        servable_changes: list[torch.Tensor] = []

        for expert_idx, (adapter_head, score_head) in enumerate(
            zip(self.quality_adapter_heads, self.quality_output_heads)
        ):
            observable = base.expert_observable_change[:, expert_idx, :]
            change_memory = base.expert_change_memory[:, expert_idx, :]
            raw_release = base.expert_raw_release[:, expert_idx, :]
            env_shift = pooled_env_shift[:, expert_idx, :]
            release_budget = base.expert_release_budget[:, expert_idx, :]
            release_confidence = base.expert_release_confidence[:, expert_idx, :]
            raw_change = base.expert_y_change_raw[:, expert_idx, :]
            change_delta = base.expert_change_delta[:, expert_idx, :]

            quality_input = torch.cat(
                [
                    observable,
                    change_memory,
                    raw_release,
                    env_shift,
                    release_budget,
                    release_confidence,
                    raw_change,
                    change_delta,
                ],
                dim=-1,
            )
            quality_state = adapter_head(quality_input)
            quality_score = torch.sigmoid(score_head(torch.cat([quality_state, quality_input], dim=-1)))
            quality_weighted_change = raw_change + quality_score * change_delta
            servable_change = torch.cat(
                [
                    quality_state,
                    observable,
                    change_memory,
                    quality_score * change_memory,
                    quality_score,
                    release_confidence,
                    release_budget,
                    quality_weighted_change,
                    raw_change,
                    change_delta,
                ],
                dim=-1,
            )
            quality_states.append(quality_state)
            quality_scores.append(quality_score)
            quality_weighted_changes.append(quality_weighted_change)
            servable_changes.append(servable_change)

        expert_quality_state = torch.stack(quality_states, dim=1)
        expert_quality_score = torch.stack(quality_scores, dim=1)
        expert_quality_weighted_change = torch.stack(quality_weighted_changes, dim=1)
        expert_servable_change = torch.stack(servable_changes, dim=1)

        observable_norm = torch.linalg.vector_norm(base.expert_observable_change, dim=-1, keepdim=True)
        memory_norm = torch.linalg.vector_norm(base.expert_change_memory, dim=-1, keepdim=True)
        quality_state_norm = torch.linalg.vector_norm(expert_quality_state, dim=-1, keepdim=True)
        router_features = torch.cat(
            [
                pooled_env_shift,
                observable_norm,
                memory_norm,
                quality_state_norm,
                expert_quality_score,
                base.expert_release_budget,
                base.expert_release_confidence,
            ],
            dim=-1,
        )
        quality_router_bias = self.router_quality_head(router_features.reshape(router_features.shape[0], -1))
        alpha_logits = base.alpha_logits + quality_router_bias
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)

        y_inv = torch.sum(alpha.unsqueeze(-1) * base.expert_y_inv, dim=1)
        y_hist = torch.sum(alpha.unsqueeze(-1) * base.expert_y_hist, dim=1)
        y_var = torch.sum(alpha.unsqueeze(-1) * expert_quality_weighted_change, dim=1)
        y_hat = y_inv + y_hist + y_var

        pooled_quality_state = torch.sum(alpha.unsqueeze(-1) * expert_quality_state, dim=1)
        pooled_quality_score = torch.sum(alpha.unsqueeze(-1) * expert_quality_score, dim=1)
        pooled_quality_weighted_change = torch.sum(alpha.unsqueeze(-1) * expert_quality_weighted_change, dim=1)
        pooled_servable_change = torch.sum(alpha.unsqueeze(-1) * expert_servable_change, dim=1)
        expert_y_hat = base.expert_y_inv + base.expert_y_hist + expert_quality_weighted_change

        return FactorOutputsV27(
            base,
            y_hat=y_hat,
            y_inv=y_inv,
            y_hist=y_hist,
            y_var=y_var,
            alpha=alpha,
            alpha_logits=alpha_logits,
            env_router_bias=base.env_router_bias + quality_router_bias,
            quality_router_bias=quality_router_bias,
            expert_y_var=expert_quality_weighted_change,
            expert_y_hat=expert_y_hat,
            expert_quality_state=expert_quality_state,
            expert_quality_score=expert_quality_score,
            expert_quality_weighted_change=expert_quality_weighted_change,
            pooled_quality_state=pooled_quality_state,
            pooled_quality_score=pooled_quality_score,
            pooled_quality_weighted_change=pooled_quality_weighted_change,
            expert_servable_change=expert_servable_change,
            pooled_servable_change=pooled_servable_change,
            expert_multiscale_change_score=expert_quality_score,
            pooled_multiscale_change_score=pooled_quality_score,
            pooled_multiscale_change=pooled_quality_state,
            pooled_raw_release=pooled_servable_change,
        )


def build_factor_mechanism_v27(cfg: FactorMechanismV27Config) -> FactorMechanismV27:
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
    quality_input_dim = 3 * cfg.repr_dim + cfg.env_count + 2 + 2 * cfg.out_dim
    quality_adapter_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(quality_input_dim, cfg.quality_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.quality_hidden_dim, cfg.repr_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    quality_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(cfg.repr_dim + quality_input_dim, cfg.quality_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.quality_hidden_dim, 1),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    router_quality_head = nn.Sequential(
        nn.Linear(cfg.n_experts * (cfg.env_count + 6), cfg.router_quality_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.router_quality_hidden_dim, cfg.n_experts),
    )
    return FactorMechanismV27(
        base_model=base_model,
        quality_adapter_heads=quality_adapter_heads,
        quality_output_heads=quality_output_heads,
        router_quality_head=router_quality_head,
    )
