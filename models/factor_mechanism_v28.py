from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.factor_mechanism_v26 import FactorMechanismV26Config, build_factor_mechanism_v26


class FactorOutputsV28:
    """Forwarding output container for the V28 support-suppression line."""

    def __init__(self, base, **kwargs) -> None:
        self.base = base
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


@dataclass
class FactorMechanismV28Config:
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
    suppression_hidden_dim: int = 128
    support_hidden_dim: int = 128
    router_support_hidden_dim: int = 64
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


class FactorMechanismV28(nn.Module):
    """V28 support-suppression dual conditioner built on the V26 correction backbone."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        suppression_adapter_heads: nn.ModuleList,
        suppression_output_heads: nn.ModuleList,
        suppression_readout_heads: nn.ModuleList,
        support_adapter_heads: nn.ModuleList,
        support_output_heads: nn.ModuleList,
        support_delta_heads: nn.ModuleList,
        router_support_head: nn.Module,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.suppression_adapter_heads = suppression_adapter_heads
        self.suppression_output_heads = suppression_output_heads
        self.suppression_readout_heads = suppression_readout_heads
        self.support_adapter_heads = support_adapter_heads
        self.support_output_heads = support_output_heads
        self.support_delta_heads = support_delta_heads
        self.router_support_head = router_support_head
        self.use_vae = getattr(base_model, "use_vae", True)
        self.vae_eval_mode = getattr(base_model, "vae_eval_mode", "mean")

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputsV28:
        base = self.base_model(x)
        pooled_env_shift = getattr(base, "pooled_env_shift", base.env_shift.mean(dim=2))

        suppression_states: list[torch.Tensor] = []
        suppression_scores: list[torch.Tensor] = []
        suppression_vectors: list[torch.Tensor] = []
        support_states: list[torch.Tensor] = []
        support_scores: list[torch.Tensor] = []
        supported_deltas: list[torch.Tensor] = []
        corrected_changes: list[torch.Tensor] = []
        servable_changes: list[torch.Tensor] = []

        for expert_idx, (
            suppression_adapter,
            suppression_output,
            suppression_readout,
            support_adapter,
            support_output,
            support_delta,
        ) in enumerate(
            zip(
                self.suppression_adapter_heads,
                self.suppression_output_heads,
                self.suppression_readout_heads,
                self.support_adapter_heads,
                self.support_output_heads,
                self.support_delta_heads,
            )
        ):
            observable = base.expert_observable_change[:, expert_idx, :]
            change_memory = base.expert_change_memory[:, expert_idx, :]
            raw_release = base.expert_raw_release[:, expert_idx, :]
            env_shift = pooled_env_shift[:, expert_idx, :]
            release_budget = base.expert_release_budget[:, expert_idx, :]
            release_confidence = base.expert_release_confidence[:, expert_idx, :]
            raw_change = base.expert_y_change_raw[:, expert_idx, :]
            change_delta = base.expert_change_delta[:, expert_idx, :]

            suppression_input = torch.cat(
                [
                    change_memory,
                    raw_release,
                    env_shift,
                    release_budget,
                    release_confidence,
                    raw_change,
                ],
                dim=-1,
            )
            suppression_state = suppression_adapter(suppression_input)
            suppression_score = torch.sigmoid(
                suppression_output(torch.cat([suppression_state, suppression_input], dim=-1))
            )
            suppression_vector = suppression_readout(torch.cat([suppression_state, suppression_input], dim=-1))
            suppressed_raw = suppression_score * suppression_vector

            support_input = torch.cat(
                [
                    observable,
                    change_memory,
                    raw_release,
                    env_shift,
                    release_budget,
                    release_confidence,
                    raw_change,
                    change_delta,
                    suppression_state,
                    suppression_score,
                    suppression_vector,
                ],
                dim=-1,
            )
            support_state = support_adapter(support_input)
            support_score = torch.sigmoid(support_output(torch.cat([support_state, support_input], dim=-1)))
            support_delta_adjustment = support_delta(torch.cat([support_state, support_input], dim=-1))
            supported_delta = support_score * change_delta + support_delta_adjustment
            corrected_change = raw_change - suppressed_raw + supported_delta

            servable_change = torch.cat(
                [
                    suppression_state,
                    support_state,
                    observable,
                    change_memory,
                    suppression_score,
                    support_score,
                    release_confidence,
                    release_budget,
                    suppression_vector,
                    supported_delta,
                    corrected_change,
                    raw_change,
                    change_delta,
                ],
                dim=-1,
            )
            suppression_states.append(suppression_state)
            suppression_scores.append(suppression_score)
            suppression_vectors.append(suppression_vector)
            support_states.append(support_state)
            support_scores.append(support_score)
            supported_deltas.append(supported_delta)
            corrected_changes.append(corrected_change)
            servable_changes.append(servable_change)

        expert_suppression_state = torch.stack(suppression_states, dim=1)
        expert_suppression_score = torch.stack(suppression_scores, dim=1)
        expert_suppression_vector = torch.stack(suppression_vectors, dim=1)
        expert_support_state = torch.stack(support_states, dim=1)
        expert_support_score = torch.stack(support_scores, dim=1)
        expert_supported_delta = torch.stack(supported_deltas, dim=1)
        expert_corrected_change = torch.stack(corrected_changes, dim=1)
        expert_servable_change = torch.stack(servable_changes, dim=1)

        observable_norm = torch.linalg.vector_norm(base.expert_observable_change, dim=-1, keepdim=True)
        memory_norm = torch.linalg.vector_norm(base.expert_change_memory, dim=-1, keepdim=True)
        suppression_norm = torch.linalg.vector_norm(expert_suppression_state, dim=-1, keepdim=True)
        support_norm = torch.linalg.vector_norm(expert_support_state, dim=-1, keepdim=True)
        router_features = torch.cat(
            [
                pooled_env_shift,
                observable_norm,
                memory_norm,
                suppression_norm,
                support_norm,
                expert_suppression_score,
                expert_support_score,
                base.expert_release_budget,
                base.expert_release_confidence,
            ],
            dim=-1,
        )
        support_router_bias = self.router_support_head(router_features.reshape(router_features.shape[0], -1))
        alpha_logits = base.alpha_logits + support_router_bias
        if gate_bias is not None:
            alpha_logits = alpha_logits + gate_bias.view(1, -1)
        alpha = torch.softmax(alpha_logits, dim=-1)

        y_inv = torch.sum(alpha.unsqueeze(-1) * base.expert_y_inv, dim=1)
        y_hist = torch.sum(alpha.unsqueeze(-1) * base.expert_y_hist, dim=1)
        y_var = torch.sum(alpha.unsqueeze(-1) * expert_corrected_change, dim=1)
        y_hat = y_inv + y_hist + y_var

        pooled_suppression_state = torch.sum(alpha.unsqueeze(-1) * expert_suppression_state, dim=1)
        pooled_suppression_score = torch.sum(alpha.unsqueeze(-1) * expert_suppression_score, dim=1)
        pooled_suppression_vector = torch.sum(alpha.unsqueeze(-1) * expert_suppression_vector, dim=1)
        pooled_support_state = torch.sum(alpha.unsqueeze(-1) * expert_support_state, dim=1)
        pooled_support_score = torch.sum(alpha.unsqueeze(-1) * expert_support_score, dim=1)
        pooled_supported_delta = torch.sum(alpha.unsqueeze(-1) * expert_supported_delta, dim=1)
        pooled_corrected_change = torch.sum(alpha.unsqueeze(-1) * expert_corrected_change, dim=1)
        pooled_servable_change = torch.sum(alpha.unsqueeze(-1) * expert_servable_change, dim=1)
        expert_y_hat = base.expert_y_inv + base.expert_y_hist + expert_corrected_change

        return FactorOutputsV28(
            base,
            y_hat=y_hat,
            y_inv=y_inv,
            y_hist=y_hist,
            y_var=y_var,
            alpha=alpha,
            alpha_logits=alpha_logits,
            env_router_bias=base.env_router_bias + support_router_bias,
            support_router_bias=support_router_bias,
            expert_y_var=expert_corrected_change,
            expert_y_hat=expert_y_hat,
            expert_suppression_state=expert_suppression_state,
            expert_suppression_score=expert_suppression_score,
            expert_suppression_vector=expert_suppression_vector,
            expert_support_state=expert_support_state,
            expert_support_score=expert_support_score,
            expert_supported_delta=expert_supported_delta,
            expert_corrected_change=expert_corrected_change,
            pooled_suppression_state=pooled_suppression_state,
            pooled_suppression_score=pooled_suppression_score,
            pooled_suppression_vector=pooled_suppression_vector,
            pooled_support_state=pooled_support_state,
            pooled_support_score=pooled_support_score,
            pooled_supported_delta=pooled_supported_delta,
            pooled_corrected_change=pooled_corrected_change,
            expert_servable_change=expert_servable_change,
            pooled_servable_change=pooled_servable_change,
            expert_multiscale_change_score=expert_support_score,
            pooled_multiscale_change_score=pooled_support_score,
            pooled_multiscale_change=pooled_support_state,
            pooled_raw_release=pooled_servable_change,
        )


def build_factor_mechanism_v28(cfg: FactorMechanismV28Config) -> FactorMechanismV28:
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
    suppression_input_dim = 2 * cfg.repr_dim + cfg.env_count + 2 + cfg.out_dim
    suppression_joint_dim = cfg.repr_dim + suppression_input_dim
    support_input_dim = 4 * cfg.repr_dim + cfg.env_count + 3 + 3 * cfg.out_dim
    support_joint_dim = cfg.repr_dim + support_input_dim
    suppression_adapter_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(suppression_input_dim, cfg.suppression_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.suppression_hidden_dim, cfg.repr_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    suppression_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(suppression_joint_dim, cfg.suppression_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.suppression_hidden_dim, 1),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    suppression_readout_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(suppression_joint_dim, cfg.suppression_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.suppression_hidden_dim, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    support_adapter_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(support_input_dim, cfg.support_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.support_hidden_dim, cfg.repr_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    support_output_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(support_joint_dim, cfg.support_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.support_hidden_dim, 1),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    support_delta_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(support_joint_dim, cfg.support_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.support_hidden_dim, cfg.out_dim),
            )
            for _ in range(cfg.n_experts)
        ]
    )
    router_support_head = nn.Sequential(
        nn.Linear(cfg.n_experts * (cfg.env_count + 8), cfg.router_support_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.router_support_hidden_dim, cfg.n_experts),
    )
    return FactorMechanismV28(
        base_model=base_model,
        suppression_adapter_heads=suppression_adapter_heads,
        suppression_output_heads=suppression_output_heads,
        suppression_readout_heads=suppression_readout_heads,
        support_adapter_heads=support_adapter_heads,
        support_output_heads=support_output_heads,
        support_delta_heads=support_delta_heads,
        router_support_head=router_support_head,
    )
