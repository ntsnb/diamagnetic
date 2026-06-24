from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models.attention_blocks import (
    TransformerEncoderConfig,
    VariableAxisExpert,
    build_temporal_encoder,
)
from models.vae import VAEConfig, VAEEncoder, VAEDecoder


@dataclass
class FactorOutputs:
    """Outputs required for offline losses (Eq. 10, 14, 25, 26)."""

    y_hat: torch.Tensor
    z: torch.Tensor
    mu_z: torch.Tensor
    log_var_z: torch.Tensor
    x_recon: torch.Tensor
    alpha: torch.Tensor
    alpha_logits: torch.Tensor
    expert_embs: List[torch.Tensor]
    x_used: torch.Tensor


class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std, mean, std

    def denormalize_all(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def denormalize_target(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_index: int,
    ) -> torch.Tensor:
        target_mean = mean[:, :, target_index]
        target_std = std[:, :, target_index]
        return y * target_std + target_mean


class FactorMechanism(nn.Module):
    """Backbone: temporal encoder + VAE + MoE experts."""

    def __init__(
        self,
        temporal_encoder: nn.Module,
        vae_encoder: nn.Module,
        vae_decoder: nn.Module,
        alpha_head: nn.Module,
        experts: nn.ModuleList,
        output_head: nn.Module,
        gate_proj: nn.Module | None = None,
        revin: RevIN | None = None,
        target_index: int = 0,
        use_vae: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.alpha_head = alpha_head
        self.experts = experts
        self.output_head = output_head
        self.gate_proj = gate_proj
        self.revin = revin
        self.target_index = target_index
        self.use_vae = use_vae

    def forward(self, x: torch.Tensor, gate_bias: torch.Tensor | None = None) -> FactorOutputs:
        if self.revin is not None:
            x_used, mean, std = self.revin.normalize(x)
        else:
            x_used = x
            mean = std = None

        h = self.temporal_encoder(x_used)
        if self.use_vae:
            mu_z, log_var_z = self.vae_encoder(h)
            z = self._reparameterize(mu_z, log_var_z)
            x_recon = self.vae_decoder(z)
            x_recon = x_recon.view(x.shape[0], x.shape[1], x.shape[2])
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
        for expert in self.experts:
            expert_embs.append(expert(x_used))
        fused = self._fuse_experts(alpha, expert_embs)
        fused = fused.mean(dim=1)  # pool over time -> [B, d_model]
        y_hat = self.output_head(fused)
        if self.revin is not None and mean is not None and std is not None:
            y_hat = self.revin.denormalize_target(y_hat, mean, std, self.target_index)

        return FactorOutputs(
            y_hat=y_hat,
            z=z,
            mu_z=mu_z,
            log_var_z=log_var_z,
            x_recon=x_recon,
            alpha=alpha,
            alpha_logits=alpha_logits,
            expert_embs=expert_embs,
            x_used=x,
        )

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _fuse_experts(alpha: torch.Tensor, expert_embs: List[torch.Tensor]) -> torch.Tensor:
        # alpha: [B, K], expert_embs: list of [B, ...]
        if len(expert_embs) == 0:
            raise ValueError("experts list is empty")
        stacked = torch.stack(expert_embs, dim=1)  # [B, K, ...]
        # expand alpha for broadcast
        while alpha.dim() < stacked.dim():
            alpha = alpha.unsqueeze(-1)
        return (alpha * stacked).sum(dim=1)


@dataclass
class FactorMechanismConfig:
    input_dim: int
    seq_len: int
    out_dim: int
    d_model: int = 128
    n_experts: int = 4
    latent_dim: int = 64
    temporal_encoder: str = "transformer"  # registry name
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_revin: bool = False
    target_index: int = 0
    use_vae: bool = True


def build_factor_mechanism(cfg: FactorMechanismConfig) -> FactorMechanism:
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
        vae_encoder = VAEEncoder(v_cfg)
        vae_decoder = VAEDecoder(cfg.latent_dim, cfg.seq_len * cfg.input_dim, hidden_dim=cfg.d_model)
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

    output_head = nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_model),
        nn.ReLU(),
        nn.Linear(cfg.d_model, cfg.out_dim),
    )
    revin = RevIN() if cfg.use_revin else None

    return FactorMechanism(
        temporal_encoder=temporal_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        alpha_head=alpha_head,
        experts=experts,
        output_head=output_head,
        gate_proj=gate_proj,
        revin=revin,
        target_index=cfg.target_index,
        use_vae=cfg.use_vae,
    )
