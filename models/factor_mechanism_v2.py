from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.attention_blocks import (
    TransformerEncoderConfig,
    VariableAxisExpert,
    build_temporal_encoder,
)
from models.factor_mechanism import FactorMechanism, RevIN
from models.vae import VAEConfig, VAEDecoder, VAEEncoder


class FactorMechanismV2(FactorMechanism):
    """Factor mechanism copy with explicit deterministic VAE evaluation policy.

    The legacy ``FactorMechanism`` is intentionally left unchanged. This V2
    subclass keeps training-time reparameterization, but uses ``mu_z`` during
    ``eval()`` unless Monte Carlo sampling is explicitly requested.
    """

    def __init__(
        self,
        *args,
        vae_eval_mode: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if vae_eval_mode not in {"mean", "sample"}:
            raise ValueError(f"Unsupported vae_eval_mode: {vae_eval_mode}")
        self.vae_eval_mode = vae_eval_mode

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training or self.vae_eval_mode == "sample":
            return FactorMechanism._reparameterize(mu, log_var)
        return mu


@dataclass
class FactorMechanismV2Config:
    input_dim: int
    seq_len: int
    out_dim: int
    d_model: int = 128
    n_experts: int = 4
    latent_dim: int = 64
    temporal_encoder: str = "transformer"
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_revin: bool = False
    target_index: int = 0
    use_vae: bool = True
    vae_eval_mode: str = "mean"


def build_factor_mechanism_v2(cfg: FactorMechanismV2Config) -> FactorMechanismV2:
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

    output_head = nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_model),
        nn.ReLU(),
        nn.Linear(cfg.d_model, cfg.out_dim),
    )
    revin = RevIN() if cfg.use_revin else None

    return FactorMechanismV2(
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
        vae_eval_mode=cfg.vae_eval_mode,
    )
