from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VAEConfig:
    in_dim: int
    latent_dim: int
    hidden_dim: int = 128


class VAEEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.log_var = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, h: torch.Tensor):
        x = self.net(h)
        return self.mu(x), self.log_var(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
