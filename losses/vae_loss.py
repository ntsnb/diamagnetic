from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float) -> torch.Tensor:
    """VAE loss (Eq. 26): recon + beta * KL."""

    recon = F.mse_loss(x_recon, x)
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl
