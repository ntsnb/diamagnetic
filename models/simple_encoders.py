from __future__ import annotations

import torch
import torch.nn as nn


class FlattenMLPEncoder(nn.Module):
    """Simple encoder for invariant module."""

    def __init__(self, input_dim: int, repr_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.net(x.view(b, -1))


class LinearRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
