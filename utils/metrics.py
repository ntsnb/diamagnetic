from __future__ import annotations

import torch


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


def mae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y))
