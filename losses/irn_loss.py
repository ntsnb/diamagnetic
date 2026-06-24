from __future__ import annotations

import torch
import torch.nn.functional as F


def irn_surrogate_loss(
    y_hat: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """IRN surrogate loss (Eq. 3-4 / Eq. 34).

    Residual normalization to mitigate Z effects on Y.
    """

    def _norm(v: torch.Tensor) -> torch.Tensor:
        mean = v.mean(dim=0, keepdim=True)
        std = v.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (v - mean) / std

    res = _norm(y) - _norm(y_hat)
    return F.mse_loss(res, torch.zeros_like(res), reduction=reduction)
