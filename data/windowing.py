from __future__ import annotations

from typing import Tuple

import torch


def build_windows(
    series: torch.Tensor, lookback: int, horizon: int, target_dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sliding windows.

    Args:
        series: [T, N]
        lookback: L
        horizon: H
        target_dim: index of target variable

    Returns:
        X: [M, L, N]
        Y: [M, H]
    """

    t, n = series.shape
    m = t - lookback - horizon + 1
    if m <= 0:
        raise ValueError("series too short for given lookback/horizon")
    xs = []
    ys = []
    for i in range(m):
        x = series[i : i + lookback]
        y = series[i + lookback : i + lookback + horizon, target_dim]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)
