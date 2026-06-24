from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    """Simple dataset for (X, Y) windows."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
