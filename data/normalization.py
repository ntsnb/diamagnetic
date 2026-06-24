from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "StandardScaler":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
