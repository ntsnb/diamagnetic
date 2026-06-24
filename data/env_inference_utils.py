from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class EnvInitConfig:
    """Warm-start pseudo environments for EnvInference.

    This is only for initialization. Online env assignment is in EnvInference.assign().
    """

    num_envs: int
    smooth_radius: int = 0


def init_env_labels_by_time(num_samples: int, cfg: EnvInitConfig) -> torch.Tensor:
    """Initialize pseudo environment labels by equal time splits.

    Returns:
        labels: LongTensor of shape (num_samples,)
    """

    if cfg.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    labels = torch.arange(num_samples) * cfg.num_envs // max(num_samples, 1)
    labels = torch.clamp(labels, min=0, max=cfg.num_envs - 1).long()
    if cfg.smooth_radius > 0:
        labels = smooth_labels(labels, cfg.smooth_radius, cfg.num_envs)
    return labels


def smooth_labels(labels: torch.Tensor, radius: int, num_envs: int) -> torch.Tensor:
    """Majority-vote temporal smoothing."""

    if radius <= 0:
        return labels
    out = labels.clone()
    for i in range(labels.numel()):
        start = max(0, i - radius)
        end = min(labels.numel(), i + radius + 1)
        window = labels[start:end]
        # Majority vote
        counts = torch.bincount(window, minlength=num_envs)
        out[i] = torch.argmax(counts)
    return out
