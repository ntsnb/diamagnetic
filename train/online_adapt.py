from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class LabelBuffer:
    """Delay buffer to enforce L-step feedback delay."""

    def __init__(self, delay_steps: int) -> None:
        if delay_steps <= 0:
            raise ValueError("delay_steps must be positive")
        self.delay_steps = delay_steps
        self._buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=delay_steps)

    def push(self, z: torch.Tensor, x: torch.Tensor, alpha_ref: torch.Tensor) -> None:
        self._buffer.append((z.detach(), x.detach(), alpha_ref.detach()))

    def pop_with_label(self, y: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return (z, x, alpha_ref, y) aligned with L-step delay if available."""

        if len(self._buffer) < self.delay_steps:
            return None
        z, x, alpha_ref = self._buffer.popleft()
        return z, x, alpha_ref, y


@dataclass
class StabilityConfig:
    gamma: float
    window_size: int


class OnlineLoop:
    """High-level online adaptation loop with Stage I/II/III hooks."""

    def __init__(self, stability_cfg: StabilityConfig) -> None:
        self.cfg = stability_cfg
        self._lrp_history: Deque[torch.Tensor] = deque(maxlen=stability_cfg.window_size)

    def update_stability(self, r_map: torch.Tensor) -> float:
        """Compute stability score S^(m)_i(t) via cosine over rolling window."""
        r = r_map.detach().clamp_min(0.0)
        if r.dim() == 3:
            # [B, L, N] -> [N, L]
            r = r.mean(dim=0).transpose(0, 1)
        elif r.dim() == 2:
            # [L, N] -> [N, L]
            r = r.transpose(0, 1)
        else:
            r = r.view(1, -1)

        mass = r.sum(dim=-1, keepdim=True)
        valid = (mass.squeeze(-1) > 1e-8)
        if not torch.any(valid):
            self._lrp_history.append(r)
            return 1.0
        r = r / (mass + 1e-6)

        # Wait until the rolling window is populated before triggering Stage II.
        if len(self._lrp_history) < self.cfg.window_size:
            self._lrp_history.append(r)
            return 1.0

        sims = []
        for r_prev in self._lrp_history:
            r_prev = r_prev / (r_prev.sum(dim=-1, keepdim=True) + 1e-6)
            var_sims = []
            for idx in range(r.shape[0]):
                if valid[idx] and torch.sum(r_prev[idx]) > 1e-8:
                    var_sims.append(F.cosine_similarity(r[idx], r_prev[idx], dim=0))
            if var_sims:
                sims.append(torch.stack(var_sims).mean())
        score = torch.stack(sims).mean().item() if sims else 1.0
        self._lrp_history.append(r)
        return score
