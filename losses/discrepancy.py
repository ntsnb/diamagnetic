from __future__ import annotations

from typing import List

import torch


def expert_discrepancy(
    relevance_maps: List[torch.Tensor], lambda_mass: float
) -> torch.Tensor:
    """Compute discrepancy across experts (Eq. 18-23).

    relevance_maps: list of [N, L] maps per expert (for a given target/time).
    """

    if len(relevance_maps) < 2:
        return torch.tensor(0.0, device=relevance_maps[0].device)

    # relevance_maps: list of [N, L]
    total = 0.0
    n_experts = len(relevance_maps)
    for p in range(n_experts):
        for q in range(p + 1, n_experts):
            r_p = relevance_maps[p].abs()
            r_q = relevance_maps[q].abs()

            # mass discrepancy (Eq. 21-22)
            m_p = r_p.sum(dim=1)  # [N]
            m_q = r_q.sum(dim=1)  # [N]
            mass = (m_p - m_q).abs().mean()

            # shape discrepancy via 1D Wasserstein on lag distributions
            # normalize to distributions along L
            p_dist = r_p / (m_p.unsqueeze(1) + 1e-6)  # [N, L]
            q_dist = r_q / (m_q.unsqueeze(1) + 1e-6)
            # cumulative sums
            cdf_p = p_dist.cumsum(dim=1)
            cdf_q = q_dist.cumsum(dim=1)
            w1 = (cdf_p - cdf_q).abs().mean()

            total = total + w1 + lambda_mass * mass

    return total
