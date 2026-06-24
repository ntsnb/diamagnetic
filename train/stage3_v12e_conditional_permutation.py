from __future__ import annotations

import torch


def build_history_norm_strata(*, history_features: torch.Tensor, bin_count: int = 4, eps: float = 1e-8) -> torch.Tensor:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    history = history_features.detach().float()
    if history.ndim != 2:
        raise ValueError("history_features must be a 2-D tensor")
    if history.shape[0] == 0:
        raise ValueError("history_features must contain at least one row")
    norms = torch.linalg.vector_norm(history, dim=1)
    if bin_count == 1 or torch.max(norms) - torch.min(norms) <= eps:
        return torch.zeros(history.shape[0], dtype=torch.long)
    quantiles = torch.linspace(0.0, 1.0, bin_count + 1, dtype=norms.dtype, device=norms.device)[1:-1]
    boundaries = torch.quantile(norms, quantiles).unique(sorted=True)
    if boundaries.numel() == 0:
        return torch.zeros(history.shape[0], dtype=torch.long)
    return torch.bucketize(norms, boundaries).long()


def _deranged_indices(indices: torch.Tensor, *, seed: int) -> torch.Tensor:
    if indices.numel() <= 1:
        return indices.clone()
    gen = torch.Generator(device=indices.device)
    gen.manual_seed(int(seed))
    order = torch.randperm(indices.numel(), generator=gen, device=indices.device)
    shuffled = indices[order]
    if torch.equal(shuffled, indices):
        shuffled = torch.roll(shuffled, shifts=1)
    return shuffled


def conditional_permute_drift(
    *,
    drift_signal: torch.Tensor,
    strata: torch.Tensor,
    seed: int = 1729,
) -> torch.Tensor:
    drift = drift_signal.detach().float()
    if drift.ndim != 2:
        raise ValueError("drift_signal must be a 2-D tensor")
    labels = strata.detach().long().flatten()
    if labels.numel() != drift.shape[0]:
        raise ValueError("strata must have one label per drift row")
    output = drift.clone()
    for offset, label in enumerate(torch.unique(labels, sorted=True)):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        permuted = _deranged_indices(indices, seed=int(seed) + offset)
        output[indices] = drift[permuted]
    return output


def conditional_random_matched_norm(
    *,
    drift_signal: torch.Tensor,
    strata: torch.Tensor,
    seed: int = 1729,
    eps: float = 1e-8,
) -> torch.Tensor:
    drift = drift_signal.detach().float()
    if drift.ndim != 2:
        raise ValueError("drift_signal must be a 2-D tensor")
    labels = strata.detach().long().flatten()
    if labels.numel() != drift.shape[0]:
        raise ValueError("strata must have one label per drift row")
    output = torch.zeros_like(drift)
    for offset, label in enumerate(torch.unique(labels, sorted=True)):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        gen = torch.Generator(device=drift.device)
        gen.manual_seed(int(seed) + offset)
        random = torch.randn((indices.numel(), drift.shape[1]), generator=gen, device=drift.device, dtype=drift.dtype)
        random_norm = torch.linalg.vector_norm(random, dim=1, keepdim=True).clamp_min(eps)
        source_norm = torch.linalg.vector_norm(drift[indices], dim=1, keepdim=True)
        output[indices] = random / random_norm * source_norm
    zero_mask = torch.linalg.vector_norm(drift, dim=1, keepdim=True) <= eps
    return torch.where(zero_mask, torch.zeros_like(output), output)
