from __future__ import annotations

import torch


def _as_matrix(drift: torch.Tensor) -> torch.Tensor:
    if drift.ndim == 1:
        return drift.unsqueeze(0)
    if drift.ndim == 2:
        return drift
    return drift.flatten(start_dim=1)


def _format_amp_name(factor: float) -> str:
    if float(factor).is_integer():
        return f"amplified_{int(factor)}x"
    return f"amplified_{str(factor).replace('.', 'p')}x"


def build_drift_interventions(
    drift: torch.Tensor,
    *,
    seed: int,
    amplification_factors: tuple[float, ...] = (2.0, 5.0),
) -> dict[str, torch.Tensor]:
    """Build deterministic causal drift-sensitivity interventions."""

    drift_matrix = _as_matrix(drift)
    generator = torch.Generator(device=drift_matrix.device)
    generator.manual_seed(seed)
    row_perm = torch.randperm(drift_matrix.shape[0], generator=generator, device=drift_matrix.device)
    col_perm = torch.randperm(drift_matrix.shape[1], generator=generator, device=drift_matrix.device)
    shuffled = drift_matrix[row_perm][:, col_perm]

    random_direction = torch.randn(
        drift_matrix.shape,
        generator=generator,
        device=drift_matrix.device,
        dtype=drift_matrix.dtype,
    )
    random_direction = torch.nn.functional.normalize(random_direction, dim=1)
    norms = torch.linalg.vector_norm(drift_matrix, dim=1, keepdim=True)
    random_matched_norm = random_direction * norms

    modes = {
        "real": drift_matrix.clone(),
        "zero": torch.zeros_like(drift_matrix),
        "shuffled": shuffled,
        "sign_flipped": -drift_matrix,
        "random_matched_norm": random_matched_norm,
    }
    for factor in amplification_factors:
        modes[_format_amp_name(factor)] = drift_matrix * factor
    return modes
