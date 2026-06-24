from __future__ import annotations

import torch

from train.stage3_sensitivity_v2 import build_drift_interventions


def test_drift_interventions_include_causal_sensitivity_modes():
    drift = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    modes = build_drift_interventions(drift, seed=123, amplification_factors=(2.0, 5.0))

    assert set(modes) == {
        "real",
        "zero",
        "shuffled",
        "sign_flipped",
        "amplified_2x",
        "amplified_5x",
        "random_matched_norm",
    }
    assert torch.allclose(modes["real"], drift)
    assert torch.allclose(modes["zero"], torch.zeros_like(drift))
    assert torch.allclose(modes["sign_flipped"], -drift)
    assert torch.allclose(modes["amplified_2x"], 2.0 * drift)
    assert torch.allclose(modes["amplified_5x"], 5.0 * drift)
    assert not torch.allclose(modes["shuffled"], drift)
    assert torch.allclose(
        torch.sort(modes["shuffled"].flatten()).values,
        torch.sort(drift.flatten()).values,
    )


def test_random_matched_norm_preserves_per_sample_norms():
    drift = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ]
    )

    modes = build_drift_interventions(drift, seed=7)

    assert torch.allclose(
        torch.linalg.vector_norm(modes["random_matched_norm"], dim=1),
        torch.linalg.vector_norm(drift, dim=1),
        atol=1e-6,
    )
    assert not torch.allclose(modes["random_matched_norm"], drift)
