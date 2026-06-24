from __future__ import annotations

import torch

from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner


def _tensor_state_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        l_value = left[key]
        r_value = right[key]
        if isinstance(l_value, torch.Tensor):
            if not torch.equal(l_value, r_value):
                return False
        elif l_value != r_value:
            return False
    return True


def _synthetic_aligned_stream(sample_count: int = 96, dim: int = 6):
    torch.manual_seed(701)
    phi_i = torch.randn(sample_count, dim)
    latent = torch.cumsum(torch.randn(sample_count, 2), dim=0)
    leak = 0.35 * phi_i @ torch.tensor(
        [
            [0.7, -0.2, 0.1, 0.0, 0.0, 0.1],
            [0.0, 0.5, -0.1, 0.2, 0.0, 0.0],
            [0.1, 0.0, 0.6, -0.2, 0.1, 0.0],
            [0.0, 0.2, 0.0, 0.4, -0.3, 0.1],
            [0.1, 0.0, 0.0, 0.2, 0.6, -0.1],
            [0.0, 0.1, -0.2, 0.0, 0.2, 0.5],
        ]
    )
    signal = torch.zeros(sample_count, dim)
    signal[:, 1] = latent[:, 0]
    signal[:, 4] = -0.8 * latent[:, 1]
    raw_delta = leak + signal + 0.01 * torch.randn(sample_count, dim)
    phi_v = phi_i + raw_delta
    y_inv = torch.zeros(sample_count, 2)
    y = torch.stack(
        [
            latent[:, 0] + 0.1 * latent[:, 1],
            -0.25 * latent[:, 0] + latent[:, 1],
        ],
        dim=1,
    )
    return phi_i, phi_v, y_inv, y


def test_residualized_rrcca_v7_removes_invariant_leakage_and_selects_active_rank():
    phi_i, phi_v, y_inv, y = _synthetic_aligned_stream()
    aligner = TrainSetChronologicalResidualizedReducedRankCCAAligner(
        ridge=1e-4,
        inner_fit_fraction=0.7,
        max_rank=4,
        min_fit_correlation=0.02,
        min_calibration_correlation=0.02,
        min_calibration_r2=0.0004,
        bootstrap_samples=24,
        min_bootstrap_sign_stability=0.55,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )

    transformed = aligner.transform_phi_pair(phi_i, phi_v)
    summary = aligner.diagnostic_summary()

    assert aligner.fit_sample_count == phi_i.shape[0]
    assert aligner.inner_fit_sample_count > aligner.inner_calib_sample_count
    assert 1 <= aligner.active_rank <= 4
    assert summary["alignment_gate_pass"] is True
    assert summary["residualizer_r2_mean"] > 0.05
    assert transformed.shape == phi_i.shape
    inactive = ~aligner.active_mask
    assert torch.max(torch.abs(transformed[:, inactive])).item() == 0.0


def test_residualized_rrcca_v7_round_trips_and_transform_is_state_free():
    phi_i, phi_v, y_inv, y = _synthetic_aligned_stream()
    aligner = TrainSetChronologicalResidualizedReducedRankCCAAligner(
        ridge=1e-4,
        inner_fit_fraction=0.75,
        max_rank=3,
        min_fit_correlation=0.02,
        min_calibration_correlation=0.02,
        min_calibration_r2=0.0004,
        bootstrap_samples=16,
        min_bootstrap_sign_stability=0.5,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )
    before_state = aligner.state_dict()
    transformed = aligner.transform_phi_pair(phi_i[:12], phi_v[:12])
    after_state = aligner.state_dict()
    restored = TrainSetChronologicalResidualizedReducedRankCCAAligner().load_state_dict(before_state)
    restored_transformed = restored.transform_phi_pair(phi_i[:12], phi_v[:12])

    assert _tensor_state_equal(before_state, after_state)
    assert torch.allclose(transformed, restored_transformed)
    assert restored.diagnostic_summary()["active_rank"] == aligner.active_rank


def test_residualized_rrcca_v7_fails_closed_when_no_stable_drift_dimensions_exist():
    torch.manual_seed(702)
    sample_count = 80
    dim = 6
    phi_i = torch.randn(sample_count, dim)
    phi_v = phi_i + 0.01 * torch.randn(sample_count, dim)
    y_inv = torch.zeros(sample_count, 2)
    y = torch.randn(sample_count, 2)

    aligner = TrainSetChronologicalResidualizedReducedRankCCAAligner(
        ridge=1e-3,
        inner_fit_fraction=0.75,
        max_rank=4,
        min_fit_correlation=0.95,
        min_calibration_correlation=0.95,
        min_calibration_r2=0.9,
        bootstrap_samples=16,
        min_bootstrap_sign_stability=0.95,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )

    transformed = aligner.transform_phi_pair(phi_i[:10], phi_v[:10])
    summary = aligner.diagnostic_summary()

    assert aligner.active_rank == 0
    assert summary["alignment_gate_pass"] is False
    assert "active_rank" in summary["alignment_gate_failure_reasons"]
    assert torch.allclose(transformed, torch.zeros_like(transformed), atol=1e-8)
