from __future__ import annotations

import torch

from models.aligned_representation_v2 import (
    AlignedRepresentationLossConfig,
    TrainSetDeltaStandardizer,
    TrainSetResidualDriftCanonicalizer,
    TrainSetTemporalResidualCCAAligner,
    TrainSetWhitenedProcrustesAligner,
    TrainSetWhitenedProcrustesTemporalResidualCCAAligner,
    SharedResidualFactorEncoder,
    TrainSetAffineAligner,
    aligned_representation_losses,
    temporal_residual_alignment_diagnostics,
)


def test_shared_residual_encoder_outputs_aligned_phi_v_as_phi_i_plus_delta():
    torch.manual_seed(21)
    encoder = SharedResidualFactorEncoder(
        input_dim=6 * 4,
        repr_dim=8,
        hidden_dim=16,
        dropout=0.0,
    )
    x = torch.randn(5, 6, 4)

    out = encoder(x)

    assert out.phi_i.shape == (5, 8)
    assert out.delta_phi_v.shape == (5, 8)
    assert out.phi_v.shape == (5, 8)
    assert torch.allclose(out.phi_v, out.phi_i + out.delta_phi_v)


def test_shared_residual_encoder_initial_variant_branch_is_identity_aligned():
    torch.manual_seed(22)
    encoder = SharedResidualFactorEncoder(
        input_dim=6 * 4,
        repr_dim=8,
        hidden_dim=16,
        dropout=0.0,
        zero_init_variant=True,
    )
    x = torch.randn(5, 6, 4)

    out = encoder(x)

    assert torch.allclose(out.delta_phi_v, torch.zeros_like(out.delta_phi_v), atol=1e-8)
    assert torch.allclose(out.phi_v, out.phi_i, atol=1e-8)


def test_shared_residual_encoder_can_hard_project_variant_residual_orthogonal_to_invariant():
    torch.manual_seed(27)
    encoder = SharedResidualFactorEncoder(
        input_dim=6 * 4,
        repr_dim=8,
        hidden_dim=16,
        dropout=0.0,
        use_projected_residual=True,
        stop_gradient_projection_anchor=True,
    )
    x = torch.randn(7, 6, 4)

    out = encoder(x)

    dot = torch.sum(out.phi_i * out.delta_phi_v, dim=1)
    assert torch.max(torch.abs(dot)).item() < 1e-5


def test_train_set_affine_aligner_fits_only_given_training_samples():
    torch.manual_seed(23)
    phi_i_train = torch.randn(32, 5)
    transform = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.1, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.3],
            [0.1, 0.0, 0.0, 0.0, 1.2],
        ]
    )
    phi_v_train = phi_i_train @ transform
    phi_v_test = torch.randn(7, 5) @ transform

    aligner = TrainSetAffineAligner(ridge=1e-4)
    aligner.fit(phi_i_train=phi_i_train, phi_v_train=phi_v_train)
    aligned = aligner.transform_phi_v(phi_v_test)

    assert aligner.fit_sample_count == 32
    assert aligned.shape == phi_v_test.shape
    assert not torch.isnan(aligned).any()


def test_train_set_whitened_procrustes_aligner_matches_phi_i_geometry_and_round_trips():
    torch.manual_seed(33)
    phi_i_train = torch.randn(64, 5)
    mixing = torch.tensor(
        [
            [1.2, 0.3, 0.0, 0.0, 0.0],
            [-0.2, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.2, 1.4, 0.3, 0.0],
            [0.0, 0.0, -0.1, 0.7, 0.2],
            [0.1, 0.0, 0.0, 0.2, 1.1],
        ]
    )
    phi_v_train = phi_i_train @ mixing + torch.tensor([0.5, -0.4, 0.2, 0.1, -0.3])
    phi_i_test = torch.randn(16, 5)
    phi_v_test = phi_i_test @ mixing + torch.tensor([0.5, -0.4, 0.2, 0.1, -0.3])

    aligner = TrainSetWhitenedProcrustesAligner(ridge=1e-4).fit(
        phi_i_train=phi_i_train,
        phi_v_train=phi_v_train,
    )
    aligned_train = aligner.transform_phi_v(phi_v_train)
    aligned_test = aligner.transform_phi_v(phi_v_test)
    restored = TrainSetWhitenedProcrustesAligner().load_state_dict(aligner.state_dict())
    restored_test = restored.transform_phi_v(phi_v_test)

    def covariance(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        return centered.T @ centered / float(centered.shape[0] - 1)

    before = torch.linalg.matrix_norm(covariance(phi_v_train) - covariance(phi_i_train))
    after = torch.linalg.matrix_norm(covariance(aligned_train) - covariance(phi_i_train))
    assert aligner.fit_sample_count == 64
    assert after < before
    assert aligned_test.shape == phi_v_test.shape
    assert torch.allclose(aligned_test, restored_test)


def test_whitened_procrustes_temporal_residual_cca_uses_phi_pair_and_round_trips():
    torch.manual_seed(34)
    latent = torch.cumsum(torch.randn(40, 2), dim=0)
    phi_i = torch.randn(40, 5)
    phi_i[:, :2] += latent
    phi_v = phi_i @ torch.tensor(
        [
            [1.1, 0.2, 0.0, 0.0, 0.0],
            [-0.1, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.1, 1.2, 0.2, 0.0],
            [0.0, 0.0, -0.1, 0.9, 0.1],
            [0.1, 0.0, 0.0, 0.1, 1.0],
        ]
    )
    phi_v[:, 0] += 0.5 * latent[:, 0]
    phi_v[:, 3] -= 0.4 * latent[:, 1]
    y_inv = torch.zeros(40, 2)
    y = torch.stack([latent[:, 0] + 0.2 * latent[:, 1], -latent[:, 1]], dim=1)

    aligner = TrainSetWhitenedProcrustesTemporalResidualCCAAligner(ridge=1e-4).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )
    transformed = aligner.transform_phi_pair(phi_i[:8], phi_v[:8])
    restored = TrainSetWhitenedProcrustesTemporalResidualCCAAligner().load_state_dict(aligner.state_dict())
    restored_transformed = restored.transform_phi_pair(phi_i[:8], phi_v[:8])

    assert aligner.fit_sample_count == 40
    assert aligner.transition_sample_count == 39
    assert transformed.shape == phi_i[:8].shape
    assert aligner.canonical_correlations[0].item() > 0.05
    assert torch.allclose(transformed, restored_transformed)


def test_train_set_delta_standardizer_fits_train_only_and_standardizes_delta():
    torch.manual_seed(26)
    train_delta = torch.randn(64, 5) * 2.0 + 3.0
    test_delta = torch.randn(7, 5) * 4.0 - 1.0

    standardizer = TrainSetDeltaStandardizer(eps=1e-6).fit(train_delta)
    standardized_train = standardizer.transform(train_delta)
    standardized_test = standardizer.transform(test_delta)

    assert standardizer.fit_sample_count == 64
    assert torch.allclose(standardized_train.mean(dim=0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(standardized_train.std(dim=0, unbiased=False), torch.ones(5), atol=1e-5)
    assert standardized_test.shape == test_delta.shape


def test_train_set_residual_drift_canonicalizer_round_trips_and_freezes_train_fit():
    torch.manual_seed(29)
    delta_train = torch.randn(16, 5)
    y_inv_train = torch.randn(16, 2)
    y_train = y_inv_train + torch.stack(
        [
            delta_train[:, 0] + 0.5 * delta_train[:, 1],
            -delta_train[:, 2] + 0.25 * delta_train[:, 3],
        ],
        dim=1,
    )
    delta_test = torch.randn(4, 5)

    canonicalizer = TrainSetResidualDriftCanonicalizer(ridge=1e-3).fit(
        delta_phi_v_train=delta_train,
        y_inv_train=y_inv_train,
        y_train=y_train,
    )
    transformed_before = canonicalizer.transform_delta(delta_test)
    restored = TrainSetResidualDriftCanonicalizer().load_state_dict(canonicalizer.state_dict())
    transformed_after = restored.transform_delta(delta_test)

    assert canonicalizer.fit_sample_count == 16
    assert transformed_before.shape == delta_test.shape
    assert torch.allclose(transformed_before, transformed_after)
    assert not torch.isnan(transformed_after).any()


def test_train_set_residual_drift_canonicalizer_uses_transition_signal():
    torch.manual_seed(30)
    delta = torch.zeros(8, 3)
    delta[:, 0] = torch.arange(8, dtype=torch.float32)
    y_inv = torch.zeros(8, 1)
    y = delta[:, :1] * 2.0

    canonicalizer = TrainSetResidualDriftCanonicalizer().fit(
        delta_phi_v_train=delta,
        y_inv_train=y_inv,
        y_train=y,
    )
    transformed = canonicalizer.transform_delta(delta)
    transition_energy = torch.linalg.vector_norm(transformed[1:] - transformed[:-1], dim=1)

    assert torch.all(transition_energy > 0.0)
    assert canonicalizer.transition_sample_count == 7


def test_train_set_temporal_residual_cca_aligner_round_trips_and_keeps_full_drift_dim():
    torch.manual_seed(31)
    latent = torch.randn(20, 2)
    mix = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.1],
            [0.0, 0.3, -0.8, 0.4, 0.0],
        ]
    )
    delta_train = latent @ mix + 0.01 * torch.randn(20, 5)
    y_inv_train = torch.randn(20, 3)
    residual = torch.stack(
        [
            latent[:, 0] + 0.25 * latent[:, 1],
            -0.5 * latent[:, 1],
            0.2 * latent[:, 0],
        ],
        dim=1,
    )
    y_train = y_inv_train + residual
    delta_test = torch.randn(6, 5)

    aligner = TrainSetTemporalResidualCCAAligner(ridge=1e-4).fit(
        delta_phi_v_train=delta_train,
        y_inv_train=y_inv_train,
        y_train=y_train,
    )
    transformed_before = aligner.transform_delta(delta_test)
    restored = TrainSetTemporalResidualCCAAligner().load_state_dict(aligner.state_dict())
    transformed_after = restored.transform_delta(delta_test)

    assert aligner.fit_sample_count == 20
    assert aligner.transition_sample_count == 19
    assert transformed_before.shape == delta_test.shape
    assert aligner.canonical_correlations.shape == (5,)
    assert aligner.canonical_correlations[0].item() > 0.1
    assert torch.allclose(transformed_before, transformed_after)


def test_temporal_residual_cca_aligner_improves_synthetic_transition_alignment():
    torch.manual_seed(32)
    latent = torch.cumsum(torch.randn(48, 2), dim=0)
    delta = torch.stack(
        [
            0.1 * torch.randn(48),
            latent[:, 0] + 0.05 * torch.randn(48),
            0.2 * torch.randn(48),
            -latent[:, 1] + 0.05 * torch.randn(48),
            0.1 * latent[:, 0] + 0.1 * torch.randn(48),
        ],
        dim=1,
    )
    y_inv = torch.zeros(48, 2)
    y = torch.stack(
        [
            latent[:, 0] + 0.1 * latent[:, 1],
            -0.2 * latent[:, 0] + latent[:, 1],
        ],
        dim=1,
    )

    before = temporal_residual_alignment_diagnostics(
        delta_phi_v=delta,
        y_inv=y_inv,
        y=y,
        prefix="pre",
    )
    aligner = TrainSetTemporalResidualCCAAligner(ridge=1e-4).fit(
        delta_phi_v_train=delta,
        y_inv_train=y_inv,
        y_train=y,
    )
    after = temporal_residual_alignment_diagnostics(
        delta_phi_v=delta,
        y_inv=y_inv,
        y=y,
        aligner=aligner,
        prefix="post",
    )

    assert after["post_top_abs_transition_residual_corr"] >= before["pre_top_abs_transition_residual_corr"]
    assert after["post_mean_best_abs_transition_residual_corr"] > 0.75
    assert after["post_delta_transition_effective_rank"] >= 1.0
    assert after["post_collapse_dim_fraction"] < 0.5


def test_aligned_representation_losses_include_forecast_orthogonal_smooth_and_norm_terms():
    torch.manual_seed(24)
    phi_i = torch.randn(6, 4, requires_grad=True)
    delta_phi_v = torch.randn(6, 4, requires_grad=True)
    y_inv = torch.randn(6, 2, requires_grad=True)
    y_var = torch.randn(6, 2, requires_grad=True)
    y = torch.randn(6, 2)

    losses = aligned_representation_losses(
        phi_i=phi_i,
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y_var=y_var,
        y=y,
        cfg=AlignedRepresentationLossConfig(
            lambda_forecast=1.0,
            lambda_residual=0.5,
            lambda_orth=0.1,
            lambda_smooth=0.2,
            lambda_norm=0.05,
            lambda_hsic=0.1,
            lambda_inv_forecast=0.2,
            lambda_barlow=0.3,
            lambda_temporal_ranking=0.4,
            lambda_deployed_temporal_ranking=0.2,
            lambda_temporal_contrastive=0.2,
            lambda_vicreg_variance=0.2,
            lambda_vicreg_covariance=0.3,
            lambda_phi_vicreg_variance=0.1,
            lambda_phi_vicreg_covariance=0.1,
        ),
    )

    assert set(losses) >= {
        "total",
        "forecast",
        "residual",
        "orthogonal",
        "smooth",
        "norm",
        "hsic",
        "inv_forecast",
        "barlow",
        "temporal_ranking",
        "deployed_temporal_ranking",
        "temporal_contrastive",
        "parallel_ratio",
        "vicreg_variance",
        "vicreg_covariance",
        "phi_vicreg_variance",
        "phi_vicreg_covariance",
    }
    assert losses["total"].requires_grad
    assert losses["total"].item() > 0.0
    assert losses["parallel_ratio"].item() >= 0.0


def test_deployment_aligned_losses_penalize_mismatched_temporal_drift():
    torch.manual_seed(28)
    phi_i = torch.randn(6, 4, requires_grad=True)
    delta_phi_v = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [3.1, 0.0, 0.0, 0.0],
            [3.2, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    y_inv = torch.zeros(6, 1, requires_grad=True)
    y = torch.tensor([[0.0], [0.1], [3.0], [3.1], [3.2], [6.0]])
    y_var = torch.zeros(6, 1, requires_grad=True)

    losses = aligned_representation_losses(
        phi_i=phi_i,
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y_var=y_var,
        y=y,
        cfg=AlignedRepresentationLossConfig(
            lambda_forecast=0.0,
            lambda_residual=0.0,
            lambda_orth=0.0,
            lambda_smooth=0.0,
            lambda_norm=0.0,
            lambda_deployed_temporal_ranking=1.0,
            lambda_temporal_contrastive=1.0,
        ),
    )

    assert losses["deployed_temporal_ranking"].requires_grad
    assert losses["temporal_contrastive"].requires_grad
    assert losses["deployed_temporal_ranking"].item() >= 0.0
    assert losses["temporal_contrastive"].item() >= 0.0


def test_aligned_representation_vicreg_variance_penalty_detects_collapsed_delta_branch():
    torch.manual_seed(25)
    phi_i = torch.randn(8, 4, requires_grad=True)
    delta_phi_v = torch.zeros(8, 4, requires_grad=True)
    y_inv = torch.randn(8, 2, requires_grad=True)
    y_var = torch.randn(8, 2, requires_grad=True)
    y = torch.randn(8, 2)

    losses = aligned_representation_losses(
        phi_i=phi_i,
        delta_phi_v=delta_phi_v,
        y_inv=y_inv,
        y_var=y_var,
        y=y,
        cfg=AlignedRepresentationLossConfig(
            lambda_forecast=1.0,
            lambda_residual=0.0,
            lambda_orth=0.0,
            lambda_smooth=0.0,
            lambda_norm=0.0,
            lambda_hsic=0.0,
            lambda_vicreg_variance=1.0,
            lambda_vicreg_covariance=0.0,
        ),
    )

    assert losses["vicreg_variance"].item() > 0.0
    assert losses["total"].requires_grad
