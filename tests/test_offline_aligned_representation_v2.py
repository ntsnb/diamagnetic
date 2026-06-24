from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from train.offline_aligned_representation_v2 import (
    AlignedRepresentationTrainConfig,
    build_aligned_representation_model,
    evaluate_aligned_representation,
    train_aligned_representation_epoch,
)


def test_aligned_representation_training_reduces_toy_forecast_loss():
    torch.manual_seed(61)
    x = torch.randn(96, 4, 3)
    y = x[:, -1, :2] + 0.5 * x[:, -2, :2]
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    model = build_aligned_representation_model(
        input_dim=4 * 3,
        repr_dim=8,
        hidden_dim=24,
        out_dim=2,
        dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    cfg = AlignedRepresentationTrainConfig(
        lambda_forecast=1.0,
        lambda_residual=0.2,
        lambda_orth=0.01,
        lambda_smooth=0.01,
        lambda_norm=0.001,
    )

    before = evaluate_aligned_representation(loader, model, torch.device("cpu"))["forecast_mse"]
    for _ in range(6):
        train_aligned_representation_epoch(loader, model, opt, cfg, torch.device("cpu"))
    after = evaluate_aligned_representation(loader, model, torch.device("cpu"))["forecast_mse"]

    assert after < before


def test_aligned_representation_model_exposes_phi_i_delta_phi_v_and_phi_v():
    torch.manual_seed(62)
    model = build_aligned_representation_model(
        input_dim=4 * 3,
        repr_dim=8,
        hidden_dim=24,
        out_dim=2,
        dropout=0.0,
    )
    x = torch.randn(5, 4, 3)

    out = model(x)

    assert out.representation.phi_i.shape == (5, 8)
    assert out.representation.delta_phi_v.shape == (5, 8)
    assert out.representation.phi_v.shape == (5, 8)
    assert out.y_inv.shape == (5, 2)
    assert out.y_var.shape == (5, 2)
    assert torch.allclose(out.representation.phi_v, out.representation.phi_i + out.representation.delta_phi_v)


def test_aligned_representation_builder_can_zero_initialize_variant_branch_and_regressor():
    torch.manual_seed(63)
    model = build_aligned_representation_model(
        input_dim=4 * 3,
        repr_dim=8,
        hidden_dim=24,
        out_dim=2,
        dropout=0.0,
        zero_init_variant=True,
        zero_init_variant_regressor=True,
    )
    x = torch.randn(5, 4, 3)

    out = model(x)

    assert torch.allclose(out.representation.delta_phi_v, torch.zeros_like(out.representation.delta_phi_v), atol=1e-8)
    assert torch.allclose(out.y_var, torch.zeros_like(out.y_var), atol=1e-8)


def test_aligned_representation_training_reports_v21_alignment_metrics():
    torch.manual_seed(64)
    x = torch.randn(32, 4, 3)
    y = x[:, -1, :2]
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    model = build_aligned_representation_model(
        input_dim=4 * 3,
        repr_dim=8,
        hidden_dim=24,
        out_dim=2,
        dropout=0.0,
        use_projected_residual=True,
        stop_gradient_projection_anchor=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    cfg = AlignedRepresentationTrainConfig(
        lambda_forecast=1.0,
        lambda_inv_forecast=0.2,
        lambda_barlow=0.1,
        lambda_temporal_ranking=0.1,
        lambda_phi_vicreg_variance=0.1,
        lambda_phi_vicreg_covariance=0.1,
    )

    metrics = train_aligned_representation_epoch(loader, model, opt, cfg, torch.device("cpu"))
    eval_metrics = evaluate_aligned_representation(loader, model, torch.device("cpu"))

    assert "inv_forecast" in metrics
    assert "barlow" in metrics
    assert "temporal_ranking" in metrics
    assert "parallel_ratio" in metrics
    assert "inv_forecast_mse" in eval_metrics
    assert "parallel_ratio" in eval_metrics
