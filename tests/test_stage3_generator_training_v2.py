from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from models.adaptation_v2 import IdentitySignedSeparableAdaptationGenerator
from models.adaptation_v2 import LayerAwareHyperAdapterGenerator
from models.factor_mechanism import FactorMechanismConfig, build_factor_mechanism
from train.stage3_generator_training import (
    Stage3GeneratorLossConfig,
    freeze_module_parameters,
    stage3_generator_train_step,
)


def test_stage3_generator_train_step_updates_only_generator_and_decreases_toy_loss():
    torch.manual_seed(41)
    base = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        base.weight.fill_(0.2)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=12,
        max_dim=8,
        max_scale=0.5,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)
    x = torch.randn(32, 3)
    y = x.sum(dim=1, keepdim=True)
    drift = torch.tensor([1.0, -0.5, 0.25])
    base_weight_before = base.weight.detach().clone()

    freeze_module_parameters(base)
    first = stage3_generator_train_step(
        model=base,
        generator=gen,
        optimizer=opt,
        x=x,
        y=y,
        drift=drift,
        loss_cfg=Stage3GeneratorLossConfig(lambda_delta=0.0, lambda_zero=0.0, lambda_shift=0.0),
    )
    later = first
    for _ in range(25):
        later = stage3_generator_train_step(
            model=base,
            generator=gen,
            optimizer=opt,
            x=x,
            y=y,
            drift=drift,
            loss_cfg=Stage3GeneratorLossConfig(lambda_delta=0.0, lambda_zero=0.0, lambda_shift=0.0),
        )

    assert later["adapted_mse"] < first["adapted_mse"]
    assert torch.allclose(base.weight, base_weight_before)
    assert all(param.grad is None for param in base.parameters())
    assert any(param.grad is not None for param in gen.parameters())


def test_stage3_generator_train_step_reports_zero_drift_regularizer():
    torch.manual_seed(42)
    base = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=2,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.01)

    metrics = stage3_generator_train_step(
        model=base,
        generator=gen,
        optimizer=opt,
        x=torch.randn(8, 2),
        y=torch.randn(8, 1),
        drift=torch.tensor([1.0, -1.0]),
        loss_cfg=Stage3GeneratorLossConfig(lambda_delta=0.1, lambda_zero=0.5, lambda_shift=0.1),
    )

    assert set(metrics) >= {
        "loss",
        "base_mse",
        "adapted_mse",
        "delta_norm",
        "zero_delta_norm",
        "candidate_shift",
    }
    assert metrics["zero_delta_norm"] == 0.0


def test_stage3_generator_train_step_reports_improvement_and_intervention_losses():
    torch.manual_seed(43)
    base = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=2,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.01)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    drift = torch.tensor([1.0, -0.5])

    metrics = stage3_generator_train_step(
        model=base,
        generator=gen,
        optimizer=opt,
        x=x,
        y=y,
        drift=drift,
        loss_cfg=Stage3GeneratorLossConfig(
            lambda_delta=0.0,
            lambda_zero=0.0,
            lambda_shift=0.0,
            lambda_improvement=2.0,
            improvement_margin=0.1,
            lambda_intervention_compare=3.0,
            intervention_margin=0.2,
            lambda_shift_budget=4.0,
            shift_budget=0.0,
        ),
        negative_drifts={"shuffled": -drift},
    )

    assert metrics["improvement_loss"] > 0.0
    assert metrics["intervention_compare_loss"] > 0.0
    assert "negative_shuffled_mse" in metrics
    assert "shift_budget_loss" in metrics
    assert metrics["loss"] > metrics["adapted_mse"]


def test_intervention_compare_can_train_against_negative_drift_without_detach():
    torch.manual_seed(431)
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[0.4, -0.2]]))
    x = torch.randn(16, 2)
    y = base(x).detach()
    drift = torch.zeros(2)
    negative = torch.tensor([1.0, -0.5])

    def run_once(detach_negative_mse: bool) -> float:
        torch.manual_seed(432)
        gen = IdentitySignedSeparableAdaptationGenerator(
            drift_dim=2,
            hidden_dim=8,
            max_dim=8,
            max_scale=0.5,
        )
        before = torch.cat([param.detach().flatten().clone() for param in gen.parameters()])
        opt = torch.optim.SGD(gen.parameters(), lr=0.1)
        stage3_generator_train_step(
            model=base,
            generator=gen,
            optimizer=opt,
            x=x,
            y=y,
            drift=drift,
            loss_cfg=Stage3GeneratorLossConfig(
                lambda_delta=0.0,
                lambda_zero=0.0,
                lambda_shift=0.0,
                lambda_intervention_compare=1.0,
                intervention_margin=10.0,
                detach_negative_mse=detach_negative_mse,
            ),
            negative_drifts={"amplified_2x": negative},
        )
        after = torch.cat([param.detach().flatten().clone() for param in gen.parameters()])
        return float(torch.linalg.vector_norm(after - before).item())

    detached_change = run_once(detach_negative_mse=True)
    attached_change = run_once(detach_negative_mse=False)

    assert detached_change == pytest.approx(0.0, abs=1e-10)
    assert attached_change > 0.0


def test_stage3_generator_train_step_reports_serving_aligned_metrics():
    torch.manual_seed(44)
    base = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=2,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.01)

    metrics = stage3_generator_train_step(
        model=base,
        generator=gen,
        optimizer=opt,
        x=torch.randn(8, 2),
        y=torch.randn(8, 1),
        drift=torch.tensor([1.0, -1.0]),
        loss_cfg=Stage3GeneratorLossConfig(
            lambda_delta=0.1,
            lambda_zero=0.5,
            lambda_shift=0.1,
            blend_alpha=0.35,
            max_delta=0.05,
        ),
    )

    assert set(metrics) >= {
        "candidate_mse",
        "blended_mse",
        "base_mae",
        "candidate_mae",
        "blended_mae",
        "candidate_shift_l1",
        "candidate_shift_mse",
        "delta_norm_pre_clamp",
        "delta_norm_post_clamp",
        "clamp_fraction",
    }
    assert metrics["adapted_mse"] == metrics["candidate_mse"]
    assert metrics["candidate_shift"] == metrics["candidate_shift_l1"]


def test_stage3_generator_train_step_rejects_stochastic_vae_eval_model():
    model = build_factor_mechanism(
        FactorMechanismConfig(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=6,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_vae=True,
        )
    )
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=32,
        max_scale=0.2,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="deterministic VAE eval"):
        stage3_generator_train_step(
            model=model,
            generator=gen,
            optimizer=opt,
            x=torch.randn(2, 4, 3),
            y=torch.randn(2, 1),
            drift=torch.randn(3),
            loss_cfg=Stage3GeneratorLossConfig(),
        )


def test_stage3_generator_train_step_can_target_bias_adapters():
    torch.manual_seed(45)
    base = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        base.weight.zero_()
        base.bias.zero_()
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes={"bias": tuple(base.bias.shape)},
        drift_dim=2,
        hidden_dim=8,
        rank=3,
        max_scale=0.5,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)
    x = torch.zeros(16, 2)
    y = torch.full((16, 1), 0.5)
    drift = torch.tensor([1.0, -0.5])

    first = stage3_generator_train_step(
        model=base,
        generator=gen,
        optimizer=opt,
        x=x,
        y=y,
        drift=drift,
        loss_cfg=Stage3GeneratorLossConfig(lambda_delta=0.0, lambda_zero=0.0, lambda_shift=0.0),
        target_names=("bias",),
    )
    later = first
    for _ in range(40):
        later = stage3_generator_train_step(
            model=base,
            generator=gen,
            optimizer=opt,
            x=x,
            y=y,
            drift=drift,
            loss_cfg=Stage3GeneratorLossConfig(lambda_delta=0.0, lambda_zero=0.0, lambda_shift=0.0),
            target_names=("bias",),
        )

    assert later["adapted_mse"] < first["adapted_mse"]
    assert torch.allclose(base.bias.detach(), torch.zeros_like(base.bias))
