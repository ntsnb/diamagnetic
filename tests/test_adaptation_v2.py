from __future__ import annotations

import torch
import torch.nn as nn

from models.adaptation_v2 import (
    IdentitySignedSeparableAdaptationGenerator,
    build_identity_param_delta,
    summarize_delta,
)


def test_v2_generator_zero_drift_produces_zero_delta_at_initialization():
    torch.manual_seed(11)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=4,
        hidden_dim=8,
        max_dim=16,
        max_scale=0.05,
    )
    params = {"layer.weight": torch.randn(3, 5)}

    delta = build_identity_param_delta(params, torch.zeros(4), gen)

    assert torch.allclose(delta["layer.weight"], torch.zeros_like(params["layer.weight"]))


def test_v2_generator_training_step_makes_nonzero_drift_effective_without_breaking_zero_drift():
    torch.manual_seed(12)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=4,
        hidden_dim=10,
        max_dim=16,
        max_scale=0.10,
    )
    params = {"layer.weight": torch.ones(3, 5)}
    drift = torch.tensor([1.0, -0.5, 0.25, 2.0])
    target_delta = torch.full((3, 5), 0.03)
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)

    before = build_identity_param_delta(params, drift, gen)["layer.weight"]
    loss = torch.mean((before - target_delta) ** 2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    after = build_identity_param_delta(params, drift, gen)["layer.weight"]
    zero_after = build_identity_param_delta(params, torch.zeros_like(drift), gen)["layer.weight"]

    assert torch.mean(torch.abs(after)) > torch.mean(torch.abs(before))
    assert torch.mean((after - target_delta) ** 2) < torch.mean((before - target_delta) ** 2)
    assert torch.allclose(zero_after, torch.zeros_like(zero_after), atol=1e-6)


def test_v2_generator_sign_flip_changes_delta_direction_after_training():
    torch.manual_seed(13)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=12,
        max_dim=16,
        max_scale=0.20,
    )
    params = {"layer.weight": torch.ones(4, 4)}
    drift = torch.tensor([1.0, -2.0, 0.5])
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)

    for _ in range(20):
        pos = build_identity_param_delta(params, drift, gen)["layer.weight"]
        neg = build_identity_param_delta(params, -drift, gen)["layer.weight"]
        loss = torch.mean((pos - 0.05) ** 2) + torch.mean((neg + 0.05) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    pos = build_identity_param_delta(params, drift, gen)["layer.weight"]
    neg = build_identity_param_delta(params, -drift, gen)["layer.weight"]

    assert pos.mean() > 0.0
    assert neg.mean() < 0.0


def test_v2_delta_summary_reports_norm_and_clamp_fraction():
    delta = {
        "a.weight": torch.tensor([[0.0, 0.2], [-0.4, 0.8]]),
        "b.weight": torch.tensor([[0.1]]),
    }

    summary = summarize_delta(delta, max_delta=0.3)

    assert summary["delta_param_count"] == 5
    assert summary["delta_norm_pre_clamp"] > summary["delta_norm_post_clamp"]
    assert summary["clamp_fraction"] == 0.4
