from __future__ import annotations

import torch
import torch.nn as nn

from models.adaptation_v2 import IdentitySignedSeparableAdaptationGenerator, build_identity_param_delta
from train.stage3_functional import functional_call_with_param_delta


def test_functional_delta_forward_preserves_original_parameters():
    torch.manual_seed(31)
    model = nn.Linear(3, 2, bias=False)
    original_weight = model.weight.detach().clone()
    x = torch.randn(4, 3)
    delta = {"weight": torch.full_like(model.weight, 0.25)}

    y_adapted = functional_call_with_param_delta(model, delta, x)
    y_original = model(x)

    assert torch.allclose(model.weight, original_weight)
    assert not torch.allclose(y_adapted, y_original)


def test_functional_delta_forward_allows_generator_gradients():
    torch.manual_seed(32)
    model = nn.Linear(3, 2, bias=False)
    for param in model.parameters():
        param.requires_grad_(False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=4,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    params = {"weight": model.weight}
    drift = torch.tensor([0.2, -1.0, 0.7, 0.5])
    x = torch.randn(5, 3)
    y = torch.randn(5, 2)

    delta = build_identity_param_delta(params, drift, gen)
    y_hat = functional_call_with_param_delta(model, delta, x)
    loss = torch.mean((y_hat - y) ** 2)
    loss.backward()

    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in gen.parameters()
        if p.grad is not None
    )
    base_grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0
    assert base_grad_count == 0
