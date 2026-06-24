from __future__ import annotations

import torch
import torch.nn as nn

from models.adaptation_v2 import (
    LayerAwareHyperAdapterGenerator,
    LayerAwareHyperLoRAGenerator,
    build_stage3_v2_param_delta,
)


def _named_shapes() -> dict[str, tuple[int, int]]:
    return {
        "experts.0.proj.weight": (4, 3),
        "output_head.0.weight": (2, 4),
    }


def test_hyperlora_zero_drift_produces_zero_delta_for_all_registered_layers():
    torch.manual_seed(101)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes=_named_shapes(),
        drift_dim=5,
        hidden_dim=12,
        rank=3,
        max_scale=0.2,
    )
    params = {name: torch.randn(shape) for name, shape in _named_shapes().items()}

    delta = build_stage3_v2_param_delta(params, torch.zeros(5), gen)

    assert set(delta) == set(params)
    for name, value in delta.items():
        assert torch.allclose(value, torch.zeros_like(params[name]), atol=1e-8)


def test_hyperlora_has_layer_specific_trainable_adapters():
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes=_named_shapes(),
        drift_dim=5,
        hidden_dim=12,
        rank=3,
        max_scale=0.2,
    )

    assert gen.adapter_count == 2
    assert set(gen.registered_param_names) == set(_named_shapes())
    assert len({id(adapter) for adapter in gen.adapters.values()}) == 2


def test_hyperlora_nonzero_drift_gives_gradient_to_zero_initialized_lora_b():
    torch.manual_seed(102)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes=_named_shapes(),
        drift_dim=5,
        hidden_dim=12,
        rank=3,
        max_scale=0.2,
    )
    params = {name: torch.randn(shape) for name, shape in _named_shapes().items()}
    drift = torch.tensor([0.5, -1.0, 0.2, 0.1, 1.5])
    target = {name: torch.full(shape, 0.03) for name, shape in _named_shapes().items()}

    delta = build_stage3_v2_param_delta(params, drift, gen)
    loss = sum(torch.mean((delta[name] - target[name]) ** 2) for name in delta)
    loss.backward()

    b_grad_norm = sum(
        adapter.lora_b.grad.detach().abs().sum().item()
        for adapter in gen.adapters.values()
        if adapter.lora_b.grad is not None
    )
    assert b_grad_norm > 0.0


def test_hyperlora_training_step_can_learn_signed_drift_direction():
    torch.manual_seed(103)
    model = nn.Linear(3, 2, bias=False)
    params = {"weight": model.weight}
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=16,
        rank=3,
        max_scale=0.5,
    )
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)
    drift = torch.tensor([1.0, -0.5, 0.25, 0.8])

    for _ in range(30):
        pos = build_stage3_v2_param_delta(params, drift, gen)["weight"]
        neg = build_stage3_v2_param_delta(params, -drift, gen)["weight"]
        loss = torch.mean((pos - 0.04) ** 2) + torch.mean((neg + 0.04) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    pos = build_stage3_v2_param_delta(params, drift, gen)["weight"]
    neg = build_stage3_v2_param_delta(params, -drift, gen)["weight"]

    assert pos.mean() > 0.0
    assert neg.mean() < 0.0


def test_hyperadapter_zero_drift_produces_zero_delta_for_weight_and_bias():
    torch.manual_seed(104)
    shapes = {
        "output_head.0.weight": (4, 3),
        "output_head.0.bias": (4,),
        "output_head.2.weight": (2, 4),
        "output_head.2.bias": (2,),
    }
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes=shapes,
        drift_dim=5,
        hidden_dim=12,
        rank=3,
        max_scale=0.2,
    )
    params = {name: torch.randn(shape) for name, shape in shapes.items()}

    delta = build_stage3_v2_param_delta(
        params,
        torch.zeros(5),
        gen,
        target_names=("weight", "bias"),
    )

    assert set(delta) == set(params)
    for name, value in delta.items():
        assert torch.allclose(value, torch.zeros_like(params[name]), atol=1e-8)


def test_hyperadapter_bias_adapter_receives_gradient_for_nonzero_drift():
    torch.manual_seed(105)
    shapes = {"output_head.2.bias": (2,)}
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes=shapes,
        drift_dim=4,
        hidden_dim=10,
        rank=3,
        max_scale=0.3,
    )
    params = {"output_head.2.bias": torch.randn(2)}
    drift = torch.tensor([0.5, -1.0, 0.2, 0.7])
    target = torch.tensor([0.05, -0.03])

    delta = build_stage3_v2_param_delta(
        params,
        drift,
        gen,
        target_names=("bias",),
    )["output_head.2.bias"]
    loss = torch.mean((delta - target) ** 2)
    loss.backward()

    bias_grad_norm = sum(
        adapter.basis.grad.detach().abs().sum().item()
        for adapter in gen.bias_adapters.values()
        if adapter.basis.grad is not None
    )
    assert bias_grad_norm > 0.0
