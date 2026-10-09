import pytest
import torch
from paper_impl.lrp import RelevanceTape
from paper_impl.model import ModelConfig, PaperForecaster


def tiny(**changes):
    options = dict(n_variables=3, lookback=4, horizon=2, hidden=8, heads=2, n_experts=2, targets=2)
    options.update(changes)
    torch.manual_seed(91)
    return PaperForecaster(ModelConfig(**options)).double()


def test_linear_rule_closed_form_and_residual():
    x = torch.tensor([[2., -1.]], dtype=torch.double)
    w = torch.tensor([[3., 2.]], dtype=torch.double, requires_grad=True)
    tape = RelevanceTape(x, 1e-5)
    y = tape.linear(0, w, torch.tensor([.7], dtype=torch.double))
    propagation = tape.propagate(y, tape.value(y))
    expected = x * w * 4.7 / (4.7 + 1e-5)
    torch.testing.assert_close(propagation.input_relevance, expected)
    torch.testing.assert_close(propagation.residual, torch.tensor([4.7 - expected.sum()], dtype=torch.double))
    assert propagation.conservation_error.abs().max() < 1e-14


def test_bilinear_equal_allocation_and_conservation():
    tape = RelevanceTape(torch.tensor([[[2.]]], dtype=torch.double))
    # 同一输入经两条分支相乘，两条relevance消息最终在input累加。
    left = tape.scale(0, 3)
    right = tape.scale(0, 4)
    output = tape.matmul(left, right)
    r = tape.propagate(output, tape.value(output))
    assert r.input_relevance.item() == pytest.approx(48, abs=2e-6)
    assert r.conservation_error.abs().max() < 1e-12


def test_softmax_rule_includes_virtual_bias_residual():
    x = torch.tensor([[.2, .8, -.1]], dtype=torch.double)
    tape = RelevanceTape(x)
    output = tape.softmax(0)
    seed = torch.tensor([[.7, 0., 0.]], dtype=torch.double)
    r = tape.propagate(output, seed)
    torch.testing.assert_close(r.input_relevance, x * (seed - x.softmax(-1) * .7))
    assert r.residual.abs().item() > 0
    assert r.conservation_error.abs().max() < 1e-14


def test_lrp_linear_rule_parameter_gradcheck():
    x = torch.tensor([[.2, .7]], dtype=torch.double)
    def attribution(weight):
        tape = RelevanceTape(x)
        output = tape.linear(0, weight, torch.tensor([.9], dtype=torch.double))
        return tape.propagate(output, tape.value(output)).input_relevance
    weight = torch.tensor([[.6, .3]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(attribution, (weight,))


@pytest.mark.parametrize("encoder", ["temporal_transformer", "inverted_transformer", "patch_transformer"])
def test_forward_and_expert_attribution_shape(encoder):
    m = tiny(encoder=encoder)
    x = torch.randn(2, 4, 3, dtype=torch.double)
    y = m(x)
    a = m.attribute(x)
    assert y.prediction.shape == (2, 2, 2)
    assert a.maps.shape == (2, 2, 2, 3, 4)
    assert len(a.channels) == len(m.expert_weights())
    assert a.conservation_error.max() < 1e-10
    torch.testing.assert_close(y.gates.sum(-1), torch.ones(2, dtype=torch.double))


def test_attribution_independent_of_router_and_head_parameter_gradients():
    m = tiny()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    a = m.attribute(x)
    with torch.no_grad():
        m.router.bias.add_(torch.tensor([8., -8.], dtype=torch.double))
    torch.testing.assert_close(a.maps, m.attribute(x).maps)
    a.maps.square().mean().backward()
    assert all(p.grad is None for p in m.output_head.parameters())
    assert m.router.weight.grad is None
    assert m.experts[0].blocks[0].q.weight.grad.abs().sum() > 0
    assert m.experts[0].blocks[0].k.weight.grad.abs().sum() > 0
    assert m.experts[0].variable_embedding.grad.abs().sum() > 0


def test_lrp_parameter_gradient_matches_finite_difference():
    m = tiny(targets=1)
    x = torch.randn(1, 4, 3, dtype=torch.double)
    parameter = m.experts[0].blocks[0].q.weight
    scalar = m.attribute(x).maps.square().sum()
    derivative = torch.autograd.grad(scalar, parameter)[0][0, 0]
    with torch.no_grad():
        old = parameter[0, 0].item()
        parameter[0, 0] = old + 1e-6
        plus = m.attribute(x).maps.square().sum()
        parameter[0, 0] = old - 1e-6
        minus = m.attribute(x).maps.square().sum()
        parameter[0, 0] = old
    torch.testing.assert_close(derivative, (plus - minus) / 2e-6, rtol=1e-3, atol=1e-7)


def test_positions_and_variable_identity_are_not_lost():
    m = tiny()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    assert not torch.allclose(m(x).prediction, m(x.flip(1)).prediction, atol=1e-9)
    assert not torch.allclose(m.expert_prediction(x, 0), m.expert_prediction(x.flip(2), 0), atol=1e-9)


def test_identity_functional_gains_and_gradient_path():
    m = tiny()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    weights = m.expert_weights()
    gains = {key: torch.ones(2, *p.shape, dtype=p.dtype) for key, p in weights.items()}
    expected = m(x).prediction
    torch.testing.assert_close(m(x, modulations=gains).prediction, expected)
    gains = {key: value * 1.1 for key, value in gains.items()}
    after = m(x, modulations=gains).prediction
    gradient = torch.autograd.grad(after.square().sum(), tuple(weights.values()))
    assert sum(g.abs().sum() for g in gradient) > 0
    torch.testing.assert_close(m(x).prediction, expected)


def test_posterior_mean_is_deterministic_in_train_and_eval():
    m = tiny()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    torch.testing.assert_close(m(x).prediction, m(x).prediction)
    assert not torch.equal(m(x, stochastic=True).gates, m(x, stochastic=True).gates)


def test_target_horizon_index_and_tape_forward_behavior():
    model = tiny(horizon=3, targets=2)
    x = torch.randn(2, 4, 3, dtype=torch.double)
    expert = model.experts[0]
    tape = RelevanceTape(x)
    node = tape.mean(expert.on_tape(tape, "experts.0."), 1)
    for layer in model.output_head:
        node = tape.linear(node, layer.weight, layer.bias) if isinstance(layer, torch.nn.Linear) else tape.gelu(node)
    flat = tape.value(node)
    prediction = model.expert_prediction(x, 0)
    for j in range(3):
        for i in range(2):
            torch.testing.assert_close(flat[:, j * 2 + i], prediction[:, j, i])
    attribution = model.attribute(x)
    torch.testing.assert_close(sum(attribution.residual_by_operation.values()), attribution.residual)


def test_asymmetric_per_sample_gain_locks_output_channel_direction():
    x = torch.tensor([[[2., 3.]], [[5., 7.]]], dtype=torch.double)
    weight = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.double)
    gain = torch.ones(2, 3, 2, dtype=torch.double)
    gain[0, 1] = 0
    gain[1, 2, 0] = 3
    bias = torch.tensor([.1, .2, .3], dtype=torch.double)
    tape = RelevanceTape(x)
    node = tape.linear(0, weight * gain, bias)
    got = tape.value(node)
    expected = torch.stack([torch.nn.functional.linear(x[b], weight * gain[b], bias) for b in range(2)])
    torch.testing.assert_close(got, expected)
    assert got[0, 0, 1] == .2  # 权重行置零仍保留bias，而不是误称输出为零。


def test_output_head_unsupported_layer_fails_explicitly():
    model = tiny()
    model.output_head[1] = torch.nn.Sigmoid()
    with pytest.raises(TypeError, match="No LRP rule"):
        model.attribute(torch.randn(1, 4, 3, dtype=torch.double))


def test_float32_zero_inputs_have_finite_maps_and_closed_ledger():
    model = tiny().float()
    a = model.attribute(torch.zeros(2, 4, 3))
    assert torch.isfinite(a.maps).all()
    assert torch.isfinite(a.residual).all()
    assert a.conservation_error.max() < 1e-4


@pytest.mark.parametrize("modulated", [False, True])
def test_expert_tape_matches_fast_forward_at_double_precision(modulated):
    model = tiny()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    expert = model.experts[0]
    gains = {name: 1 + .2 * torch.randn(2, *weight.shape, dtype=torch.double)
             for name, weight in model.expert_weights().items()} if modulated else None
    tape = RelevanceTape(x)
    node = expert.on_tape(tape, "experts.0.", gains)
    fast = expert(x, "experts.0.", gains)
    torch.testing.assert_close(tape.value(node), fast, rtol=1e-12, atol=1e-12)
