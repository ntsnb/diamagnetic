"""Operator identities and complete-network LRP, separate from forecast smoke tests."""
import pytest
import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

from paper_impl.official_backbones import OFFICIAL_MODEL_NAMES, OfficialForecastConfig, OfficialForecaster
from paper_impl.official_lrp import (
    OperatorRelevanceTape, UnsupportedLRPOperator, attribute_official, _sampling,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(15)
    yield
    torch.set_num_threads(previous)


def config(name, **kwargs):
    settings = dict(model_name=name, lookback=16, horizon=2, n_variables=2,
                    target_indices=(1,), hidden=8, heads=8, depth=2, dropout=0.,
                    moving_avg=3, patch_len=4, patch_stride=2, crossformer_seg_len=4,
                    fedformer_modes=2)
    settings.update(kwargs)
    return OfficialForecastConfig(**settings)


def explain_function(fn, x, relevance=None, epsilon=1e-6):
    graph = make_fx(fn)(x)
    tape = OperatorRelevanceTape(graph, x, epsilon, create_graph=True)
    r = tape.output if relevance is None else relevance
    return tape, tape.propagate(r)


def stable(x, eps=1e-6):
    return x + eps * torch.where(x >= 0, 1., -1.)


def test_softmax_uses_taylor_bias_not_gradient_times_input():
    x = torch.tensor([[0.2, -0.6, 1.4]], requires_grad=True)
    r = torch.tensor([[0.3, -0.2, 0.7]])
    tape, (actual, _, residual, error, ledger) = explain_function(lambda a: a.softmax(-1), x, r)
    expected = x * (r - x.softmax(-1) * r.sum(-1, keepdim=True))
    torch.testing.assert_close(actual, expected)
    assert any("softmax_taylor" in name for name in ledger)
    torch.testing.assert_close(residual, r.sum() - expected.sum())
    assert error.abs() < 1e-6
    gradient = torch.autograd.grad((x.softmax(-1) * r).sum(), x)[0]
    assert not torch.allclose(actual, x * gradient)


def test_affine_bias_and_stabilizer_accounting():
    weight = torch.tensor([[2., -1.], [0.5, 3.]], requires_grad=True)
    bias = torch.tensor([0.4, -0.2], requires_grad=True)
    x = torch.tensor([[1., 2.]])
    tape, (actual, _, residual, error, ledger) = explain_function(lambda a: a @ weight + bias, x)
    # Graph has separate mm and bias-add operations, each with its own epsilon.
    linear = x @ weight
    r_linear = linear * tape.output / stable(tape.output)
    expected = x * ((r_linear / stable(linear)) @ weight.T)
    torch.testing.assert_close(actual, expected)
    assert any("stabilization" in key for key in ledger)
    assert any("bias_or_constant" in key for key in ledger)
    torch.testing.assert_close(actual.sum() + residual, tape.output.sum())
    actual.square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_bilinear_self_matmul_applies_one_half_to_both_paths():
    x = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    r = torch.ones(1, 2, 2)
    tape, (actual, _, _, _, _) = explain_function(lambda a: a @ a, x, r)
    s = r / stable(x @ x)
    expected = 0.5 * x * (s @ x.transpose(-1, -2) + x.transpose(-1, -2) @ s)
    torch.testing.assert_close(actual, expected)


def test_layernorm_centering_and_normalization_identity_formula():
    layer = nn.LayerNorm(3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.1, 0.7, 1.3]))
        layer.bias.copy_(torch.tensor([0.1, -0.3, 0.2]))
    x = torch.tensor([[1., -2., 4.]], requires_grad=True)
    r = torch.tensor([[0.3, -0.7, 0.9]])
    tape, (actual, _, _, error, ledger) = explain_function(layer, x, r)
    centered = x - x.mean(-1, keepdim=True)
    normalized = centered / torch.sqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
    rc = normalized * layer.weight * r / stable(layer(x))
    s = rc / stable(centered)
    expected = x * (s - s.mean(-1, keepdim=True))
    torch.testing.assert_close(actual, expected)
    assert any("normalization_identity" in key for key in ledger)
    assert error.abs() < 1e-5


def test_fft_real_coordinate_rule_matches_explicit_real_dft_matrix():
    x = torch.tensor([[0.2, -0.7, 1.3, 0.5]], requires_grad=True)
    tape, result = explain_function(lambda a: torch.fft.rfft(a), x,
                                    torch.tensor([[[0.1, 0.0], [0.3, -0.2], [0.5, 0.0]]]))
    r = torch.tensor([[[0.1, 0.0], [0.3, -0.2], [0.5, 0.0]]])
    matrix = torch.view_as_real(torch.fft.rfft(torch.eye(4))).reshape(4, -1).T
    output = x @ matrix.T
    expected = x * ((r.reshape(1, -1) / stable(output)) @ matrix)
    torch.testing.assert_close(result[0], expected)
    assert any("complex_real_coordinate__fft_r2c" in key for key in result[4])


def test_complex_linear_epsilon_rule_matches_explicit_real_imaginary_products():
    x = torch.tensor([[1.2 + 0.3j, -0.8 + 0.7j]], requires_grad=True)
    weight = torch.tensor([[0.8 + 0.2j, -0.4 + 0.9j], [0.5 - 0.3j, 0.1 + 0.6j]], requires_grad=True)
    r = torch.tensor([[[0.4, -0.1], [0.3, 0.2]]])
    tape, result = explain_function(lambda a: a @ weight, x, r)
    output = x @ weight
    sr = r[..., 0] / stable(output.real)
    si = r[..., 1] / stable(output.imag)
    real_part = x.real * (sr @ weight.real.T + si @ weight.imag.T)
    imag_part = x.imag * (-sr @ weight.imag.T + si @ weight.real.T)
    torch.testing.assert_close(result[0], torch.stack((real_part, imag_part), -1))
    result[0].square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_convolution_rule_matches_transposed_convolution_epsilon_formula():
    x = torch.tensor([[[0.2, -0.4, 0.6, 0.9]]], requires_grad=True)
    weight = torch.tensor([[[0.8, 0.3, -0.7]]], requires_grad=True)
    bias = torch.tensor([0.3])
    fn = lambda a: torch.nn.functional.conv1d(a, weight, bias, padding=1)
    tape, result = explain_function(fn, x)
    expected = x * torch.nn.functional.conv_transpose1d(tape.output / stable(tape.output), weight, padding=1)
    torch.testing.assert_close(result[0], expected)


def test_unsupported_operator_never_falls_back_to_network_gradient():
    x = torch.ones(1, 3)
    with pytest.raises(UnsupportedLRPOperator, match="sin"):
        explain_function(torch.sin, x)


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_complete_model_relevance_prediction_gradients_and_head_freeze(name):
    model = OfficialForecaster(config(name)).eval()
    x = torch.randn(2, 16, 2)
    with _sampling(x, 42):
        expected = model(x)
    result = attribute_official(model, x, freeze_head=True)
    assert result.maps.shape == (2, 1, 2, 16)
    torch.testing.assert_close(result.predictions, expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(result.maps).all() and result.maps.abs().sum() > 0
    assert result.conservation_error.abs().max() < 2e-4
    assert set(result.channels) == {n + ".weight" for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    loss = result.maps.square().mean() + 0.01 * sum(v.square().mean() for v in result.channels.values())
    loss.backward()
    head_ids = {id(p) for p in model.head_parameters()}
    assert all(p.grad is None for p in model.head_parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters() if id(p) not in head_ids)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if name == "fedformer":
        fourier = model.model.encoder.attn_layers[0].attention.inner_correlation.weights1
        assert fourier.is_complex() and fourier.grad is not None and fourier.grad.abs().sum() > 0
    for key, value in result.channels.items():
        assert value.shape == (2, 1, dict(model.named_parameters())[key].shape[0])


def test_absolute_value_is_taken_per_horizon_before_averaging():
    from paper_impl.official_lrp import _trace
    model = OfficialForecaster(config("transformer", depth=1)).eval()
    x = torch.randn(1, 16, 2)
    result = attribute_official(model, x)
    graph = _trace(model, x, True, 42)
    with _sampling(x, 42):
        tape = OperatorRelevanceTape(graph, x)
    per_step = []
    for h in range(2):
        mask = torch.zeros_like(tape.output)
        mask[:, h, 0] = 1
        per_step.append(tape.propagate(tape.output * mask)[0][0].T)
    torch.testing.assert_close(result.maps[0, 0], torch.stack(per_step).abs().mean(0))
    assert not torch.allclose(result.maps[0, 0], torch.stack(per_step).mean(0).abs())


def test_no_grad_diagnostics_and_training_mode_are_preserved():
    model = OfficialForecaster(config("patchtst")).train()
    x = torch.randn(1, 16, 2)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        result = attribute_official(model, x)
    assert not result.maps.requires_grad and not result.predictions.requires_grad
    assert model.training and all(m.training for m in model.modules())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(buffers[n], b) for n, b in model.named_buffers())


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_full_mixture_channels_exclude_gate_and_heads_and_integrate_parent_contract(name):
    from paper_impl.model import ModelConfig, PaperForecaster
    cfg = ModelConfig(n_variables=2, lookback=16, horizon=2, targets=1, n_experts=2,
                      hidden=8, heads=8, depth=1, expert_backbone=name, target_indices=(1,),
                      official_options=dict(dropout=0., moving_avg=3, patch_len=4, patch_stride=2,
                                            crossformer_seg_len=4, fedformer_modes=2))
    model = PaperForecaster(cfg).eval()
    result = model.attribute(torch.randn(1, 16, 2))
    assert result.maps.shape == (1, 1, 2, 2, 16)
    assert result.residual.shape == (1, 1, 2)
    assert all(key.startswith("experts.") for key in result.channels)
    assert set(result.channels) == set(model.expert_weights())
    head_ids = {id(p) for expert in model.experts for p in expert.head_parameters()}
    assert all(id(dict(model.named_parameters())[key]) not in head_ids for key in result.channels)
    result.maps.square().sum().backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("experts."))
    assert all(p.grad is None for expert in model.experts for p in expert.head_parameters())


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_default_stage_ii_and_iii_preserve_complete_forecast_heads(name):
    from paper_impl.adaptation import CenteredBoundedGenerator, OnlineConfig, OnlineController
    from paper_impl.model import ModelConfig, PaperForecaster
    cfg = ModelConfig(n_variables=2, lookback=16, horizon=2, targets=1, n_experts=1,
                      hidden=8, heads=8, depth=1, expert_backbone=name, target_indices=(1,),
                      official_options=dict(dropout=0., moving_avg=3, patch_len=4, patch_stride=2,
                                            crossformer_seg_len=4, fedformer_modes=2, fedformer_mode_select="low"))
    model = PaperForecaster(cfg).eval()
    head_ids = {id(p) for expert in model.experts for p in expert.head_parameters()}
    heads = {key: p for key, p in model.named_parameters() if id(p) in head_ids}
    original_heads = {key: p.detach().clone() for key, p in heads.items()}
    eligible = model.expert_weights()
    assert eligible and set(eligible).isdisjoint(heads)
    generator = CenteredBoundedGenerator({key: tuple(p.shape) for key, p in eligible.items()},
                                        contrast_dim=2, hidden_dim=4)
    with torch.no_grad():
        for block in generator.blocks.values():
            block.w2.weight.uniform_(-0.2, 0.2)
    x = torch.randn(1, 16, 2)
    gains = generator(torch.ones(1, 2))
    assert set(gains) == set(eligible) and set(gains).isdisjoint(heads)
    assert any(not torch.equal(gain, torch.ones_like(gain)) for gain in gains.values())
    original_weights = {key: p.detach().clone() for key, p in model.named_parameters()}
    prediction = model(x, modulations=gains).prediction
    prediction.square().sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in generator.parameters())
    for key, p in model.named_parameters():
        torch.testing.assert_close(p, original_weights[key], rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    controller = OnlineController(model, OnlineConfig(stage_i=False, stage_ii=True, stage_iii=True,
                                  selector="gradient_only", learning_rate_ii=0.01),
                                  generator=generator, contrast_fn=lambda inputs: inputs[:, -1, 1:2].repeat(1, 2))
    issued = controller.predict(x, 15)
    update = controller.release(17, {15: issued.prediction + 2.0}, current_x=x)
    assert update.experts.accepted
    assert any(not torch.equal(p, original_weights[key]) for key, p in eligible.items())
    for key, p in heads.items():
        torch.testing.assert_close(p, original_heads[key], rtol=0, atol=0)


def test_informer_batch_modulation_identity_gradients_and_state_preservation():
    import copy
    from paper_impl.model import ModelConfig, PaperForecaster
    cfg = ModelConfig(n_variables=2, lookback=16, horizon=2, targets=1, n_experts=2,
                      hidden=8, heads=8, depth=1, expert_backbone="informer", target_indices=(1,))
    model = PaperForecaster(cfg).eval()
    x = torch.randn(3, 16, 2)
    original_state = copy.deepcopy(model.state_dict())
    original_rng = torch.random.get_rng_state()
    original = model._official_prediction(x, 0)
    gains = {key: torch.ones((len(x), *weight.shape), requires_grad=True)
             for key, weight in model.expert_weights().items() if key.startswith("experts.0.")}
    modulated = model._official_prediction(x, 0, gains)
    torch.testing.assert_close(modulated, original, rtol=2e-5, atol=2e-6)
    gradients = torch.autograd.grad(modulated.square().sum(), tuple(gains.values()), allow_unused=True)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
        else:
            assert value == original_state[key]
    assert not model.training and not any(m.training for m in model.modules())


@pytest.mark.parametrize("name, options", [
    ("pyraformer", {"pyraformer_decoder": "attention", "pyraformer_truncate": True}),
    ("patchtst", {"patch_decomposition": True, "patch_kernel_size": 3}),
])
def test_optional_complete_author_decoder_paths(name, options):
    model = OfficialForecaster(config(name, depth=1, **options)).eval()
    result = attribute_official(model, torch.randn(1, 16, 2))
    result.maps.square().sum().backward()
    assert torch.isfinite(result.maps).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_cuda_full_operator_lrp_parameter_gradients(name):
    model = OfficialForecaster(config(name, depth=1)).cuda().eval()
    x = torch.randn(1, 16, 2, device="cuda")
    result = attribute_official(model, x)
    assert result.maps.is_cuda and torch.isfinite(result.maps).all()
    result.maps.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.head_parameters())
    assert result.conservation_error.abs().max() < 2e-4
