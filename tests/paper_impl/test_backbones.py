"""核对真实旧编码器接入、边界长度、梯度、状态保存与 CPU/CUDA。"""

from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models import attention_blocks as legacy
from paper_impl.backbones import LEGACY_DEPENDENCY_FILES, LEGACY_ENCODERS, build_legacy_encoder


@pytest.fixture(autouse=True, scope="module")
def _small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(encoder="transformer_style", **kwargs):
    values = dict(encoder=encoder, n_variables=3, lookback=7, hidden=8, heads=2,
                  depth=2, patch_size=4, horizon=3, targets=2)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_registry_contains_exact_existing_style_classes_and_cannot_be_mutated():
    expected = {
        "transformer_style": legacy.TransformerTemporalEncoder,
        "itransformer_style": legacy.InvertedTransformerTemporalEncoder,
        "patchtst_style": legacy.PatchTSTTemporalEncoder,
        "informer_style": legacy.InformerTemporalEncoder,
        "autoformer_style": legacy.AutoformerTemporalEncoder,
        "fedformer_style": legacy.FEDformerTemporalEncoder,
        "pyraformer_style": legacy.PyraformerTemporalEncoder,
        "crossformer_style": legacy.CrossformerTemporalEncoder,
    }
    assert dict(LEGACY_ENCODERS) == expected
    with pytest.raises(TypeError):
        LEGACY_ENCODERS["official_transformer"] = nn.Linear
    assert LEGACY_DEPENDENCY_FILES == ("models/__init__.py", "models/attention_blocks.py")


@pytest.mark.parametrize("name", tuple(LEGACY_ENCODERS))
@pytest.mark.parametrize("lookback,variables,horizon,targets", [(1, 1, 1, 1), (3, 4, 5, 3), (7, 3, 3, 2), (8, 2, 2, 4)])
def test_summary_and_forecast_head_have_finite_forward_gradients(name, lookback, variables, horizon, targets):
    torch.manual_seed(13)
    config = _config(name, lookback=lookback, n_variables=variables, horizon=horizon, targets=targets)
    encoder = build_legacy_encoder(config).double()
    head = nn.Linear(config.hidden, horizon * targets).double()
    # 非连续输入同样合法；包含变量轴与时间轴不同长度的情况。
    x = torch.randn(2, variables, lookback, dtype=torch.float64).transpose(1, 2).requires_grad_()
    summary = encoder(x)
    assert summary.shape == (2, config.hidden)
    assert summary.dtype == x.dtype and summary.device == x.device
    assert isinstance(encoder.backbone, LEGACY_ENCODERS[name])
    forecast = head(summary).reshape(2, horizon, targets)
    target = torch.randn_like(forecast)
    loss = (forecast - target).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert any(parameter.grad.abs().sum() > 0 for parameter in encoder.parameters())


@pytest.mark.parametrize("name", tuple(LEGACY_ENCODERS))
def test_dropout_zero_and_deterministic_eval_save_load(name):
    torch.manual_seed(21)
    config = _config(name)
    encoder = build_legacy_encoder(config).double()
    x = torch.randn(3, config.lookback, config.n_variables, dtype=torch.float64)
    assert encoder.legacy_config.dropout == 0
    assert encoder.legacy_config.dim_feedforward == 4 * config.hidden
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0
        if isinstance(module, legacy.SimpleMultiheadAttention):
            assert module.dropout == 0 and module.capture_grads is False
    with torch.no_grad():
        encoder.train()
        training_output = encoder(x)
        encoder.eval()
        expected = encoder(x)
        assert torch.equal(training_output, expected)
        assert torch.equal(encoder(x), expected)
    storage = BytesIO()
    torch.save(encoder.state_dict(), storage)
    storage.seek(0)
    restored = build_legacy_encoder(config).double().eval()
    restored.load_state_dict(torch.load(storage, weights_only=True))
    with torch.no_grad():
        assert torch.equal(restored(x), expected)


@pytest.mark.parametrize("lookback", [1, 3, 7, 8, 9])
def test_patch_style_pads_endpoints_and_keeps_tail_gradient(lookback):
    torch.manual_seed(25)
    config = _config("patchtst_style", lookback=lookback)
    encoder = build_legacy_encoder(config).double()
    x = torch.randn(2, lookback, config.n_variables, dtype=torch.float64, requires_grad=True)
    padding = (-lookback) % config.patch_size
    padded = F.pad(x.transpose(1, 2), (0, padding), mode="replicate").transpose(1, 2) if padding else x
    assert encoder.right_padding == padding
    assert encoder.backbone.patch_len == encoder.backbone.stride == config.patch_size
    assert encoder.legacy_config.seq_len == lookback + padding
    output = encoder(x)
    torch.testing.assert_close(output, encoder.backbone(padded), rtol=0, atol=0)
    (output * torch.arange(config.hidden)).sum().backward()
    assert torch.isfinite(x.grad[:, -1]).all() and x.grad[:, -1].abs().sum() > 0


@pytest.mark.parametrize("lookback", [1, 2, 3, 6, 7, 9])
def test_fedformer_frequency_projection_matches_available_short_window_bins(lookback):
    config = _config("fedformer_style", lookback=lookback)
    encoder = build_legacy_encoder(config)
    bins = min(4, lookback // 2 + 1)
    assert encoder.backbone.topk_freq == bins
    assert encoder.backbone.freq_proj.in_features == 2 * bins * config.n_variables
    assert encoder(torch.randn(2, lookback, config.n_variables)).shape == (2, config.hidden)


@pytest.mark.parametrize("name", tuple(LEGACY_ENCODERS))
def test_depth_one_and_hidden_size_one_are_supported_as_existing_architectures(name):
    config = _config(name, hidden=1, heads=1, depth=1, lookback=1, n_variables=1)
    encoder = build_legacy_encoder(config)
    result = encoder(torch.randn(2, 1, 1))
    assert result.shape == (2, 1) and torch.isfinite(result).all()


@pytest.mark.parametrize("name", ["transformer", "fedformer", "temporal_transformer", "not_a_model", None])
def test_builder_requires_explicit_style_name(name):
    with pytest.raises(ValueError, match="style encoder"):
        build_legacy_encoder(_config(name))


@pytest.mark.parametrize("field,value", [("hidden", 0), ("heads", 3), ("depth", 0), ("lookback", -1),
                                          ("n_variables", 0), ("lookback", 2.5), ("patch_size", 0)])
def test_builder_rejects_invalid_dimensions(field, value):
    config = _config("patchtst_style", **{field: value})
    with pytest.raises(ValueError):
        build_legacy_encoder(config)


def test_input_shape_cannot_change_the_fixed_window_or_variables():
    encoder = build_legacy_encoder(_config())
    for invalid in (torch.randn(2, 6, 3), torch.randn(2, 7, 2), torch.randn(7, 3), torch.empty(0, 7, 3)):
        with pytest.raises(ValueError, match="expected nonempty"):
            encoder(invalid)
    with pytest.raises(TypeError, match="floating point"):
        encoder(torch.ones(2, 7, 3, dtype=torch.long))
    with pytest.raises(TypeError, match="tensor"):
        encoder([[[1.0]]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", tuple(LEGACY_ENCODERS))
@pytest.mark.parametrize("lookback", [3, 7])
def test_cuda_matches_cpu_forward_and_keeps_input_and_parameter_gradients(name, lookback):
    torch.manual_seed(29)
    config = _config(name, lookback=lookback, n_variables=2)
    cpu = build_legacy_encoder(config).eval()
    gpu = build_legacy_encoder(config).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    values = torch.randn(2, config.lookback, config.n_variables)
    x_cpu = values.clone().requires_grad_()
    x_gpu = values.cuda().requires_grad_()
    y_cpu, y_gpu = cpu(x_cpu), gpu(x_gpu)
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, rtol=3e-5, atol=3e-5)
    weights = torch.arange(config.hidden).float()
    (y_cpu * weights).sum().backward()
    (y_gpu * weights.cuda()).sum().backward()
    torch.testing.assert_close(x_gpu.grad.cpu(), x_cpu.grad, rtol=3e-4, atol=3e-5)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters())
