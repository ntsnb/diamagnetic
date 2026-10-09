"""Source parity and persistent-gradient-memory FSNet integration checks."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from paper_impl.official_fsnet import (OfficialFSNet, OfficialFSNetConfig,
    OnlineTSFSamePadConv, AuthorSamePadConv, author_onenet_branches)
from paper_impl.official_backbones import OfficialForecaster, OfficialForecastConfig, OFFICIAL_MODEL_NAMES
from paper_impl.online_baselines import OnlineBaselineConfig, ReleasedBatch, make_online_baseline, _cpu_clone


ROOT = Path(__file__).resolve().parents[2]


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def source_class(source):
    path = ROOT / "third_party/official_online" / (
        "OnlineTSF/layers/ts2vec/fsnet_.py" if source == "onlinetsf" else "OneNet/models/ts2vec/fsnet_.py")
    spec = importlib.util.spec_from_file_location("_source_oracle_" + source, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SamePadConv


@pytest.mark.parametrize("source", ["onenet", "onlinetsf"])
def test_samepad_source_parity_including_trigger_read_write_and_gradient_ema(source):
    if source == "onlinetsf" and not torch.cuda.is_available():
        pytest.skip("unmodified OnlineTSF SamePadConv explicitly allocates CUDA memory")
    device = "cuda" if source == "onlinetsf" else "cpu"
    cls = source_class(source)
    author = cls(4, 8, 3, **({"device": torch.device(device)} if source == "onenet" else {})).to(device)
    port = (OnlineTSFSamePadConv if source == "onlinetsf" else AuthorSamePadConv)(4, 8, 3).to(device)
    port.load_state_dict(author.state_dict(), strict=False)
    # Direct audit: upstream grad EMAs, query EMA and trigger are absent.
    assert "grads" not in author.state_dict() and "f_grads" not in author.state_dict()
    for model in (author, port):
        model.eval()
    if source == "onlinetsf":
        author.try_trigger = True
        port.try_trigger.fill_(True)
    for step in range(3):
        x = torch.randn(2, 4, 13, device=device)
        if step == 2:
            author.trigger = 1
            port.trigger.fill_(True)
        a, b = author(x), port(x)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        a.square().mean().backward()
        b.square().mean().backward()
        author.store_grad()
        port.store_grad()
        for key in ("grads", "f_grads", "q_ema", "W"):
            torch.testing.assert_close(getattr(author, key), getattr(port, key), rtol=2e-5, atol=2e-6)
        author.zero_grad(set_to_none=True)
        port.zero_grad(set_to_none=True)
    assert set(("grads", "f_grads", "q_ema", "trigger", "try_trigger", "q_initialized")) <= port.state_dict().keys()


def test_full_default_author_encoder_has_eleven_residual_blocks():
    model = OfficialFSNet(OfficialFSNetConfig(24, 8, 3, (0,)))
    assert len(model.encoder.feature_extractor.net) == 11
    assert len(list(model.memory_layers())) == 22
    model.requires_grad_(True)
    assert all(not layer.W.requires_grad for layer in model.memory_layers())
    x = torch.randn(2, 24, 3)
    assert model(x).shape == (2, 8, 1)


@pytest.mark.parametrize("source", ["onenet", "onlinetsf"])
def test_grad_memory_is_restored_before_next_forced_trigger(source, tmp_path):
    cfg = OfficialFSNetConfig(12, 4, 3, (0,), source=source, hidden_dims=8, output_dims=16,
                              depth=1, context_length=12, revin=source == "onlinetsf")
    first = OfficialFSNet(cfg).eval()
    x = torch.randn(2, 12, 3)
    first(x).square().mean().backward()
    first.store_grad()
    first.zero_grad(set_to_none=True)
    for layer in first.memory_layers():
        layer.trigger.fill_(True)
        layer.try_trigger.fill_(True)
    path = tmp_path / "fsnet.pt"
    torch.save(first.state_dict(), path)
    restored = OfficialFSNet(cfg).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    for model in (first, restored):
        pred = model(x)
        pred.square().mean().backward()
        model.store_grad()
        model.zero_grad(set_to_none=True)
    same(first.state_dict(), restored.state_dict())
    torch.testing.assert_close(first(x), restored(x), rtol=0, atol=0)


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_eight_full_forecasters_with_official_fsnet_onenet_next_step(name):
    primary = OfficialForecaster(OfficialForecastConfig(name, 24, 8, 3, (0,), hidden=16, heads=8,
        depth=1, dropout=0., patch_len=4, patch_stride=2, moving_avg=3, fedformer_modes=4,
        crossformer_seg_len=4))
    secondary = OfficialFSNet(OfficialFSNetConfig(24, 8, 3, (0,), hidden_dims=8, output_dims=16, depth=1))
    config = OnlineBaselineConfig(24, 8, (0,), seed=31)
    method = make_online_baseline("OneNet", primary, config, secondary_model=secondary)
    x, y = torch.randn(1, 24, 3), torch.randn(1, 8, 1)
    method.forecast(x, 23)
    method.update(ReleasedBatch(x, y, (23,), (31,)), time=31)
    memories = list(method.secondary_model.memory_layers())
    assert any(torch.count_nonzero(layer.grads) for layer in memories)
    for layer in memories:
        layer.trigger.fill_(True)
    saved = method.state_dict()
    restored = make_online_baseline("OneNet", primary, config, secondary_model=secondary)
    restored.load_state_dict(saved)
    for control in (method, restored):
        control.forecast(x, 32)
        control.update(ReleasedBatch(x, y, (24,), (32,)), time=32)
    same(method.state_dict(), restored.state_dict())
    assert all(not layer.W.requires_grad for layer in method.secondary_model.memory_layers())


def test_original_two_fsnet_branches_and_numpy_rng_resume():
    first, second = author_onenet_branches(12, 4, 3, (0,), hidden_dims=8, output_dims=16, depth=1)
    config = OnlineBaselineConfig(12, 4, (0,), seed=9)
    method = make_online_baseline("OneNet", first, config, secondary_model=second)
    x, y = torch.randn(4, 12, 3), torch.randn(4, 4, 1)
    numpy_before = copy.deepcopy(np.random.get_state())
    method.fit_offline(x, y, batch_size=2)
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    state = method.state_dict()
    restored = make_online_baseline("OneNet", first, config, secondary_model=second)
    restored.load_state_dict(state)
    for control in (method, restored):
        control.forecast(x[:1], 11)
        control.update(ReleasedBatch(x[:1], y[:1], (11,), (15,)), time=15)
    same(method.state_dict(), restored.state_dict())
    assert method.model.training is False and method.secondary_model.training is False
