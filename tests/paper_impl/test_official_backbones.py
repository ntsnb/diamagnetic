"""Executable smoke evidence for the complete pinned author networks."""
from dataclasses import replace
import copy
import hashlib
import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from paper_impl.official_backbones import (
    OFFICIAL_MODEL_NAMES, OfficialForecastConfig, OfficialForecaster, official_source_info,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(name, **changes):
    values = dict(model_name=name, lookback=24, horizon=8, n_variables=3,
                  target_indices=(2, 0), hidden=16, heads=8, depth=2, d_ff=32,
                  dropout=0.0, moving_avg=3, patch_len=4, patch_stride=2,
                  crossformer_seg_len=4, fedformer_modes=4)
    values.update(changes)
    return OfficialForecastConfig(**values)


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_full_forward_backward_checkpoint_target_selection(name):
    torch.manual_seed(19)
    np.random.seed(19)
    cfg = config(name)
    model = OfficialForecaster(cfg)
    assert type(model.model).__module__.startswith("third_party.official_forecasters.runtime.")
    x = torch.randn(2, cfg.lookback, cfg.n_variables, requires_grad=True)
    before = {key: p.detach().clone() for key, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = model(x)
    assert y.shape == (2, cfg.horizon, 2)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    heads = tuple(model.head_parameters())
    assert heads and all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[key], p) for key, p in model.named_parameters())
    model.eval()
    torch.manual_seed(40)  # Official Informer retains random sparse sampling in eval.
    expected = model(x.detach())
    state_file = io.BytesIO()
    torch.save(model.state_dict(), state_file)
    state_file.seek(0)
    # Deliberately change both RNGs: FEDformer random frequency selection must
    # be restored from the adapter's extra_state, not its construction RNG.
    torch.manual_seed(101)
    np.random.seed(101)
    restored = OfficialForecaster(cfg).eval()
    state = torch.load(state_file, weights_only=True)
    restored.load_pretrained(state)
    torch.manual_seed(40)
    actual = restored(x.detach())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    all_columns = OfficialForecaster(replace(cfg, target_indices=None)).eval()
    all_columns.load_state_dict(state)
    torch.manual_seed(40)
    full = all_columns(x.detach())
    torch.testing.assert_close(actual, full[:, :, [2, 0]], rtol=0, atol=0)


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_head_gd_uses_author_heads_only(name):
    model = OfficialForecaster(config(name)).eval()
    names = model.head_parameter_names()
    heads = list(model.head_parameters())
    assert names and len(names) == len(heads)
    ids = {id(p) for p in heads}
    fixed = {}
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(id(parameter) in ids)
        if id(parameter) not in ids:
            fixed[key] = parameter.detach().clone()
    old_heads = [p.detach().clone() for p in heads]
    optimizer = torch.optim.SGD(heads, lr=0.05)
    model(torch.randn(2, 24, 3)).square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(old_heads, heads))
    assert all(torch.equal(fixed[key], p) for key, p in model.named_parameters() if key in fixed)


def test_models_coexist_without_global_import_aliases():
    import models.attention_blocks as legacy
    paths = list(sys.path)
    aliases = {name: sys.modules.get(name) for name in ("layers", "models", "utils", "cross_models", "pyraformer")}
    instances = [OfficialForecaster(config(name)).eval() for name in OFFICIAL_MODEL_NAMES]
    for model in reversed(instances):
        assert model(torch.randn(2, 24, 3)).shape == (2, 8, 2)
    assert paths == sys.path
    assert all(sys.modules.get(name) is previous for name, previous in aliases.items())
    assert sys.modules["models.attention_blocks"] is legacy


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_pinned_sources_and_runtime_hashes(name):
    root = Path(__file__).resolve().parents[2] / "third_party" / "official_forecasters"
    info = official_source_info(name)
    assert len(info["commit"]) == 40
    assert Path(info["license_path"]).is_file()
    assert hashlib.sha256(Path(info["patch_path"]).read_bytes()).hexdigest() == info["patch_sha256"]
    for folder, key in (("upstream", "upstream_files"), ("runtime", "runtime_files")):
        for relative, expected_hash in info[key].items():
            assert hashlib.sha256((root / folder / name / relative).read_bytes()).hexdigest() == expected_hash
    assert info["implementation_kind"] == ("full_time_series_reference" if name == "transformer" else "full_author_forecaster")


@pytest.mark.parametrize("name", ("autoformer", "transformer", "informer", "fedformer"))
def test_time_mark_decoder_construction(name):
    model = OfficialForecaster(config(name)).eval()
    x = torch.randn(2, 24, 3)
    past = torch.randn(2, 24, 4)
    future = torch.randn(2, 8, 4)
    torch.manual_seed(17)
    pred = model(x, past, future)
    torch.manual_seed(17)
    expanded = model(x, past, torch.cat((past[:, -12:], future), dim=1))
    torch.testing.assert_close(pred, expanded, rtol=0, atol=0)
    with pytest.raises(ValueError, match="x_mark_dec"):
        model(x, past, torch.randn(2, 9, 4))


@pytest.mark.parametrize("changes, message", [
    ({"model_name": "patchtst", "lookback": 3}, "patch_len"),
    ({"model_name": "pyraformer", "lookback": 3}, "pyramid"),
    ({"model_name": "fedformer", "heads": 4}, "heads=8"),
    ({"model_name": "fedformer", "label_len": 5}, "label_len"),
    ({"model_name": "fedformer", "fedformer_version": "Wavelets"}, "Fourier"),
    ({"model_name": "autoformer", "moving_avg": 4}, "odd"),
    ({"model_name": "autoformer", "lookback": 1}, "Autoformer"),
    ({"model_name": "crossformer", "lookback": 3, "crossformer_win_size": 4}, "merging"),
    ({"model_name": "informer", "lookback": 1}, "lookback"),
    ({"model_name": "informer", "lookback": 2, "depth": 2}, "distillation"),
    ({"model_name": "transformer", "target_indices": (0, 0)}, "duplicates"),
    ({"model_name": "transformer", "target_indices": (3,)}, "target_indices"),
    ({"model_name": "transformer", "dropout": float("nan")}, "dropout"),
])
def test_invalid_model_constraints_are_explicit(changes, message):
    changes = dict(changes)
    name = changes.pop("model_name")
    with pytest.raises(ValueError, match=message):
        config(name, **changes)


def test_pyraformer_full_attention_decoder_and_batchnorm_guard():
    model = OfficialForecaster(config("pyraformer", pyraformer_decoder="attention", pyraformer_truncate=True))
    pred = model(torch.randn(2, 24, 3), torch.randn(2, 24, 4), torch.randn(2, 8, 4))
    pred.square().mean().backward()
    assert model.model.decoder.layers[0].slf_attn.w_qs.weight.grad is not None
    short = OfficialForecaster(config("pyraformer", lookback=8))
    with pytest.raises(ValueError, match="coarsest pyramid"):
        short(torch.randn(1, 8, 3))
    assert short.eval()(torch.randn(1, 8, 3)).shape == (1, 8, 2)


def test_patchtst_complete_decomposition_variant():
    model = OfficialForecaster(config("patchtst", patch_decomposition=True, patch_kernel_size=3, patch_individual=True))
    model(torch.randn(2, 24, 3)).square().mean().backward()
    assert len(tuple(model.head_parameters())) == 12
    assert any("model_res.head" in name for name in model.head_parameter_names())
    assert any("model_trend.head" in name for name in model.head_parameter_names())


def test_fedformer_raw_random_checkpoint_requires_frequency_metadata():
    model = OfficialForecaster(config("fedformer"))
    raw = model.model.state_dict()
    restored = OfficialForecaster(config("fedformer"))
    with pytest.raises(ValueError, match="frequency_indices"):
        restored.load_pretrained(raw)
    with pytest.raises(ValueError, match="frequency_indices"):
        restored.load_pretrained(raw, frequency_indices={})
    restored.load_pretrained(raw, frequency_indices=model.get_extra_state()["frequency_indices"])
    x = torch.randn(2, 24, 3)
    torch.testing.assert_close(model.eval()(x), restored.eval()(x), rtol=0, atol=0)


@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_deepcopy_functional_parameter_modulation_and_input_gradient(name):
    model = OfficialForecaster(config(name)).eval()
    clone = copy.deepcopy(model)
    x = torch.randn(2, 24, 3, requires_grad=True)
    scale = torch.tensor(0.9, requires_grad=True)
    adapted = {key: parameter * scale for key, parameter in clone.named_parameters()}
    buffers = dict(clone.named_buffers())
    pred = torch.func.functional_call(clone, (adapted, buffers), (x,))
    pred.square().mean().backward()
    assert scale.grad is not None and torch.isfinite(scale.grad) and scale.grad.abs() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.model is not clone.model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_cuda_full_forward_backward(name):
    model = OfficialForecaster(config(name)).cuda()
    pred = model(torch.randn(2, 24, 3, device="cuda"))
    pred.square().mean().backward()
    assert pred.is_cuda and torch.isfinite(pred).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head_parameters())
