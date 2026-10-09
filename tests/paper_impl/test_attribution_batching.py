"""归因分块只改变高阶计算图驻留量，不改变固定专家对目标或单步优化。"""
import copy
import json
import math
import os
from pathlib import Path
import time

import pytest
import torch
from torch.nn import functional as F

from paper_impl.discrepancy import specialization_loss
from paper_impl.model import ModelConfig, PaperForecaster
from paper_impl.training import TrainingConfig, seed_everything, train_forecaster


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(backbone):
    """固定相同初始化；两个目标、三个专家确保全部专家对与目标都参与均值。"""
    torch.manual_seed(271)
    model = PaperForecaster(ModelConfig(
        n_variables=2, lookback=4, horizon=2, targets=2, target_indices=(0, 1),
        n_experts=3, hidden=4, heads=1, depth=1, latent_block=2,
        use_vae=True, use_revin=backbone == "native", expert_backbone=backbone,
        official_options={"dropout": 0.0} if backbone != "native" else None,
    )).double()
    assert all(p.dtype == torch.float64 for p in model.parameters())
    return model


def fixed_samples(count=7):
    generator = torch.Generator().manual_seed(314)
    x = torch.randn(count, 4, 2, generator=generator, dtype=torch.float64) + .4
    y = torch.randn(count, 2, 2, generator=generator, dtype=torch.float64)
    return x, y


def gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()}


def assert_gradients_close(actual, expected, *, rtol=1e-9, atol=1e-10):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert (actual[name] is None) == (expected[name] is None), name
        if actual[name] is not None:
            assert torch.isfinite(actual[name]).all(), name
            torch.testing.assert_close(actual[name], expected[name], rtol=rtol, atol=atol,
                                       msg=lambda message: f"{name}: {message}")


def chunked_spec_backward(model, x, chunk_size, margin=.95):
    """每块立即反传，数值汇总使用 detach，不能把旧图保留至批末。"""
    model.zero_grad(set_to_none=True)
    total = x.new_zeros(())
    for part in x.split(chunk_size):
        attribution = model.attribute(part, freeze_head=True)
        value = specialization_loss(attribution.maps, margin) * (len(part) / len(x))
        value.backward()
        total += value.detach()
        del attribution, value
    return total, gradients(model)


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_fixed_pair_mean_is_sample_additive_even_with_uneven_zero_map_pairs(chunk_size):
    generator = torch.Generator().manual_seed(97)
    maps = torch.rand(5, 2, 3, 2, 4, generator=generator, dtype=torch.float64) + .2
    # 不同样本的有效专家对数不同；零地图对仍占固定分母，不能按有效对重算权重。
    maps[0, 0, 0] = 0
    maps[1, 1, 2] = 0
    maps[4, :, 1] = 0
    whole = maps.clone().requires_grad_()
    reference = specialization_loss(whole, .95)
    reference.backward()
    split = maps.clone().requires_grad_()
    total = split.new_zeros(())
    for part in split.split(chunk_size):
        loss = specialization_loss(part, .95) * (len(part) / len(split))
        loss.backward()
        total += loss.detach()
    torch.testing.assert_close(total, reference.detach(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(split.grad, whole.grad, rtol=1e-13, atol=1e-14)
    assert whole.grad.abs().sum() > 0


@pytest.mark.parametrize("backbone", ["native", "transformer"])
def test_parameter_spec_gradients_match_chunks_one_two_and_whole_double(backbone):
    initial = tiny_model(backbone).train()
    x, _ = fixed_samples(5)
    reference_value, reference_grad = chunked_spec_backward(copy.deepcopy(initial), x, len(x))
    assert reference_value > 0
    assert any(value is not None and value.abs().sum() > 0
               for name, value in reference_grad.items() if name.startswith("experts."))
    for chunk_size in (1, 2):  # 5 = 2 + 2 + 1，显式覆盖不足整块的尾部。
        value, actual_grad = chunked_spec_backward(copy.deepcopy(initial), x, chunk_size)
        torch.testing.assert_close(value, reference_value, rtol=1e-10, atol=1e-12)
        assert_gradients_close(actual_grad, reference_grad)


def whole_batch_epoch(model, x, y, config):
    """原整批目标的独立基准：全部损失求和，一次 backward/clip/Adam step。"""
    seed_everything(config.seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    per_step_gradients, norms, component_totals = [], [], dict(prediction=0., reconstruction=0., kl=0., specialization=0., loss=0.)
    for ids in torch.randperm(len(x), generator=generator).split(config.batch_size):
        xb, yb = x[ids], y[ids]
        optimizer.zero_grad(set_to_none=True)
        output = model(xb, stochastic=True)
        prediction = F.mse_loss(output.prediction, yb)
        reconstruction = F.mse_loss(output.reconstruction, xb)
        kl = -.5 * (1 + output.log_var - output.mu.square() - output.log_var.exp()).sum(-1).mean()
        spec = specialization_loss(model.attribute(xb, freeze_head=True).maps, config.margin)
        loss = prediction + config.vae_weight * (reconstruction + config.kl_weight * kl) + config.specialization_weight * spec
        loss.backward()
        per_step_gradients.append(gradients(model))
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)))
        optimizer.step()
        for name, value in zip(component_totals, (prediction, reconstruction, kl, spec, loss)):
            component_totals[name] += float(value.detach()) * len(ids) / len(x)
    return per_step_gradients, norms, component_totals


@pytest.mark.parametrize("backbone", ["native", "transformer"])
def test_single_epoch_matches_original_combined_loss_and_single_clip_step(backbone, monkeypatch):
    initial = tiny_model(backbone)
    x, y = fixed_samples()
    settings = dict(epochs=1, batch_size=5, learning_rate=.002, vae_weight=.07,
                    kl_weight=.09, specialization_weight=.23, margin=.95, grad_clip=.02, seed=519)
    reference_model = copy.deepcopy(initial)
    reference_grad, reference_norms, reference_totals = whole_batch_epoch(reference_model, x, y, TrainingConfig(**settings))
    assert any(norm > settings["grad_clip"] for norm in reference_norms), "combined-gradient clipping must be exercised"
    original_clip = torch.nn.utils.clip_grad_norm_
    original_adam = torch.optim.Adam
    for chunk_size in (1, 2, 5):
        candidate = copy.deepcopy(initial)
        observed_gradients, step_count = [], []

        class CountedAdam(original_adam):
            def step(self, *args, **kwargs):
                step_count.append(True)
                return super().step(*args, **kwargs)

        def record_clip(parameters, *args, **kwargs):
            observed_gradients.append(gradients(candidate))
            return original_clip(parameters, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(torch.optim, "Adam", CountedAdam)
            context.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
            result = train_forecaster(candidate, x, y, x[:3], y[:3],
                                      TrainingConfig(**settings, attribution_batch_size=chunk_size))
        assert len(step_count) == len(observed_gradients) == math.ceil(len(x) / settings["batch_size"]) == 2
        for actual_grad, expected_grad in zip(observed_gradients, reference_grad):
            assert_gradients_close(actual_grad, expected_grad)
        for (name, actual), (expected_name, expected) in zip(candidate.named_parameters(), reference_model.named_parameters()):
            assert name == expected_name
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10,
                                       msg=lambda message: f"chunk={chunk_size}, {name}: {message}")
        for name, expected in reference_totals.items():
            assert result["history"][0][name] == pytest.approx(expected, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("bad_size", [0, -1, 1.5, True])
def test_invalid_attribution_batch_size_is_rejected(bad_size):
    with pytest.raises((TypeError, ValueError)):
        TrainingConfig(attribution_batch_size=bad_size)


def profile_single_sample(backbone, device):
    """默认矩阵尺寸的一次归因和分工反传；只诊断运行成本，不训练或评估精度。"""
    seed_everything(20261007)
    model = PaperForecaster(ModelConfig(
        n_variables=8, lookback=24, horizon=12, targets=1, target_indices=(0,),
        n_experts=3, hidden=64, heads=8, depth=2, latent_block=8,
        expert_backbone=backbone, use_revin=backbone == "native",
        official_options={"dropout": 0.0} if backbone != "native" else None,
    )).to(device).train()
    x = torch.randn(1, 24, 8, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        initial_bytes = torch.cuda.memory_allocated(device)
    else:
        initial_bytes = None
    start = time.perf_counter()
    attribution = model.attribute(x, freeze_head=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attribute_seconds = time.perf_counter() - start
    loss = specialization_loss(attribution.maps, .3)
    backward_start = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_start
    nonzero_gradients = sum(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                            for parameter in model.parameters())
    result = {"backbone": backbone, "device": str(device), "dtype": str(x.dtype),
              "shape": {"batch": 1, "lookback": 24, "horizon": 12, "variables": 8,
                        "targets": 1, "experts": 3, "hidden": 64, "heads": 8, "depth": 2},
              "attribute_seconds": attribute_seconds, "spec_backward_seconds": backward_seconds,
              "total_seconds": attribute_seconds + backward_seconds, "specialization_loss": float(loss.detach()),
              "initial_allocated_bytes": initial_bytes,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
              "nonzero_parameter_gradient_count": nonzero_gradients,
              "finite_maps": bool(torch.isfinite(attribution.maps).all()),
              "finite_gradients": all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()),
              "max_signed_conservation_error": float(attribution.conservation_error.detach().max())}
    if initial_bytes is not None:
        result["incremental_peak_allocated_bytes"] = result["peak_allocated_bytes"] - initial_bytes
    return result


@pytest.mark.skipif(os.environ.get("ATTRIBUTION_PROFILE") != "1", reason="explicit bounded cost profile only")
@pytest.mark.parametrize("backbone", ["native", "transformer"])
def test_default_matrix_single_sample_profile(backbone):
    device = torch.device(os.environ.get("ATTRIBUTION_PROFILE_DEVICE", "cpu"))
    if device.type == "cuda":
        progress = Path(__file__).resolve().parents[2] / "results/full_experiment/20261007_full_framework_cuda_v2/progress.json"
        records = json.loads(progress.read_text(encoding="utf8"))
        assert len(records) == 8 and all(item["status"] == "passed" for item in records), "wait for the shared CUDA smoke run"
    report = profile_single_sample(backbone, device)
    print("ATTRIBUTION_PROFILE " + json.dumps(report, sort_keys=True))
    assert report["finite_maps"] and report["finite_gradients"]
    assert report["nonzero_parameter_gradient_count"] > 0
