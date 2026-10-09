"""Baseline runner causality, missing observations and real checkpoint resumes."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from paper_impl.baseline_run import (DelayedBaselineStream, evaluate_baseline_stream,
                                     normalize_baseline_config, run_baseline, run_baseline_matrix)
from paper_impl.data import DataConfig, prepare_data
from paper_impl.online_baselines import OnlineBaselineConfig, ReleasedBatch, make_online_baseline


def make_data(path, *, missing=False):
    n = 160
    t = np.arange(n)
    load = np.sin(t / 9) + t / 90
    frame = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=n, freq="h"),
                          "load": load, "temperature": np.cos(t / 13), "mask": np.ones(n, dtype=int)})
    if missing:
        frame.loc[[130, 131, 134], "load"] = np.nan
        frame.loc[[130, 131, 134], "mask"] = 0
    frame.to_csv(path, index=False)
    return frame


def configuration(path, method="GD", device="cpu"):
    config = {"seed": 23, "device": device, "cpu_threads": 1,
              "data": {"path": str(path), "timestamp_column": "time", "target_columns": ["load"],
                       "feature_columns": ["load", "temperature"], "target_mask_columns": ["mask"],
                       "lookback": 12, "horizon": 4, "train_fraction": .6, "validation_fraction": .2},
              "model": {"model_name": "itransformer", "hidden": 8, "heads": 2, "depth": 1, "dropout": 0.},
              "method": method,
              "training": {"epochs": 1, "adapter_epochs": 1, "batch_size": 4,
                           "max_train_windows": 8, "validation_windows": 4},
              "online": {"learning_rate": .0001, "extra_delay": 2, "concept_dim": 4,
                         "bottleneck_dim": 2, "memory_size": 12, "selected_data_num": 2},
              "test_windows": 10}
    if method == "OneNet":
        config["fsnet"] = {"hidden_dims": 8, "output_dims": 16, "depth": 1}
    return config


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(24, 4)

    def forward(self, x):
        return self.head(x.flatten(1)).unsqueeze(-1)

    def head_parameters(self):
        return self.head.parameters()


def test_missing_target_windows_are_issued_but_not_used_for_supervision(tmp_path):
    path = tmp_path / "missing.csv"
    make_data(path, missing=True)
    cfg = configuration(path)
    data = prepare_data(DataConfig(**cfg["data"]))
    method = make_online_baseline("GD", Tiny(), OnlineBaselineConfig(12, 4, (0,), extra_delay=2))
    result = evaluate_baseline_stream(method, data, data.test, max_windows=10)
    assert result["metrics"]["issued_forecasts"] == 10
    assert result["metrics"]["missing_feedback_windows"] > 0
    assert result["metrics"]["updated_windows"] < 10
    mask = np.isfinite(result["targets"])
    expected = np.square(result["predictions"][mask] - result["targets"][mask]).mean()
    assert result["metrics"]["normalized_mse"] == pytest.approx(expected, rel=1e-6)
    assert result["metrics"]["valid_target_points"] == int(mask.sum())
    assert result["metrics"]["last_segment_peak_cuda_bytes"] == 0
    assert "peak_cuda_bytes" not in result["metrics"]


def test_no_test_target_slice_or_future_target_read_in_stream(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path)
    data = prepare_data(DataConfig(**cfg["data"]))
    first = int(data.test.issue_times[0])
    method = make_online_baseline("GD", Tiny(), OnlineBaselineConfig(12, 4, (0,), extra_delay=2))
    issue = first - 7
    method.seed_history(ReleasedBatch(data.values[issue - 11:issue + 1].unsqueeze(0),
                                     data.targets[issue + 1:issue + 5].unsqueeze(0), (issue,), (issue + 6,)),
                        time=first - 1)
    targets = data.targets
    seen = []

    class ObservedOnly:
        def __getitem__(self, key):
            assert isinstance(key, int), "live target slicing would reveal a future window"
            assert key == first + len(seen), "live target reads must be strictly sequential"
            seen.append(key)
            return targets[key]

    data.targets = ObservedOnly()
    result = evaluate_baseline_stream(method, data, data.test, max_windows=10, stop_after_observations=3)
    assert seen == [first, first + 1, first + 2]
    assert result["metrics"]["updated_windows"] == 0
    assert len(result["state"]["pending"]) == 3
    assert np.isnan(result["targets"][0, 2:]).all()


def test_unseen_future_labels_cannot_change_issued_predictions(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path)
    original = prepare_data(DataConfig(**cfg["data"]))
    altered = copy.deepcopy(original)
    first = int(original.test.issue_times[0])
    altered.targets[first + 3:] += 100000
    torch.manual_seed(11)
    model = Tiny()
    a = make_online_baseline("GD", model, OnlineBaselineConfig(12, 4, (0,), extra_delay=2))
    b = make_online_baseline("GD", model, OnlineBaselineConfig(12, 4, (0,), extra_delay=2))
    ra = evaluate_baseline_stream(a, original, original.test, max_windows=8, stop_after_observations=3)
    rb = evaluate_baseline_stream(b, altered, altered.test, max_windows=8, stop_after_observations=3)
    np.testing.assert_array_equal(ra["predictions"], rb["predictions"])


@pytest.mark.parametrize("name", ["GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED"])
def test_full_runner_and_same_stream_checkpoint_resume(name, tmp_path):
    path = tmp_path / "series.csv"
    make_data(path, missing=True)
    cfg = configuration(path, name)
    full = run_baseline(cfg, tmp_path / "full")
    partial = run_baseline(cfg, tmp_path / "partial", stop_after_observations=7)
    assert partial["metrics"]["completed"] is False
    resumed = run_baseline(None, tmp_path / "resumed", resume=partial["checkpoint"])
    assert resumed["metrics"]["completed"] is True
    full_npz, resumed_npz = np.load(Path(full["output"]) / "forecasts.npz"), np.load(Path(resumed["output"]) / "forecasts.npz")
    for key in ("issue_times", "predictions", "targets", "target_mask"):
        np.testing.assert_array_equal(full_npz[key], resumed_npz[key])
    for key in ("normalized_mse", "native_mae", "updated_windows", "missing_feedback_windows", "optimizer_steps"):
        assert full["metrics"][key] == resumed["metrics"][key]
    protocol = json.loads((Path(full["output"]) / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["diagnostic_only"] is False
    assert protocol["online_validation_updates"] is False
    assert protocol["cuda_memory_metric"]["name"] == "last_segment_peak_cuda_bytes"
    assert protocol["cuda_memory_metric"]["aggregated_across_resumes"] is False
    counts = json.loads((Path(full["output"]) / "resource_counts.json").read_text(encoding="utf-8"))
    assert counts["logical_parameter_count"] > 0
    if name in ("Head-GD", "SOLID"):
        assert counts["trainable_logical_parameter_count"] < counts["logical_parameter_count"]
    if name == "OneNet":
        assert "branches.secondary" in counts["components"] and "generator.ensemble_weight" in counts["components"]
    input_protocol = json.loads((Path(full["output"]) / "backbone_input_protocol.json").read_text(encoding="utf-8"))
    assert "None" in input_protocol["backbones"]["primary"]["separate_calendar_marks"]
    with pytest.raises(FileExistsError):
        run_baseline(cfg, tmp_path / "full")


def test_informer_sampling_rng_restores_exactly(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path)
    cfg["model"]["model_name"] = "informer"
    full = run_baseline(cfg, tmp_path / "full")
    partial = run_baseline(cfg, tmp_path / "part", stop_after_observations=6)
    resumed = run_baseline(None, tmp_path / "resume", resume=partial["checkpoint"])
    a, b = np.load(Path(full["output"]) / "forecasts.npz"), np.load(Path(resumed["output"]) / "forecasts.npz")
    np.testing.assert_array_equal(a["predictions"], b["predictions"])


@pytest.mark.parametrize("name,mode,reduction,variant", [
    ("SOLID", "eval", "sum", "solid_original_eval_sum"),
    ("SOLID", "train", "mean", "onlinetsf_train_mean_full_state_reset"),
    ("SOLID", "train", "sum", "explicit_solid_mode_reduction_variant"),
    ("SOLID++", "eval", "sum", "onlinetsf_continual_train_mean"),
])
def test_solid_mode_configuration_and_source_variant_are_recorded(name, mode, reduction, variant, tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path, name)
    assert normalize_baseline_config(cfg)["online"]["solid_adaptation_mode"] == "eval"
    cfg["online"].update(solid_adaptation_mode=mode, solid_gradient_reduction=reduction)
    cfg["model"]["dropout"] = .2
    result = run_baseline(cfg, tmp_path / "run")
    output = Path(result["output"])
    saved_config = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert saved_config["online"]["solid_adaptation_mode"] == mode
    assert protocol["source_variant"] == variant
    assert protocol["solid_adaptation_mode"] == ("train" if name == "SOLID++" else mode)
    assert protocol["solid_gradient_reduction"] == ("mean" if name == "SOLID++" else reduction)
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert state["stream_state"]["method"]["config"]["solid_adaptation_mode"] == mode


def test_resume_rejects_changed_source_data(tmp_path):
    path = tmp_path / "data.csv"
    frame = make_data(path)
    partial = run_baseline(configuration(path), tmp_path / "part", stop_after_observations=2)
    frame.loc[150, "load"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source data"):
        run_baseline(None, tmp_path / "invalid", resume=partial["checkpoint"])


def test_explicit_matrix_and_onenet_branch_validation(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path)
    results = run_baseline_matrix(cfg, tmp_path / "matrix", models=["itransformer"], methods=["GD", "Head-GD"], seeds=[23])
    assert len(results) == 2 and all(record["metrics"]["completed"] for record in results)
    cfg["method"] = "OneNet"
    assert normalize_baseline_config(cfg)["onenet_mode"] == "onlinetsf_cross_backbone"
    cfg["onenet_mode"] = "explicit_backbone_variant"
    with pytest.raises(ValueError, match="secondary_model"):
        normalize_baseline_config(cfg)
    cfg["secondary_model"] = copy.deepcopy(cfg["model"])
    with pytest.raises(ValueError, match="different"):
        normalize_baseline_config(cfg)


def test_original_author_two_fsnet_runner_resumes_exactly(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path, "OneNet")
    cfg["onenet_mode"] = "author_architecture"
    cfg["model"] = {"model_name": "fsnet_time"}
    full = run_baseline(cfg, tmp_path / "full")
    paused = run_baseline(cfg, tmp_path / "paused", stop_after_observations=7)
    resumed = run_baseline(None, tmp_path / "resumed", resume=paused["checkpoint"])
    a, b = np.load(Path(full["output"]) / "forecasts.npz"), np.load(Path(resumed["output"]) / "forecasts.npz")
    np.testing.assert_array_equal(a["predictions"], b["predictions"])
    protocol = json.loads((Path(full["output"]) / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["onenet_variant"] == "author_architecture"
    assert protocol["secondary_backbone"] == "fsnet_channel"


def test_data_and_online_delay_are_one_contract(tmp_path):
    cfg = configuration(tmp_path / "not_read.csv")
    normalized = normalize_baseline_config(cfg)
    assert normalized["data"]["extra_delay"] == normalized["online"]["extra_delay"] == 2
    cfg["data"]["extra_delay"] = 1
    with pytest.raises(ValueError, match="same protocol"):
        normalize_baseline_config(cfg)


def test_legacy_shifted_is_explicitly_diagnostic(tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    cfg = configuration(path)
    cfg["data"]["protocol"] = "legacy_shifted"
    cfg["data"]["online_test_fraction"] = .1
    result = run_baseline(cfg, tmp_path / "legacy")
    protocol = json.loads((Path(result["output"]) / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["diagnostic_only"] and protocol["name"] == "legacy_shifted"
    assert result["metrics"]["completed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", ["PROCEED", "OneNet"])
def test_complete_cuda_runner_with_exact_resume(name, tmp_path):
    path = tmp_path / "data.csv"
    make_data(path)
    # Exercise the same-process API after an unrelated CUDA GEMM has already
    # created cuBLAS handles; late environment initialization used to fail here.
    previous_cuda_model = torch.ones(2, 2, device="cuda")
    _ = previous_cuda_model @ previous_cuda_model
    cfg = configuration(path, name, "cuda")
    result = run_baseline(cfg, tmp_path / "cuda")
    assert result["metrics"]["completed"]
    assert result["metrics"]["last_segment_peak_cuda_bytes"] > 0
    assert "peak_cuda_bytes" not in result["metrics"]
    partial = run_baseline(cfg, tmp_path / "partial", stop_after_observations=7)
    resumed = run_baseline(None, tmp_path / "resumed", resume=partial["checkpoint"])
    a, b = np.load(Path(result["output"]) / "forecasts.npz"), np.load(Path(resumed["output"]) / "forecasts.npz")
    np.testing.assert_array_equal(a["predictions"], b["predictions"])
    manifest = json.loads((Path(result["output"]) / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["deterministic_algorithms"] and manifest["cudnn_deterministic"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_peak_is_last_stream_segment_not_maximum_across_resumes(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    make_data(path)
    data = prepare_data(DataConfig(**configuration(path)["data"]))
    model = Tiny().cuda()
    cfg = OnlineBaselineConfig(12, 4, (0,), extra_delay=2)
    first = make_online_baseline("GD", model, cfg)
    # Deterministic allocator readings isolate aggregation semantics from GPU
    # caching and model size; forward/backward and checkpoint resume are real.
    peaks = iter((900, 300))
    resets = []
    original_reset = torch.cuda.reset_peak_memory_stats
    def reset(device):
        resets.append(device)
        original_reset(device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: next(peaks))
    partial = evaluate_baseline_stream(first, data, data.test, max_windows=10, stop_after_observations=7)
    second = make_online_baseline("GD", model, cfg)
    resumed = evaluate_baseline_stream(second, data, data.test, max_windows=10, state=partial["state"])
    assert partial["metrics"]["last_segment_peak_cuda_bytes"] == 900
    assert resumed["metrics"]["last_segment_peak_cuda_bytes"] == 300
    assert resumed["metrics"]["completed"] and len(resets) == 2
    assert "peak_cuda_bytes" not in resumed["metrics"]


@pytest.mark.parametrize("fail", [False, True])
def test_runner_restores_callers_determinism_threads_and_rng_on_success_or_failure(fail, tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    make_data(path)
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                torch.get_num_threads())
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        if fail:
            def rejected(*args, **kwargs):
                raise RuntimeError("controlled preparation failure")
            monkeypatch.setattr("paper_impl.baseline_run.prepare_data", rejected)
            with pytest.raises(RuntimeError, match="controlled preparation failure"):
                run_baseline(configuration(path), tmp_path / "failed")
        else:
            run_baseline(configuration(path), tmp_path / "successful")
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
        assert torch.get_num_threads() == 2
        assert random.getstate() == python_rng
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        assert np.random.get_state()[2:] == numpy_rng[2:]
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        if cuda_rng is not None:
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
        torch.set_num_threads(previous[4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_package_configures_cublas_before_first_handle_without_external_env():
    # Start without a pytest/shell-provided workaround. Importing this package
    # must install the workspace policy before another model creates cuBLAS.
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    program = """
import os
import torch
assert not torch.cuda.is_initialized()
import paper_impl
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
assert not torch.cuda.is_initialized()
x = torch.randn(8, 8, device='cuda')
y = x @ x
from paper_impl.baseline_run import run_baseline
torch.use_deterministic_algorithms(True)
z = x @ x
assert torch.equal(y, z)
torch.cuda.synchronize()
"""
    result = subprocess.run([sys.executable, "-c", program], env=environment,
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
