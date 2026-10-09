"""Single-author-backbone Frozen behavior and a bounded complete runner check."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl.baseline_run import METHOD_NAMES, run_baseline
from paper_impl.data import DataConfig, prepare_data
from paper_impl.official_backbones import OFFICIAL_MODEL_NAMES, OfficialForecastConfig, OfficialForecaster
from paper_impl.online_baselines import Frozen, OnlineBaselineConfig, ReleasedBatch, make_online_baseline


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_same_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual):
            assert_same_state(left, right)
    else:
        assert actual == expected


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", OFFICIAL_MODEL_NAMES)
def test_frozen_matches_official_model_and_ignores_released_label_values(name, device):
    torch.manual_seed(37)
    np.random.seed(37)
    cfg = OfficialForecastConfig(model_name=name, lookback=24, horizon=8, n_variables=3,
                                 target_indices=(2, 0), hidden=16, heads=8, depth=2, d_ff=32,
                                 dropout=.2, moving_avg=3, patch_len=4, patch_stride=2,
                                 crossformer_seg_len=4, fedformer_modes=4)
    # Match the inference-only parameter flags as well as eval mode: some
    # torch kernels select a different numeric fast path when gradients exist.
    source = OfficialForecaster(cfg).to(device).eval().requires_grad_(False)
    initial = copy.deepcopy(source.state_dict())
    online = OnlineBaselineConfig(24, 8, (2, 0), extra_delay=2, seed=43)
    first = make_online_baseline("Frozen", source, online)
    second = make_online_baseline("Frozen", source, online)
    assert isinstance(first, Frozen)
    assert not first.optimizers and not any(p.requires_grad for p in first.model.parameters())
    x = torch.randn(2, 24, 3).to(device)
    labels = torch.randn(2, 8, 2)
    for event, time in enumerate((0, 11)):
        if event:
            for method, y in ((first, labels), (second, labels * -1000 + 50000)):
                report = method.update(ReleasedBatch(x, y, (0, 1), (10, 11)), time=time,
                                       current_context=x)
                assert report.steps == 0 and report.loss is None and report.issue_times == (0, 1)
        # Informer uses random sparse sampling in eval; use the runner's event
        # seed to compare the same official inference operation exactly.
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(online.seed + event)
            expected = source(x)
        a, b = first.forecast(x, time), second.forecast(x, time)
        torch.testing.assert_close(a, expected, rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert_same_state(initial, first.model.state_dict())
        assert_same_state(initial, second.model.state_dict())
        assert_same_state(initial, source.state_dict())
    assert first.total_released_samples == second.total_released_samples == 2
    assert first.total_update_steps == second.total_update_steps == 0
    assert all(p.grad is None for p in first.model.parameters())
    assert all(not module.training for module in first.model.modules())


def test_frozen_rejects_early_feedback_without_changing_state():
    source = OfficialForecaster(OfficialForecastConfig(
        model_name="itransformer", lookback=12, horizon=4, n_variables=2,
        target_indices=(0,), hidden=8, heads=2, depth=1))
    method = make_online_baseline("Frozen", source, OnlineBaselineConfig(12, 4, (0,), extra_delay=2))
    initial = method.state_dict()
    with pytest.raises(ValueError, match="future/unreleased"):
        method.update(ReleasedBatch(torch.ones(1, 12, 2), torch.ones(1, 4, 1), (0,), (6,)), time=5)
    assert_same_state(initial, method.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_full_runner_trains_selects_scores_and_saves_without_online_updates(tmp_path, device):
    assert "Frozen" in METHOD_NAMES
    path = tmp_path / "series.csv"
    clock = np.arange(160)
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=160, freq="h"),
                          "load": np.sin(clock / 8) + clock / 100,
                          "temperature": np.cos(clock / 13), "mask": 1})
    frame.loc[130, ["load", "mask"]] = [np.nan, 0]
    frame.to_csv(path, index=False)
    config = {"seed": 23, "device": device, "cpu_threads": 1, "method": "Frozen",
              "data": {"path": str(path), "timestamp_column": "time", "target_columns": ["load"],
                       "feature_columns": ["load", "temperature"], "target_mask_columns": ["mask"],
                       "lookback": 12, "horizon": 4, "train_fraction": .6, "validation_fraction": .2},
              "model": {"model_name": "itransformer", "hidden": 8, "heads": 2, "depth": 1, "dropout": .2},
              "training": {"epochs": 2, "batch_size": 4, "max_train_windows": 8, "validation_windows": 4},
              "online": {"extra_delay": 2}, "test_windows": 10}
    result = run_baseline(config, tmp_path / "run")
    output = Path(result["output"])
    for filename in ("configuration.json", "data_metadata.json", "source_manifest.json", "training_log.json",
                     "metrics.json", "updates.json", "forecasts.npz", "protocol.json", "resource_counts.json",
                     "backbone_input_protocol.json", "offline_checkpoint.pt", "stream_checkpoint.pt", "run_record.json"):
        assert (output / filename).is_file()
    metrics = result["metrics"]
    assert metrics["completed"] and metrics["issued_forecasts"] == 10
    assert metrics["released_windows"] > 0 and metrics["missing_feedback_windows"] > 0
    assert metrics["updated_windows"] == metrics["optimizer_steps"] == 0
    offline = torch.load(output / "offline_checkpoint.pt", map_location="cpu", weights_only=True)
    final = torch.load(output / "stream_checkpoint.pt", map_location="cpu", weights_only=True)
    assert offline["secondary"] is None and offline["training_log"]["method"] is None
    assert_same_state(offline["primary"], offline["method"]["model"])
    assert_same_state(offline["primary"], final["stream_state"]["method"]["model"])
    assert final["stream_state"]["method"]["optimizers"] == {}
    log = offline["training_log"]["primary"]
    assert log["best_validation_mse"] == min(row["validation_mse"] for row in log["history"])
    assert 1 <= log["best_epoch"] <= 2 and log["used_train_windows"] == 8
    data = prepare_data(DataConfig(**offline["configuration"]["data"]))
    cfg = OfficialForecastConfig(**config["model"], lookback=12, horizon=4, n_variables=2, target_indices=(0,))
    with torch.random.fork_rng():
        torch.manual_seed(config["seed"])
        expected_model = OfficialForecaster(cfg)
    assert any(not torch.equal(parameter.cpu(), offline["primary"][name])
               for name, parameter in expected_model.named_parameters())
    expected_model.load_state_dict(offline["primary"])
    expected_model.to(device).eval().requires_grad_(False)
    forecasts = np.load(output / "forecasts.npz")
    with torch.no_grad():
        expected = torch.cat([expected_model(data.values[t - 11:t + 1].unsqueeze(0).to(device)).cpu()
                              for t in forecasts["issue_times"]]).numpy()
    np.testing.assert_array_equal(forecasts["predictions"], expected)
    mask = forecasts["target_mask"]
    error = forecasts["predictions"].astype(np.float64) - forecasts["targets"].astype(np.float64)
    assert metrics["normalized_mse"] == pytest.approx(np.square(error[mask]).mean(), rel=1e-12)
    updates = json.loads((output / "updates.json").read_text(encoding="utf-8"))
    assert updates and all(row["updated_windows"] == row["optimizer_steps"] == 0 for row in updates)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source_variant"] == "single_official_frozen"
    assert "stages_000" in protocol["predictor_scope"]
    assert protocol["feedback_usage"] == "release bookkeeping and scoring only"
    counts = json.loads((output / "resource_counts.json").read_text(encoding="utf-8"))
    assert counts["logical_parameter_count"] > 0 and counts["trainable_logical_parameter_count"] == 0
