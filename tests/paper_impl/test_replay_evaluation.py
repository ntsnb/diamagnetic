from dataclasses import replace
import torch
from paper_impl.model import ModelConfig, PaperForecaster, ForecastBranches
from paper_impl.adaptation import CenteredBoundedGenerator, OnlineConfig
from paper_impl.replay import ReplayConfig, train_generator
from paper_impl.evaluation import evaluate_stream, stage_variants, validation_error_threshold
from paper_impl.data import DataConfig, prepare_data
import pandas as pd
import numpy as np


def components(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    path = tmp_path / "signal.csv"
    pd.DataFrame({"t": pd.date_range("2024-01-01", periods=50, freq="h"),
                  "load": np.sin(np.arange(50) / 3), "weather": np.cos(np.arange(50) / 7)}).to_csv(path, index=False)
    data = prepare_data(DataConfig(str(path), "t", ["load"], lookback=3, horizon=2))
    config = ModelConfig(2, 3, 2, n_experts=2, hidden=4, heads=1)
    model = PaperForecaster(config)
    branches = ForecastBranches(config).freeze()
    gen = CenteredBoundedGenerator({k: tuple(p.shape) for k,p in model.expert_weights().items()}, 2, hidden_dim=4)
    online = OnlineConfig(reference_windows=2, reference_interval=2, history_capacity=8, extra_delay=1, error_threshold=0.)
    return data, model, branches, gen, online


def test_generator_replay_learns_and_does_not_change_initial_predictor(tmp_path):
    data, model, branches, gen, online = components(tmp_path)
    before = {k: p.clone() for k,p in model.state_dict().items()}
    history = train_generator(model, branches, gen, data.values, data.targets,
                              data.metadata["train_end"], online, ReplayConfig(episodes=2, max_windows=4))
    assert any(block.w2.weight.abs().sum() > 0 for block in gen.blocks.values())
    assert history["episodes"][0]["max_gradient_norm"] > 0
    assert history["episodes"][0]["events"][0]["time"] == 2 + 2 + 1
    assert history["episodes"][1]["events"][0] == history["episodes"][0]["events"][0]
    for key, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[key])


def test_all_eight_variants_have_same_targets(tmp_path):
    data, model, branches, gen, online = components(tmp_path)
    first = None
    for flags in stage_variants().values():
        cfg = replace(online, stage_i=flags[0], stage_ii=flags[1], stage_iii=flags[2])
        result = evaluate_stream(model, branches, gen, data, data.test, cfg, max_windows=2)
        if first is None:
            first = result["targets"]
        np.testing.assert_array_equal(first, result["targets"])
        assert result["metrics"]["issued_forecasts"] == 2


def test_validation_does_not_observe_test_prefix(tmp_path):
    data, model, branches, gen, online = components(tmp_path)
    config = replace(online, extra_delay=4)
    first = evaluate_stream(model, branches, gen, data, data.validation, config)
    data.values[data.metadata["validation_end"]:] += 1000
    data.targets = data.values[:, :1]
    second = evaluate_stream(model, branches, gen, data, data.validation, config)
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert first["updates"] == second["updates"]
    assert first["metrics"]["unreleased_forecasts"] > 0


def test_threshold_computed_on_complete_forecaster(tmp_path):
    data, model, branches, gen, online = components(tmp_path)
    value = validation_error_threshold(model, data.validation.inputs, data.validation.targets,
                                       generator=gen, branches=branches)
    assert value >= 0 and np.isfinite(value)


def test_nonconsecutive_issue_times_do_not_insert_extra_forecasts(tmp_path):
    from paper_impl.data import WindowSplit
    data, model, branches, gen, online = components(tmp_path)
    indices = np.array([0, 2, 4])
    split = WindowSplit(data.test.inputs[indices], data.test.targets[indices], data.test.issue_times[indices])
    result = evaluate_stream(model, branches, gen, data, split, replace(online, stage_i=False, stage_ii=False, stage_iii=False))
    np.testing.assert_array_equal(result["issue_times"], split.issue_times)
    np.testing.assert_array_equal(result["targets"], split.targets.numpy())
    assert result["metrics"]["issued_forecasts"] == 3
