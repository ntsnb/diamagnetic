import numpy as np
import pandas as pd
import torch
import pytest
from paper_impl.data import DataConfig, prepare_data
from paper_impl.model import ModelConfig, PaperForecaster
from paper_impl.training import TrainingConfig, train_forecaster, evaluate_offline


def make_csv(path, count=90):
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=count, freq="h"),
                          "load": np.sin(np.arange(count) / 8) + np.arange(count) / count,
                          "temp": np.cos(np.arange(count) / 11)})
    frame.to_csv(path, index=False)
    return frame


def test_split_target_boundaries_and_training_only_statistics(tmp_path):
    path = tmp_path / "series.csv"
    frame = make_csv(path)
    config = DataConfig(str(path), "time", ["load"], lookback=4, horizon=2)
    original = prepare_data(config)
    train_end, val_end = original.metadata["train_end"], original.metadata["validation_end"]
    assert original.train.issue_times[-1] + 2 < train_end
    assert original.validation.issue_times[0] + 1 == train_end
    assert original.validation.issue_times[-1] + 2 < val_end
    assert original.test.issue_times[0] + 1 == val_end
    frame.loc[val_end:, "load"] += 1e6
    frame.to_csv(path, index=False)
    revised = prepare_data(config)
    assert original.metadata["mean"] == revised.metadata["mean"]
    torch.testing.assert_close(original.train.inputs, revised.train.inputs)
    torch.testing.assert_close(original.validation.targets, revised.validation.targets)


def test_missing_feature_no_future_interpolation(tmp_path):
    path = tmp_path / "series.csv"
    frame = make_csv(path)
    frame.loc[60, "temp"] = np.nan
    frame.loc[61, "temp"] = 10000
    frame.to_csv(path, index=False)
    data = prepare_data(DataConfig(str(path), "time", ["load"], lookback=4, horizon=2))
    assert data.values[60, 1] == data.values[59, 1]
    frame.loc[60, "load"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing forecast targets"):
        prepare_data(DataConfig(str(path), "time", ["load"], lookback=4, horizon=2))


def test_end_to_end_training_uses_actual_specialization(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    path = tmp_path / "series.csv"
    make_csv(path, 50)
    data = prepare_data(DataConfig(str(path), "time", ["load"], lookback=3, horizon=2))
    model = PaperForecaster(ModelConfig(2, 3, 2, n_experts=2, hidden=4, heads=1))
    original = model.experts[0].embedding.weight.detach().clone()
    history = train_forecaster(model, data.train.inputs[:8], data.train.targets[:8],
                               data.validation.inputs, data.validation.targets,
                               TrainingConfig(epochs=1, batch_size=4))
    assert history["history"][0]["specialization"] >= 0
    assert history["diagnostic"]["max_signed_conservation_error"] < 1e-4
    assert not torch.equal(original, model.experts[0].embedding.weight)


def test_clock_gaps_and_declared_sampling_interval(tmp_path):
    path = tmp_path / "series.csv"
    frame = make_csv(path)
    config = DataConfig(str(path), "time", ["load"], lookback=4, horizon=2,
                        expected_interval_seconds=3600)
    assert prepare_data(config).metadata["interval_seconds"] == 3600
    config.expected_interval_seconds = 600
    with pytest.raises(ValueError, match="declared interval"):
        prepare_data(config)
    config.expected_interval_seconds = 3600
    frame.drop(index=45).to_csv(path, index=False)
    with pytest.raises(ValueError, match="regular sampling"):
        prepare_data(config)


def test_constant_training_feature_is_recorded_without_future_fitting(tmp_path):
    path = tmp_path / "series.csv"
    frame = make_csv(path)
    frame.loc[:53, "temp"] = 0
    frame.loc[54:, "temp"] = 3
    frame.to_csv(path, index=False)
    data = prepare_data(DataConfig(str(path), "time", ["load"], lookback=4, horizon=2))
    assert data.metadata["unit_std_columns"] == ["temp"]
    assert data.metadata["original_training_std"][1] == 0
    assert data.metadata["std"][1] == 1
    assert data.values[54, 1] == 3


def test_nonfinite_validation_stops_before_checkpoint_selection():
    model = PaperForecaster(ModelConfig(2, 3, 2, hidden=4, heads=1))
    x = torch.randn(2, 3, 2)
    y = torch.full((2, 2, 1), float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite offline validation"):
        evaluate_offline(model, x, y)
