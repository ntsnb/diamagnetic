import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl.data import DataConfig, prepare_data
from paper_impl.model import ModelConfig, PaperForecaster, ForecastBranches
from paper_impl.adaptation import OnlineConfig, OnlineController, CenteredBoundedGenerator
from paper_impl.stream import DelayedFeedbackStream
from paper_impl.evaluation import evaluate_stream


def dataset(tmp_path, *, delay=2, protocol="chronological"):
    target = np.arange(100, dtype=float)
    target[[15, 54, 79]] = np.nan
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="h"),
                          "load": target, "load_input": pd.Series(target).ffill(),
                          "observed": np.isfinite(target).astype(int)})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    config = DataConfig(str(path), "timestamp", ["load"], lookback=3, horizon=2,
                        feature_columns=["load_input"], target_input_columns={"load": "load_input"},
                        target_mask_columns=["observed"], extra_delay=delay, protocol=protocol,
                        train_fraction=.5, validation_fraction=.2)
    return prepare_data(config)


def test_masked_truth_never_becomes_imputed_label_and_test_windows_are_not_filtered(tmp_path):
    data = dataset(tmp_path)
    assert torch.isfinite(data.values).all()
    assert torch.isnan(data.targets[[15, 54, 79]]).all()
    assert torch.isfinite(data.train.targets).all() and torch.isfinite(data.validation.targets).all()
    assert len(data.test) == 29
    assert 77 in data.test.issue_times and 78 in data.test.issue_times
    assert torch.isnan(data.test.targets).sum() == 2
    for name, cutoff in (("train", "offline_training_cutoff"), ("validation", "offline_validation_cutoff")):
        split = getattr(data, name)
        assert np.all(split.issue_times + 2 + 2 <= data.metadata[cutoff])
    assert data.metadata["excluded_unreleased_supervised_horizons"] == {"train": 2, "validation": 2}


def test_missing_label_horizon_is_retired_only_at_release_time():
    torch.manual_seed(2)
    model = PaperForecaster(ModelConfig(1, 3, 2, n_experts=1, hidden=8, heads=2, use_vae=False))
    controller = OnlineController(model, OnlineConfig(stage_i=False, stage_ii=False, stage_iii=False, extra_delay=1))
    stream = DelayedFeedbackStream(controller, allow_missing=True)
    x = torch.randn(1, 3, 1)
    forecast, _ = stream.step(2, x, torch.tensor([2.]))
    stream.observe(3, torch.tensor([float("nan")]))
    stream.observe(4, torch.tensor([4.]))
    assert len(controller.pending_records) == 1 and not stream.skipped_forecasts
    stream.observe(5, torch.tensor([5.]))
    assert not controller.pending_records
    assert stream.skipped_forecasts == [{"issue_time": 2, "time": 5, "reason": "incomplete_observed_horizon"}]
    assert controller.update_occasions == 0
    assert torch.isfinite(forecast.prediction).all()


def test_evaluation_scores_observed_points_without_changing_issued_windows(tmp_path):
    data = dataset(tmp_path, delay=0)
    model = PaperForecaster(ModelConfig(1, 3, 2, n_experts=1, hidden=8, heads=2))
    branches = ForecastBranches(model.config, mode="disabled")
    generator = CenteredBoundedGenerator({k: tuple(v.shape) for k, v in model.expert_weights().items()}, 2, mode="identity")
    result = evaluate_stream(model, branches, generator, data, data.test,
                            OnlineConfig(stage_i=False, stage_ii=False, stage_iii=False))
    metrics = result["metrics"]
    assert metrics["issued_forecasts"] == len(data.test) == 29
    assert metrics["missing_target_values"] == metrics["skipped_incomplete_label_updates"] == 2
    assert metrics["unreleased_forecasts"] == 0
    assert metrics["observed_target_values"] == 56
    assert np.isfinite(metrics["normalized_mse"])


def test_legacy_selection_keeps_original_clock_and_marks_retrospective_selection(tmp_path):
    data = dataset(tmp_path, protocol="legacy_shifted")
    assert np.all(np.diff(data.test.issue_times) == 1)
    assert data.metadata["selection"]["future_inputs_for_selection"] is True
    for name, cutoff in (("train", "offline_training_cutoff"), ("validation", "offline_validation_cutoff")):
        assert np.all(getattr(data, name).issue_times + 4 <= data.metadata[cutoff])
    torch.testing.assert_close(data.test.inputs[0], data.values[data.test.issue_times[0] - 2:data.test.issue_times[0] + 1])


def test_target_mask_cannot_be_an_input_feature(tmp_path):
    data = dataset(tmp_path)
    config = DataConfig(**data.metadata["config"])
    config.feature_columns.append("observed")
    with pytest.raises(ValueError, match="evaluation masks"):
        prepare_data(config)
