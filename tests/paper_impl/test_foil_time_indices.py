"""Filtered-window time IDs survive FOIL training, shuffling and saved runs."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from paper_impl import foil, run as cli
from paper_impl.data import DataConfig, prepare_data
from paper_impl.model import ForecastBranches, ModelConfig


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def quick_config(**kwargs):
    return foil.FoilConfig(**{
        "n_environments": 2, "rounds": 1, "mil_epochs": 1, "mei_epochs": 1,
        "batch_size": 2, "smoothing_radius": 2, **kwargs,
    })


def test_inference_does_not_smooth_across_removed_time_positions_and_keeps_row_order():
    branch = nn.Module()
    branch.config = SimpleNamespace(horizon=3, targets=1)
    branch.encoder = nn.Identity()
    branch.head = nn.Linear(1, 3).double()
    heads = nn.ModuleList([nn.Linear(1, 3, bias=False).double() for _ in range(2)])
    pattern = torch.tensor([-1., 0., 1.], dtype=torch.double)
    with torch.no_grad():
        heads[0].weight[:, 0] = pattern
        heads[1].weight[:, 0] = -pattern
    labels = torch.tensor([0, 1, 1, 0, 0])
    ids = torch.tensor([0, 100, 101, 102, 103])
    x = torch.ones(5, 1, dtype=torch.double)
    y = torch.stack([pattern, -pattern])[labels].unsqueeze(-1)

    def infer(order, sample_ids):
        return foil.infer_environments(
            branch, heads, x[order], y[order], labels[order], quick_config(),
            generator=torch.Generator().manual_seed(73), sample_ids=sample_ids,
        )

    ordered = infer(torch.arange(5), ids)
    compacted = infer(torch.arange(5), None)
    assert ordered.labels.tolist() == [0, 1, 1, 0, 0]
    assert compacted.labels.tolist() == [1, 1, 0, 0, 0]
    assert ordered.repaired == compacted.repaired == 0
    # Permute all row-associated data, independently of the internal batch shuffle.
    order = torch.tensor([4, 1, 0, 3, 2])
    shuffled = infer(order, ids[order])
    assert torch.equal(shuffled.labels, ordered.labels[order])
    torch.testing.assert_close(shuffled.losses, ordered.losses[order])


@pytest.mark.parametrize("ids", [
    torch.arange(4), torch.arange(5).reshape(5, 1), torch.arange(5).float(),
    torch.ones(5, dtype=torch.bool), torch.tensor([0, 1, 1, 3, 4]),
    torch.tensor([-1, 0, 1, 2, 3]), [0, 1, 2, 3, 4],
])
def test_training_rejects_invalid_sample_id_mapping_before_optimization(ids):
    branches = ForecastBranches(ModelConfig(1, 3, 3, hidden=4, heads=1))
    x, y = torch.randn(5, 3, 1), torch.randn(5, 3, 1)
    before = deepcopy(branches.state_dict())
    with pytest.raises(ValueError, match="sample_ids"):
        foil.train_foil_branches(branches, x, y, x, y, quick_config(), sample_ids=ids)
    for name, value in branches.state_dict().items():
        assert torch.equal(value, before[name])


def test_omitted_ids_retain_contiguous_behavior_and_record_the_fallback():
    torch.manual_seed(54)
    branches = ForecastBranches(ModelConfig(1, 3, 3, hidden=4, heads=1))
    duplicate = deepcopy(branches)
    x, y = torch.randn(5, 3, 1), torch.randn(5, 3, 1)
    fallback = foil.train_foil_branches(branches, x, y, x, y, quick_config())
    supplied = foil.train_foil_branches(
        duplicate, x, y, x, y, quick_config(), sample_ids=torch.arange(5, dtype=torch.int32),
    )
    assert fallback["sample_id_source"] == "contiguous_row_fallback"
    assert supplied["sample_id_source"] == "caller_supplied"
    assert fallback["training_sample_ids"] == supplied["training_sample_ids"] == list(range(5))
    for name in ("invariant", "variant"):
        assert fallback[name] == supplied[name]
    for actual, expected in zip(branches.parameters(), duplicate.parameters()):
        assert torch.equal(actual, expected)


@pytest.fixture(scope="module")
def gapped_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("foil_original_time")
    path = directory / "series.csv"
    raw = np.sin(np.arange(60) / 3.) + np.arange(60) / 20.
    raw[12:19] = np.nan
    pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=60, freq="h"),
        "load": raw, "load_input": pd.Series(raw).ffill(),
        "observed": np.isfinite(raw).astype(int),
    }).to_csv(path, index=False)
    configuration = {
        "seed": 17, "device": "cpu", "cpu_threads": 1,
        "data": {"path": str(path), "timestamp_column": "time", "target_columns": ["load"],
                 "feature_columns": ["load_input"], "target_input_columns": {"load": "load_input"},
                 "target_mask_columns": ["observed"], "lookback": 3, "horizon": 3,
                 "train_fraction": .6, "validation_fraction": .2},
        "model": {"hidden": 4, "heads": 1, "n_experts": 2, "depth": 1, "latent_block": 2},
        "foil": {"n_environments": 2, "rounds": 1, "mil_epochs": 1, "mei_epochs": 1,
                 "batch_size": 4, "smoothing_radius": 2},
        "training": {"epochs": 1, "batch_size": 8, "specialization_weight": 0.},
        "generator": {"hidden_dim": 3, "mode": "identity"},
        "online": {"extra_delay": 1}, "variants": ["stages_000"], "test_windows": 2,
    }
    resolved = cli._normalize(configuration)
    data = prepare_data(DataConfig(**resolved["data"]))
    expected_ids = torch.as_tensor(data.train.issue_times)
    calls = []

    def observe(function, name):
        def wrapped(first, *args, **kwargs):
            sample_ids = args[0] if name == "smooth" else args[1]
            assert torch.equal(sample_ids, expected_ids)
            result = function(first, *args, **kwargs)
            calls.append(name)
            return result
        return wrapped

    output = directory / "trained"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(foil, "smooth_environment_labels", observe(foil.smooth_environment_labels, "smooth"))
        patch.setattr(foil, "repair_empty_environments", observe(foil.repair_empty_environments, "repair"))
        cli.run(configuration, output)
    return configuration, data, calls, output


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_passes_filtered_original_ids_through_both_branches_and_saves_mapping(gapped_run):
    configuration, data, calls, output = gapped_run
    assert calls == ["smooth", "repair", "smooth", "repair"]
    assert np.any(np.diff(data.train.issue_times) > configuration["foil"]["smoothing_radius"])
    assert torch.isnan(data.targets[12:19]).all()
    assert torch.isfinite(data.train.targets).all()
    assert not set(range(9, 18)) & set(data.train.issue_times.tolist())
    assert data.metadata["excluded_incomplete_supervised_horizons"]["train"] == 9
    history = read_json(output / "foil_history.json")
    assert history["training_sample_ids"] == data.train.issue_times.tolist()
    assert history["sample_id_source"] == "caller_supplied"
    assert history["smoothing_radius_unit"] == "sample_id_difference"
    assert history["environment_label_order"] == "training_sample_ids row order"
    for name in ("invariant", "variant"):
        assert len(history[name]["final_environment_labels"]) == len(data.train)
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    assert saved["history"]["foil"] == history
    assert saved["data_metadata"]["issue_times"]["train"] == history["training_sample_ids"]


def test_new_run_sampling_seed_inherits_global_seed_and_preserves_explicit_override(gapped_run):
    configuration, _, _, output = gapped_run
    assert read_json(output / "effective_config.json")["model"]["official_sampling_seed"] == 17
    assert read_json(output / "seed_plan.json")["official_sampling_root"] == 17
    changed = deepcopy(configuration)
    changed["model"]["official_sampling_seed"] = 93
    assert cli._normalize(changed)["model"]["official_sampling_seed"] == 93


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_run_rejects_invalid_official_sampling_seed(gapped_run, seed):
    configuration = deepcopy(gapped_run[0])
    configuration["model"]["official_sampling_seed"] = seed
    with pytest.raises(ValueError, match="official_sampling_seed"):
        cli._normalize(configuration)


@pytest.mark.parametrize("stored_seed", [None, 42, 7])
def test_legacy_checkpoint_reevaluation_keeps_saved_sampling_seed(gapped_run, tmp_path, stored_seed):
    original = gapped_run[3]
    saved = torch.load(original / "model.pt", map_location="cpu", weights_only=False)
    saved["format_version"] = 1
    saved["configuration"]["model"].pop("official_sampling_seed")
    if stored_seed is None:
        saved["model_config"].pop("official_sampling_seed")
    else:
        saved["model_config"]["official_sampling_seed"] = stored_seed
    checkpoint = tmp_path / "legacy.pt"
    torch.save(saved, checkpoint)
    output = tmp_path / "reevaluated"
    cli.run({}, output, checkpoint)
    expected_seed = 42 if stored_seed is None else stored_seed
    effective = read_json(output / "effective_config.json")
    assert effective["seed"] == 17
    assert effective["model"]["official_sampling_seed"] == expected_seed
    assert read_json(output / "seed_plan.json")["official_sampling_root"] == expected_seed
    # Native predictions have no sparse attention; this checks the complete
    # checkpoint path and retained outputs, not random-attention behavior.
    with np.load(original / "stages_000_forecasts.npz") as old, np.load(output / "stages_000_forecasts.npz") as new:
        for name in ("issue_times", "predictions", "targets"):
            np.testing.assert_array_equal(old[name], new[name])
