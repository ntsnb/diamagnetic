"""真实小规模训练→保存→再评估，及拒绝不可复现的配置错配。"""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl import run as cli


def small_configuration(path):
    return {
        "seed": 17, "device": "cpu", "cpu_threads": 1,
        "data": {"path": str(path), "timestamp_column": "time", "target_columns": ["load"],
                 "feature_columns": ["load", "temperature"], "lookback": 3, "horizon": 2,
                 "train_fraction": .6, "validation_fraction": .2},
        "model": {"hidden": 4, "heads": 1, "n_experts": 2, "depth": 1, "latent_block": 2},
        "foil": {"n_environments": 2, "rounds": 1, "mil_epochs": 1, "mei_epochs": 1, "batch_size": 8},
        "training": {"epochs": 1, "batch_size": 8},
        "generator": {"hidden_dim": 3, "epsilon": .1},
        "online": {"reference_windows": 2, "reference_interval": 2, "history_capacity": 4, "extra_delay": 1},
        "replay": {"episodes": 1, "max_windows": 2}, "test_windows": 2,
        "variants": ["stages_000", "stages_111"],
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cli_roundtrip")
    path = directory / "series.csv"
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=36, freq="h"),
                  "load": np.sin(np.arange(36) / 3.) + np.arange(36) / 20.,
                  "temperature": np.cos(np.arange(36) / 5.)}).to_csv(path, index=False)
    configuration = small_configuration(path)
    output = directory / "trained"
    summary = cli.run(configuration, output)
    return {"configuration": configuration, "output": output, "checkpoint": output / "model.pt",
            "summary": summary, "path": path}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_complete_training_checkpoint_roundtrip_inherits_seeds_and_is_identical(trained, tmp_path):
    output = tmp_path / "reevaluated"
    cli.run({}, output, trained["checkpoint"])
    original, restored = read_json(trained["output"] / "effective_config.json"), read_json(output / "effective_config.json")
    assert original == restored
    assert original["seed"] == original["foil"]["seed"] == original["training"]["seed"] == 17
    assert read_json(trained["output"] / "foil_history.json")["config"]["seed"] == 17
    assert read_json(trained["output"] / "forecaster_history.json")["config"]["seed"] == 17
    assert original["online"]["error_threshold"] is not None
    for variant in original["variants"]:
        with np.load(trained["output"] / f"{variant}_forecasts.npz") as first, np.load(output / f"{variant}_forecasts.npz") as second:
            for key in ("issue_times", "predictions", "targets"):
                np.testing.assert_array_equal(first[key], second[key])
        assert read_json(trained["output"] / f"{variant}_updates.json") == read_json(output / f"{variant}_updates.json")
    assert {"models/attention_blocks.py", "models/__init__.py"}.issubset(read_json(output / "source_manifest.json")["code_sha256"])
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    assert saved["format_version"] == 2
    assert saved["generator_config"] == original["generator"]
    assert saved["online_config"] == original["online"]
    replay = read_json(trained["output"] / "replay_history.json")
    assert np.isfinite(replay["error_threshold"])
    assert replay["deployment_error_threshold"] == original["online"]["error_threshold"]
    assert "before generator replay" in replay["threshold_source"]


def test_evaluation_accepts_equal_partial_contract_and_explicit_run_settings(trained, tmp_path):
    request = {"model": {"hidden": 4}, "generator": {"epsilon": .1},
               "online": {"reference_windows": 2}, "variants": ["stages_111"], "test_windows": 1}
    summary = cli.run(request, tmp_path / "partial", trained["checkpoint"])
    assert len(summary["results"]) == 1 and summary["results"][0]["issued_forecasts"] == 1


@pytest.mark.parametrize("requested_configuration", [
    {"model": {"heads": 2}}, {"generator": {"epsilon": .2}}, {"generator": {"mode": "identity"}},
    {"online": {"radius_ii": .4}}, {"online": {"stage_ii": False}}, {"seed": 99},
    {"data": {"validation_fraction": .25}}, {"data": {"feature_columns": ["temperature", "load"]}},
    {"data": {"target_columns": ["temperature"]}}, {"training": {"seed": 1}},
])
def test_checkpoint_rejects_implicit_architecture_algorithm_seed_and_data_mismatch(trained, tmp_path, requested_configuration):
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        cli.run(requested_configuration, output, trained["checkpoint"])
    assert not output.exists()


@pytest.mark.parametrize("variants", [[], ["stages_111", "stages_111"], ["missing"], "stages_111"])
def test_invalid_variant_selection_is_rejected_before_training(trained, tmp_path, monkeypatch, variants):
    def forbidden(*args, **kwargs):
        raise AssertionError("training was reached")
    monkeypatch.setattr(cli, "train_foil_branches", forbidden)
    configuration = copy.deepcopy(trained["configuration"])
    configuration["variants"] = variants
    with pytest.raises(ValueError, match="[Vv]ariants"):
        cli.run(configuration, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_evaluation_data_relocation_requires_identical_content(trained, tmp_path):
    relocated = tmp_path / "relocated.csv"
    shutil.copyfile(trained["path"], relocated)
    cli.run({"data": {"path": str(relocated)}, "variants": ["stages_000"], "test_windows": 1},
            tmp_path / "moved", trained["checkpoint"])
    frame = pd.read_csv(relocated)
    frame.loc[len(frame) - 1, "load"] += 1
    frame.to_csv(relocated, index=False)
    with pytest.raises(ValueError, match="source_sha256"):
        cli.run({"data": {"path": str(relocated)}}, tmp_path / "changed", trained["checkpoint"])


@pytest.mark.parametrize("field", ["model", "generator", "online", "columns", "target_indices", "validation_end",
                                  "original_training_std", "unit_std_columns", "interval_seconds"])
def test_checkpoint_internal_contract_tampering_is_detected(trained, tmp_path, field):
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    if field == "model":
        saved["model_config"]["heads"] = 2
    elif field == "generator":
        saved["generator_config"] = {**saved["generator_config"], "epsilon": .2}
    elif field == "online":
        saved["online_config"] = {**saved["online_config"], "radius_i": .2}
    elif field == "columns":
        saved["data_metadata"]["columns"] = list(reversed(saved["data_metadata"]["columns"]))
    elif field == "target_indices":
        saved["data_metadata"]["target_indices"] = [1]
    elif field == "original_training_std":
        saved["data_metadata"]["original_training_std"][0] += .5
    elif field == "unit_std_columns":
        saved["data_metadata"]["unit_std_columns"] = ["load"]
    elif field == "interval_seconds":
        saved["data_metadata"]["interval_seconds"] *= 2
    else:
        saved["data_metadata"]["validation_end"] -= 1
    checkpoint = tmp_path / "tampered.pt"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint"):
        cli.run({}, tmp_path / "bad", checkpoint)


def test_declared_selector_and_identity_ablation_is_recorded(trained, tmp_path):
    request = {"evaluation_overrides": {"online": {"selector": "gradient_only"}, "generator": {"mode": "identity"}},
               "variants": ["stages_111"]}
    output = tmp_path / "ablation"
    cli.run(request, output, trained["checkpoint"])
    effective = read_json(output / "effective_config.json")
    assert effective["online"]["selector"] == "gradient_only"
    assert effective["generator"]["mode"] == "identity"
    assert read_json(output / "evaluation_overrides.json") == request["evaluation_overrides"]
    assert read_json(output / "checkpoint_config.json")["generator"]["mode"] == "centered_bounded"
    for event in read_json(output / "stages_111_updates.json"):
        assert event["experts"]["reason"].startswith("gradient_only")


@pytest.mark.parametrize("overrides", [
    {"online": {"error_threshold": 0}}, {"online": {"extra_delay": 0}},
    {"generator": {"epsilon": .2}}, {"model": {"heads": 2}}, {"data": {"train_fraction": .5}},
])
def test_evaluation_overrides_cannot_change_training_or_control_hyperparameters(trained, tmp_path, overrides):
    with pytest.raises(ValueError, match="Unknown"):
        cli.run({"evaluation_overrides": overrides}, tmp_path / "bad", trained["checkpoint"])


def test_seed_defaults_inherit_but_explicit_stage_seeds_are_recorded(trained):
    configuration = copy.deepcopy(trained["configuration"])
    configuration["foil"]["seed"], configuration["training"]["seed"] = 5, 6
    resolved = cli._normalize(configuration)
    assert (resolved["seed"], resolved["foil"]["seed"], resolved["training"]["seed"]) == (17, 5, 6)
    configuration["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        cli._normalize(configuration)


def test_equal_but_wrong_type_overrides_are_not_silently_accepted(trained, tmp_path):
    with pytest.raises(ValueError, match="heads"):
        cli.run({"model": {"heads": True}}, tmp_path / "bad", trained["checkpoint"])


def test_existing_output_and_existing_artifacts_are_never_overwritten(trained, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_bytes(b"preserve raw evidence")
    with pytest.raises(FileExistsError):
        cli.run({}, output, trained["checkpoint"])
    with pytest.raises(FileExistsError):
        cli.write_json(sentinel, {"new": "data"})
    assert sentinel.read_bytes() == b"preserve raw evidence"


def test_cli_checkpoint_evaluation_without_config(trained, tmp_path, monkeypatch):
    log = tmp_path / "experiment_log.md"
    monkeypatch.setattr(cli, "EXPERIMENT_LOG", log)
    summary = cli.main(["--checkpoint", str(trained["checkpoint"]), "--output", str(tmp_path / "cli")])
    assert summary["status"] == "completed"
    assert "completed" in log.read_text(encoding="utf-8")


def test_identity_training_is_an_explicit_complete_ablation(trained, tmp_path):
    configuration = copy.deepcopy(trained["configuration"])
    configuration["generator"]["mode"] = "identity"
    configuration["variants"] = ["stages_111"]
    configuration["test_windows"] = 1
    output = tmp_path / "identity"
    cli.run(configuration, output)
    assert read_json(output / "replay_history.json")["reason"] == "identity_generator_ablation"
    assert read_json(output / "effective_config.json")["generator"]["mode"] == "identity"
