"""Auditable counts, complex tensors, shared parameters and CPU-only plans."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from paper_impl.resource_report import framework_parameter_counts, isolated_cpu_initialization
from tools.report_full_experiment_resources import dataset_schema, report_matrix


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.real = nn.Parameter(torch.zeros(2, 3))
        self.complex = nn.Parameter(torch.zeros(4, dtype=torch.complex64))
        self.frozen = nn.Parameter(torch.zeros(5), requires_grad=False)
        self.register_buffer("memory", torch.zeros(7, dtype=torch.float64))


def test_complex_real_scalar_and_frozen_parameter_counts():
    model = Mixed()
    report = framework_parameter_counts(model, include_parameter_details=True)
    assert report["logical_parameter_count"] == 15
    assert report["real_scalar_count"] == 19
    assert report["trainable_logical_parameter_count"] == 10
    assert report["trainable_real_scalar_count"] == 14
    assert report["parameter_bytes"] == 6 * 4 + 4 * 8 + 5 * 4
    assert report["buffer_bytes"] == 7 * 8
    assert len(report["components"]["model"]["parameters"]) == 3
    json.dumps(report)


def test_shared_parameters_are_globally_deduplicated_and_contributions_sum():
    model = Mixed()
    branch = nn.Module()
    branch.shared = model.real
    branch.extra = nn.Parameter(torch.ones(2))
    report = framework_parameter_counts(model, {"first": branch, "alias": branch}, {"same": model.real})
    assert report["logical_parameter_count"] == 17
    assert report["components"]["branches.first"]["inclusive"]["logical_parameter_count"] == 8
    assert report["components"]["branches.first"]["exclusive_contribution"]["logical_parameter_count"] == 2
    assert report["components"]["branches.alias"]["exclusive_contribution"]["logical_parameter_count"] == 0
    for key in ("logical_parameter_count", "real_scalar_count", "trainable_logical_parameter_count", "parameter_bytes"):
        assert sum(component["exclusive_contribution"][key] for component in report["components"].values()) == report[key]


def test_distinct_parameters_sharing_storage_remain_distinct_logical_parameters():
    first = nn.Parameter(torch.ones(3))
    second = nn.Parameter(first.detach())
    assert first.data_ptr() == second.data_ptr()
    report = framework_parameter_counts(first, second)
    assert report["logical_parameter_count"] == 6 and report["parameter_tensor_count"] == 2


def test_reporting_does_not_change_parameters_gradients_rng_or_train_mode():
    model = Mixed().eval()
    model.real.grad = torch.ones_like(model.real)
    state = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    framework_parameter_counts(model)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    assert torch.equal(model.real.grad, torch.ones_like(model.real))
    assert torch.equal(rng, torch.random.get_rng_state()) and not model.training


def test_lazy_parameters_must_not_be_reported_as_zero():
    with pytest.raises(ValueError, match="materialize"):
        framework_parameter_counts(nn.LazyLinear(3))


def test_cpu_initialization_restores_all_rng_even_after_exception():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    cuda_was_initialized = torch.cuda.is_initialized()
    device = torch.get_default_device()
    with pytest.raises(RuntimeError, match="probe"):
        with isolated_cpu_initialization(99):
            random.random()
            np.random.randn(10)
            torch.randn(10)
            assert nn.Linear(3, 2).weight.device.type == "cpu"
            raise RuntimeError("probe")
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert torch.get_default_device() == device and torch.cuda.is_initialized() == cuda_was_initialized


def tiny_matrix(tmp_path):
    csv_path = tmp_path / "source.csv"
    # Explicit registry schema means the invalid data row need never be parsed.
    csv_path.write_text("time,load,temp,load_mask\nnot,a,numeric,row\n", encoding="utf-8")
    data = {"path": str(csv_path), "timestamp_column": "time", "target_columns": ["load"],
            "feature_columns": ["load", "temp"], "target_mask_columns": ["load_mask"], "lookback": 8, "horizon": 2}
    configuration = {"seed": 42, "data": data,
        "model": {"hidden": 8, "heads": 2, "depth": 1, "latent_block": 3, "n_experts": 2},
        "foil": {"mode": "foil"}, "generator": {"hidden_dim": 3}}
    jobs = []
    for name in ("full", "noSpec", "noVAE"):
        cfg = copy.deepcopy(configuration)
        if name == "noSpec":
            cfg["training"] = {"specialization_weight": 0.}
        if name == "noVAE":
            cfg["model"]["use_vae"] = False
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(cfg), encoding="utf-8")
        jobs.append({"dataset": "demo", "backbone": "native", "ablation": name, "seed": 42, "config": path.name})
    matrix = {"datasets": [{"id": "demo", "input_data": data}], "jobs": jobs}
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    return path


def test_matrix_caches_equal_structures_without_data_scan_or_forward(tmp_path, monkeypatch):
    from paper_impl.model import PaperForecaster
    matrix = tiny_matrix(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("counting must not read rows or call model.forward")
    monkeypatch.setattr("pandas.read_csv", forbidden)
    monkeypatch.setattr(PaperForecaster, "forward", forbidden)
    before = torch.random.get_rng_state().clone()
    report = report_matrix(matrix, tmp_path / "report")
    assert report["framework_jobs_counted"] == 3 and report["unique_structures_instantiated"] == 2
    assert torch.equal(before, torch.random.get_rng_state())
    saved = json.loads((tmp_path / "report/resource_counts.json").read_text(encoding="utf-8"))
    assert saved["forward_calls"] == saved["training_steps"] == 0
    assert saved["framework_jobs"][0]["logical_parameter_count"] == saved["framework_jobs"][1]["logical_parameter_count"]
    assert saved["framework_jobs"][0]["logical_parameter_count"] != saved["framework_jobs"][2]["logical_parameter_count"]
    with pytest.raises(FileExistsError):
        report_matrix(matrix, tmp_path / "report")


def test_unspecified_numeric_features_require_opt_in_and_match_dtype_logic(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("time,load,label,load_mask\n2020,1,hot,1\n2021,2,cold,1\n", encoding="utf-8")
    dataset = {"id": "demo", "input_data": {"path": str(path), "timestamp_column": "time",
        "target_columns": ["load"], "feature_columns": None, "target_mask_columns": ["load_mask"]}}
    with pytest.raises(ValueError, match="dtype-scan"):
        dataset_schema(dataset)
    result = dataset_schema(dataset, allow_dtype_scan=True)
    assert result["columns"] == ["load"] and result["target_indices"] == [0]
    assert result["dtypes"]["label"] == "object"
