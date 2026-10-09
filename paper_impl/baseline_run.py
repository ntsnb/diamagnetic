"""Complete-author-model baseline training and causal delayed-feedback runner.

Examples::

    python -m paper_impl.baseline_run --config CONFIG.json --out NEW_DIRECTORY
    python -m paper_impl.baseline_run --resume RUN/stream_checkpoint.pt --out NEW_DIRECTORY
    python -m paper_impl.baseline_run --config CONFIG.json --out NEW_MATRIX --matrix \
        --models itransformer patchtst --methods GD PROCEED --seeds 7 2026

The stream loop accesses test targets exclusively as ``data.targets[t]`` at the
current observation time. It never selects forecast origins using future masks.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import DataConfig, PreparedData, WindowSplit, prepare_data
from .official_backbones import OfficialForecastConfig, OfficialForecaster, OFFICIAL_MODEL_NAMES
from .official_fsnet import OfficialFSNet, OfficialFSNetConfig, author_onenet_branches
from .resource_report import framework_parameter_counts
from .online_baselines import (OnlineBaselineConfig, ReleasedBatch, SOLID, PROCEED, OneNet,
                               make_online_baseline, _cpu_clone)


ROOT = Path(__file__).resolve().parents[1]
METHOD_NAMES = ("Frozen", "GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED")


@dataclass(frozen=True)
class BaselineTrainingConfig:
    epochs: int = 10
    adapter_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    adapter_learning_rate: float = 1e-3
    patience: int = 3
    max_train_windows: int | None = None
    validation_windows: int | None = None
    max_grad_norm: float | None = 1.0

    def __post_init__(self):
        for name in ("epochs", "adapter_epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        for name in ("max_train_windows", "validation_windows"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"training.{name} must be null or a positive integer")
        for name in ("learning_rate", "adapter_learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"training.{name} must be finite and positive")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def baseline_source_manifest():
    paths = [ROOT / "paper_impl" / name for name in
             ("__init__.py", "baseline_run.py", "online_baselines.py", "official_backbones.py", "official_fsnet.py", "resource_report.py", "data.py")]
    paths.extend(ROOT / "third_party" / name for name in
                 ("official_online/SOURCE_MANIFEST.json", "official_forecasters/SOURCE_MANIFEST.json"))
    # Include actual vendored implementation bytes; the manifest filename of a
    # forecaster bundle is allowed to evolve without hiding code provenance.
    paths.extend((ROOT / "third_party/official_forecasters").rglob("*.py"))
    for source in ("OnlineTSF/layers/ts2vec", "OneNet/models/ts2vec"):
        paths.extend(ROOT / "third_party/official_online" / source / filename for filename in ("fsnet.py", "fsnet_.py"))
    return {"files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(set(paths)) if path.exists()},
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "prediction_rng": "isolated per-event torch+numpy seed=config.seed+saved_rng_counter"}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_baseline_config(configuration: Mapping):
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration must be an object")
    config = copy.deepcopy(dict(configuration))
    allowed = {"seed", "device", "cpu_threads", "deterministic", "data", "model", "secondary_model", "onenet_secondary_models", "onenet_mode", "fsnet", "method",
               "online", "training", "test_windows"}
    if set(config) - allowed:
        raise ValueError(f"unknown baseline configuration fields: {sorted(set(config) - allowed)}")
    config.setdefault("seed", 2026)
    config.setdefault("cpu_threads", 1)
    config.setdefault("device", "cpu")
    config.setdefault("deterministic", True)
    if not isinstance(config["deterministic"], bool):
        raise ValueError("deterministic must be boolean")
    config.setdefault("method", "GD")
    config.setdefault("test_windows", None)
    for name, minimum in (("seed", 0), ("cpu_threads", 1)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["method"] not in METHOD_NAMES:
        raise ValueError(f"method must be one of {METHOD_NAMES}")
    if config["test_windows"] is not None and (isinstance(config["test_windows"], bool)
            or not isinstance(config["test_windows"], int) or config["test_windows"] < 1):
        raise ValueError("test_windows must be a positive integer or null")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = torch.device(config["device"])
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config.setdefault("online", {})
    config["online"].setdefault("solid_adaptation_mode", "eval")
    config["online"].setdefault("solid_gradient_reduction", "sum")
    data_values = dict(config["data"])
    delay = config["online"].get("extra_delay", data_values.get("extra_delay", 0))
    if "extra_delay" in data_values and data_values["extra_delay"] != delay:
        raise ValueError("data.extra_delay and online.extra_delay must describe the same protocol")
    data_values["extra_delay"] = delay
    config["online"]["extra_delay"] = delay
    data_config = DataConfig(**data_values)
    data_config.path = str(Path(data_config.path).resolve())
    config["data"] = asdict(data_config)
    config["training"] = asdict(BaselineTrainingConfig(**config.get("training", {})))
    config.setdefault("model", {"model_name": "itransformer"})
    config.setdefault("secondary_model", None)
    config.setdefault("onenet_mode", "explicit_backbone_variant" if config["secondary_model"] is not None
                      else "onlinetsf_cross_backbone")
    if config["onenet_mode"] not in ("onlinetsf_cross_backbone", "author_architecture", "explicit_backbone_variant"):
        raise ValueError("unknown onenet_mode")
    config.setdefault("fsnet", {})
    if not isinstance(config["fsnet"], dict) or set(config["fsnet"]) - {
            "hidden_dims", "output_dims", "depth", "gamma", "context_length"}:
        raise ValueError("fsnet only accepts hidden_dims/output_dims/depth/gamma/context_length")
    config.setdefault("onenet_secondary_models", {})
    if not isinstance(config["onenet_secondary_models"], dict):
        raise ValueError("onenet_secondary_models must map primary model names to explicit secondary configurations")
    if set(config["onenet_secondary_models"]) - set(OFFICIAL_MODEL_NAMES):
        raise ValueError("unknown primary architecture in onenet_secondary_models")
    if config["method"] == "OneNet" and config["onenet_mode"] == "explicit_backbone_variant" and config["model"]["model_name"] in config["onenet_secondary_models"]:
        config["secondary_model"] = copy.deepcopy(config["onenet_secondary_models"][config["model"]["model_name"]])
    for key in ("model", "secondary_model"):
        model = config[key]
        if model is None:
            continue
        if not isinstance(model, dict) or "model_name" not in model:
            raise ValueError(f"{key} needs an explicit model_name")
        if key == "model" and config["method"] == "OneNet" and config["onenet_mode"] == "author_architecture":
            if model != {"model_name": "fsnet_time"}:
                raise ValueError("author_architecture requires model={'model_name':'fsnet_time'} and uses fsnet settings")
            continue
        if {"lookback", "horizon", "n_variables", "target_indices"} & model.keys():
            raise ValueError(f"{key} input/output dimensions must be derived from data")
        if set(model) - {field.name for field in fields(OfficialForecastConfig)}:
            raise ValueError(f"unknown {key} field")
        if model["model_name"] not in OFFICIAL_MODEL_NAMES:
            raise ValueError(f"unsupported complete official forecaster: {model['model_name']}")
    if config["method"] == "OneNet" and config["onenet_mode"] == "explicit_backbone_variant":
        if config["secondary_model"] is None:
            raise ValueError("OneNet needs an explicit secondary_model architecture")
        if config["secondary_model"]["model_name"] == config["model"]["model_name"]:
            raise ValueError("OneNet requires explicitly different complementary branch architectures")
    elif config["method"] == "OneNet" and config["secondary_model"] is not None:
        raise ValueError("secondary_model is only accepted for explicit_backbone_variant; official modes construct FSNet")
    if {"lookback", "horizon", "target_indices", "seed"} & config["online"].keys():
        raise ValueError("online dimensions and seed are derived from the common experiment configuration")
    return config


def _construct(config, data):
    dims = {"lookback": config["data"]["lookback"], "horizon": config["data"]["horizon"],
            "n_variables": int(data.values.shape[-1]), "target_indices": tuple(data.metadata["target_indices"])}
    if config["method"] == "OneNet" and config["onenet_mode"] == "author_architecture":
        primary, secondary = author_onenet_branches(**dims, **config["fsnet"])
        primary, secondary = primary.to(config["device"]), secondary.to(config["device"])
    else:
        primary = OfficialForecaster(OfficialForecastConfig(**config["model"], **dims)).to(config["device"])
        secondary = None
    if config["method"] == "OneNet" and config["onenet_mode"] == "explicit_backbone_variant":
        _seed(config["seed"] + 1)
        secondary = OfficialForecaster(OfficialForecastConfig(**config["secondary_model"], **dims)).to(config["device"])
    elif config["method"] == "OneNet" and config["onenet_mode"] == "onlinetsf_cross_backbone":
        _seed(config["seed"] + 1)
        secondary = OfficialFSNet(OfficialFSNetConfig(**dims, **config["fsnet"])).to(config["device"])
    online_dims = {key: dims[key] for key in ("lookback", "horizon", "target_indices")}
    online = OnlineBaselineConfig(**config["online"], **online_dims, seed=config["seed"])
    return primary, secondary, online


def _training_windows(data, training):
    # Missing labels can be excluded from offline supervision; forecast origins
    # in validation/test are never filtered on future target validity.
    valid = torch.isfinite(data.train.targets).all(dim=(1, 2))
    indices = valid.nonzero(as_tuple=False).flatten()
    if training.max_train_windows is not None:
        indices = indices[:training.max_train_windows]
    if not len(indices):
        raise ValueError("no fully observed training target window")
    x, y = data.train.inputs[indices], data.train.targets[indices]
    issue_times = data.train.issue_times[indices.cpu().numpy()]
    return x, y, issue_times, {"available_train_windows": len(data.train), "used_train_windows": len(indices),
                             "excluded_missing_train_windows": int((~valid).sum()),
                             "train_budget_selection": "earliest fully observed training windows"}


def _validation_loss(model_or_method, split, training, *, method=False):
    limit = len(split) if training.validation_windows is None else min(len(split), training.validation_windows)
    if not limit:
        raise ValueError("empty validation split")
    snapshot = model_or_method.state_dict() if method else None
    device = model_or_method.device if method else next(model_or_method.parameters()).device
    squared, count = 0., 0
    try:
        if not method:
            model_or_method.eval()
        with torch.no_grad():
            for start in range(0, limit, training.batch_size):
                end = min(start + training.batch_size, limit)
                x, y = split.inputs[start:end].to(device), split.targets[start:end].to(device)
                prediction = (model_or_method.forecast(x, int(split.issue_times[end - 1])) if method
                              else model_or_method(x))
                mask = torch.isfinite(y)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("validation prediction contains nonfinite values")
                squared += float((prediction[mask] - y[mask]).square().sum())
                count += int(mask.sum())
    finally:
        if method:
            model_or_method.load_state_dict(snapshot)
    if not count:
        raise ValueError("validation has no observed target points")
    return squared / count


def _fit_backbone(model, data, training, seed):
    # A method's extra branch must not change the common backbone's dropout or
    # Informer sampling stream during its pretraining stage.
    _seed(seed)
    x, y, _, data_record = _training_windows(data, training)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    history = []
    for epoch in range(training.epochs):
        model.train()
        losses = []
        for indices in torch.randperm(len(x), generator=generator).split(training.batch_size):
            prediction = model(x[indices].to(device))
            loss = F.mse_loss(prediction, y[indices].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("offline training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
            optimizer.step()
            if hasattr(model, "store_grad"):
                model.store_grad()
            losses.append(float(loss.detach()))
        validation = _validation_loss(model, data.validation, training)
        history.append({"epoch": epoch + 1, "training_mse": sum(losses) / len(losses), "validation_mse": validation})
        if validation < best:
            best, best_state, best_epoch, stale = validation, _cpu_clone(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if stale >= training.patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_mse": best, "history": history, **data_record}


def _fit_method(method, data, training):
    if not isinstance(method, (PROCEED, OneNet)):
        return None
    x, y, issues, _ = _training_windows(data, training)
    best, selected, best_epoch, stale = math.inf, None, 0, 0
    history = []
    for epoch in range(training.adapter_epochs):
        report = method.fit_offline(x, y, epochs=1, batch_size=training.batch_size,
                                    learning_rate=training.adapter_learning_rate,
                                    issue_times=tuple(int(t) for t in issues),
                                    cutoff=int(data.metadata.get("offline_training_cutoff", int(data.metadata["train_end"]) - 1)))
        score = _validation_loss(method, data.validation, training, method=True)
        history.append({"epoch": epoch + 1, "training": report, "validation_mse": score})
        if score < best:
            best, selected, best_epoch, stale = score, method.state_dict(), epoch + 1, 0
        else:
            stale += 1
        if stale >= training.patience:
            break
    method.load_state_dict(selected)
    method.set_reset_point()
    return {"best_epoch": best_epoch, "best_validation_mse": best, "history": history}


def _seed_observed_history(method, data, first_issue):
    latest = first_issue - method.config.horizon - method.config.extra_delay - 1
    earliest = max(method.config.lookback - 1, latest - method.config.memory_size + 1)
    rows = []
    for issue in range(earliest, latest + 1):
        # These targets end strictly before the first online timestamp.
        y = data.targets[issue + 1:issue + method.config.horizon + 1]
        if torch.isfinite(y).all():
            rows.append((issue, data.values[issue - method.config.lookback + 1:issue + 1], y))
    if rows:
        issues = tuple(row[0] for row in rows)
        history = ReleasedBatch(torch.stack([row[1] for row in rows]), torch.stack([row[2] for row in rows]),
                                issues, tuple(s + method.config.horizon + method.config.extra_delay for s in issues))
        method.seed_history(history, time=first_issue - 1)
    method.set_reset_point()
    return {"observed_history_windows": len(rows), "pretest_gradient_updates": 0,
            "history_latest_issue": rows[-1][0] if rows else None}


def _data_identity(data):
    metadata = data.metadata
    return _hash_json({"source_sha256": metadata.get("source_sha256"), "config": metadata.get("config"),
                       "rows": metadata["rows"], "mean": metadata.get("mean"), "std": metadata.get("std"),
                       "target_mean": metadata.get("target_mean"), "target_std": metadata.get("target_std"),
                       "train_end": metadata["train_end"], "validation_end": metadata["validation_end"]})


def _baseline_resource_counts(method):
    branches, generators = None, None
    if isinstance(method, OneNet):
        branches = {"secondary": method.secondary_model}
        generators = {"decision": method.decision, "ensemble_weight": method.weight}
    elif isinstance(method, PROCEED):
        generators = {"proceed_adapter": method.adapter}
    return {**framework_parameter_counts(method.model, branches, generators),
            "phase": "online controller after offline initialization/restoration",
            "method": method.name,
            "additional_excluded_storage": ["controller reset snapshot", "replay/released buffers", "checkpoint copies"]}


def _backbone_input_protocol(config, data, primary, secondary):
    records = {}
    for name, model in (("primary", primary), ("secondary", secondary)):
        if model is None:
            continue
        if isinstance(model, OfficialFSNet):
            marks = "channel branch appends seven zero calendar channels; time branch has no separate marks"
        elif model.config.model_name == "itransformer":
            marks = "x_mark_enc remains None; no additional calendar tokens"
        elif model.config.model_name in ("crossformer", "patchtst"):
            marks = "author forward has no separate marks input"
        else:
            marks = "author paths requiring encoder/decoder marks receive explicit zero marks"
        records[name] = {"class": type(model).__name__, "configuration": asdict(model.config),
                         "separate_calendar_marks": marks,
                         "internal_normalization": "preserved according to the recorded complete author configuration"}
    return {"schema_version": 1, "observed_feature_columns": data.metadata["columns"],
            "calendar_information": "only explicitly selected observed input columns; runner supplies no additional calendar tensors",
            "global_normalization": "training-prefix global z-score from PreparedData; targets retain explicit missingness",
            "global_normalization_metadata": "data_metadata.json",
            "target_input_indices": data.metadata["target_indices"], "backbones": records}


class DelayedBaselineStream:
    """Streaming state machine; stores only labels actually observed so far."""

    version = 1

    def __init__(self, method, issue_times, *, rows, native_scale, data_identity):
        issues = tuple(int(t) for t in issue_times)
        if not issues or any(b <= a for a, b in zip(issues, issues[1:])):
            raise ValueError("forecast origins must be nonempty and strictly chronological")
        if issues[0] < method.config.lookback - 1 or issues[-1] + method.config.horizon >= rows:
            raise ValueError("forecast origins exceed available data boundaries")
        self.method = method
        self.issue_times = issues
        self._issue_set = set(issues)
        self.first_time = issues[0]
        self.end_time = min(rows - 1, issues[-1] + method.config.horizon + method.config.extra_delay)
        self.current_time = self.first_time - 1
        self.data_identity = data_identity
        self.native_scale = torch.as_tensor(native_scale, dtype=torch.float64)
        if tuple(self.native_scale.shape) != (len(method.config.target_indices),) or not torch.isfinite(self.native_scale).all():
            raise ValueError("native target scales must be finite and match output channels")
        self.pending = []
        self.records = []
        self.events = []
        self.squared = self.absolute = self.native_squared = self.native_absolute = 0.
        self.valid_target_points = 0
        self.released_windows = 0
        self.updated_windows = 0
        self.missing_feedback_windows = 0
        self.elapsed_seconds = 0.

    def step(self, time, current_x, observed_target):
        if time != self.current_time + 1 or time > self.end_time:
            raise ValueError("stream observations must advance exactly one time step")
        observed = torch.as_tensor(observed_target).detach().cpu().reshape(-1)
        if len(observed) != len(self.method.config.target_indices):
            raise ValueError("observed target channel count mismatch")
        ready, retained, valid_ready = [], [], []
        for pending in self.pending:
            row = self.records[pending["record_index"]]
            step = time - row["issue_time"] - 1
            if 0 <= step < self.method.config.horizon:
                row["truth"][step] = observed
                mask = torch.isfinite(observed)
                error = (row["prediction"][step].double() - observed.double())[mask]
                native_error = error * self.native_scale[mask]
                self.squared += float(error.square().sum())
                self.absolute += float(error.abs().sum())
                self.native_squared += float(native_error.square().sum())
                self.native_absolute += float(native_error.abs().sum())
                self.valid_target_points += int(mask.sum())
            if time >= pending["available_at"]:
                ready.append(pending)
                if torch.isfinite(row["truth"]).all():
                    valid_ready.append(pending)
                else:
                    self.missing_feedback_windows += 1
            else:
                retained.append(pending)
        self.pending = retained
        self.released_windows += len(ready)
        before = self.method.total_update_steps
        report = None
        updated_windows = 0
        if valid_ready:
            targets = [self.records[p["record_index"]] for p in valid_ready]
            released = ReleasedBatch(torch.cat([p["x"] for p in valid_ready]),
                                     torch.stack([r["truth"] for r in targets]),
                                     tuple(r["issue_time"] for r in targets),
                                     tuple(p["available_at"] for p in valid_ready),
                                     torch.stack([r["prediction"] for r in targets]))
            report = self.method.update(released, time=time,
                                        current_context=current_x if time in self._issue_set else None)
            updated_windows = len(valid_ready) if self.method.online_updates_enabled else 0
            self.updated_windows += updated_windows
        issued = time in self._issue_set
        if issued:
            prediction = self.method.forecast(current_x, time=time).detach().cpu()
            if prediction.shape != (1, self.method.config.horizon, len(self.method.config.target_indices)):
                raise ValueError("online forecast shape mismatch")
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("issued prediction contains nonfinite values")
            self.records.append({"issue_time": time, "prediction": prediction[0].clone(),
                                 "truth": torch.full_like(prediction[0], float("nan"))})
            self.pending.append({"record_index": len(self.records) - 1,
                                 "x": current_x.detach().cpu().clone(),
                                 "available_at": time + self.method.config.horizon + self.method.config.extra_delay})
        steps = self.method.total_update_steps - before
        if ready or steps:
            self.events.append({"time": time, "released_windows": len(ready), "updated_windows": updated_windows,
                                "missing_feedback_windows": len(ready) - len(valid_ready), "optimizer_steps": steps,
                                "update": (report or self.method.last_update).to_dict()})
        self.current_time = time

    @property
    def completed(self):
        return self.current_time == self.end_time

    def state_dict(self):
        return _cpu_clone({"version": self.version, "issue_times": self.issue_times,
                          "first_time": self.first_time, "end_time": self.end_time,
                          "current_time": self.current_time, "data_identity": self.data_identity,
                          "native_scale": self.native_scale, "pending": self.pending,
                          "records": self.records, "events": self.events,
                          "squared": self.squared, "absolute": self.absolute,
                          "native_squared": self.native_squared, "native_absolute": self.native_absolute,
                          "valid_target_points": self.valid_target_points, "released_windows": self.released_windows,
                          "updated_windows": self.updated_windows, "missing_feedback_windows": self.missing_feedback_windows,
                          "elapsed_seconds": self.elapsed_seconds,
                          "method": self.method.state_dict(), "method_reset_state": self.method._initial_state})

    def load_state_dict(self, state):
        if state["version"] != self.version or state["data_identity"] != self.data_identity:
            raise ValueError("stream checkpoint source data or version differs")
        if tuple(state["issue_times"]) != self.issue_times or state["end_time"] != self.end_time:
            raise ValueError("stream checkpoint forecast schedule differs")
        if not torch.equal(state["native_scale"], self.native_scale):
            raise ValueError("stream checkpoint normalization differs")
        for key in ("current_time", "pending", "records", "events", "squared", "absolute", "native_squared",
                    "native_absolute", "valid_target_points", "released_windows", "updated_windows",
                    "missing_feedback_windows", "elapsed_seconds"):
            setattr(self, key, _cpu_clone(state[key]))
        self.method.load_state_dict(state["method"])
        self.method._initial_state = _cpu_clone(state["method_reset_state"])

    def result(self):
        count = self.valid_target_points
        metrics = {"normalized_mse": self.squared / count if count else None,
                   "normalized_mae": self.absolute / count if count else None,
                   "native_mse": self.native_squared / count if count else None,
                   "native_mae": self.native_absolute / count if count else None,
                   "valid_target_points": count,
                   "issued_target_points": len(self.records) * self.method.config.horizon * len(self.native_scale),
                   "issued_forecasts": len(self.records), "released_windows": self.released_windows,
                   "updated_windows": self.updated_windows, "missing_feedback_windows": self.missing_feedback_windows,
                   "optimizer_steps": self.method.total_update_steps,
                   "unreleased_forecasts": len(self.pending), "seconds": self.elapsed_seconds,
                   "milliseconds_per_forecast": self.elapsed_seconds * 1000 / len(self.records) if self.records else None,
                   "completed": self.completed, "current_time": self.current_time, "end_time": self.end_time}
        return {"metrics": metrics, "issue_times": np.asarray([r["issue_time"] for r in self.records], dtype=np.int64),
                "predictions": torch.stack([r["prediction"] for r in self.records]).numpy(),
                "targets": torch.stack([r["truth"] for r in self.records]).numpy(),
                "updates": copy.deepcopy(self.events), "state": self.state_dict()}


def evaluate_baseline_stream(method, data: PreparedData, split: WindowSplit, *, max_windows=None,
                             state=None, stop_after_observations=None):
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    if stop_after_observations is not None and stop_after_observations < 1:
        raise ValueError("stop_after_observations must be positive")
    issues = split.issue_times if max_windows is None else split.issue_times[:max_windows]
    scale = data.metadata.get("target_std")
    if scale is None:
        scale = np.asarray(data.metadata["std"])[data.metadata["target_indices"]].tolist()
    stream = DelayedBaselineStream(method, issues, rows=len(data.values), native_scale=scale,
                                   data_identity=_data_identity(data))
    if state is not None:
        stream.load_state_dict(state)
    elif stream.method._last_time is None:
        _seed_observed_history(method, data, stream.first_time)
    start = stream.current_time + 1
    stop = stream.end_time + 1
    if stop_after_observations is not None:
        stop = min(stop, start + stop_after_observations)
    if method.device.type == "cuda":
        torch.cuda.synchronize(method.device)
        torch.cuda.reset_peak_memory_stats(method.device)
    started = time.perf_counter()
    for t in range(start, stop):
        current_x = data.values[t - method.config.lookback + 1:t + 1].unsqueeze(0)
        # The only test-label access in the live protocol. No future slice and
        # no truth-mask lookahead is used when deciding whether to issue.
        stream.step(t, current_x, data.targets[t])
    if method.device.type == "cuda":
        torch.cuda.synchronize(method.device)
    stream.elapsed_seconds += time.perf_counter() - started
    result = stream.result()
    # reset_peak_memory_stats above starts a fresh allocator peak for this
    # invocation. A resumed run reports its final segment, never an aggregate
    # over previous segments or the earlier offline-training phase.
    result["metrics"]["last_segment_peak_cuda_bytes"] = torch.cuda.max_memory_allocated(method.device) if method.device.type == "cuda" else 0
    return result


def run_baseline(configuration, output_dir, *, resume=None, stop_after_observations=None):
    """Run one experiment without leaking deterministic flags or caller RNGs."""
    previous = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cpu_threads": torch.get_num_threads()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    # Capture every already-initialized device that _seed() may touch. Avoid
    # creating a CUDA context merely to execute an otherwise CPU-only job.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            return _run_baseline_impl(configuration, output_dir, resume=resume,
                                      stop_after_observations=stop_after_observations)
        finally:
            torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
            torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
            torch.set_num_threads(previous["cpu_threads"])
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


def _run_baseline_impl(configuration, output_dir, *, resume=None, stop_after_observations=None):
    """Train/select/stream one baseline, or resume it into a new output directory."""
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    if saved is not None and configuration is None:
        configuration = saved["configuration"]
    config = normalize_baseline_config(configuration)
    if saved is not None and config != saved["configuration"]:
        raise ValueError("resume configuration must exactly match the saved protocol")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.backends.cudnn.benchmark = False
    _seed(config["seed"])
    data = prepare_data(DataConfig(**config["data"]))
    primary, secondary, online_config = _construct(config, data)
    training = BaselineTrainingConfig(**config["training"])
    source = baseline_source_manifest()
    if saved is not None and saved["source_manifest"]["files"] != source["files"]:
        raise ValueError("resume implementation sources differ from saved checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "configuration.json", config)
    _write_json(output / "data_metadata.json", data.metadata)
    _write_json(output / "source_manifest.json", source)
    started = datetime.now(timezone.utc).isoformat()
    try:
        if saved is None:
            training_log = {"primary": _fit_backbone(primary, data, training, config["seed"])}
            if secondary is not None:
                training_log["secondary"] = _fit_backbone(secondary, data, training, config["seed"] + 1)
            method = make_online_baseline(config["method"], primary, online_config, secondary_model=secondary)
            training_log["method"] = _fit_method(method, data, training)
            torch.save({"configuration": config, "primary": _cpu_clone(primary.state_dict()),
                        "secondary": _cpu_clone(secondary.state_dict()) if secondary is not None else None,
                        "method": method.state_dict(), "training_log": training_log}, output / "offline_checkpoint.pt")
        else:
            method = make_online_baseline(config["method"], primary, online_config, secondary_model=secondary)
            training_log = saved["training_log"]
        result = evaluate_baseline_stream(method, data, data.test, max_windows=config["test_windows"],
                                          state=saved["stream_state"] if saved is not None else None,
                                          stop_after_observations=stop_after_observations)
        _write_json(output / "resource_counts.json", _baseline_resource_counts(method))
        _write_json(output / "backbone_input_protocol.json", _backbone_input_protocol(config, data, primary, secondary))
        protocol = {"name": data.metadata.get("config", {}).get("protocol", "chronological"),
                    "official_backbone": config["model"]["model_name"], "online_method": config["method"],
                    "secondary_backbone": ("fsnet_channel" if isinstance(secondary, OfficialFSNet) else
                                           config["secondary_model"]["model_name"]) if secondary is not None else None,
                    "onenet_variant": config["onenet_mode"] if secondary is not None else None,
                    "source_variant": (config["onenet_mode"] if secondary is not None else
                                       method.source_variant if isinstance(method, SOLID) else "source_audited_port"),
                    "solid_adaptation_mode": method.adaptation_mode if isinstance(method, SOLID) else None,
                    "solid_gradient_reduction": method.gradient_reduction if isinstance(method, SOLID) else None,
                    "solid_reset_policy": ("continual" if method.continual else "complete_pretrained_state_per_new_input")
                                          if isinstance(method, SOLID) else None,
                    "fsnet_configuration": asdict(secondary.config) if isinstance(secondary, OfficialFSNet) else None,
                    "feedback_rule": "issue+horizon+extra_delay; all target points must be observed for supervision",
                    "issue_rule": "fixed split.issue_times; independent of future target validity",
                    "missing_target_policy": "full-window supervision skip, pointwise finite-mask scoring",
                    "online_validation_updates": False,
                    "pretest_history": "latest fully observed history for context/memory, no gradient updates",
                    "validation_selection": "minimum pointwise finite-mask validation MSE; never uses test labels",
                    "prediction_rng": source["prediction_rng"],
                    "deterministic_algorithms": config["deterministic"],
                    "cuda_memory_metric": {"name": "last_segment_peak_cuda_bytes",
                                           "scope": "current stream invocation; includes resident tensors, excludes peaks in previous segments and offline training",
                                           "aggregated_across_resumes": False},
                    "offline_branch_seeds": {"primary": config["seed"], "secondary": config["seed"] + 1 if secondary is not None else None},
                    "diagnostic_only": data.metadata.get("config", {}).get("protocol") == "legacy_shifted"}
        if not method.online_updates_enabled:
            protocol.update(source_variant="single_official_frozen",
                            predictor_scope="one complete official forecaster; distinct from framework stages_000 MoE",
                            online_update_policy="none; parameters and buffers fixed at the selected offline checkpoint",
                            feedback_usage="release bookkeeping and scoring only")
        _write_json(output / "protocol.json", protocol)
        _write_json(output / "training_log.json", training_log)
        _write_json(output / "metrics.json", result["metrics"])
        _write_json(output / "updates.json", result["updates"])
        np.savez_compressed(output / "forecasts.npz", issue_times=result["issue_times"],
                            predictions=result["predictions"], targets=result["targets"],
                            target_mask=np.isfinite(result["targets"]))
        checkpoint = {"configuration": config, "data_identity": _data_identity(data),
                      "source_manifest": source, "training_log": training_log, "stream_state": result["state"]}
        torch.save(checkpoint, output / "stream_checkpoint.pt")
        _write_json(output / "run_record.json", {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                                                "command": sys.argv, "seed": config["seed"], "device": config["device"],
                                                "resume_from": str(Path(resume).resolve()) if resume else None,
                                                "status": "completed" if result["metrics"]["completed"] else "paused_at_requested_limit",
                                                "output": str(output), "protocol": protocol})
        return {"output": str(output), "metrics": result["metrics"], "checkpoint": str(output / "stream_checkpoint.pt")}
    except Exception as error:
        _write_json(output / "failure.json", {"started_utc": started, "failed_utc": datetime.now(timezone.utc).isoformat(),
                                               "type": type(error).__name__, "error": str(error)})
        raise


def run_baseline_matrix(configuration, output_dir, *, models=None, methods=None, seeds=None):
    """Explicit bounded matrix; each arm has its own directory and no overrides."""
    config = normalize_baseline_config(configuration)
    models = list(models or [config["model"]["model_name"]])
    methods = list(methods or [config["method"]])
    seeds = list(seeds or [config["seed"]])
    if len(set(models)) != len(models) or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("matrix axes must not contain duplicates")
    arms = []
    for model in models:
        for method in methods:
            for seed in seeds:
                arm = copy.deepcopy(config)
                arm["model"]["model_name"], arm["method"], arm["seed"] = model, method, seed
                # An explicit OneNet branch selection is required; the runner
                # never silently swaps architecture merely to make an arm run.
                arms.append(normalize_baseline_config(arm))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "matrix_plan.json", {"models": models, "methods": methods, "seeds": seeds, "arms": len(arms)})
    records = []
    for arm in arms:
        key = f"{arm['model']['model_name']}__{arm['method']}__seed{arm['seed']}"
        records.append(run_baseline(arm, output / key))
    _write_json(output / "matrix_results.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-observations", type=int)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--models", nargs="+", choices=OFFICIAL_MODEL_NAMES)
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.config is None and args.resume is None:
        parser.error("--config or --resume is required")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    if args.matrix:
        if args.resume is not None or args.stop_after_observations is not None:
            parser.error("matrix cannot resume or pause an individual stream")
        result = run_baseline_matrix(config, args.out, models=args.models, methods=args.methods, seeds=args.seeds)
    else:
        result = run_baseline(config, args.out, resume=args.resume, stop_after_observations=args.stop_after_observations)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
