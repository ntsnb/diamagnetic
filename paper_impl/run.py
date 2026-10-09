"""完整训练和checkpoint再评估；测试期I/II仍按固定规则使用已释放标签。

再评估继承checkpoint，普通运行设置可改；selector/identity消融需要
evaluation_overrides显式声明，不允许改变训练模型或数据处理协议。
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from .model import ModelConfig, PaperForecaster, ForecastBranches
from .foil import FoilConfig, train_foil_branches
from .training import TrainingConfig, seed_everything, train_forecaster
from .resource_report import framework_parameter_counts
from .data import DataConfig, prepare_data
from .adaptation import CenteredBoundedGenerator, OnlineConfig
from .replay import ReplayConfig, train_generator
from .evaluation import evaluate_stream, validation_error_threshold, stage_variants, selection_variants, selection_comparison_summary
from .diagnostics import ExpertDiagnosticsConfig
from .backbones import LEGACY_DEPENDENCY_FILES

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_LOG = ROOT / "docs" / "experiment_log.md"
TOP_KEYS = {"seed", "device", "cpu_threads", "data", "model", "foil", "training", "generator",
            "online", "replay", "error_quantile", "variants", "test_windows", "evaluation_overrides",
            "selection_comparisons", "expert_diagnostics"}
GENERATOR_DEFAULTS = {"hidden_dim": 16, "epsilon": .1, "mode": "centered_bounded", "share_matching_shapes": False}
DERIVED_MODEL_KEYS = {"n_variables", "lookback", "horizon", "targets", "target_indices"}
EVALUATION_SETTINGS = {"device", "cpu_threads", "variants", "test_windows", "selection_comparisons", "expert_diagnostics"}


def write_json(path, value):
    """结果只创建一次，不覆盖现有实验文件。"""
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def source_manifest():
    sources = list((ROOT / "paper_impl").glob("*.py"))
    for dependency in LEGACY_DEPENDENCY_FILES:
        path = ROOT / dependency
        if path.exists():
            sources.append(path)
    for subtree in ("official_forecasters", "official_online"):
        root = ROOT / "third_party" / subtree
        if root.exists():
            sources.extend(p for p in root.rglob("*") if p.is_file()
                           and ".git" not in p.parts and (p.suffix in (".py", ".json") or p.name.startswith("LICENSE")))
    records = {file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
               for file in sorted(sources)}
    return {"code_sha256": records, "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return copy.deepcopy(value)


def _keys(value, allowed, name):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer in the permitted range")


def _number(value, name, *, minimum=None, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (minimum is not None and (value <= minimum if strict else value < minimum))):
        raise ValueError(f"{name} must be a finite number in the permitted range")


def _finite_values(value, name="configuration"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a nonfinite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_values(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_values(item, f"{name}[{index}]")


def _section(configuration, key, cls):
    values = _mapping(configuration.get(key, {}), key)
    _keys(values, {field.name for field in fields(cls)}, key)
    return values


def _online(values):
    values = dict(values)
    # JSON null means the training threshold has not yet been calibrated.
    if values.get("error_threshold") is None:
        values.pop("error_threshold", None)
    return OnlineConfig(**values)


def _online_json(config):
    values = asdict(config)
    if math.isinf(values["error_threshold"]):
        values["error_threshold"] = None
    return values


def _normalize(configuration):
    """解析所有默认值；阶段seed仅在未明确给定时继承总seed。"""
    configuration = _mapping(configuration, "configuration")
    _keys(configuration, TOP_KEYS, "configuration")
    _finite_values(configuration)
    if "evaluation_overrides" in configuration:
        _evaluation_overrides(configuration["evaluation_overrides"])
    seed, threads = configuration.get("seed", 42), configuration.get("cpu_threads", 1)
    _integer(seed, "seed", 0, 2**32 - 1)
    _integer(threads, "cpu_threads")
    device = configuration.get("device", "cpu")
    if not isinstance(device, str):
        raise ValueError("device must be a torch device name or auto")
    if device != "auto":
        try:
            torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("Invalid device") from error
    try:
        data = asdict(DataConfig(**_section(configuration, "data", DataConfig)))
    except TypeError as error:
        raise ValueError("data requires path, timestamp_column and target_columns") from error
    for key in ("path", "timestamp_column", "encoding"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key} must be a nonempty string")
    for key in ("target_columns", "feature_columns"):
        columns = data[key]
        if key == "feature_columns" and columns is None:
            continue
        if (not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f"data.{key} must be a nonempty list of unique column names")
    for key in ("lookback", "horizon"):
        _integer(data[key], f"data.{key}")
    if data["max_rows"] is not None:
        _integer(data["max_rows"], "data.max_rows")
    if data.get("expected_interval_seconds") is not None:
        _number(data["expected_interval_seconds"], "data.expected_interval_seconds", minimum=0, strict=True)
    for key in ("train_fraction", "validation_fraction"):
        _number(data[key], f"data.{key}", minimum=0, strict=True)
    if data["train_fraction"] + data["validation_fraction"] >= 1:
        raise ValueError("Invalid chronological split fractions")
    data["path"] = str(Path(data["path"]).resolve())
    model_values = _section(configuration, "model", ModelConfig)
    model_values.setdefault("official_sampling_seed", seed)
    _integer(model_values["official_sampling_seed"], "model.official_sampling_seed", 0, 2**32 - 1)
    if DERIVED_MODEL_KEYS & model_values.keys():
        raise ValueError("model dimensions n_variables/lookback/horizon/targets are derived from data")
    for key in ("n_experts", "hidden", "heads", "depth", "latent_block", "patch_size"):
        if key in model_values:
            _integer(model_values[key], f"model.{key}")
    if "lrp_epsilon" in model_values:
        _number(model_values["lrp_epsilon"], "model.lrp_epsilon", minimum=0, strict=True)
    model = asdict(ModelConfig(n_variables=max(1, len(data["target_columns"])), lookback=data["lookback"], horizon=data["horizon"],
                               targets=len(data["target_columns"]), **model_values))
    model = {k: v for k, v in model.items() if k not in DERIVED_MODEL_KEYS}
    foil_values = _section(configuration, "foil", FoilConfig)
    foil_values.setdefault("seed", seed)
    _integer(foil_values["seed"], "foil.seed", 0, 2**32 - 1)
    foil = asdict(FoilConfig(**foil_values))
    training_values = _section(configuration, "training", TrainingConfig)
    training_values.setdefault("seed", seed)
    _integer(training_values["seed"], "training.seed", 0, 2**32 - 1)
    for key in ("epochs", "batch_size", "attribution_batch_size"):
        if key in training_values:
            _integer(training_values[key], f"training.{key}")
    for key in ("learning_rate", "grad_clip", "vae_weight", "kl_weight", "specialization_weight", "margin"):
        if key in training_values:
            _number(training_values[key], f"training.{key}", minimum=0, strict=key in {"learning_rate", "grad_clip", "margin"})
    training = asdict(TrainingConfig(**training_values))
    replay_values = _section(configuration, "replay", ReplayConfig)
    if "episodes" in replay_values:
        _integer(replay_values["episodes"], "replay.episodes")
    if replay_values.get("max_windows") is not None:
        _integer(replay_values["max_windows"], "replay.max_windows")
    for key in ("learning_rate", "grad_clip"):
        if key in replay_values:
            _number(replay_values[key], f"replay.{key}", minimum=0, strict=True)
    replay = asdict(ReplayConfig(**replay_values))
    generator_values = _mapping(configuration.get("generator", {}), "generator")
    _keys(generator_values, GENERATOR_DEFAULTS, "generator")
    generator = {**GENERATOR_DEFAULTS, **generator_values}
    if foil["mode"] == "disabled" and generator["mode"] != "identity":
        raise ValueError("removing the invariant module requires generator.mode=identity and Stage III disabled")
    _integer(generator["hidden_dim"], "generator.hidden_dim")
    _number(generator["epsilon"], "generator.epsilon", minimum=0, strict=True)
    if generator["epsilon"] >= 1 or generator["mode"] not in {"centered_bounded", "identity"}:
        raise ValueError("Invalid generator epsilon or mode")
    if not isinstance(generator["share_matching_shapes"], bool):
        raise ValueError("generator.share_matching_shapes must be bool")
    online_values = _section(configuration, "online", OnlineConfig)
    online_values.setdefault("selection_seed", seed)
    _integer(online_values["selection_seed"], "online.selection_seed", 0, 2**32 - 1)
    for key in ("stage_i", "stage_ii", "stage_iii"):
        if key in online_values and not isinstance(online_values[key], bool):
            raise ValueError(f"online.{key} must be bool")
    for key in ("max_backtracks", "reference_windows", "reference_interval", "history_capacity",
                "pending_capacity", "record_capacity", "extra_delay"):
        if key in online_values:
            _integer(online_values[key], f"online.{key}", 0 if key == "extra_delay" else 1)
    if "channel_budget" in online_values:
        budgets = online_values["channel_budget"]
        if isinstance(budgets, dict):
            if not budgets or any(not isinstance(key, str) for key in budgets):
                raise ValueError("online.channel_budget requires named layer budgets")
            for key, value in budgets.items():
                _integer(value, f"online.channel_budget.{key}", 0)
        else:
            _integer(budgets, "online.channel_budget", 0)
    for key in ("learning_rate_i", "learning_rate_ii", "radius_i", "radius_ii", "armijo_c", "backtrack_factor",
                "stability_threshold", "relevance_floor", "error_threshold", "reference_ema"):
        if key in online_values and not (key == "error_threshold" and online_values[key] is None):
            _number(online_values[key], f"online.{key}", minimum=0,
                    strict=key not in {"stability_threshold", "error_threshold"})
    online = _online_json(_online(online_values))
    if "extra_delay" in configuration.get("data", {}) and data["extra_delay"] != online["extra_delay"]:
        raise ValueError("data.extra_delay must match online.extra_delay for offline-label embargo")
    data["extra_delay"] = online["extra_delay"]
    if online["pending_capacity"] < data["horizon"] + online["extra_delay"] + 1:
        raise ValueError("online.pending_capacity must cover horizon + delay + 1")
    variants = configuration.get("variants", list(stage_variants()))
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) for v in variants):
        raise ValueError("variants must be a nonempty list")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must not contain duplicates")
    unknown = set(variants) - set(stage_variants())
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    comparisons = configuration.get("selection_comparisons", [])
    if (not isinstance(comparisons, list) or any(not isinstance(v, str) for v in comparisons)
            or len(set(comparisons)) != len(comparisons) or set(comparisons) - set(selection_variants())):
        raise ValueError("selection_comparisons must be a unique list of defined matched controls")
    diagnostics = asdict(ExpertDiagnosticsConfig(**_section(configuration, "expert_diagnostics", ExpertDiagnosticsConfig)))
    test_windows = configuration.get("test_windows")
    if test_windows is not None:
        _integer(test_windows, "test_windows")
    quantile = configuration.get("error_quantile", .9)
    _number(quantile, "error_quantile", minimum=0, strict=True)
    if quantile >= 1:
        raise ValueError("error_quantile must be in (0,1)")
    return {"seed": seed, "device": device, "cpu_threads": threads, "data": data, "model": model,
            "foil": foil, "training": training, "generator": generator, "online": online, "replay": replay,
            "error_quantile": quantile, "variants": list(variants), "test_windows": test_windows,
            "selection_comparisons": list(comparisons), "expert_diagnostics": diagnostics}


def _evaluation_overrides(value):
    overrides = _mapping(value, "evaluation_overrides")
    _keys(overrides, {"online", "generator"}, "evaluation_overrides")
    online = _mapping(overrides.get("online", {}), "evaluation_overrides.online")
    generator = _mapping(overrides.get("generator", {}), "evaluation_overrides.generator")
    _keys(online, {"selector"}, "evaluation_overrides.online")
    _keys(generator, {"mode"}, "evaluation_overrides.generator")
    if online.get("selector", "relevance_error") not in {"relevance_error", "gradient_only"}:
        raise ValueError("Invalid explicit selector ablation")
    if generator and generator.get("mode") != "identity":
        raise ValueError("The only generator evaluation override is mode=identity")
    return {key: values for key, values in (("online", online), ("generator", generator)) if values}


def _load_checkpoint(path):
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format_version", "model_config", "model", "branches", "generator", "configuration", "data_metadata", "history"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Checkpoint is missing its model/configuration/data contract")
    if saved["format_version"] not in {1, 2}:
        raise ValueError("Unsupported checkpoint format version")
    saved_configuration = _mapping(saved["configuration"], "checkpoint.configuration")
    saved_model = _mapping(saved_configuration.get("model", {}), "checkpoint.configuration.model")
    model_contract = _mapping(saved["model_config"], "checkpoint.model_config")
    # Older checkpoints used the constructor's fixed seed, even when their
    # global training seed differed. Do not reinterpret those predictions.
    saved_model.setdefault("official_sampling_seed", model_contract.get("official_sampling_seed", 42))
    saved_configuration["model"] = saved_model
    saved_configuration.setdefault("online", {}).setdefault("selection_seed", (saved.get("online_config") or {}).get("selection_seed", 42))
    configuration = _normalize(saved_configuration)
    if configuration["online"]["error_threshold"] is None:
        raise ValueError("Checkpoint lacks a finite validation-calibrated error threshold")
    metadata = saved["data_metadata"]
    if not isinstance(metadata, dict) or not {"columns", "target_indices", "config"}.issubset(metadata):
        raise ValueError("Checkpoint is missing its data column contract")
    if configuration["data"]["feature_columns"] is None:
        configuration["data"]["feature_columns"] = list(metadata["columns"])
    if saved["format_version"] == 2:
        if saved.get("generator_config") != configuration["generator"]:
            raise ValueError("Checkpoint generator configuration is internally inconsistent")
        if _online_json(_online(saved.get("online_config", {}))) != configuration["online"]:
            raise ValueError("Checkpoint online configuration is internally inconsistent")
    provenance = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "format_version": saved["format_version"], "legacy_constructor_defaults": saved["format_version"] == 1,
                  "training_source_manifest": saved.get("source_manifest")}
    return saved, configuration, provenance


def _resolve_evaluation(request, saved_configuration):
    request = _mapping(request, "configuration")
    _keys(request, TOP_KEYS, "configuration")
    _finite_values(request)
    effective = copy.deepcopy(saved_configuration)
    for key, value in request.items():
        if key in EVALUATION_SETTINGS:
            effective[key] = value
        elif key == "evaluation_overrides":
            continue
        elif isinstance(saved_configuration.get(key), dict):
            values = _mapping(value, key)
            _keys(values, saved_configuration[key], key)
            for field, requested in values.items():
                if key == "data" and field == "path":
                    effective[key][field] = requested
                elif requested != saved_configuration[key][field]:
                    raise ValueError(f"Checkpoint mismatch: {key}.{field}; use a permitted explicit evaluation override")
                else:
                    effective[key][field] = requested
        elif value != saved_configuration.get(key):
            raise ValueError(f"Checkpoint mismatch: {key}")
        else:
            effective[key] = value
    overrides = _evaluation_overrides(request.get("evaluation_overrides", {}))
    for section, values in overrides.items():
        effective[section].update(values)
    return _normalize(effective), overrides


def _check_data_contract(saved_metadata, current_metadata):
    # Hash/moments cannot detect a changed validation boundary or a permutation
    # of identically distributed feature/target columns.
    for key in ("source_sha256", "rows", "columns", "target_indices", "train_end", "validation_end", "mean",
                "std", "original_training_std", "unit_std_columns", "interval_seconds", "imputation_medians",
                "imputation", "first_timestamp", "last_timestamp"):
        if key not in saved_metadata or saved_metadata[key] != current_metadata[key]:
            raise ValueError(f"Checkpoint data mismatch: {key}")
    old = {k: v for k, v in asdict(DataConfig(**saved_metadata["config"])).items() if k != "path"}
    new = {k: v for k, v in asdict(DataConfig(**current_metadata["config"])).items() if k != "path"}
    if old.get("feature_columns") is None:
        old["feature_columns"] = list(saved_metadata["columns"])
    if new.get("feature_columns") is None:
        new["feature_columns"] = list(current_metadata["columns"])
    if old != new:
        raise ValueError("Checkpoint data preprocessing/split configuration differs")


def build(configuration, data, device):
    cfg = ModelConfig(n_variables=len(data.metadata["columns"]), lookback=configuration["data"]["lookback"],
                      horizon=configuration["data"]["horizon"], targets=len(data.metadata["target_indices"]),
                      target_indices=tuple(data.metadata["target_indices"]),
                      **configuration.get("model", {}))
    model = PaperForecaster(cfg).to(device)
    branches = ForecastBranches(cfg, mode=configuration.get("foil", {}).get("mode", "foil"),
                                contrast=configuration.get("foil", {}).get("contrast", "difference")).to(device)
    generator = CenteredBoundedGenerator({k: tuple(p.shape) for k, p in model.expert_weights().items()},
                                         cfg.horizon * cfg.targets, **configuration.get("generator", {})).to(device)
    return model, branches, generator


def _save_checkpoint(path, model, branches, generator, configuration, metadata, history, manifest):
    payload = {"format_version": 2, "model_config": asdict(model.config), "model": model.state_dict(),
               "branches": branches.state_dict(), "generator": generator.state_dict(),
               "generator_config": configuration["generator"], "online_config": configuration["online"],
               "configuration": configuration, "data_metadata": metadata, "history": history, "source_manifest": manifest}
    with Path(path).open("xb") as file:
        torch.save(payload, file)


def run(configuration, output, checkpoint=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new run directory: {output}")
    requested = _mapping(configuration, "configuration")
    saved = provenance = None
    if checkpoint is None:
        if requested.get("evaluation_overrides"):
            raise ValueError("evaluation_overrides require an existing checkpoint")
        effective, overrides = _normalize(requested), {}
        if not effective["online"]["stage_iii"] and effective["generator"]["mode"] != "identity":
            raise ValueError("Train with Stage III enabled; use variants for stage ablations")
    else:
        saved, saved_configuration, provenance = _load_checkpoint(checkpoint)
        effective, overrides = _resolve_evaluation(requested, saved_configuration)
    data = prepare_data(DataConfig(**effective["data"]))
    effective["data"]["feature_columns"] = list(data.metadata["columns"])
    if saved is not None:
        _check_data_contract(saved["data_metadata"], data.metadata)
    device_name = effective["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(effective["cpu_threads"])
    seed_everything(effective["seed"])
    model, branches, generator = build(effective, data, device)
    if branches.mode == "disabled" and any(stage_variants()[name][2] for name in effective["variants"]):
        raise ValueError("w/o invariant module has no Stage III contrast; choose variants ending in 0")
    online = _online(effective["online"])
    if saved is not None:
        previous_model_config = dict(saved["model_config"])
        previous_model_config.setdefault("target_indices", tuple(data.metadata["target_indices"]))
        if asdict(ModelConfig(**previous_model_config)) != asdict(model.config):
            raise ValueError("Checkpoint architecture differs from its configuration/data contract")
        model.load_state_dict(saved["model"], strict=True)
        branches.load_state_dict(saved["branches"], strict=True)
        if generator.mode != "identity":
            generator.load_state_dict(saved["generator"], strict=True)
        elif saved["configuration"]["generator"]["mode"] != "identity" and not overrides.get("generator"):
            raise ValueError("removing saved generator parameters requires the explicit identity override")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        manifest = source_manifest()
        write_json(output / "config.json", requested)
        write_json(output / "source_manifest.json", manifest)
        write_json(output / "data_metadata.json", data.metadata)
        write_json(output / "resource_counts.json", framework_parameter_counts(model, branches, generator))
        write_json(output / "backbone_input_protocol.json", {
            "calendar_information": "explicit observed input columns when present; required time-mark inputs use zeros, optional marks remain absent",
            "dataset_scaling": "training-only column statistics",
            "instance_normalization": "author model options on official paths; model.use_revin on native path",
            "official_model_options": effective["model"].get("official_options"),
            "target_indices": data.metadata["target_indices"],
        })
        write_json(output / "evaluation_overrides.json", overrides)
        if provenance is not None:
            original_sources = (provenance.get("training_source_manifest") or {}).get("code_sha256", {})
            provenance["changed_source_files"] = sorted(name for name in set(original_sources) | set(manifest["code_sha256"])
                                                        if original_sources.get(name) != manifest["code_sha256"].get(name))
            write_json(output / "checkpoint_provenance.json", provenance)
            write_json(output / "checkpoint_config.json", saved_configuration)
        history = {}
        if saved is None:
            print("Training invariant and variant branches", flush=True)
            history["foil"] = train_foil_branches(branches, data.train.inputs, data.train.targets,
                                                  data.validation.inputs, data.validation.targets, FoilConfig(**effective["foil"]),
                                                  sample_ids=torch.as_tensor(data.train.issue_times))
            write_json(output / "foil_history.json", history["foil"])
            print("Training forecasting experts with differentiable relevance", flush=True)
            history["forecaster"] = train_forecaster(model, data.train.inputs, data.train.targets,
                                                      data.validation.inputs, data.validation.targets,
                                                      TrainingConfig(**effective["training"]))
            write_json(output / "forecaster_history.json", history["forecaster"])
            threshold = validation_error_threshold(model, data.validation.inputs, data.validation.targets, effective["error_quantile"])
            online = replace(online, error_threshold=threshold)
            if generator.mode == "identity":
                history["replay"] = {"status": "skipped", "reason": "identity_generator_ablation", "config": effective["replay"]}
            else:
                print("Training temporary modulation on chronological delayed replay", flush=True)
                history["replay"] = train_generator(model, branches, generator, data.values, data.targets,
                                                    data.metadata["offline_training_cutoff"] + 1, online, ReplayConfig(**effective["replay"]))
            history["replay"]["error_threshold"] = online.error_threshold
            history["replay"]["threshold_source"] = "validation quantile of pretrained forecasts before generator replay"
            threshold = validation_error_threshold(model, data.validation.inputs, data.validation.targets,
                                                    effective["error_quantile"], generator=generator, branches=branches)
            online = replace(online, error_threshold=threshold)
            history["replay"]["deployment_error_threshold"] = threshold
            history["replay"]["deployment_threshold_source"] = "validation quantile after generator training; no test targets"
            write_json(output / "replay_history.json", history["replay"])
            effective["online"] = _online_json(online)
            _save_checkpoint(output / "model.pt", model, branches, generator, effective, data.metadata, history, manifest)
        model.eval()
        branches.freeze()
        generator.eval()
        write_json(output / "effective_config.json", effective)
        write_json(output / "online_config.json", _online_json(online))
        write_json(output / "seed_plan.json", {"initialization": effective["seed"], "foil": effective["foil"]["seed"],
                                               "forecaster": effective["training"]["seed"],
                                               "official_sampling_root": effective["model"]["official_sampling_seed"],
                                               "official_sampling_rule": "root + expert index; fixed for evaluation, attribution and replay",
                                               "selection": effective["online"]["selection_seed"],
                                               "replay": "deterministic chronological updates; inherits initialized parameters"})
        variants, rows = stage_variants(), []
        expert_rows, pair_rows = [], []
        evaluation_configs = {}
        for name in effective["variants"]:
            flags = variants[name]
            evaluation_configs[name] = replace(online, stage_i=flags[0], stage_ii=flags[1], stage_iii=flags[2])
        for name in effective["selection_comparisons"]:
            evaluation_configs[name] = replace(online, **selection_variants()[name])
        write_json(output / "evaluation_protocols.json", {name: _online_json(cfg) for name, cfg in evaluation_configs.items()})
        comparison_records = {}
        for name, cfg in evaluation_configs.items():
            print(f"Online test evaluation with fixed hyperparameters: {name}", flush=True)
            result = evaluate_stream(model, branches, generator, data, data.test, cfg, max_windows=effective["test_windows"],
                                     diagnostics_config=ExpertDiagnosticsConfig(**effective["expert_diagnostics"]))
            row = {"variant": name, **result["metrics"]}
            rows.append(row)
            write_json(output / f"{name}_updates.json", result["updates"])
            write_json(output / f"{name}_skipped_feedback.json", result["skipped_feedback"])
            with (output / f"{name}_forecasts.npz").open("xb") as file:
                np.savez_compressed(file, issue_times=result["issue_times"], predictions=result["predictions"], targets=result["targets"])
            if "expert_diagnostics" in result:
                diagnostic = result["expert_diagnostics"]
                write_json(output / f"{name}_expert_diagnostics.json", diagnostic["summary"])
                with (output / f"{name}_expert_diagnostics.npz").open("xb") as file:
                    np.savez_compressed(file, **diagnostic["arrays"])
                details = diagnostic["summary"]
                row.update({key: details[key] for key in ("mean_routing_entropy", "mean_effective_experts", "mean_specialization_distance")})
                for expert in details["per_expert_error"]:
                    m = expert["expert_index"]
                    expert_rows.append({"variant": name, **expert, "mean_gate_weight": details["mean_gate_weight"][m],
                                        "top1_share": details["top1_share"][m], "top1_count": details["top1_count"][m]})
                for pair in details["pair_distances"]:
                    pair_rows.append({"variant": name, "expert_m": pair["experts"][0], "expert_n": pair["experts"][1],
                                      "mean_distance": pair["mean_distance"], "valid_target_windows": pair["valid_target_windows"]})
            if name in effective["selection_comparisons"]:
                comparison_records[name] = result["updates"]
            print(json.dumps(row), flush=True)
        if comparison_records:
            write_json(output / "selection_comparison.json", selection_comparison_summary(comparison_records))
        for filename, records in (("expert_metrics.csv", expert_rows), ("expert_pair_metrics.csv", pair_rows)):
            if records:
                with (output / filename).open("x", newline="", encoding="utf-8-sig") as file:
                    writer = csv.DictWriter(file, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"status": "completed", "output": str(output), "device": str(device), "seconds": time.perf_counter() - started,
                   "results": rows, "checkpoint": provenance["path"] if provenance else str(output / "model.pt"),
                   "evaluation_overrides": overrides, "new_implementation": True,
                   "test_protocol": "fixed pretrained modules/hyperparameters; enabled Stages I/II adapt only on released labels",
                   "performance_claim": "No superiority claim; validation scope is defined by this run config."}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        failure = output / "failure.json"
        if not failure.exists():
            write_json(failure, {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="training config or optional checkpoint evaluation settings")
    parser.add_argument("--output", required=True, help="new directory; existing outputs are never overwritten")
    parser.add_argument("--checkpoint", help="evaluate a saved local checkpoint; omitted fields are inherited")
    args = parser.parse_args(argv)
    if not args.config and not args.checkpoint:
        parser.error("training requires --config; checkpoint evaluation may omit it")
    configuration = {}
    if args.config:
        config_path = Path(args.config).resolve()
        configuration = _mapping(json.loads(config_path.read_text(encoding="utf-8-sig")), "configuration")
        data_request = _mapping(configuration.get("data", {}), "data")
        path = data_request.get("path")
        if isinstance(path, str) and not Path(path).is_absolute():
            configuration["data"]["path"] = str((config_path.parent / path).resolve())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new run directory: {output}")
    command = [sys.executable, "-m", "paper_impl.run", "--output", str(output)]
    if args.config:
        command += ["--config", str(Path(args.config).resolve())]
    if args.checkpoint:
        command += ["--checkpoint", str(Path(args.checkpoint).resolve())]
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as file:
        file.write(f"\n\n## {output.name}\n\n- 类型：完整训练或 checkpoint 再评估。\n- 状态：running。\n"
                   f"- 命令：`{subprocess.list2cmdline(command)}`\n"
                   f"- 数据、种子和实际设置：见 effective_config.json；不复用历史结果声称新方法性能。\n"
                   f"- 开始：{datetime.now(timezone.utc).isoformat()}\n")
    try:
        summary = run(configuration, output, args.checkpoint)
    except Exception as error:
        with EXPERIMENT_LOG.open("a", encoding="utf-8") as file:
            file.write(f"- 状态：failed；{type(error).__name__}: {error}\n")
        raise
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as file:
        file.write(f"- 状态：completed；耗时{summary['seconds']:.1f}秒。结果：`{output / 'metrics.csv'}`。\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
