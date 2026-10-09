"""Count matrix parameter structures on CPU; never train or run a forecast.

Explicit feature schemas need only CSV headers. For feature_columns=null, an
explicit --allow-dtype-scan reads each such CSV once to reproduce prepare_data's
numeric-column selection, then retains only its schema, never raw rows.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn

from paper_impl.resource_report import framework_parameter_counts, isolated_cpu_initialization


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def dataset_schema(dataset, *, allow_dtype_scan=False):
    config = dataset["input_data"]
    path = Path(config["path"])
    encoding = config.get("encoding", "utf-8")
    with path.open(encoding=encoding, newline="") as file:
        header = next(csv.reader(file))
    header[0] = header[0].lstrip("\ufeff")
    mapping = config.get("target_input_columns") or {target: target for target in config["target_columns"]}
    excluded = set(config.get("target_mask_columns") or []) | {
        target for target, source in mapping.items() if source != target}
    columns, dtypes = config.get("feature_columns"), None
    if columns is None:
        if not allow_dtype_scan:
            raise ValueError(f"{dataset['id']}: numeric feature inference requires --allow-dtype-scan or explicit registry columns")
        frame = pd.read_csv(path, encoding=encoding)
        columns = [column for column in frame.select_dtypes(include="number").columns
                   if column != config["timestamp_column"] and column not in excluded]
        dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        del frame
        evidence = "one read_csv dtype inference; same numeric selection as prepare_data; raw rows discarded"
    else:
        columns = list(columns)
        evidence = "explicit registry columns verified against CSV header only"
    if not columns or len(columns) != len(set(columns)) or not set(columns).issubset(header):
        raise ValueError(f"{dataset['id']}: invalid input column schema")
    if set(config["target_columns"]) != set(mapping) or not set(mapping.values()).issubset(columns):
        raise ValueError(f"{dataset['id']}: invalid target-input mapping")
    return {"dataset": dataset["id"], "path": path.resolve().as_posix(), "header": header,
            "columns": columns, "target_indices": [columns.index(mapping[target]) for target in config["target_columns"]],
            "n_variables": len(columns), "n_targets": len(config["target_columns"]),
            "dtypes": dtypes, "evidence": evidence,
            "matrix_declared_source_sha256": dataset.get("source_sha256"),
            "source_content_hash_revalidated": False}


def structure_key(configuration, schema):
    model = dict(configuration.get("model", {}))
    model.pop("official_sampling_seed", None)
    mode = configuration.get("foil", {}).get("mode", "foil")
    branch_count = 0 if mode == "disabled" else 1 if mode == "variant_only" else 2
    generator = dict(configuration.get("generator", {}))
    generator.pop("epsilon", None)
    return {"model": model, "lookback": configuration["data"]["lookback"],
            "horizon": configuration["data"]["horizon"], "n_variables": schema["n_variables"],
            "target_indices": schema["target_indices"], "branch_count": branch_count, "generator": generator}


def count_framework(configuration, schema):
    from paper_impl.run import build
    data = SimpleNamespace(metadata={"columns": schema["columns"], "target_indices": schema["target_indices"]})
    with isolated_cpu_initialization(), redirect_stdout(io.StringIO()):
        model, branches, generator = build(configuration, data, torch.device("cpu"))
        construction = framework_parameter_counts(model, branches, generator)
        model.eval()
        branches.freeze()
        generator.eval()
        online = framework_parameter_counts(model, branches, generator)
        if any(parameter.device.type != "cpu" for component in (model, branches, generator) for parameter in component.parameters()):
            raise AssertionError("resource counting created a non-CPU parameter")
    return {"construction": construction, "online_requires_grad_snapshot": online,
            "phase_note": "online snapshot freezes auxiliary branches as run.py does; stages/masks are not parameter counts"}


def count_baseline(configuration, schema, method):
    from paper_impl.official_backbones import OfficialForecaster, OfficialForecastConfig
    from paper_impl.official_fsnet import OfficialFSNet, OfficialFSNetConfig
    from paper_impl.online_baselines import OnlineBaselineConfig, ProceedAdapter, _DecisionNetwork
    model_values = configuration["model"]
    dims = {"lookback": configuration["data"]["lookback"], "horizon": configuration["data"]["horizon"],
            "n_variables": schema["n_variables"], "target_indices": tuple(schema["target_indices"])}
    options = {"hidden": model_values.get("hidden", 64), "heads": model_values.get("heads", 8),
               "depth": model_values.get("depth", 2), **model_values.get("official_options", {})}
    with isolated_cpu_initialization(), redirect_stdout(io.StringIO()):
        model = OfficialForecaster(OfficialForecastConfig(model_values["expert_backbone"], **dims, **options))
        controller_config = OnlineBaselineConfig(dims["lookback"], dims["horizon"], dims["target_indices"])
        branches, generator = None, None
        if method in ("Head-GD", "SOLID"):
            model.requires_grad_(False)
            for parameter in model.head_parameters():
                parameter.requires_grad_(True)
        elif method == "OneNet":
            branches = {"secondary": OfficialFSNet(OfficialFSNetConfig(**dims))}
            generator = {"decision": _DecisionNetwork(dims["horizon"], controller_config.decision_dropout),
                         "ensemble_weight": nn.Parameter(torch.zeros(len(dims["target_indices"])))}
        elif method == "PROCEED":
            adapter = ProceedAdapter(model, controller_config)
            adapter.online_mode()
            generator = {"proceed_adapter": adapter}
        elif method not in ("GD", "SOLID++"):
            raise ValueError(f"unknown planned baseline {method}")
        counts = framework_parameter_counts(model, branches, generator)
    return {"online_requires_grad_snapshot": counts,
            "configuration_basis": "full framework job's common L/H/input schema/official backbone size; current baseline adapter defaults",
            "adapter_defaults": asdict(controller_config),
            "not_a_training_result": True}


def report_matrix(matrix_path, output, *, allow_dtype_scan=False, include_baselines=True):
    matrix_path, output = Path(matrix_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"resource report output already exists: {output}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    schemas = {dataset["id"]: dataset_schema(dataset, allow_dtype_scan=allow_dtype_scan)
               for dataset in matrix["datasets"]}
    cache, rows, baseline_rows, full_jobs = {}, [], [], {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for job in matrix["jobs"]:
            config_path = matrix_path.parent / job["config"]
            configuration = json.loads(config_path.read_text(encoding="utf-8"))
            schema = schemas[job["dataset"]]
            signature = structure_key(configuration, schema)
            cache_id = "framework_" + _hash(signature)[:20]
            if cache_id not in cache:
                cache[cache_id] = {"signature": signature, **count_framework(configuration, schema)}
            counts = cache[cache_id]["construction"]
            online = cache[cache_id]["online_requires_grad_snapshot"]
            rows.append({"dataset": job["dataset"], "backbone": job["backbone"], "ablation": job["ablation"],
                         "seed": job["seed"], "config": job["config"], "structure_id": cache_id,
                         "logical_parameter_count": counts["logical_parameter_count"], "real_scalar_count": counts["real_scalar_count"],
                         "construction_trainable_logical_parameter_count": counts["trainable_logical_parameter_count"],
                         "online_requires_grad_logical_parameter_count": online["trainable_logical_parameter_count"],
                         "parameter_bytes": counts["parameter_bytes"], "buffer_bytes": counts["buffer_bytes"]})
            if job["ablation"] == "full":
                full_jobs[job["dataset"], job["backbone"], job["seed"]] = configuration
        if include_baselines:
            for job in matrix.get("baseline_jobs", []):
                configuration = full_jobs[job["dataset"], job["backbone"], job["seed"]]
                schema = schemas[job["dataset"]]
                signature = {"method": job["method"], "base": structure_key(configuration, schema)}
                cache_id = "baseline_" + _hash(signature)[:20]
                if cache_id not in cache:
                    cache[cache_id] = {"signature": signature, **count_baseline(configuration, schema, job["method"])}
                counts = cache[cache_id]["online_requires_grad_snapshot"]
                baseline_rows.append({"dataset": job["dataset"], "backbone": job["backbone"], "method": job["method"],
                    "seed": job["seed"], "structure_id": cache_id,
                    **{name: counts[name] for name in ("logical_parameter_count", "real_scalar_count",
                        "trainable_logical_parameter_count", "trainable_real_scalar_count", "parameter_bytes", "buffer_bytes")}})
    finally:
        torch.set_num_threads(previous_threads)
    source_files = [Path(__file__), ROOT / "paper_impl/resource_report.py", ROOT / "paper_impl/model.py",
                    ROOT / "paper_impl/adaptation.py", ROOT / "paper_impl/official_backbones.py",
                    ROOT / "paper_impl/online_baselines.py", ROOT / "paper_impl/official_fsnet.py", ROOT / "paper_impl/run.py"]
    report = {"schema_version": 1, "status": "parameter_counts_only_no_training_or_performance_profiling",
              "matrix_path": str(matrix_path), "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
              "framework_jobs_counted": len(rows), "baseline_jobs_counted": len(baseline_rows),
              "unique_structures_instantiated": len(cache), "device": "cpu", "forward_calls": 0, "training_steps": 0,
              "datasets": schemas, "structures": cache, "framework_jobs": rows, "baseline_jobs": baseline_rows,
              "sources": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
              "cost_scope": "parameter and registered buffer storage only; runtime and peak allocation require a separate measured profile"}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "resource_counts.json", report)
    _write(output / "dataset_schemas.json", schemas)
    for filename, records in (("framework_parameter_counts.csv", rows), ("baseline_parameter_counts.csv", baseline_rows)):
        if records:
            with (output / filename).open("x", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return {key: report[key] for key in ("status", "framework_jobs_counted", "baseline_jobs_counted", "unique_structures_instantiated")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/full_experiment/structural/matrix.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-dtype-scan", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report_matrix(args.matrix, args.out, allow_dtype_scan=args.allow_dtype_scan,
                                   include_baselines=not args.no_baselines), ensure_ascii=False))


if __name__ == "__main__":
    main()
