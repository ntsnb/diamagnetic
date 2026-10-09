"""One experiment dispatcher for the framework and six source-defined baselines.

Matrix invocations write a concrete plan by default. --execute runs that plan.
All jobs retain their own source/data provenance; existing outputs are refused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .run import run, _normalize
from .baseline_run import run_baseline, normalize_baseline_config, METHOD_NAMES


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)


def dispatch(kind, configuration, output, checkpoint=None):
    if kind == "framework":
        return run(configuration or {}, output, checkpoint=checkpoint)
    if kind == "baseline":
        return run_baseline(configuration, output, resume=checkpoint)
    raise ValueError("kind must be framework or baseline")


def baseline_configuration(configuration, *, backbone, method, device=None):
    """Derive the runnable baseline config from the same full-framework protocol."""
    base = _normalize(configuration)
    model = base["model"]
    options = {key: model[key] for key in ("hidden", "heads", "depth")}
    options.update(model.get("official_options") or {})
    options["model_name"] = backbone
    training = base["training"]
    interval = base["data"].get("expected_interval_seconds")
    if interval is None:
        raise ValueError(
            "baseline matrix requires a finite, positive data.expected_interval_seconds "
            "to convert the daily SOLID period into sampling steps"
        )
    config = {"seed": base["seed"], "device": device or base["device"], "cpu_threads": base["cpu_threads"],
        "data": copy.deepcopy(base["data"]), "method": method, "model": options,
        "onenet_mode": "onlinetsf_cross_backbone",
        "training": {"epochs": training["epochs"], "adapter_epochs": training["epochs"],
            "batch_size": training["batch_size"], "learning_rate": training["learning_rate"],
            "adapter_learning_rate": training["learning_rate"], "max_grad_norm": training["grad_clip"]},
        "online": {"extra_delay": base["online"]["extra_delay"], "period": 86400. / interval}}
    normalize_baseline_config(config)
    return config


def _check_seed_coverage(groups, seeds, expected_scopes=()):
    """Require every requested seed for each applicable selected task, not just globally."""
    if seeds is None:
        return
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
                   for seed in seeds)):
        raise ValueError("seeds must be a nonempty unique list of integers in [0, 2**32)")
    scopes = {scope: set() for scope in expected_scopes}
    for kind, jobs, variant in groups:
        for job in jobs:
            key = (kind, job["dataset"], job["backbone"], job[variant])
            scopes.setdefault(key, set()).add(job["seed"])
    for seed in seeds:
        missing = ["/".join(key) for key, available in scopes.items() if seed not in available]
        if missing:
            examples = ", ".join(missing[:4])
            suffix = f" ... ({len(missing)} tasks)" if len(missing) > 4 else ""
            raise ValueError(f"种子 {seed} 缺少配置 (missing configuration for seed {seed}): {examples}{suffix}")


def _framework_config(path, job, device=None):
    config_path = path.parent / job["config"]
    if not config_path.is_file():
        raise ValueError(f"种子 {job['seed']} 缺少配置文件: {config_path}")
    config = _load(config_path)
    if device is not None:
        config["device"] = device
    effective = _normalize(config)
    for name, value in (("seed", effective["seed"]), ("foil.seed", effective["foil"]["seed"]),
                        ("training.seed", effective["training"]["seed"]),
                        ("online.selection_seed", effective["online"]["selection_seed"]),
                        ("model.official_sampling_seed", effective["model"]["official_sampling_seed"])):
        if value != job["seed"]:
            raise ValueError(f"seed mismatch for {config_path}: {name}={value}, task seed={job['seed']}")
    return config


def plan_matrix(path, *, datasets=None, backbones=None, ablations=None, methods=None,
                families=("framework", "baseline"), seeds=None, device=None):
    path = Path(path).resolve()
    matrix = _load(path)
    known_datasets = {d["id"] for d in matrix["datasets"]}
    known_backbones = set(matrix.get("backbones", [job["backbone"] for job in matrix["jobs"]]))
    definitions = matrix.get("structural_definitions", {job["ablation"]: {} for job in matrix["jobs"]})
    known_ablations = set(definitions)
    for selection, known, name in ((datasets, known_datasets, "dataset"), (backbones, known_backbones, "backbone"),
                                     (ablations, known_ablations, "ablation"), (methods, set(METHOD_NAMES), "method"),
                                     (families, {"framework", "baseline"}, "family")):
        if selection is not None and set(selection) - known:
            raise ValueError(f"unknown {name}: {sorted(set(selection) - known)}")
    full = {(j["dataset"], j["backbone"], j["seed"]): j for j in matrix["jobs"] if j["ablation"] == "full"}
    source_by_dataset = {d["id"]: d["source_sha256"] for d in matrix["datasets"]}
    selected = []

    def in_scope(job):
        return ((datasets is None or job["dataset"] in datasets) and
                (backbones is None or job["backbone"] in backbones))

    framework_jobs = [job for job in matrix["jobs"] if "framework" in families and in_scope(job)
                      and (ablations is None or job["ablation"] in ablations)]
    baseline_jobs = [job for job in matrix["baseline_jobs"] if "baseline" in families and in_scope(job)
                     and (methods is None or job["method"] in methods)]
    # Build the requested task grid from declarations before inspecting job rows.
    # Otherwise a task with every seed row missing would disappear from validation.
    expected_scopes = []
    baseline_definitions = {item["name"]: item for item in matrix.get("baseline_inventory", [])}
    if not baseline_definitions:
        baseline_definitions = {job["method"]: {} for job in matrix["baseline_jobs"]}
    for dataset in sorted(known_datasets if datasets is None else datasets):
        for backbone in sorted(known_backbones if backbones is None else backbones):
            if "framework" in families:
                for variant in sorted(known_ablations if ablations is None else ablations):
                    definition = definitions[variant]
                    if backbone not in definition.get("applicable_backbones", [backbone]):
                        continue
                    if definition.get("native_only", False) and backbone != "native":
                        continue
                    expected_scopes.append(("framework", dataset, backbone, variant))
            if "baseline" in families and backbone != "native":
                for method in sorted(baseline_definitions if methods is None else methods):
                    definition = baseline_definitions.get(method, {})
                    if backbone in definition.get("applicable_backbones", [backbone]):
                        expected_scopes.append(("baseline", dataset, backbone, method))
    requested_seeds = matrix.get("seeds") if seeds is None else seeds
    _check_seed_coverage((("framework", framework_jobs, "ablation"),
                          ("baseline", baseline_jobs, "method")), requested_seeds, expected_scopes)

    if "framework" in families:
        for job in framework_jobs:
            if seeds is not None and job["seed"] not in seeds:
                continue
            config = _framework_config(path, job, device)
            selected.append({"kind": "framework", "dataset": job["dataset"], "backbone": job["backbone"],
                "variant": job["ablation"], "seed": job["seed"], "configuration": config,
                "data_sha256": source_by_dataset[job["dataset"]]})
    if "baseline" in families:
        for job in baseline_jobs:
            if seeds is not None and job["seed"] not in seeds:
                continue
            if "config" in job:
                config_path = path.parent / job["config"]
                if not config_path.is_file():
                    raise ValueError(f"种子 {job['seed']} 缺少配置文件: {config_path}")
                config = _load(config_path)
                if device is not None:
                    config["device"] = device
            else:
                key = (job["dataset"], job["backbone"], job["seed"])
                if key not in full:
                    raise ValueError(f"种子 {job['seed']} 缺少基线所需的Full配置: {key}")
                base = _framework_config(path, full[key])
                config = baseline_configuration(base, backbone=job["backbone"], method=job["method"], device=device)
            effective = normalize_baseline_config(config)
            if effective["seed"] != job["seed"]:
                raise ValueError(f"seed mismatch for baseline {job['method']}: expected {job['seed']}, got {effective['seed']}")
            if effective["method"] != job["method"] or effective["model"]["model_name"] != job["backbone"]:
                raise ValueError(f"baseline configuration does not match its task: {job['method']}/{job['backbone']}")
            selected.append({"kind": "baseline", "dataset": job["dataset"], "backbone": job["backbone"],
                "variant": job["method"], "seed": job["seed"], "configuration": config,
                "data_sha256": source_by_dataset[job["dataset"]]})
    if not selected:
        raise ValueError("matrix selection is empty; requested seeds must already exist in the frozen matrix")
    for job in selected:
        job["id"] = f"{job['kind']}__{job['dataset']}__{job['backbone']}__{job['variant']}__seed{job['seed']}"
    return {"schema_version": 1, "matrix_path": str(path), "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seeds": sorted({job["seed"] for job in selected}), "seed_coverage_validated": True,
            "job_count": len(selected), "status": "planned_not_executed", "jobs": selected,
            "calendar_input_convention": "declared observed input columns; required mark interfaces receive zeros, optional marks remain absent",
            "normalization_convention": "training-prefix scaling plus each architecture's explicit normalization options",
            "comparison": "same data, horizon and label availability; architecture/capacity and required offline stages are recorded separately"}


def execute_plan(plan, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "plan.json", plan)
    records = []
    for job in plan["jobs"]:
        source = Path(job["configuration"]["data"]["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != job["data_sha256"]:
            raise ValueError(f"dataset changed since matrix creation: {job['dataset']}")
        result = dispatch(job["kind"], job["configuration"], output / job["id"])
        records.append({"id": job["id"], "result": result})
        _save(output / f"completed_{len(records):04d}.json", records[-1])
    _save(output / "results.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--matrix", type=Path)
    parser.add_argument("--kind", choices=("framework", "baseline"), default="framework")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--ablations", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES)
    parser.add_argument("--families", nargs="+", choices=("framework", "baseline"), default=["framework", "baseline"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.matrix:
        if args.checkpoint:
            parser.error("matrix jobs cannot share a single checkpoint")
        plan = plan_matrix(args.matrix, datasets=args.datasets, backbones=args.backbones, ablations=args.ablations,
                           methods=args.methods, families=args.families, seeds=args.seeds, device=args.device)
        if args.execute:
            execute_plan(plan, args.output)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            _save(args.output / "plan.json", plan)
        print(json.dumps({"jobs": plan["job_count"], "executed": args.execute, "output": str(args.output)}))
    else:
        if not args.config and not args.checkpoint:
            parser.error("specify --config, --checkpoint or --matrix")
        config = _load(args.config) if args.config else None
        if args.device:
            config = {} if config is None else config
            config["device"] = args.device
        result = dispatch(args.kind, config, args.output, args.checkpoint)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
