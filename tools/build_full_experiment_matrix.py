"""生成可审计的完整数据实验配置；只写配置，不启动训练。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_impl.evaluation import stage_variants, selection_variants
from paper_impl.official_backbones import OFFICIAL_MODEL_NAMES, OfficialForecastConfig
from paper_impl.run import _normalize
from paper_impl.experiment import baseline_configuration


STRUCTURAL_ABLATIONS = {
    "full": {"description": "Full three-stage method; all eight stage combinations share this trained checkpoint.",
             "overrides": {}, "variants": list(stage_variants()), "native_only": False},
    "noSpec": {"description": "Remove differentiable expert-specialization loss only; keep expert count, VAE and three stages.",
               "overrides": {"training": {"specialization_weight": 0.0}}, "variants": ["stages_111"], "native_only": False},
    "noVAE": {"description": "Remove mu/log_var/decoder and replace latent sampling by deterministic MLP gate features; reconstruction and KL are absent.",
              "overrides": {"model": {"use_vae": False}}, "variants": ["stages_111"], "native_only": False},
    "Single": {"description": "Exactly one expert and a one-output router; expert-pair specialization is identically zero.",
               "overrides": {"model": {"n_experts": 1}}, "variants": ["stages_111"], "native_only": False},
    "woInvariantDisabled": {"description": "Remove BOTH auxiliary forecast branches and all generator parameters; Stage III has no contrast and is disabled.",
                            "overrides": {"foil": {"mode": "disabled"}, "generator": {"mode": "identity"},
                                          "online": {"stage_iii": False}}, "variants": ["stages_110"], "native_only": False},
    "ERMpair": {"description": "Keep two independently initialized/trained auxiliary branches; replace FOIL objectives/environment inference by ordinary MSE; contrast is variant minus invariant.",
                "overrides": {"foil": {"mode": "erm_pair"}}, "variants": ["stages_111"], "native_only": False},
    "VariantOnly": {"description": "Remove invariant predictor; train one variant predictor by ERM; contrast is variant forecast relative to a fixed zero anchor.",
                    "overrides": {"foil": {"mode": "variant_only"}}, "variants": ["stages_111"], "native_only": False},
    "woRevIN": {"description": "Disable native RevIN, or only the explicitly supported PatchTST/iTransformer author normalization; no no-op configuration for other backbones.",
                "overrides": {"model": {"use_revin": False}}, "variants": ["stages_111"], "native_only": False,
                "applicable_backbones": ["native", "patchtst", "itransformer"]},
    "woContrast": {"description": "Keep and train BOTH auxiliary branches but use variant forecast instead of variant-minus-invariant as generator input; capacity-controlled contrast ablation, not the legacy latent-z proxy.",
                   "overrides": {"foil": {"contrast": "variant"}}, "variants": ["stages_111"], "native_only": False},
    "DenseStageII": {"description": "Stage II can update all forecaster parameters; other release and acceptance rules remain explicit.",
                     "overrides": {"online": {"stage_ii_scope": "all_forecaster"}}, "variants": ["stages_111"], "native_only": False},
    "DenseExperts": {"description": "Stage II updates all expert parameters without sparse channel selection; router/head remain outside this scope.",
                     "overrides": {"online": {"stage_ii_scope": "dense_experts"}}, "variants": ["stages_111"], "native_only": False},
    "SyncAllExperts": {"description": "Stage II jointly updates experts, router and forecast head; it does not remove forecast-label delay.",
                       "overrides": {"online": {"stage_ii_scope": "experts_router_head"}}, "variants": ["stages_111"], "native_only": False},
    "NoEMA_current": {"description": "Recompute current and historical relevance under a current-model snapshot at each scoring event; do not use an EMA reference.",
                      "overrides": {"online": {"reference_mode": "current"}}, "variants": ["stages_111"], "native_only": False},
    "ReferenceFrozen": {"description": "Keep the initial common reference fixed; differs from current-snapshot NoEMA.",
                        "overrides": {"online": {"reference_mode": "frozen"}}, "variants": ["stages_111"], "native_only": False},
    "woAcceptanceCheck": {"description": "Disable Armijo/worse-loss rejection only; continue rejecting nonfinite proposals; no nonexistent acceptance buffer is removed.",
                          "overrides": {"online": {"rollback_on_worse": False}}, "variants": ["stages_111"], "native_only": False},
    "NoExtraDelay": {"description": "Set the EXTRA label delay to zero; complete horizon labels still become available at issue+H.",
                     "overrides": {"data": {"extra_delay": 0}, "online": {"extra_delay": 0}},
                     "variants": ["stages_111"], "native_only": False},
}

BASELINES = [
    {"name": "Frozen", "class": "Frozen", "scope": "one offline-trained complete official backbone; no online parameter updates"},
    {"name": "GD", "class": "GradientDescent", "scope": "all persistent forecaster parameters"},
    {"name": "Head-GD", "class": "HeadGradientDescent", "scope": "author prediction-head parameters only"},
    {"name": "SOLID", "class": "SOLID", "scope": "retrieve released historical samples; reset and adapt head per forecast"},
    {"name": "SOLID++", "class": "SOLIDPlusPlus", "scope": "continual retrieval updates of all forecaster parameters"},
    {"name": "OneNet", "class": "OneNet", "scope": "complete target backbone plus official channel FSNet; global weights and decision bias",
     "onenet_mode": "onlinetsf_cross_backbone", "secondary_backbone": "fsnet_channel",
     "combination_note": "OnlineTSF cross-backbone protocol; other explicit two-backbone combinations are named variants"},
    {"name": "PROCEED", "class": "PROCEED", "scope": "author concept encoders and modulation pretraining; frozen online generator"},
]


def apply_structural_ablation(configuration: dict, name: str) -> dict:
    """返回显式结构配置；绝不把移除模块退化为某个损失权重设零。"""
    if name not in STRUCTURAL_ABLATIONS:
        raise ValueError(f"unknown structural ablation: {name}")
    spec = STRUCTURAL_ABLATIONS[name]
    backbone = configuration.get("model", {}).get("expert_backbone", "native")
    if backbone not in spec.get("applicable_backbones", [backbone]):
        raise ValueError(f"{name} is not a defined ablation for {backbone}")
    result = copy.deepcopy(configuration)
    for section, patch in spec["overrides"].items():
        result.setdefault(section, {}).update(patch)
    if name == "woRevIN" and backbone != "native":
        option = "patch_revin" if backbone == "patchtst" else "itransformer_use_norm"
        result["model"].setdefault("official_options", {})[option] = False
    result["variants"] = list(spec["variants"])
    result["selection_comparisons"] = list(selection_variants()) if name == "full" else []
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_inventory(dataset_root: Path, lishui_registry: Path, spain_source: Path) -> list[dict]:
    """显式真实数据清单；不把候选市级和旧插补九县重复纳入。"""
    dataset_root, lishui_registry, spain_source = (Path(p).resolve() for p in (dataset_root, lishui_registry, spain_source))
    existing = [
        ("panama", dataset_root / "panama.csv", "datetime", ["nat_demand"], 3600),
        # Preserve the manuscript's Zone 1 target; other zones remain available inputs.
        ("tetouan", dataset_root / "tetouan.csv", "DateTime", ["Zone 1 Power Consumption"], 600),
        ("spain", spain_source, "time", ["actual_load"], 3600),
    ]
    inventory = []
    for name, path, timestamp, targets, interval in existing:
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory.append({"id": name, "source_sha256": sha256(path), "path": path.as_posix(),
                          "identity": "existing_prepared_real_load_dataset",
                          "input_data": {"path": path.as_posix(), "timestamp_column": timestamp,
                                         "target_columns": targets, "feature_columns": None,
                                         "expected_interval_seconds": interval},
                          "scope_note": "retain the existing data audit; source preparation is not newly certified as raw observations"})
    registry = json.loads(lishui_registry.read_text(encoding="utf8"))
    if registry.get("schema_version") != 2 or registry.get("independent_task_count") != 10:
        raise ValueError("the confirmed ten-task Lishui v2 registry is required")
    if "user_confirmed" not in registry.get("city_identity_status", ""):
        raise ValueError("city source must be explicitly confirmed by the user")
    for task in registry["tasks"]:
        path = lishui_registry.parent / task["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        config = {key: copy.deepcopy(task[key]) for key in
                  ("timestamp_column", "target_columns", "feature_columns", "target_input_columns",
                   "target_mask_columns", "expected_interval_seconds")}
        config["path"] = path.resolve().as_posix()
        inventory.append({"id": task["entity_id"], "identity": task["scope"], "path": path.resolve().as_posix(),
                          "source_sha256": sha256(path), "raw_source_sha256": task["source_sha256"],
                          "unit": task["unit"], "first_timestamp": task["first_timestamp"],
                          "last_timestamp": task["last_timestamp"], "input_data": config,
                          "scope_note": "independent task chronology; load is never imputed as an evaluation target"})
    ids = [item["id"] for item in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset identities must be unique")
    return inventory


def base_configuration(dataset: dict, backbone: str, seed: int, *, extra_delay: int = 2) -> dict:
    config = {
        "seed": seed, "device": "auto", "cpu_threads": 1,
        "data": {**copy.deepcopy(dataset["input_data"]), "lookback": 24, "horizon": 12,
                 "train_fraction": .55, "validation_fraction": .10, "max_rows": None,
                 "protocol": "chronological", "extra_delay": extra_delay},
        "model": {"expert_backbone": backbone, "encoder": "temporal_transformer", "n_experts": 3,
                  "hidden": 64, "heads": 8, "depth": 2, "latent_block": 8,
                  "use_vae": True, "use_revin": backbone == "native", "official_sampling_seed": seed},
        "foil": {"mode": "foil", "contrast": "difference", "n_environments": 3, "rounds": 2, "mil_epochs": 2,
                 "mei_epochs": 1, "batch_size": 128, "seed": seed},
        "training": {"epochs": 5, "batch_size": 32, "learning_rate": .001,
                     "specialization_weight": .05, "seed": seed},
        "generator": {"mode": "centered_bounded", "hidden_dim": 16, "epsilon": .1},
        "online": {"stage_i": True, "stage_ii": True, "stage_iii": True,
                   "reference_windows": 4, "reference_interval": 16, "history_capacity": 64,
                   "extra_delay": extra_delay, "channel_budget": 1, "selection_seed": seed},
        "expert_diagnostics": {"enabled": True, "distance_interval": 1},
        "replay": {"episodes": 2, "max_windows": None}, "test_windows": None,
        "variants": list(stage_variants()),
    }
    if backbone != "native":
        # Author architectures remain whole; these are explicit, fixed size choices.
        config["model"]["official_options"] = {"dropout": .1, "moving_avg": 25, "patch_len": 16,
                                               "patch_stride": 8, "crossformer_seg_len": 6,
                                               "fedformer_modes": 12}
        OfficialForecastConfig(model_name=backbone, lookback=24, horizon=12, n_variables=8,
                               hidden=64, heads=8, depth=2, **config["model"]["official_options"])
    return config


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def generate(output: Path, inventory: list[dict], *, seeds: list[int] | None = None,
             backbones: list[str] | None = None, extra_delay: int = 2) -> dict:
    """写训练配置与基线计划清单；全阶段组合复用同一结构的离线 checkpoint。"""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("matrix output must be new or empty")
    seeds = [42] if seeds is None else seeds
    backbones = [*OFFICIAL_MODEL_NAMES, "native"] if backbones is None else backbones
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be a nonempty unique list of integers in [0, 2**32)")
    if not backbones or len(set(backbones)) != len(backbones) or set(backbones) - {*OFFICIAL_MODEL_NAMES, "native"}:
        raise ValueError("use complete official backbones or native, never legacy *_style aliases")
    jobs, baseline_jobs = [], []
    for dataset in inventory:
        for backbone in backbones:
            for seed in seeds:
                base = base_configuration(dataset, backbone, seed, extra_delay=extra_delay)
                for name, specification in STRUCTURAL_ABLATIONS.items():
                    if backbone not in specification.get("applicable_backbones", [backbone]):
                        continue
                    config = apply_structural_ablation(base, name)
                    # Parse all current contract fields before writing a runnable file.
                    _normalize(config)
                    relative = Path(dataset["id"]) / backbone / f"{name}_seed{seed}.json"
                    _write(output / relative, config)
                    jobs.append({"dataset": dataset["id"], "backbone": backbone, "seed": seed,
                                 "ablation": name, "config": relative.as_posix(),
                                 "config_sha256": sha256(output / relative),
                                 "evaluation_variants": config["variants"] + config["selection_comparisons"],
                                 "status": "configured_not_run", "training_required": True,
                                 "entrypoint": "python -m paper_impl.run --config <config> --output <new_directory>"})
                if backbone != "native":
                    for baseline in BASELINES:
                        config = baseline_configuration(base, backbone=backbone, method=baseline["name"])
                        relative = Path("baselines") / dataset["id"] / backbone / f"{baseline['name']}_seed{seed}.json"
                        _write(output / relative, config)
                        job = {"dataset": dataset["id"], "backbone": backbone, "seed": seed,
                               "method": baseline["name"], "adapter_class": baseline["class"],
                               "config": relative.as_posix(), "config_sha256": sha256(output / relative),
                               "offline_forecaster": "one complete author forecaster (not the MoE)",
                               "protocol": "same dataset chronology, lookback, horizon, observed labels and extra delay",
                               "extra_delay": extra_delay, "status": "configured_not_run",
                               "dispatch_note": "runnable baseline configuration; dispatch through paper_impl.baseline_run, not paper_impl.run"}
                        if baseline["name"] == "OneNet":
                            job.update({"offline_forecaster": "complete target author backbone plus independent official channel FSNet",
                                        "onenet_mode": "onlinetsf_cross_backbone", "secondary_backbone": "fsnet_channel",
                                        "secondary_adapter": "paper_impl.official_fsnet.OfficialFSNet"})
                        baseline_jobs.append(job)
    provenance = {}
    for name in ("official_forecasters", "official_online"):
        path = ROOT / "third_party" / name / "SOURCE_MANIFEST.json"
        provenance[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path) if path.exists() else None}
    manifest = {
        "schema_version": 1, "status": "configuration_matrix_only_no_training_started",
        "dataset_count": len(inventory), "datasets": inventory, "backbones": backbones, "seeds": seeds,
        "structural_definitions": STRUCTURAL_ABLATIONS,
        "stage_bit_order": ["stage_i_router", "stage_ii_expert_weights", "stage_iii_temporary_modulation"],
        "stage_variants": {key: list(value) for key, value in stage_variants().items()},
        "selection_comparisons": selection_variants(),
        "expert_diagnostics": {"enabled": True, "distance_interval": 1,
                               "scope": "routing and persistent expert predictions at every issue; ungated persistent LRP distance"},
        "jobs": jobs, "job_count": len(jobs), "evaluation_count": sum(len(job["evaluation_variants"]) for job in jobs),
        "baseline_inventory": BASELINES, "baseline_jobs": baseline_jobs, "baseline_job_count": len(baseline_jobs),
        "source_manifests": provenance, "builder_sha256": sha256(Path(__file__)),
        "budget_status": "fixed reproducible starting budget, not a tuned or completed publication experiment",
        "data_scope": "full chronological datasets; no max_rows, replay max_windows, or test_windows truncation",
        "comparison_caveats": ["different structures have different capacities and optimizer paths",
                               "ERMpair is the two-branch objective control; disabled removes both branches and Stage III",
                               f"{len(seeds)} configured seed(s); configurations are not completed multi-seed experiments",
                               "L/H/delay count observations; Tetouan is 10-minute and other tasks are hourly",
                               "Lishui physical units remain null; do not aggregate physical errors across tasks",
                               "native RevIN does not replace the official authors' model-specific normalization"],
    }
    _write(output / "matrix.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lishui-registry", type=Path, default=ROOT / "data/lishui_online/v2/registry.json")
    parser.add_argument("--spain-source", type=Path, default=ROOT / "data/paper_impl_prepared/spain_energy_weather_clock_corrected.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/full_experiment/structural")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--backbones", nargs="+", default=[*OFFICIAL_MODEL_NAMES, "native"])
    parser.add_argument("--extra-delay", type=int, default=2)
    args = parser.parse_args()
    inventory = dataset_inventory(args.dataset_root, args.lishui_registry, args.spain_source)
    matrix = generate(args.output, inventory, seeds=args.seeds, backbones=args.backbones, extra_delay=args.extra_delay)
    print(json.dumps({"output": args.output.resolve().as_posix(), "datasets": matrix["dataset_count"],
                      "training_configs": matrix["job_count"], "stage_evaluations": matrix["evaluation_count"],
                      "baseline_jobs": matrix["baseline_job_count"], "status": matrix["status"]}))


if __name__ == "__main__":
    main()
