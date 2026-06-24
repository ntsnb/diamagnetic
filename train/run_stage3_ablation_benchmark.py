from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lishui_dataset import load_lishui_registry
from train.online_loop_stage3_ablation import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from train.stage3_signals import STAGE3_SIGNAL_MODES, VECTOR_STAGE3_MODES
from utils.config import load_yaml
from utils.seed import set_seed


STAGE_VARIANTS = {
    "stage3": (False, False, True),
    "stage1_stage2_stage3": (True, True, True),
}

SIGNAL_MODES = ["level_scalar", "delta_scalar", "level_vector", "delta_vector"]

TRACE_SIGNAL_SLUGS = {
    "level_scalar": "ls",
    "delta_scalar": "ds",
    "level_vector": "lv",
    "delta_vector": "dv",
}

TRACE_VARIANT_SLUGS = {
    "stage3": "s3",
    "stage1_stage2_stage3": "full",
}

TUNE_GRIDS = {
    "smoke": {
        "lr_scale": [0.0005],
        "max_delta": [0.02],
        "blend_alpha": [0.28],
        "max_pred_shift": [0.15],
    },
    "public": {
        "lr_scale": [0.0003, 0.0005, 0.0008],
        "max_delta": [0.015, 0.02, 0.03],
        "blend_alpha": [0.22, 0.28, 0.34],
        "max_pred_shift": [0.15],
    },
    "lishui": {
        "lr_scale": [0.0006, 0.0008, 0.001],
        "max_delta": [0.03, 0.04, 0.05],
        "blend_alpha": [0.32, 0.42, 0.50],
        "max_pred_shift": [0.22],
    },
}
TUNE_GRIDS["full"] = TUNE_GRIDS["public"]


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    existing_rows: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
        rows = existing_rows + rows
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_slug(value: str, max_chars: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    if not slug:
        slug = "x"
    return slug[:max_chars].rstrip("_") or slug[:max_chars]


def make_trace_path(
    *,
    trace_dir: Path,
    run_id: str,
    dataset_name: str,
    seed: int,
    split_name: str,
    variant_name: str,
    signal_mode: str,
    settings: dict,
) -> Path:
    payload = json.dumps(
        {
            "run_id": run_id,
            "dataset": dataset_name,
            "seed": seed,
            "split": split_name,
            "variant": variant_name,
            "signal": signal_mode,
            "settings": settings,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    filename = (
        f"{compact_slug(run_id, 18)}_"
        f"{compact_slug(dataset_name, 14)}_"
        f"s{seed}_"
        f"{compact_slug(split_name, 4)}_"
        f"{TRACE_VARIANT_SLUGS.get(variant_name, compact_slug(variant_name, 6))}_"
        f"{TRACE_SIGNAL_SLUGS.get(signal_mode, compact_slug(signal_mode, 6))}_"
        f"{digest}.csv"
    )
    return trace_dir / filename


def prepare_dataset_cfg(
    base_cfg: dict,
    ds: dict,
    run_id: str,
    seed: int,
    epochs: int,
    invariant_epochs: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["train"]["epochs"] = epochs
    cfg["invariant"]["epochs"] = invariant_epochs
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
    cfg["online_run"]["enable"] = False
    cfg["experiment_run_id"] = run_id
    return cfg


def build_stream(test_ds, batch_size: int, device: torch.device, invariant_bundle):
    online_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def generator():
        for x, y in online_dl:
            x = x.to(device)
            y = y.to(device)
            if invariant_bundle is None:
                yield x, y
                continue
            with torch.no_grad():
                phi_i = invariant_bundle["encoder_i"](x)
                phi_v = invariant_bundle["encoder_v"](x)
            yield x, y, phi_i, phi_v

    return generator


def fit_train_pca_projection(
    train_ds,
    batch_size: int,
    device: torch.device,
    invariant_bundle,
    output_dim: int,
    max_samples: int | None = None,
) -> list[list[float]] | None:
    if invariant_bundle is None:
        return None

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    drift_parts: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            phi_i = invariant_bundle["encoder_i"](x)
            phi_v = invariant_bundle["encoder_v"](x)
            drift = (phi_v - phi_i).detach().cpu()
            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                drift = drift[:remaining]
            drift_parts.append(drift)
            seen += int(drift.shape[0])
            if max_samples is not None and seen >= max_samples:
                break
    if not drift_parts:
        return None

    drift_matrix = torch.cat(drift_parts, dim=0).float()
    if drift_matrix.shape[0] < 2:
        return None
    centered = drift_matrix - drift_matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = min(output_dim, vh.shape[0])
    projection = vh[:keep].T.contiguous()
    if keep < output_dim:
        pad = torch.zeros(projection.shape[0], output_dim - keep)
        projection = torch.cat([projection, pad], dim=1)
    return projection.tolist()


def projection_for_mode(signal_mode: str, projection: list[list[float]] | None) -> list[list[float]] | None:
    if signal_mode in VECTOR_STAGE3_MODES:
        return projection
    return None


def tune_grid(name: str) -> list[dict]:
    grid = TUNE_GRIDS[name]
    keys = list(grid.keys())
    rows = [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]
    if name == "lishui":
        for row in rows:
            row["blend_alpha_min_delta"] = 0.10
            row["blend_alpha_max_delta"] = 0.08
    return rows


def apply_stage3_settings(cfg: dict, signal_mode: str, settings: dict) -> None:
    cfg["online"]["stage3"]["signal_mode"] = signal_mode
    cfg["online"]["stage3"]["lr_scale"] = float(settings["lr_scale"])
    cfg["online"]["stage3"]["max_delta"] = float(settings["max_delta"])
    cfg["online"]["stage3"]["blend_alpha"] = float(settings["blend_alpha"])
    alpha = float(settings["blend_alpha"])
    cfg["online"]["stage3"]["blend_alpha_min"] = max(0.0, alpha - float(settings.get("blend_alpha_min_delta", 0.08)))
    cfg["online"]["stage3"]["blend_alpha_max"] = alpha + float(settings.get("blend_alpha_max_delta", 0.06))
    cfg["online"]["stage3"]["max_pred_shift"] = float(settings["max_pred_shift"])


def run_online_once(
    *,
    base_cfg: dict,
    offline_model,
    invariant_bundle,
    dataset,
    device: torch.device,
    variant_name: str,
    signal_mode: str,
    settings: dict,
    run_id: str,
    dataset_name: str,
    seed: int,
    split_name: str,
    trace_dir: Path,
    vector_projection: list[list[float]] | None = None,
) -> tuple[dict, Path]:
    s1, s2, s3 = STAGE_VARIANTS[variant_name]
    cfg = copy.deepcopy(base_cfg)
    cfg["online_run"]["enable"] = True
    cfg["online"]["stage1"]["enable"] = s1
    cfg["online"]["stage2"]["enable"] = s2
    cfg["online"]["stage3"]["enable"] = s3
    apply_stage3_settings(cfg, signal_mode, settings)
    projection = projection_for_mode(signal_mode, vector_projection)
    if projection is not None:
        cfg["online"]["stage3"]["vector_projection"] = projection
        cfg["online"]["stage3"]["projection_method"] = "train_pca_to_lrp_bottleneck"
    else:
        cfg["online"]["stage3"].pop("vector_projection", None)
        cfg["online"]["stage3"]["projection_method"] = "none"

    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = make_trace_path(
        trace_dir=trace_dir,
        run_id=run_id,
        dataset_name=dataset_name,
        seed=seed,
        split_name=split_name,
        variant_name=variant_name,
        signal_mode=signal_mode,
        settings=settings,
    )
    cfg["online"]["compare_log_path"] = str(trace_path)
    cfg["online"]["stage1_log_path"] = str(trace_path.with_name(trace_path.stem + "_stage1.csv"))
    cfg["online"]["stage2_log_path"] = str(trace_path.with_name(trace_path.stem + "_stage2.csv"))

    set_seed(seed)
    stream_fn = build_stream(dataset, cfg["train"]["batch_size"], device, invariant_bundle)
    result = run_online_adaptation(
        stream_fn(),
        copy.deepcopy(offline_model).to(device),
        None,
        cfg,
        device,
    )
    return result, trace_path


def result_row(
    *,
    timestamp: str,
    run_id: str,
    config_path: str,
    config_anchor: str,
    dataset_family: str,
    dataset_name: str,
    seed: int,
    split_name: str,
    variant_name: str,
    signal_mode: str,
    settings: dict,
    result: dict,
    offline: dict,
    trace_path: Path,
) -> dict:
    adapted_mse = result["online_stage3_mse"]
    adapted_mae = result["online_stage3_mae"]
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "script": "train/run_stage3_ablation_benchmark.py",
        "config": config_path,
        "config_anchor": config_anchor,
        "dataset": dataset_name,
        "dataset_family": dataset_family,
        "seed": seed,
        "split": split_name,
        "variant": variant_name,
        "stage3_signal_mode": signal_mode,
        "signal_level_or_delta": "delta" if signal_mode.startswith("delta_") else "level",
        "signal_scalar_or_vector": "vector" if signal_mode.endswith("_vector") else "scalar",
        "projection_method": "train_pca_to_lrp_bottleneck" if signal_mode in VECTOR_STAGE3_MODES else "none",
        "stage3_lr_scale": f"{settings['lr_scale']:.8f}",
        "stage3_lr_decay": f"{offline.get('stage3_lr_decay', float('nan')):.8f}",
        "stage3_max_delta": f"{settings['max_delta']:.8f}",
        "stage3_blend_alpha": f"{settings['blend_alpha']:.8f}",
        "stage3_blend_alpha_min": f"{max(0.0, float(settings['blend_alpha']) - float(settings.get('blend_alpha_min_delta', 0.08))):.8f}",
        "stage3_blend_alpha_max": f"{float(settings['blend_alpha']) + float(settings.get('blend_alpha_max_delta', 0.06)):.8f}",
        "stage3_max_pred_shift": f"{settings['max_pred_shift']:.8f}",
        "split_mode": offline.get("split_mode", ""),
        "train_ratio": offline.get("train_ratio", ""),
        "val_ratio": offline.get("val_ratio", ""),
        "online_test_ratio": offline.get("online_test_ratio", ""),
        "shift_pool_ratio": offline.get("shift_pool_ratio", ""),
        "split_info_json": offline.get("split_info_json", ""),
        "offline_train_mse": f"{offline['offline_train_mse']:.6f}",
        "offline_train_mae": f"{offline['offline_train_mae']:.6f}",
        "offline_val_mse": f"{offline['offline_val_mse']:.6f}",
        "offline_test_mse": f"{offline['offline_test_mse']:.6f}",
        "offline_test_mae": f"{offline['offline_test_mae']:.6f}",
        "online_base_mse": f"{result['online_base_mse']:.6f}",
        "online_base_mae": f"{result['online_base_mae']:.6f}",
        "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
        "online_stage2_mae": f"{result['online_stage2_mae']:.6f}",
        "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
        "online_stage3_mae": f"{result['online_stage3_mae']:.6f}",
        "adapted_online_mse": f"{adapted_mse:.6f}",
        "adapted_online_mae": f"{adapted_mae:.6f}",
        "gain_mse_vs_base": f"{(result['online_base_mse'] - adapted_mse):.6f}",
        "gain_mae_vs_base": f"{(result['online_base_mae'] - adapted_mae):.6f}",
        "stage3_use_rate": f"{result.get('online_stage3_use_rate', float('nan')):.6f}",
        "stage3_blend_mean": f"{result.get('online_stage3_blend', float('nan')):.6f}",
        "stage3_pred_shift_mean": f"{result.get('online_stage3_pred_shift', float('nan')):.6f}",
        "stage3_recon_error_mean": f"{result.get('online_stage3_recon_error', float('nan')):.6f}",
        "stage3_drift_dim": f"{result.get('online_stage3_drift_dim', float('nan')):.6f}",
        "stage3_signal_dim": f"{result.get('online_stage3_signal_dim', float('nan')):.6f}",
        "stage3_raw_abs_mean": f"{result.get('online_stage3_raw_abs_mean', float('nan')):.6f}",
        "stage3_signal_abs_mean": f"{result.get('online_stage3_signal_abs_mean', float('nan')):.6f}",
        "stage3_signal_norm": f"{result.get('online_stage3_signal_norm', float('nan')):.6f}",
        "stage3_scalar_ratio": f"{result.get('online_stage3_scalar_ratio', float('nan')):.6f}",
        "trace_path": str(trace_path),
    }


def pick_best(rows: list[dict]) -> dict:
    return min(
        rows,
        key=lambda row: (
            float(row["adapted_online_mse"]),
            float(row["adapted_online_mae"]),
        ),
    )


def selected_dataset_records(args) -> list[dict]:
    wanted = set(args.datasets)
    include_all = not wanted or "all" in wanted
    records: list[dict] = []
    if args.dataset_family in {"public", "all"}:
        for ds in DATASETS:
            if include_all or ds["name"] in wanted:
                records.append({"family": "public", "config_path": args.config, "dataset": ds})
    if args.dataset_family in {"lishui", "all"}:
        for ds in load_lishui_registry(args.lishui_prepared_dir):
            if include_all or ds["name"] in wanted:
                records.append({"family": "lishui", "config_path": args.lishui_config, "dataset": ds})
    return records


def grid_name_for_family(requested_grid: str, family: str) -> str:
    if requested_grid != "auto":
        return requested_grid
    return "lishui" if family == "lishui" else "public"


def settings_from_row(row: dict) -> dict:
    return {
        "lr_scale": float(row["stage3_lr_scale"]),
        "max_delta": float(row["stage3_max_delta"]),
        "blend_alpha": float(row["stage3_blend_alpha"]),
        "max_pred_shift": float(row["stage3_max_pred_shift"]),
        "blend_alpha_min_delta": float(row["stage3_blend_alpha"]) - float(row["stage3_blend_alpha_min"]),
        "blend_alpha_max_delta": float(row["stage3_blend_alpha_max"]) - float(row["stage3_blend_alpha"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--lishui-config", default="configs/tuning/lishui_candidate_m_stage3strong.yaml")
    parser.add_argument("--dataset-family", choices=["public", "lishui", "all"], default="public")
    parser.add_argument("--lishui-prepared-dir", default="dataset/load_benchmarks/lishui")
    parser.add_argument("--datasets", nargs="*", default=["all"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--invariant-epochs", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=["stage3", "stage1_stage2_stage3"])
    parser.add_argument("--signal-modes", nargs="*", default=SIGNAL_MODES)
    parser.add_argument("--tune-grid", choices=sorted([*TUNE_GRIDS, "auto"]), default="auto")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--out-dir", default="logs/load_benchmarks/stage3_ablation")
    args = parser.parse_args()

    unknown_variants = set(args.variants) - set(STAGE_VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    unknown_modes = set(args.signal_modes) - STAGE3_SIGNAL_MODES
    if unknown_modes:
        raise ValueError(f"unknown signal modes: {sorted(unknown_modes)}")

    selected_records = selected_dataset_records(args)
    if not selected_records:
        raise ValueError("no datasets selected")

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    tuning_path = out_dir / f"{args.run_id}_tuning.csv"
    test_path = out_dir / f"{args.run_id}_test.csv"
    summary_path = out_dir / f"{args.run_id}_summary.json"
    test_rows: list[dict] = []
    summary: dict[str, dict] = {
        "run_id": args.run_id,
        "config": args.config,
        "lishui_config": args.lishui_config,
        "dataset_family": args.dataset_family,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "invariant_epochs": args.invariant_epochs,
        "variants": args.variants,
        "signal_modes": args.signal_modes,
        "tune_grid": args.tune_grid,
        "test_results": [],
    }

    for record in selected_records:
        family = record["family"]
        ds = record["dataset"]
        config_path = record["config_path"]
        base_cfg = load_yaml(config_path)
        grid_name = grid_name_for_family(args.tune_grid, family)
        grid = tune_grid(grid_name)
        summary.setdefault("grid_sizes", {})[family] = len(grid)
        for seed in args.seeds:
            ds_cfg = prepare_dataset_cfg(
                base_cfg,
                ds,
                args.run_id,
                seed,
                args.epochs,
                args.invariant_epochs,
            )
            offline_tag = f"stage3_ablation_{args.run_id}_{ds['name']}_seed{seed}_offline"
            offline = run_single_experiment(
                ds_cfg,
                save_ckpt=False,
                variant_name=offline_tag,
                return_artifacts=True,
            )
            split_info = offline.get("split_info", {})
            offline.update(
                {
                    "stage3_lr_decay": ds_cfg["online"]["stage3"].get("lr_decay", float("nan")),
                    "split_mode": split_info.get("split_mode", ds_cfg["data"].get("split_mode", "")),
                    "train_ratio": ds_cfg["data"].get("train_ratio", ""),
                    "val_ratio": ds_cfg["data"].get("val_ratio", ""),
                    "online_test_ratio": ds_cfg["data"].get("online_test_ratio", ""),
                    "shift_pool_ratio": ds_cfg["data"].get("shift_pool_ratio", ""),
                    "split_info_json": json.dumps(split_info, ensure_ascii=False),
                }
            )
            vector_projection = fit_train_pca_projection(
                offline["train_ds"],
                ds_cfg["train"]["batch_size"],
                offline["device"],
                offline["invariant_bundle"],
                output_dim=int(ds_cfg["online"]["stage2"]["lrp_bottleneck"]),
            )
            for variant_name in args.variants:
                for signal_mode in args.signal_modes:
                    validation_rows: list[dict] = []
                    for settings in grid:
                        result, trace_path = run_online_once(
                            base_cfg=ds_cfg,
                            offline_model=offline["model"],
                            invariant_bundle=offline["invariant_bundle"],
                            dataset=offline["val_ds"],
                            device=offline["device"],
                            variant_name=variant_name,
                            signal_mode=signal_mode,
                            settings=settings,
                            run_id=args.run_id,
                            dataset_name=ds["name"],
                            seed=seed,
                            split_name="val",
                            trace_dir=trace_dir,
                            vector_projection=vector_projection,
                        )
                        row = result_row(
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                            run_id=args.run_id,
                            config_path=config_path,
                            config_anchor=grid_name,
                            dataset_family=family,
                            dataset_name=ds["name"],
                            seed=seed,
                            split_name="val",
                            variant_name=variant_name,
                            signal_mode=signal_mode,
                            settings=settings,
                            result=result,
                            offline=offline,
                            trace_path=trace_path,
                        )
                        validation_rows.append(row)
                        append_rows(tuning_path, [row])
                    best = pick_best(validation_rows)
                    best_settings = settings_from_row(best)
                    test_result, test_trace_path = run_online_once(
                        base_cfg=ds_cfg,
                        offline_model=offline["model"],
                        invariant_bundle=offline["invariant_bundle"],
                        dataset=offline["test_ds"],
                        device=offline["device"],
                        variant_name=variant_name,
                        signal_mode=signal_mode,
                        settings=best_settings,
                        run_id=args.run_id,
                        dataset_name=ds["name"],
                        seed=seed,
                        split_name="test",
                        trace_dir=trace_dir,
                        vector_projection=vector_projection,
                    )
                    test_row = result_row(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        run_id=args.run_id,
                        config_path=config_path,
                        config_anchor=grid_name,
                        dataset_family=family,
                        dataset_name=ds["name"],
                        seed=seed,
                        split_name="test",
                        variant_name=variant_name,
                        signal_mode=signal_mode,
                        settings=best_settings,
                        result=test_result,
                        offline=offline,
                        trace_path=test_trace_path,
                    )
                    test_row["selected_val_mse"] = best["adapted_online_mse"]
                    test_row["selected_val_mae"] = best["adapted_online_mae"]
                    append_rows(test_path, [test_row])
                    test_rows.append(test_row)
                    summary["test_results"].append(test_row)
                    print(
                        f"[{ds['name']}:seed{seed}:{variant_name}:{signal_mode}] "
                        f"val={best['adapted_online_mse']} "
                        f"test_mse={test_row['adapted_online_mse']} "
                        f"test_mae={test_row['adapted_online_mae']} "
                        f"gain={test_row['gain_mse_vs_base']}"
                    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote tuning rows to {tuning_path}")
    print(f"wrote test rows to {test_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
