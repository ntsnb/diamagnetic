from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.online_loop import run_online_adaptation as run_level_stage3
from train.online_loop_stage3_delta import run_online_adaptation as run_delta_stage3
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml


STAGE_VARIANTS = {
    "stage3": (False, False, True),
    "stage1_stage2_stage3": (True, True, True),
}


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    existing_rows: list[dict] = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
            fieldnames = list(existing_fieldnames)
            for name in rows[0].keys():
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in rows:
                for name in row:
                    if name not in fieldnames:
                        fieldnames.append(name)
        rows = existing_rows + rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_optional(result: dict, key: str) -> str:
    value = result.get(key)
    if value is None or value != value:
        return ""
    return f"{value:.6f}"


def prepare_dataset_cfg(
    base_cfg: dict,
    ds: dict,
    run_id: str,
    epochs: int | None,
    invariant_epochs: int | None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if invariant_epochs is not None:
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


def run_online_variant(
    runner,
    signal_name: str,
    variant_name: str,
    base_model,
    invariant_bundle,
    base_cfg: dict,
    device: torch.device,
    test_ds,
    run_id: str,
    dataset_name: str,
) -> dict:
    s1, s2, s3 = STAGE_VARIANTS[variant_name]
    cur = copy.deepcopy(base_cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    log_dir = Path("logs/load_benchmarks/stage3_delta_compare")
    cur["online"]["compare_log_path"] = str(log_dir / f"{run_id}_{dataset_name}_{signal_name}_{variant_name}_trace.csv")
    cur["online"]["stage1_log_path"] = str(log_dir / f"{run_id}_{dataset_name}_{signal_name}_{variant_name}_stage1.csv")
    cur["online"]["stage2_log_path"] = str(log_dir / f"{run_id}_{dataset_name}_{signal_name}_{variant_name}_stage2.csv")
    seed = int(cur.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    stream_fn = build_stream(test_ds, cur["train"]["batch_size"], device, invariant_bundle)
    return runner(stream_fn(), copy.deepcopy(base_model).to(device), None, cur, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--results", default="logs/load_benchmarks/stage3_delta_compare/results.csv")
    parser.add_argument("--datasets", nargs="*", default=[ds["name"] for ds in DATASETS])
    parser.add_argument("--variants", nargs="*", default=["stage3", "stage1_stage2_stage3"])
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--invariant-epochs", type=int)
    args = parser.parse_args()

    selected_datasets = [ds for ds in DATASETS if ds["name"] in set(args.datasets)]
    if not selected_datasets:
        raise ValueError("no datasets selected")
    unknown_variants = set(args.variants) - set(STAGE_VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")

    base_cfg = load_yaml(args.config)
    rows: list[dict] = []
    for ds in selected_datasets:
        ds_cfg = prepare_dataset_cfg(
            base_cfg,
            ds,
            args.run_id,
            args.epochs,
            args.invariant_epochs,
        )
        offline_tag = f"stage3_delta_compare_{args.run_id}_{ds['name']}_offline"
        offline = run_single_experiment(
            ds_cfg,
            save_ckpt=False,
            variant_name=offline_tag,
            return_artifacts=True,
        )
        for variant_name in args.variants:
            for signal_name, runner in (
                ("level_phi_v_minus_phi_i", run_level_stage3),
                ("temporal_delta_phi_v_minus_phi_i", run_delta_stage3),
            ):
                result = run_online_variant(
                    runner,
                    signal_name,
                    variant_name,
                    offline["model"],
                    offline["invariant_bundle"],
                    ds_cfg,
                    offline["device"],
                    offline["test_ds"],
                    args.run_id,
                    ds["name"],
                )
                adapted_mse = result["online_stage3_mse"]
                adapted_mae = result["online_stage3_mae"]
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_id": args.run_id,
                    "dataset": ds["name"],
                    "variant": variant_name,
                    "stage3_signal": signal_name,
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
                    "gain_mse_vs_base": f"{result['online_base_mse'] - adapted_mse:.6f}",
                    "gain_mae_vs_base": f"{result['online_base_mae'] - adapted_mae:.6f}",
                    "stage3_level_mean": fmt_optional(result, "online_stage3_level_mean"),
                    "stage3_level_abs_mean": fmt_optional(result, "online_stage3_level_abs_mean"),
                    "stage3_delta_mean": fmt_optional(result, "online_stage3_delta_mean"),
                    "stage3_delta_abs_mean": fmt_optional(result, "online_stage3_delta_abs_mean"),
                    "stage3_delta_ratio": fmt_optional(result, "online_stage3_delta_ratio"),
                }
                rows.append(row)
                append_rows(args.results, [row])
                print(
                    f"[{ds['name']}:{variant_name}:{signal_name}] "
                    f"mse={adapted_mse:.6f} mae={adapted_mae:.6f} "
                    f"gain_mse={result['online_base_mse'] - adapted_mse:.6f}"
                )
    print(f"wrote {len(rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
