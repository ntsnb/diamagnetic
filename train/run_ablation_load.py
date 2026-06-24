from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.readers import load_tabular_timeseries
from train.run_experiment import run_single_experiment
from utils.config import load_yaml


DATASETS = [
    {
        "name": "panama",
        "path": "dataset/load_benchmarks/panama.csv",
        "timestamp_col": "datetime",
        "target_col": "nat_demand",
        "encoding": "utf-8",
    },
    {
        "name": "tetouan",
        "path": "dataset/load_benchmarks/tetouan.csv",
        "timestamp_col": "DateTime",
        "target_col": "Zone 1 Power Consumption",
        "encoding": "utf-8",
    },
    {
        "name": "spain_energy_weather",
        "path": "dataset/load_benchmarks/spain_energy_weather.csv",
        "timestamp_col": "time",
        "target_col": "actual_load",
        "encoding": "utf-8",
        "impute_method": "time_interpolate",
    },
]

STAGE_VARIANTS = {
    "baseline": (False, False, False),
    "stage1": (True, False, False),
    "stage2": (False, True, False),
    "stage3": (False, False, True),
    "stage1_stage2": (True, True, False),
    "stage1_stage3": (True, False, True),
    "stage2_stage3": (False, True, True),
    "stage1_stage2_stage3": (True, True, True),
}


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def target_dim_for_dataset(ds: dict) -> int:
    _, df = load_tabular_timeseries(
        ds["path"],
        timestamp_col=ds["timestamp_col"],
        feature_cols=None,
        encoding=ds.get("encoding"),
        missing_sentinels=ds.get("missing_sentinels"),
        impute_method=ds.get("impute_method", "drop"),
    )
    feature_cols = [c for c in df.columns if c != ds["timestamp_col"]]
    return feature_cols.index(ds["target_col"])


def prepare_dataset_cfg(base_cfg: dict, ds: dict, epochs: int | None, invariant_epochs: int | None) -> dict:
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
    return cfg


def make_stage_cfg(cfg: dict, variant: str) -> dict:
    s1, s2, s3 = STAGE_VARIANTS[variant]
    cur = copy.deepcopy(cfg)
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    return cur


def make_component_cfg(cfg: dict, ablation: str) -> tuple[dict, str]:
    cur = copy.deepcopy(cfg)
    cur["online"]["stage1"]["enable"] = True
    cur["online"]["stage2"]["enable"] = True
    cur["online"]["stage3"]["enable"] = True
    label = ablation

    if ablation == "full_ref":
        return cur, label
    if ablation == "no_invariant":
        cur["invariant"]["enable"] = False
        return cur, label
    if ablation == "dense_stage2_all_layers":
        cur["online"]["stage2"]["use_sparse_mask"] = False
        cur["online"]["stage2"]["target_modules"] = []
        cur["online"]["stage2"]["target_modules_no_stage1"] = []
        return cur, label
    if ablation == "grad_stage2_no_lrp":
        cur["online"]["stage2"]["relevance_method"] = "grad"
        return cur, label
    if ablation == "mlp_gate_no_vae":
        cur["model"]["use_vae"] = False
        cur["train"]["beta_vae"] = 0.0
        cur["online"]["stage2"]["gate_by_recon"] = False
        cur["online"]["stage3"]["gate_by_recon"] = False
        return cur, label
    if ablation == "no_revin":
        cur["model"]["use_revin"] = False
        return cur, label
    if ablation == "no_invdelta_stage3":
        cur["online"]["stage3"]["use_invariant_delta"] = False
        return cur, label
    if ablation == "single_expert":
        cur["model"]["n_experts"] = 1
        return cur, label
    raise ValueError(f"unknown ablation: {ablation}")


def run_variant(cur: dict, dataset_name: str, label: str, group: str) -> dict:
    tag = f"{dataset_name}_{group}_{label}"
    result = run_single_experiment(cur, save_ckpt=False, variant_name=tag)
    adapted = result["online_stage3_mse"] if cur["online"]["stage3"]["enable"] else result["online_stage2_mse"]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "group": group,
        "variant": label,
        "offline_train_mse": f"{result['offline_train_mse']:.6f}",
        "offline_test_mse": f"{result['offline_test_mse']:.6f}",
        "offline_test_mae": f"{result.get('offline_test_mae', float('nan')):.6f}",
        "online_base_mse": f"{result['online_base_mse']:.6f}",
        "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
        "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
        "adapted_online_mse": f"{adapted:.6f}",
        "improve_vs_base": f"{result['online_base_mse'] - adapted:.6f}",
        "shift_score": f"{result.get('split_info', {}).get('shift_score', float('nan')):.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tuning/load_candidate_i_shifted.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--invariant-epochs", type=int, default=None)
    parser.add_argument("--results", default="logs/load_benchmarks/ablation_results.csv")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    rows: list[dict] = []

    component_variants = [
        "full_ref",
        "no_invariant",
        "dense_stage2_all_layers",
        "grad_stage2_no_lrp",
        "mlp_gate_no_vae",
        "no_revin",
        "no_invdelta_stage3",
        "single_expert",
    ]

    for ds in DATASETS:
        ds_cfg = prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs)

        for label in STAGE_VARIANTS:
            cur = make_stage_cfg(ds_cfg, label)
            row = run_variant(cur, ds["name"], label, "stage_combo")
            rows.append(row)
            print(f"[{ds['name']}:stage_combo:{label}] online={row['adapted_online_mse']} gain={row['improve_vs_base']}")

        for label in component_variants:
            cur, tag = make_component_cfg(ds_cfg, label)
            row = run_variant(cur, ds["name"], tag, "component")
            rows.append(row)
            print(f"[{ds['name']}:component:{tag}] online={row['adapted_online_mse']} gain={row['improve_vs_base']}")

    append_rows(args.results, rows)

    df = pd.DataFrame(rows)
    summary = (
        df.assign(adapted_online_mse=lambda x: x["adapted_online_mse"].astype(float))
        .groupby(["group", "variant"], as_index=False)["adapted_online_mse"]
        .mean()
        .sort_values(["group", "adapted_online_mse"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
