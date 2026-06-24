from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime

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

VARIANTS = [
    ("baseline", False, False, False),
    ("stage1", True, False, False),
    ("stage2", False, True, False),
    ("stage3", False, False, True),
    ("stage1_stage2_stage3", True, True, True),
]

VARIANT_MAP = {name: flags for name, *flags in VARIANTS}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_online.yaml")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--results", default="logs/load_benchmarks/results.csv")
    parser.add_argument("--variants", nargs="*", default=[v[0] for v in VARIANTS])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}
    for ds in DATASETS:
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["epochs"] = args.epochs
        cfg["invariant"]["epochs"] = args.invariant_epochs
        cfg["data"]["path"] = ds["path"]
        cfg["data"]["timestamp_col"] = ds["timestamp_col"]
        cfg["data"]["csv_encoding"] = ds.get("encoding")
        cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
        cfg["data"]["target_col"] = ds["target_col"]
        cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
        cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
        summary[ds["name"]] = {}
        for variant_name in args.variants:
            if variant_name not in VARIANT_MAP:
                raise ValueError(f"unknown variant: {variant_name}")
            s1, s2, s3 = VARIANT_MAP[variant_name]
            cur = copy.deepcopy(cfg)
            cur["online"]["stage1"]["enable"] = s1
            cur["online"]["stage2"]["enable"] = s2
            cur["online"]["stage3"]["enable"] = s3
            tag = f"{ds['name']}_{variant_name}"
            result = run_single_experiment(cur, save_ckpt=False, variant_name=tag)
            adapted = result["online_stage3_mse"] if s3 else result["online_stage2_mse"]
            overfit_gap = result["offline_test_mse"] - result["offline_train_mse"]
            improve_vs_base = result["online_base_mse"] - adapted
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dataset": ds["name"],
                    "variant": variant_name,
                    "offline_train_mse": f"{result['offline_train_mse']:.6f}",
                    "offline_train_mae": f"{result.get('offline_train_mae', float('nan')):.6f}",
                    "offline_val_mse": f"{result['offline_val_mse']:.6f}",
                    "offline_test_mse": f"{result['offline_test_mse']:.6f}",
                    "offline_test_mae": f"{result.get('offline_test_mae', float('nan')):.6f}",
                    "overfit_gap": f"{overfit_gap:.6f}",
                    "online_base_mse": f"{result['online_base_mse']:.6f}",
                    "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
                    "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
                    "adapted_online_mse": f"{adapted:.6f}",
                    "improve_vs_base": f"{improve_vs_base:.6f}",
                }
            )
            summary[ds["name"]][variant_name] = adapted
            print(
                f"[{ds['name']}:{variant_name}] "
                f"train={result['offline_train_mse']:.6f} "
                f"test={result['offline_test_mse']:.6f} "
                f"online={adapted:.6f} "
                f"gain={improve_vs_base:.6f}"
            )
    append_rows(args.results, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
