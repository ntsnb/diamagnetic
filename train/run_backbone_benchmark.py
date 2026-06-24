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

from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from train.run_experiment import run_single_experiment
from utils.config import load_yaml


BACKBONES = [
    "transformer",
    "itransformer",
    "patchtst",
    "informer",
    "autoformer",
    "fedformer",
    "pyraformer",
    "crossformer",
]

VARIANTS = [
    ("baseline", False, False, False),
    ("stage1", True, False, False),
    ("stage2", False, True, False),
    ("stage3", False, False, True),
    ("stage1_stage2_stage3", True, True, True),
]


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--results", default="logs/load_benchmarks/backbone_suite.csv")
    parser.add_argument("--backbones", nargs="*", default=BACKBONES)
    parser.add_argument("--variants", nargs="*", default=[name for name, *_ in VARIANTS])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    summary: dict[str, dict[str, float]] = {}
    fieldnames = [
        "timestamp",
        "backbone",
        "dataset",
        "variant",
        "offline_test_mse",
        "offline_test_mae",
        "online_base_mse",
        "adapted_online_mse",
        "improve_vs_base",
    ]

    variant_map = {name: (s1, s2, s3) for name, s1, s2, s3 in VARIANTS}
    selected_variants = []
    for name in args.variants:
        if name not in variant_map:
            raise ValueError(f"unknown variant: {name}")
        s1, s2, s3 = variant_map[name]
        selected_variants.append((name, s1, s2, s3))

    for backbone in args.backbones:
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        summary[backbone] = {}
        for ds in DATASETS:
            cfg = copy.deepcopy(base_cfg)
            cfg["train"]["epochs"] = args.epochs
            cfg["invariant"]["epochs"] = args.invariant_epochs
            cfg["model"]["temporal_encoder"] = backbone
            cfg["data"]["path"] = ds["path"]
            cfg["data"]["timestamp_col"] = ds["timestamp_col"]
            cfg["data"]["csv_encoding"] = ds.get("encoding")
            cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
            cfg["data"]["target_col"] = ds["target_col"]
            cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
            cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
            for variant_name, s1, s2, s3 in selected_variants:
                cur = copy.deepcopy(cfg)
                cur["online"]["stage1"]["enable"] = s1
                cur["online"]["stage2"]["enable"] = s2
                cur["online"]["stage3"]["enable"] = s3
                tag = f"{backbone}_{ds['name']}_{variant_name}"
                result = run_single_experiment(cur, save_ckpt=False, variant_name=tag)
                adapted = result["online_stage3_mse"] if s3 else result["online_stage2_mse"]
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "backbone": backbone,
                    "dataset": ds["name"],
                    "variant": variant_name,
                    "offline_test_mse": f"{result['offline_test_mse']:.6f}",
                    "offline_test_mae": f"{result.get('offline_test_mae', float('nan')):.6f}",
                    "online_base_mse": f"{result['online_base_mse']:.6f}",
                    "adapted_online_mse": f"{adapted:.6f}",
                    "improve_vs_base": f"{(result['online_base_mse'] - adapted):.6f}",
                }
                append_rows(args.results, [row])
                print(
                    f"[{backbone}:{ds['name']}:{variant_name}] "
                    f"offline={result['offline_test_mse']:.6f}/{result.get('offline_test_mae', float('nan')):.6f} "
                    f"online={adapted:.6f}"
                )
                if variant_name == "stage1_stage2_stage3":
                    summary[backbone][ds["name"]] = adapted

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
