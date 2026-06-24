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

from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--invariant-epochs", type=int)
    parser.add_argument("--results", default="logs/load_benchmarks/backbone_suite_v2_with_mae.csv")
    parser.add_argument("--backbones", nargs="*", default=BACKBONES)
    parser.add_argument("--variants", nargs="*", default=[name for name, *_ in VARIANTS])
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    summary: dict[str, dict[str, float]] = {}
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
            if args.epochs is not None:
                cfg["train"]["epochs"] = args.epochs
            if args.invariant_epochs is not None:
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
                tag = f"{args.run_id}_{backbone}_{ds['name']}_{variant_name}_with_mae"
                result = run_single_experiment(cur, save_ckpt=False, variant_name=tag)
                adapted_mse = result["online_stage3_mse"] if s3 else result["online_stage2_mse"]
                adapted_mae = result["online_stage3_mae"] if s3 else result["online_stage2_mae"]
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_id": args.run_id,
                    "backbone": backbone,
                    "dataset": ds["name"],
                    "variant": variant_name,
                    "offline_test_mse": f"{result['offline_test_mse']:.6f}",
                    "offline_test_mae": f"{result.get('offline_test_mae', float('nan')):.6f}",
                    "online_base_mse": f"{result['online_base_mse']:.6f}",
                    "online_base_mae": f"{result['online_base_mae']:.6f}",
                    "adapted_online_mse": f"{adapted_mse:.6f}",
                    "adapted_online_mae": f"{adapted_mae:.6f}",
                    "improve_mse_vs_base": f"{(result['online_base_mse'] - adapted_mse):.6f}",
                    "improve_mae_vs_base": f"{(result['online_base_mae'] - adapted_mae):.6f}",
                }
                append_rows(args.results, [row])
                print(
                    f"[{backbone}:{ds['name']}:{variant_name}] "
                    f"offline={result['offline_test_mse']:.6f}/{result.get('offline_test_mae', float('nan')):.6f} "
                    f"online={adapted_mse:.6f}/{adapted_mae:.6f}"
                )
                if variant_name == "stage1_stage2_stage3":
                    summary[backbone][ds["name"]] = adapted_mse

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
