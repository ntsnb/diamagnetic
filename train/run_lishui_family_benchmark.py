from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lishui_dataset import load_lishui_registry
from train.run_experiment import run_single_experiment
from train.run_ablation_load import STAGE_VARIANTS
from utils.config import load_yaml


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_completed(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["variant"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--prepared-dir", default="dataset/load_benchmarks/lishui")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--results", default="logs/lishui_benchmarks/family_results.csv")
    parser.add_argument("--variants", nargs="*", default=list(STAGE_VARIANTS.keys()))
    args = parser.parse_args()

    datasets = load_lishui_registry(args.prepared_dir)
    base_cfg = load_yaml(args.config)
    rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}
    completed = load_completed(args.results)

    for ds in datasets:
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["epochs"] = args.epochs
        cfg["invariant"]["epochs"] = args.invariant_epochs
        cfg["data"]["path"] = ds["path"]
        cfg["data"]["timestamp_col"] = ds["timestamp_col"]
        cfg["data"]["csv_encoding"] = ds.get("encoding")
        cfg["data"]["target_col"] = ds["target_col"]
        cfg["data"]["target_dim"] = None
        cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
        cfg["data"]["impute_method"] = ds.get("impute_method", "time_interpolate")
        cfg["data"]["split_mode"] = "chronological_shifted"
        cfg["data"]["online_test_ratio"] = 0.20
        cfg["data"]["shift_pool_ratio"] = 0.35
        summary[ds["name"]] = {}

        for variant_name in args.variants:
            if (ds["name"], variant_name) in completed:
                print(f"[skip] {ds['name']}:{variant_name} already exists in {args.results}")
                continue
            s1, s2, s3 = STAGE_VARIANTS[variant_name]
            cur = copy.deepcopy(cfg)
            cur["online"]["stage1"]["enable"] = s1
            cur["online"]["stage2"]["enable"] = s2
            cur["online"]["stage3"]["enable"] = s3
            cur["online_run"]["enable"] = True
            tag = f"{ds['name']}_{variant_name}"
            result = run_single_experiment(cur, save_ckpt=False, variant_name=tag)
            adapted = result["online_stage3_mse"] if s3 else result["online_stage2_mse"]
            overfit_gap = result["offline_test_mse"] - result["offline_train_mse"]
            improve_vs_base = result["online_base_mse"] - adapted
            split_info = result.get("split_info", {})
            row = {
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
                "shift_score": f"{float(split_info.get('shift_score', 0.0)):.6f}",
                "test_start": split_info.get("test_start", ""),
                "test_end": split_info.get("test_end", ""),
            }
            rows.append(row)
            append_rows(args.results, [row])
            summary[ds["name"]][variant_name] = adapted
            print(
                f"[{ds['name']}:{variant_name}] "
                f"train={result['offline_train_mse']:.6f} test={result['offline_test_mse']:.6f} "
                f"online={adapted:.6f} gain={improve_vs_base:.6f} shift={float(split_info.get('shift_score', 0.0)):.4f}"
            )

    summary_path = Path(args.results).with_name("family_latest_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
