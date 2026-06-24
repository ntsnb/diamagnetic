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

from data.lishui_dataset import load_lishui_registry
from train.run_experiment import run_single_experiment
from utils.config import load_yaml


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_completed_pairs(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    import pandas as pd

    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["ablation"]))


def apply_ablation(cfg: dict, name: str) -> dict:
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = True
    cur["online"]["stage2"]["enable"] = True
    cur["online"]["stage3"]["enable"] = True

    if name == "full":
        return cur
    if name == "wo_invariant":
        cur["invariant"]["enable"] = False
        return cur
    if name == "wo_revin":
        cur["model"]["use_revin"] = False
        return cur
    if name == "single_expert":
        cur["model"]["n_experts"] = 1
        return cur
    if name == "mlp_gate_no_vae":
        cur["model"]["use_vae"] = False
        return cur
    if name == "dense_stage2":
        cur["online"]["stage2"]["mask_threshold"] = 0.0
        cur["online"]["stage2"]["topk_layer"] = 100000
        cur["online"]["stage2"]["topk_omega"] = 100000
        cur["online"]["stage2"]["min_omega_density"] = 0.0
        cur["online"]["stage2"]["target_modules"] = ["experts", "output_head"]
        return cur
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--prepared-dir", default="dataset/load_benchmarks/lishui")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--results", default="logs/lishui_benchmarks/component_ablation.csv")
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=["full", "wo_invariant", "wo_revin", "single_expert", "mlp_gate_no_vae", "dense_stage2"],
    )
    args = parser.parse_args()

    datasets = load_lishui_registry(args.prepared_dir)
    base_cfg = load_yaml(args.config)
    completed = load_completed_pairs(args.results)
    rows = []
    summary = {}
    for ds in datasets:
        summary[ds["name"]] = {}
        for ab in args.ablations:
            if (ds["name"], ab) in completed:
                print(f"[skip] {ds['name']}:{ab} already finished")
                continue
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
            cfg = apply_ablation(cfg, ab)
            tag = f"{ds['name']}_{ab}"
            result = run_single_experiment(cfg, save_ckpt=False, variant_name=tag)
            adapted = result["online_stage3_mse"]
            split_info = result.get("split_info", {})
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": ds["name"],
                "ablation": ab,
                "offline_test_mse": f"{result['offline_test_mse']:.6f}",
                "offline_test_mae": f"{result['offline_test_mae']:.6f}",
                "online_base_mse": f"{result['online_base_mse']:.6f}",
                "online_mse": f"{adapted:.6f}",
                "gain_vs_base": f"{(result['online_base_mse'] - adapted):.6f}",
                "shift_score": f"{float(split_info.get('shift_score', 0.0)):.6f}",
            }
            rows.append(row)
            append_rows(args.results, [row])
            summary[ds["name"]][ab] = adapted
            print(f"[{ds['name']}:{ab}] online={adapted:.6f} gain={result['online_base_mse'] - adapted:.6f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
