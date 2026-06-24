from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lishui_dataset import load_lishui_registry
from train.run_experiment import run_single_experiment
from train.run_shifted_sota_methods import (
    _prepare_memory,
    build_online_loader,
    evaluate_frozen,
    evaluate_online_gd,
    evaluate_onenet,
    evaluate_proceed_style,
    evaluate_solid,
    train_proceed_adapter,
)
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
    import pandas as pd

    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["method"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--prepared-dir", default="dataset/load_benchmarks/lishui")
    parser.add_argument("--results", default="logs/lishui_benchmarks/shifted_sota_methods.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["frozen", "gd_full", "gd_last_layer", "onenet", "solid", "solidpp", "proceed", "ours_stage3", "ours_full"],
    )
    args = parser.parse_args()

    datasets = load_lishui_registry(args.prepared_dir)
    base_cfg = load_yaml(args.config)
    selected = set(args.methods)
    completed = load_completed(args.results)
    rows = []
    summary = {}

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
        cfg["online_run"]["enable"] = False

        tag = f"{ds['name']}_lishui_shifted_sota"
        result = run_single_experiment(cfg, save_ckpt=False, variant_name=tag, return_artifacts=True)
        model = result["model"]
        device = result["device"]
        train_ds = result["train_ds"]
        test_ds = result["test_ds"]
        invariant_bundle = result["invariant_bundle"]
        split_info = result["split_info"]
        online_dl = build_online_loader(test_ds, cfg["train"]["batch_size"])
        train_memory = None
        if {"solid", "solidpp"} & selected:
            train_memory = _prepare_memory(train_ds, max_items=1024, device=device)
        proceed_adapter = proceed_backbone = None
        if "proceed" in selected:
            proceed_adapter, proceed_backbone = train_proceed_adapter(model, train_ds, device)

        methods = {}
        if "frozen" in selected and (ds["name"], "frozen") not in completed:
            methods["frozen"] = evaluate_frozen(copy.deepcopy(model).to(device), online_dl, device)
        if "gd_full" in selected and (ds["name"], "gd_full") not in completed:
            methods["gd_full"] = evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=None, weight_decay=1e-5, delay_steps=cfg["data"]["lookback"])
        if "gd_last_layer" in selected and (ds["name"], "gd_last_layer") not in completed:
            methods["gd_last_layer"] = evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=["output_head"], weight_decay=1e-5, delay_steps=cfg["data"]["lookback"])
        if "onenet" in selected and (ds["name"], "onenet") not in completed:
            methods["onenet"] = evaluate_onenet(copy.deepcopy(model), online_dl, device, lr=1e-4, delay_steps=cfg["data"]["lookback"])
        if "solid" in selected and (ds["name"], "solid") not in completed:
            methods["solid"] = evaluate_solid(copy.deepcopy(model), online_dl, train_memory, device, k=32, lr=1e-4, delay_steps=cfg["data"]["lookback"], continual=False)
        if "solidpp" in selected and (ds["name"], "solidpp") not in completed:
            methods["solidpp"] = evaluate_solid(copy.deepcopy(model), online_dl, train_memory, device, k=32, lr=1e-4, delay_steps=cfg["data"]["lookback"], continual=True)
        if "proceed" in selected and (ds["name"], "proceed") not in completed:
            methods["proceed"] = evaluate_proceed_style(proceed_backbone, proceed_adapter, online_dl, device, delay_steps=cfg["data"]["lookback"])

        from train.online_loop import run_online_adaptation

        def ours_stream():
            for x, y in online_dl:
                x = x.to(device)
                y = y.to(device)
                if invariant_bundle is None:
                    yield x, y
                else:
                    with torch.no_grad():
                        phi_i = invariant_bundle["encoder_i"](x)
                        phi_v = invariant_bundle["encoder_v"](x)
                    yield x, y, phi_i, phi_v

        if "ours_stage3" in selected and (ds["name"], "ours_stage3") not in completed:
            ours_stage3_cfg = copy.deepcopy(cfg)
            ours_stage3_cfg["online_run"]["enable"] = True
            ours_stage3_cfg["online"]["stage1"]["enable"] = False
            ours_stage3_cfg["online"]["stage2"]["enable"] = False
            ours_stage3_cfg["online"]["stage3"]["enable"] = True
            ours_stage3 = run_online_adaptation(ours_stream(), copy.deepcopy(model).to(device), None, ours_stage3_cfg, device)
            methods["ours_stage3"] = {"mse": ours_stage3["online_stage3_mse"], "mae": ours_stage3["online_stage3_mae"]}

        if "ours_full" in selected and (ds["name"], "ours_full") not in completed:
            ours_full_cfg = copy.deepcopy(cfg)
            ours_full_cfg["online_run"]["enable"] = True
            ours_full_cfg["online"]["stage1"]["enable"] = True
            ours_full_cfg["online"]["stage2"]["enable"] = True
            ours_full_cfg["online"]["stage3"]["enable"] = True
            ours_full = run_online_adaptation(ours_stream(), copy.deepcopy(model).to(device), None, ours_full_cfg, device)
            methods["ours_full"] = {"mse": ours_full["online_stage3_mse"], "mae": ours_full["online_stage3_mae"]}

        summary[ds["name"]] = methods
        for method, metrics in methods.items():
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": ds["name"],
                "backbone": cfg["model"]["temporal_encoder"],
                "method": method,
                "offline_test_mse": f"{result['offline_test_mse']:.6f}",
                "offline_test_mae": f"{result['offline_test_mae']:.6f}",
                "online_mse": f"{metrics['mse']:.6f}",
                "online_mae": f"{metrics['mae']:.6f}",
                "shift_score": f"{float(split_info.get('shift_score', 0.0)):.6f}",
                "test_start": split_info.get("test_start", ""),
                "test_end": split_info.get("test_end", ""),
            }
            rows.append(row)
            append_rows(args.results, [row])

        print(json.dumps({ds["name"]: methods, "shift": split_info}, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
