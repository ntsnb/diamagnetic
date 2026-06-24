from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.online_loop import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml


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

REUSE_ABLATIONS = [
    "full_ref",
    "dense_stage2_all_layers",
    "grad_stage2_no_lrp",
    "no_invdelta_stage3",
    "no_ema_reference",
    "no_rollback_accept",
    "sync_all_experts",
    "no_delay_feedback",
]

RETRAIN_ABLATIONS = [
    "no_invariant",
    "mlp_gate_no_vae",
    "no_revin",
    "single_expert",
]


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def build_stream(test_ds, batch_size: int, device, invariant_bundle):
    online_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def generator():
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

    return online_dl, generator


def run_variant_from_artifacts(
    model,
    invariant_bundle,
    cfg: dict,
    device,
    test_ds,
    s1: bool,
    s2: bool,
    s3: bool,
) -> dict:
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    _, stream_fn = build_stream(test_ds, cur["train"]["batch_size"], device, invariant_bundle)
    return run_online_adaptation(stream_fn(), copy.deepcopy(model).to(device), None, cur, device)


def apply_component_ablation(cfg: dict, label: str) -> dict:
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = False
    cur["online"]["stage1"]["enable"] = True
    cur["online"]["stage2"]["enable"] = True
    cur["online"]["stage3"]["enable"] = True
    if label == "full_ref":
        return cur
    if label == "dense_stage2_all_layers":
        cur["online"]["stage2"]["use_sparse_mask"] = False
        cur["online"]["stage2"]["target_modules"] = []
        cur["online"]["stage2"]["target_modules_no_stage1"] = []
        return cur
    if label == "grad_stage2_no_lrp":
        cur["online"]["stage2"]["relevance_method"] = "grad"
        return cur
    if label == "no_invdelta_stage3":
        cur["online"]["stage3"]["use_invariant_delta"] = False
        return cur
    if label == "no_ema_reference":
        cur["online"]["stage2"]["use_ema_reference"] = False
        return cur
    if label == "no_rollback_accept":
        cur["online"]["stage1"]["rollback_on_worse"] = False
        cur["online"]["stage2"]["rollback_on_worse"] = False
        cur["online"]["stage2"]["use_accept_buffer"] = False
        cur["online"]["stage2"]["accept_buffer_size"] = 1
        return cur
    if label == "sync_all_experts":
        cur["online"]["stage2"]["use_sparse_mask"] = False
        cur["online"]["stage2"]["target_modules"] = ["experts", "alpha_head", "output_head"]
        cur["online"]["stage2"]["target_modules_no_stage1"] = ["experts", "alpha_head", "output_head"]
        return cur
    if label == "no_delay_feedback":
        cur["online"]["feedback_delay_steps"] = 0
        return cur
    if label == "no_invariant":
        cur["invariant"]["enable"] = False
        return cur
    if label == "mlp_gate_no_vae":
        cur["model"]["use_vae"] = False
        cur["train"]["beta_vae"] = 0.0
        cur["online"]["stage2"]["gate_by_recon"] = False
        cur["online"]["stage3"]["gate_by_recon"] = False
        return cur
    if label == "no_revin":
        cur["model"]["use_revin"] = False
        return cur
    if label == "single_expert":
        cur["model"]["n_experts"] = 1
        return cur
    raise ValueError(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tuning/load_candidate_i_shifted.yaml")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--results", default="logs/load_benchmarks/ablation_results.csv")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    done: set[tuple[str, str, str]] = set()
    if os.path.exists(args.results):
        existing = pd.read_csv(args.results)
        for _, row in existing.iterrows():
            done.add((str(row["dataset"]), str(row["group"]), str(row["variant"])))

    all_rows = []
    for ds in DATASETS:
        ds_cfg = prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs)
        need_reuse_base = any((ds["name"], "stage_combo", label) not in done for label in STAGE_VARIANTS) or any(
            (ds["name"], "component", label) not in done for label in REUSE_ABLATIONS
        )
        offline_result = None
        if need_reuse_base:
            offline_cfg = copy.deepcopy(ds_cfg)
            offline_cfg["online_run"]["enable"] = False
            offline_result = run_single_experiment(
                offline_cfg,
                save_ckpt=False,
                variant_name=f"{ds['name']}_ablation_base_offline",
                return_artifacts=True,
            )
            base_model = offline_result["model"]
            invariant_bundle = offline_result["invariant_bundle"]
            test_ds = offline_result["test_ds"]
            device = offline_result["device"]
            split_info = offline_result["split_info"]
            offline_test_mse = offline_result["offline_test_mse"]
            offline_test_mae = offline_result["offline_test_mae"]

        if need_reuse_base:
            for label, (s1, s2, s3) in STAGE_VARIANTS.items():
                if (ds["name"], "stage_combo", label) in done:
                    continue
                result = run_variant_from_artifacts(base_model, invariant_bundle, ds_cfg, device, test_ds, s1, s2, s3)
                adapted = result["online_stage3_mse"] if s3 else result["online_stage2_mse"]
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dataset": ds["name"],
                    "group": "stage_combo",
                    "variant": label,
                    "offline_test_mse": f"{offline_test_mse:.6f}",
                    "offline_test_mae": f"{offline_test_mae:.6f}",
                    "online_base_mse": f"{result['online_base_mse']:.6f}",
                    "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
                    "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
                    "adapted_online_mse": f"{adapted:.6f}",
                    "improve_vs_base": f"{result['online_base_mse'] - adapted:.6f}",
                    "shift_score": f"{float(split_info.get('shift_score', 0.0)):.6f}",
                }
                append_row(args.results, row)
                all_rows.append(row)
                print(f"[{ds['name']}:stage_combo:{label}] online={adapted:.6f} gain={result['online_base_mse'] - adapted:.6f}")

            for label in REUSE_ABLATIONS:
                if (ds["name"], "component", label) in done:
                    continue
                ab_cfg = apply_component_ablation(ds_cfg, label)
                result = run_variant_from_artifacts(base_model, invariant_bundle, ab_cfg, device, test_ds, True, True, True)
                adapted = result["online_stage3_mse"]
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dataset": ds["name"],
                    "group": "component",
                    "variant": label,
                    "offline_test_mse": f"{offline_test_mse:.6f}",
                    "offline_test_mae": f"{offline_test_mae:.6f}",
                    "online_base_mse": f"{result['online_base_mse']:.6f}",
                    "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
                    "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
                    "adapted_online_mse": f"{adapted:.6f}",
                    "improve_vs_base": f"{result['online_base_mse'] - adapted:.6f}",
                    "shift_score": f"{float(split_info.get('shift_score', 0.0)):.6f}",
                }
                append_row(args.results, row)
                all_rows.append(row)
                print(f"[{ds['name']}:component:{label}] online={adapted:.6f} gain={result['online_base_mse'] - adapted:.6f}")

        for label in RETRAIN_ABLATIONS:
            if (ds["name"], "component", label) in done:
                continue
            ab_cfg = apply_component_ablation(ds_cfg, label)
            ab_result = run_single_experiment(
                ab_cfg,
                save_ckpt=False,
                variant_name=f"{ds['name']}_{label}",
                return_artifacts=True,
            )
            ab_model = ab_result["model"]
            ab_inv = ab_result["invariant_bundle"]
            ab_test_ds = ab_result["test_ds"]
            ab_device = ab_result["device"]
            ab_split_info = ab_result["split_info"]
            result = run_variant_from_artifacts(ab_model, ab_inv, ab_cfg, ab_device, ab_test_ds, True, True, True)
            adapted = result["online_stage3_mse"]
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": ds["name"],
                "group": "component",
                "variant": label,
                "offline_test_mse": f"{ab_result['offline_test_mse']:.6f}",
                "offline_test_mae": f"{ab_result['offline_test_mae']:.6f}",
                "online_base_mse": f"{result['online_base_mse']:.6f}",
                "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
                "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
                "adapted_online_mse": f"{adapted:.6f}",
                "improve_vs_base": f"{result['online_base_mse'] - adapted:.6f}",
                "shift_score": f"{float(ab_split_info.get('shift_score', 0.0)):.6f}",
            }
            append_row(args.results, row)
            all_rows.append(row)
            print(f"[{ds['name']}:component:{label}] online={adapted:.6f} gain={result['online_base_mse'] - adapted:.6f}")

    df = pd.DataFrame(all_rows)
    summary = (
        df.assign(adapted_online_mse=lambda x: x["adapted_online_mse"].astype(float))
        .groupby(["group", "variant"], as_index=False)["adapted_online_mse"]
        .mean()
        .sort_values(["group", "adapted_online_mse"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
