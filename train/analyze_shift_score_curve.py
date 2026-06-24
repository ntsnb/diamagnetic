from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import WindowedDataset
from data.loaders import _shift_score
from data.normalization import StandardScaler
from data.readers import load_tabular_timeseries
from data.windowing import build_windows
from train.online_loop import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml
from utils.plot_style import apply_ieee_plot_style


SHIFT_QUANTILES = [0.15, 0.40, 0.65, 0.90]
VARIANTS = {
    "Frozen": (False, False, False),
    "Stage III": (False, False, True),
    "Full": (True, True, True),
}


def prepare_dataset_cfg(base_cfg: dict, ds: dict, epochs: int, invariant_epochs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["epochs"] = epochs
    cfg["invariant"]["epochs"] = invariant_epochs
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
    cfg["online"]["compare_log_path"] = "logs/tmp_online_compare.csv"
    cfg["online"]["stage1_log_path"] = "logs/tmp_online_stage1.csv"
    cfg["online"]["stage2_log_path"] = "logs/tmp_online_stage2.csv"
    return cfg


def build_series_and_windows(cfg: dict, ds: dict):
    series, _ = load_tabular_timeseries(
        ds["path"],
        timestamp_col=ds["timestamp_col"],
        feature_cols=None,
        encoding=ds.get("encoding"),
        missing_sentinels=ds.get("missing_sentinels"),
        impute_method=ds.get("impute_method", "drop"),
    )
    split_idx = int(series.shape[0] * cfg["data"]["train_ratio"])
    scaler = StandardScaler.fit(series[:split_idx])
    series = scaler.transform(series)
    target_dim = target_dim_for_dataset(ds)
    x, y = build_windows(series, cfg["data"]["lookback"], cfg["data"]["horizon"], target_dim)
    return WindowedDataset(x, y)


def choose_shift_blocks(ds_obj: WindowedDataset, cfg: dict):
    n = len(ds_obj)
    n_train = int(n * cfg["data"]["train_ratio"])
    n_val = int(n * cfg["data"]["val_ratio"])
    n_test = max(1, int(n * cfg["data"]["online_test_ratio"]))
    pool_size = max(n_test, int(n * cfg["data"]["shift_pool_ratio"]))
    pool_start = max(n_train + n_val, n - pool_size)
    pool_end = n
    train_x = ds_obj.x[:n_train]

    candidates = []
    for start in range(pool_start, pool_end - n_test + 1):
        stop = start + n_test
        score = _shift_score(train_x, ds_obj.x[start:stop])
        candidates.append((start, stop, score))
    cand_df = pd.DataFrame(candidates, columns=["start", "stop", "shift_score"])
    chosen = []
    used = set()
    for q in SHIFT_QUANTILES:
        target = cand_df["shift_score"].quantile(q)
        idx = (cand_df["shift_score"] - target).abs().idxmin()
        row = cand_df.loc[idx].copy()
        key = (int(row["start"]), int(row["stop"]))
        if key in used:
            continue
        used.add(key)
        row["quantile"] = q
        chosen.append(row)
    chosen_df = pd.DataFrame(chosen).sort_values("shift_score").reset_index(drop=True)
    return chosen_df


def build_stream(test_ds, batch_size: int, device, invariant_bundle):
    online_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def generator():
        for x, y in online_dl:
            x = x.to(device)
            y = y.to(device)
            if invariant_bundle is None:
                yield x, y
            else:
                import torch
                with torch.no_grad():
                    phi_i = invariant_bundle["encoder_i"](x)
                    phi_v = invariant_bundle["encoder_v"](x)
                yield x, y, phi_i, phi_v

    return generator


def run_variant(model, invariant_bundle, cfg: dict, test_ds, device, s1: bool, s2: bool, s3: bool):
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    for p in [cur["online"]["compare_log_path"], cur["online"]["stage1_log_path"], cur["online"]["stage2_log_path"]]:
        if os.path.exists(p):
            os.remove(p)
    stream = build_stream(test_ds, cur["train"]["batch_size"], device, invariant_bundle)
    return run_online_adaptation(stream(), copy.deepcopy(model).to(device), None, cur, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tuning/load_candidate_i_shifted.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--online-test-ratio", type=float, default=0.03)
    parser.add_argument("--shift-pool-ratio", type=float, default=0.18)
    parser.add_argument("--results", default="logs/load_benchmarks/shift_score_curve.csv")
    args = parser.parse_args()

    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})

    rows = []
    base_cfg = load_yaml(args.config)
    for ds in DATASETS:
        cfg = prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs)
        cfg["data"]["online_test_ratio"] = args.online_test_ratio
        cfg["data"]["shift_pool_ratio"] = args.shift_pool_ratio
        offline_cfg = copy.deepcopy(cfg)
        offline_cfg["online_run"]["enable"] = False
        result = run_single_experiment(
            offline_cfg,
            save_ckpt=False,
            variant_name=f"{ds['name']}_shift_curve",
            return_artifacts=True,
        )
        model = result["model"]
        invariant_bundle = result["invariant_bundle"]
        device = result["device"]
        ds_obj = build_series_and_windows(cfg, ds)
        chosen_blocks = choose_shift_blocks(ds_obj, cfg)

        for _, row in chosen_blocks.iterrows():
            test_ds = Subset(ds_obj, range(int(row["start"]), int(row["stop"])))
            frozen_summary = None
            for variant_name, (s1, s2, s3) in VARIANTS.items():
                summary = run_variant(model, invariant_bundle, cfg, test_ds, device, s1, s2, s3)
                adapted = summary["online_stage3_mse"] if s3 else summary["online_stage2_mse"]
                if variant_name == "Frozen":
                    frozen_summary = adapted
                rows.append(
                    {
                        "dataset": ds["name"],
                        "shift_score": float(row["shift_score"]),
                        "quantile": float(row["quantile"]),
                        "variant": variant_name,
                        "online_mse": float(adapted),
                        "gain_vs_frozen": float(frozen_summary - adapted) if frozen_summary is not None else 0.0,
                    }
                )
                print(
                    f"[{ds['name']} q={row['quantile']:.2f} {variant_name}] "
                    f"shift={row['shift_score']:.4f} online={adapted:.6f}"
                )

    out = pd.DataFrame(rows)
    out.to_csv(args.results, index=False)

    gain_summary = (
        out[out["variant"] != "Frozen"]
        .groupby(["dataset", "variant"])
        .agg(
            mean_gain=("gain_vs_frozen", "mean"),
            max_gain=("gain_vs_frozen", "max"),
            min_gain=("gain_vs_frozen", "min"),
            mean_online_mse=("online_mse", "mean"),
        )
        .reset_index()
    )
    gain_summary["gain_shift_corr"] = 0.0
    for idx, row in gain_summary.iterrows():
        sub = out[(out["dataset"] == row["dataset"]) & (out["variant"] == row["variant"])]
        corr = sub["gain_vs_frozen"].corr(sub["shift_score"], method="spearman")
        gain_summary.loc[idx, "gain_shift_corr"] = float(corr) if pd.notna(corr) else 0.0
    gain_summary.to_csv(ROOT / "logs" / "load_benchmarks" / "shift_score_curve_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13.2, 3.8), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, ds in zip(axes, DATASETS):
        sub = out[out["dataset"] == ds["name"]].sort_values("shift_score")
        sns.lineplot(data=sub, x="shift_score", y="online_mse", hue="variant", marker="o", ax=ax)
        ax.set_title(ds["name"].replace("_", " ").title())
        ax.set_xlabel("Shift score")
        if ax is axes[0]:
            ax.set_ylabel("Online MSE")
        else:
            ax.set_ylabel("")
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(ROOT / "logs" / "load_benchmarks" / "paper_shift_score_curve.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13.2, 3.8), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, ds in zip(axes, DATASETS):
        sub = out[(out["dataset"] == ds["name"]) & (out["variant"] != "Frozen")].sort_values("shift_score")
        sns.lineplot(data=sub, x="shift_score", y="gain_vs_frozen", hue="variant", marker="o", ax=ax)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ds["name"].replace("_", " ").title())
        ax.set_xlabel("Shift score")
        if ax is axes[0]:
            ax.set_ylabel("Gain vs Frozen")
        else:
            ax.set_ylabel("")
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(ROOT / "logs" / "load_benchmarks" / "paper_shift_score_gain_curve.png", dpi=300)
    plt.close(fig)

    report_lines = [
        "# Quantitative Shift-Score Analysis",
        "",
        "This analysis replaces the coarse low/medium/high grouping with a denser shift-score sweep on each of the three main load datasets.",
        "",
        "## Summary",
        "",
        gain_summary.to_markdown(index=False),
        "",
    ]
    (ROOT / "logs" / "load_benchmarks" / "shift_score_curve_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
