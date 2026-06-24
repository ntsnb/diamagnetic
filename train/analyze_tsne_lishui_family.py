from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lishui_dataset import load_lishui_registry
from train.analyze_tsne_components import (
    _collect_expert_space,
    _collect_phi_alpha,
    _centroid_gap,
    _mean_pairwise_centroid_distance,
    _run_stage1_alpha_stream,
    _run_stage2_drift_stream,
    _run_stage3_gain_stream,
    _tsne_embed,
)
from train.run_experiment import run_single_experiment
from utils.config import load_yaml
from utils.plot_style import apply_ieee_plot_style


COUNTY_LABELS = {
    "lishui_云和": "Yunhe",
    "lishui_庆元": "Qingyuan",
    "lishui_景宁": "Jingning",
    "lishui_松阳": "Songyang",
    "lishui_缙云": "Jinyun",
    "lishui_莲都": "Liandu",
    "lishui_遂昌": "Suichang",
    "lishui_青田": "Qingtian",
    "lishui_龙泉": "Longquan",
}


def _sample_subset(ds, n: int) -> Subset:
    idx = np.linspace(0, len(ds) - 1, num=min(n, len(ds)), dtype=int).tolist()
    return Subset(ds, idx)


def _prepare_cfg(base_cfg: dict, ds: dict, epochs: int, invariant_epochs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["target_dim"] = None
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "time_interpolate")
    cfg["online_run"]["enable"] = False
    cfg["train"]["epochs"] = epochs
    cfg["invariant"]["epochs"] = invariant_epochs
    return cfg


def _plot_grid(results: dict[str, dict], key: str, labels_key: str, outpath: Path, title_mode: str) -> None:
    apply_ieee_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0))
    palette = {
        "Train": "#4C78A8",
        "Online": "#E45756",
        "Selected": "#54A24B",
        "Skipped": "#B0B0B0",
        "Strong gain": "#54A24B",
        "Mild gain": "#F58518",
        "No gain": "#B0B0B0",
    }
    for ax, (dataset, item) in zip(axes.flatten(), results.items()):
        emb = item[key]
        labels = np.asarray(item[labels_key])
        for lab in np.unique(labels):
            sel = labels == lab
            ax.scatter(emb[sel, 0], emb[sel, 1], s=7, alpha=0.68, c=palette.get(str(lab), "#4C78A8"), label=str(lab))
        ax.set_title(COUNTY_LABELS.get(dataset, dataset), fontsize=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    handles, labs = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, frameon=False, ncol=min(4, len(labs)), loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_lishui_online.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--outdir", default="logs/lishui_benchmarks/tsne")
    args = parser.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(args.config)
    results = {}
    rows = []

    for ds in load_lishui_registry("dataset/load_benchmarks/lishui"):
        cfg = _prepare_cfg(base_cfg, ds, args.epochs, args.invariant_epochs)
        run = run_single_experiment(cfg, save_ckpt=False, variant_name=f"{ds['name']}_tsne_lishui", return_artifacts=True)
        model = run["model"]
        device = run["device"]
        invariant_bundle = run["invariant_bundle"]

        train_loader = DataLoader(_sample_subset(run["train_ds"], args.samples), batch_size=1, shuffle=False)
        online_subset = _sample_subset(run["test_ds"], min(args.samples, 120))
        online_loader = DataLoader(online_subset, batch_size=1, shuffle=False)

        phi_train, drift_train, alpha_train = _collect_phi_alpha(model, invariant_bundle, train_loader, device, args.samples)
        phi_online, drift_online, alpha_online = _collect_phi_alpha(model, invariant_bundle, online_loader, device, args.samples)
        alpha_stage1 = _run_stage1_alpha_stream(model, online_subset, device, cfg, min(args.samples, 120))
        stage2_drift, stage2_labels = _run_stage2_drift_stream(model, online_subset, device, cfg, min(args.samples, 120))
        stage3_drift, stage3_gain, stage3_labels = _run_stage3_gain_stream(model, invariant_bundle, online_subset, device, cfg, min(args.samples, 120))
        expert_points, expert_labels, _ = _collect_expert_space(model, train_loader, online_loader, device, min(args.samples, 80))

        drift_emb = _tsne_embed([drift_train, drift_online])
        stage2_emb = _tsne_embed([stage2_drift])[0]
        stage3_emb = _tsne_embed([stage3_drift])[0]
        expert_emb = _tsne_embed([expert_points])[0]

        results[ds["name"]] = {
            "drift_train_online": np.concatenate(drift_emb, axis=0),
            "drift_train_online_labels": np.array(["Train"] * len(drift_emb[0]) + ["Online"] * len(drift_emb[1])),
            "stage2_emb": stage2_emb,
            "stage2_labels": stage2_labels,
            "stage3_emb": stage3_emb,
            "stage3_labels": stage3_labels,
            "expert_emb": expert_emb,
            "expert_labels": expert_labels,
        }

        rows.extend(
            [
                {"dataset": ds["name"], "metric": "phi_gap", "value": _centroid_gap(phi_train, phi_online)},
                {"dataset": ds["name"], "metric": "drift_gap", "value": _centroid_gap(drift_train, drift_online)},
                {"dataset": ds["name"], "metric": "alpha_gap_reduction", "value": _centroid_gap(alpha_train, alpha_online) - _centroid_gap(alpha_train, alpha_stage1)},
                {"dataset": ds["name"], "metric": "stage2_selected_ratio", "value": float(np.mean(np.asarray(stage2_labels) == "Selected"))},
                {"dataset": ds["name"], "metric": "stage3_positive_gain_ratio", "value": float(np.mean(stage3_gain > 0))},
                {"dataset": ds["name"], "metric": "expert_margin", "value": _mean_pairwise_centroid_distance(expert_points, expert_labels)},
            ]
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "tsne_lishui_summary.csv", index=False)
    _plot_grid(results, "drift_train_online", "drift_train_online_labels", outdir / "paper_tsne_lishui_drift.png", "drift")
    _plot_grid(results, "stage2_emb", "stage2_labels", outdir / "paper_tsne_lishui_stage2.png", "stage2")
    _plot_grid(results, "stage3_emb", "stage3_labels", outdir / "paper_tsne_lishui_stage3.png", "stage3")

    heat = summary.pivot(index="dataset", columns="metric", values="value")
    heat.index = [COUNTY_LABELS.get(idx, idx) for idx in heat.index]
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(outdir / "paper_tsne_lishui_metrics.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
