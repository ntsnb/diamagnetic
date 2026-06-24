from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.plot_style import apply_ieee_plot_style

METHOD_LABELS = {
    "frozen": "Frozen",
    "gd_full": "GD",
    "gd_last_layer": "Head-GD",
    "onenet": "OneNet",
    "solid": "SOLID",
    "solidpp": "SOLID++",
    "proceed": "PROCEED",
    "ours_stage3": "Ours-Stage III",
    "ours_full": "Ours-Full",
}

METHOD_ORDER = list(METHOD_LABELS.keys())


def build_report(df: pd.DataFrame) -> str:
    lines = [
        "# Shifted Online Baselines: Second-Wave Methods",
        "",
        "This report summarizes the second-wave online-learning comparison under the distribution-shifted online protocol on the iTransformer backbone.",
        "",
    ]

    for dataset, dset_df in df.groupby("dataset", sort=False):
        frozen = float(dset_df.loc[dset_df["method"] == "frozen", "online_mse"].iloc[0])
        best_row = dset_df.loc[dset_df["online_mse"].idxmin()]
        strongest_external = (
            dset_df.loc[
                dset_df["method"].isin(
                    ["gd_full", "gd_last_layer", "onenet", "solid", "solidpp", "proceed"]
                )
            ]
            .sort_values("online_mse")
            .iloc[0]
        )
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Shift score: `{float(dset_df['shift_score'].iloc[0]):.4f}`",
                f"- Best overall: `{METHOD_LABELS[best_row['method']]}` with online MSE `{float(best_row['online_mse']):.6f}`",
                f"- Strongest external baseline: `{METHOD_LABELS[strongest_external['method']]}` with online MSE `{float(strongest_external['online_mse']):.6f}`",
                f"- Frozen baseline: `{frozen:.6f}`",
                "",
            ]
        )

        ranked = dset_df.assign(
            label=dset_df["method"].map(METHOD_LABELS),
            gain_vs_frozen=frozen - dset_df["online_mse"],
        ).sort_values("online_mse")
        lines.append("| Method | Online MSE | Gain vs Frozen |")
        lines.append("|---|---:|---:|")
        for _, row in ranked.iterrows():
            lines.append(
                f"| {row['label']} | {float(row['online_mse']):.6f} | {float(row['gain_vs_frozen']):.6f} |"
            )
        lines.append("")

    mean_df = (
        df.groupby("method", as_index=False)["online_mse"]
        .mean()
        .assign(label=lambda x: x["method"].map(METHOD_LABELS))
        .sort_values("online_mse")
    )
    lines.extend(
        [
            "## Mean Ranking Across Datasets",
            "",
            "| Method | Mean Online MSE |",
            "|---|---:|",
        ]
    )
    for _, row in mean_df.iterrows():
        lines.append(f"| {row['label']} | {float(row['online_mse']):.6f} |")

    return "\n".join(lines) + "\n"


def make_plot(df: pd.DataFrame, output: Path) -> None:
    apply_ieee_plot_style()
    plot_df = df.copy()
    plot_df["label"] = plot_df["method"].map(METHOD_LABELS)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER]

    datasets = list(plot_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = plot_df[plot_df["dataset"] == dataset].copy()
        sub["label"] = pd.Categorical(sub["label"], categories=order, ordered=True)
        sub = sub.sort_values("label")
        colors = ["#4c78a8" if not l.startswith("Ours") else "#f58518" for l in sub["label"]]
        ax.bar(sub["label"], sub["online_mse"], color=colors)
        ax.set_ylabel("Online MSE")
        ax.set_xlabel(dataset.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)
        best_idx = sub["online_mse"].idxmin()
        best_x = list(sub.index).index(best_idx)
        ax.get_xticklabels()[best_x].set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    Path(args.report).write_text(build_report(df), encoding="utf-8")
    make_plot(df, Path(args.figure))

    summary = (
        df.pivot(index="dataset", columns="method", values="online_mse")
        .reset_index()
        .rename(columns=METHOD_LABELS)
    )
    summary.to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
