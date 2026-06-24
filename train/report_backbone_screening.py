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


BACKBONE_ORDER = [
    "itransformer",
    "fedformer",
    "crossformer",
    "informer",
    "autoformer",
    "pyraformer",
    "transformer",
    "patchtst",
]
BACKBONE_LABELS = {
    "transformer": "Transformer",
    "itransformer": "iTransformer",
    "patchtst": "PatchTST",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "fedformer": "FEDformer",
    "pyraformer": "Pyraformer",
    "crossformer": "Crossformer",
}
VARIANTS = {
    "baseline": "Baseline",
    "stage3": "Stage III",
    "stage1_stage2_stage3": "Full",
}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        df.sort_values("timestamp")
        .groupby(["backbone", "dataset", "variant"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest = latest[latest["variant"].isin(VARIANTS)]
    return (
        latest.groupby(["backbone", "variant"], as_index=False)[
            ["offline_test_mse", "offline_test_mae", "adapted_online_mse"]
        ]
        .mean()
    )


def plot(summary: pd.DataFrame, output: Path) -> None:
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    width = 0.22
    xlabels = [BACKBONE_LABELS[b] for b in BACKBONE_ORDER]
    x = range(len(xlabels))
    colors = {
        "Baseline": "#4C78A8",
        "Stage III": "#F58518",
        "Full": "#54A24B",
    }
    for i, variant in enumerate(["baseline", "stage3", "stage1_stage2_stage3"]):
        vals = []
        for backbone in BACKBONE_ORDER:
            sub = summary[(summary["backbone"] == backbone) & (summary["variant"] == variant)]
            vals.append(float(sub["adapted_online_mse"].iloc[0]))
        label = VARIANTS[variant]
        ax.bar([xx + (i - 1) * width for xx in x], vals, width=width, label=label, color=colors[label])
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("Mean Online MSE")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/load_benchmarks/backbone_suite_v2.csv")
    parser.add_argument("--figure", default="logs/load_benchmarks/paper_backbone_screening.png")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    summary = summarize(df)
    plot(summary, Path(args.figure))


if __name__ == "__main__":
    main()
