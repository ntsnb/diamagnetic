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
    "ours_stage3": "Ours-Stage III",
    "ours_full": "Ours-Full",
}
METHOD_ORDER = ["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"]
DATASET_ORDER = ["panama", "tetouan", "spain_energy_weather"]
DATASET_LABELS = {
    "panama": "Panama",
    "tetouan": "Tetouan",
    "spain_energy_weather": "Spain Energy--Weather",
}


def plot(df: pd.DataFrame, output: Path) -> None:
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    width = 0.15
    x = range(len(DATASET_ORDER))
    colors = {
        "Frozen": "#4C78A8",
        "GD": "#E45756",
        "Head-GD": "#72B7B2",
        "Ours-Stage III": "#F58518",
        "Ours-Full": "#54A24B",
    }

    for i, method in enumerate(METHOD_ORDER):
        vals = []
        for dataset in DATASET_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            vals.append(float(sub["online_mse"].iloc[0]))
        label = METHOD_LABELS[method]
        ax.bar([xx + (i - 2) * width for xx in x], vals, width=width, label=label, color=colors[label])

    ax.set_xticks(list(x))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Online MSE")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/load_benchmarks/shifted_online_baselines.csv")
    parser.add_argument("--figure", default="logs/load_benchmarks/paper_shifted_baselines.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "method"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    plot(latest, Path(args.figure))


if __name__ == "__main__":
    main()
