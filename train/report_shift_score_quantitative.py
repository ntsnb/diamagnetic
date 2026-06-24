from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import apply_ieee_plot_style


def main() -> None:
    src = ROOT / "logs" / "load_benchmarks" / "shift_severity_analysis.csv"
    df = pd.read_csv(src)
    df = df.sort_values(["dataset", "shift_score", "variant"]).reset_index(drop=True)

    pivot = (
        df.pivot_table(
            index=["dataset", "shift_score"],
            columns="variant",
            values="adapted_online_mse",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.to_csv(ROOT / "logs" / "load_benchmarks" / "shift_score_quantitative_table.csv", index=False)

    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})
    order = ["panama", "tetouan", "spain_energy_weather"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharey=False)
    for ax, ds in zip(axes, order):
        sub = df[df["dataset"] == ds].copy().sort_values("shift_score")
        sns.lineplot(data=sub, x="shift_score", y="adapted_online_mse", hue="variant", marker="o", ax=ax)
        ax.set_title(ds.replace("_", " ").title())
        ax.set_xlabel("Shift score")
        ax.set_ylabel("Online MSE")
        if ax.legend_ is not None:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(ROOT / "logs" / "load_benchmarks" / "paper_shift_score_quantitative.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharey=False)
    for ax, ds in zip(axes, order):
        sub = df[df["dataset"] == ds].copy().sort_values("shift_score")
        frozen = sub[sub["variant"] == "Frozen"][["shift_score", "adapted_online_mse"]].rename(
            columns={"adapted_online_mse": "Frozen"}
        )
        merged = sub.merge(frozen, on="shift_score")
        merged["gain_vs_frozen"] = merged["Frozen"] - merged["adapted_online_mse"]
        merged = merged[merged["variant"] != "Frozen"]
        sns.lineplot(data=merged, x="shift_score", y="gain_vs_frozen", hue="variant", marker="o", ax=ax)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ds.replace("_", " ").title())
        ax.set_xlabel("Shift score")
        ax.set_ylabel("Gain vs Frozen")
        if ax.legend_ is not None:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(ROOT / "logs" / "load_benchmarks" / "paper_shift_score_gain.png", dpi=300)
    plt.close(fig)

    report = [
        "# Quantitative Shift-Score Report",
        "",
        "This report rewrites the previous low/medium/high severity analysis into exact per-dataset shift-score comparisons.",
        "",
        pivot.to_markdown(index=False),
    ]
    (ROOT / "logs" / "load_benchmarks" / "shift_score_quantitative_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
