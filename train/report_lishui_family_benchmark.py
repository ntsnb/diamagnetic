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


def main() -> None:
    result_path = ROOT / "logs" / "lishui_benchmarks" / "family_results.csv"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    df = pd.read_csv(result_path)
    df["adapted_online_mse"] = pd.to_numeric(df["adapted_online_mse"])
    df["offline_test_mse"] = pd.to_numeric(df["offline_test_mse"])
    df["offline_test_mae"] = pd.to_numeric(df["offline_test_mae"])
    df["improve_vs_base"] = pd.to_numeric(df["improve_vs_base"])

    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "variant"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    summary = (
        latest.groupby("variant", as_index=False)[["adapted_online_mse", "offline_test_mse", "offline_test_mae", "improve_vs_base"]]
        .mean()
        .sort_values("adapted_online_mse")
    )
    summary.to_csv(ROOT / "logs" / "lishui_benchmarks" / "family_summary.csv", index=False, encoding="utf-8-sig")

    pivot = latest.pivot(index="dataset", columns="variant", values="adapted_online_mse").reset_index()
    pivot.to_csv(ROOT / "logs" / "lishui_benchmarks" / "family_by_county.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", context="paper", font="serif")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=summary, x="variant", y="adapted_online_mse", ax=ax, palette="Blues_r")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Online MSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "lishui_benchmarks" / "paper_family_summary.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    heat = latest.pivot(index="dataset", columns="variant", values="adapted_online_mse")
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "lishui_benchmarks" / "paper_family_heatmap.png", dpi=300)
    plt.close(fig)

    lines = [
        "# Lishui-9 Family Benchmark",
        "",
        "## Mean over 9 county-level datasets",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-county online MSE",
        "",
        pivot.to_markdown(index=False),
    ]
    (ROOT / "logs" / "lishui_benchmarks" / "family_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
