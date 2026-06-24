from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    path = ROOT / "logs" / "lishui_benchmarks" / "component_ablation.csv"
    df = pd.read_csv(path)
    df["online_mse"] = pd.to_numeric(df["online_mse"])
    df["gain_vs_base"] = pd.to_numeric(df["gain_vs_base"])
    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "ablation"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    summary = latest.groupby("ablation", as_index=False)[["online_mse", "gain_vs_base"]].mean().sort_values("online_mse")
    summary.to_csv(ROOT / "logs" / "lishui_benchmarks" / "component_ablation_summary.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", context="paper", font="serif")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=summary, x="ablation", y="online_mse", hue="ablation", legend=False, palette="magma_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Online MSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "lishui_benchmarks" / "paper_component_ablation.png", dpi=300)
    plt.close(fig)

    lines = [
        "# Lishui-9 Component Ablation",
        "",
        summary.to_markdown(index=False),
    ]
    (ROOT / "logs" / "lishui_benchmarks" / "component_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
