from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISPLAY_NAMES = {
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


def main() -> None:
    path = ROOT / "logs" / "lishui_benchmarks" / "shifted_sota_methods.csv"
    df = pd.read_csv(path)
    df["online_mse"] = pd.to_numeric(df["online_mse"])
    df["shift_score"] = pd.to_numeric(df["shift_score"])
    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "method"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["dataset_display"] = latest["dataset"].map(DISPLAY_NAMES).fillna(latest["dataset"])
    summary = latest.groupby("method", as_index=False)["online_mse"].mean().sort_values("online_mse")
    summary.to_csv(ROOT / "logs" / "lishui_benchmarks" / "shifted_sota_summary.csv", index=False, encoding="utf-8-sig")
    pivot = latest.pivot(index="dataset_display", columns="method", values="online_mse").reset_index()
    pivot.to_csv(ROOT / "logs" / "lishui_benchmarks" / "shifted_sota_by_county.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", context="paper", font="serif")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=summary, x="method", y="online_mse", hue="method", legend=False, ax=ax, palette="viridis_r")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Online MSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "lishui_benchmarks" / "paper_shifted_sota_summary.png", dpi=300)
    plt.close(fig)

    report = [
        "# Lishui-9 Shifted Online SOTA Comparison",
        "",
        "## Mean over 9 counties",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-county online MSE",
        "",
        pivot.to_markdown(index=False),
    ]
    (ROOT / "logs" / "lishui_benchmarks" / "shifted_sota_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
