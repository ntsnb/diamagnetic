from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import apply_ieee_plot_style


OUT_DIR = ROOT / "logs" / "lishui_benchmarks"

COUNTY_ORDER = [
    "Yunhe",
    "Qingyuan",
    "Jingning",
    "Songyang",
    "Jinyun",
    "Liandu",
    "Suichang",
    "Qingtian",
    "Longquan",
]

METHOD_ORDER = [
    "frozen",
    "gd_full",
    "gd_last_layer",
    "onenet",
    "solid",
    "solidpp",
    "proceed",
    "ours_stage3",
    "ours_full",
]
METHOD_LABELS = {
    "frozen": "Frozen",
    "gd_full": "GD",
    "gd_last_layer": "Head-GD",
    "onenet": "OneNet",
    "solid": "SOLID",
    "solidpp": "SOLID++",
    "proceed": "PROCEED",
    "ours_stage3": "Ours-III",
    "ours_full": "Ours-Full",
}


def _load_county_table() -> pd.DataFrame:
    table_path = OUT_DIR / "shifted_sota_by_county.csv"
    if table_path.exists():
        return pd.read_csv(table_path)

    raw_candidates = [
        OUT_DIR / "shifted_sota_methods_with_mae.csv",
        OUT_DIR / "shifted_sota_methods.csv",
    ]
    for raw_path in raw_candidates:
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            df["online_mse"] = pd.to_numeric(df["online_mse"])
            latest = (
                df.sort_values("timestamp")
                .groupby(["dataset", "method"], as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )
            latest["dataset_display"] = latest["dataset"]
            for raw, disp in [
                ("lishui_浜戝拰", "Yunhe"),
                ("lishui_搴嗗厓", "Qingyuan"),
                ("lishui_鏅畞", "Jingning"),
                ("lishui_鏉鹃槼", "Songyang"),
                ("lishui_缂欎簯", "Jinyun"),
                ("lishui_鑾查兘", "Liandu"),
                ("lishui_閬傛槍", "Suichang"),
                ("lishui_闈掔敯", "Qingtian"),
                ("lishui_榫欐硥", "Longquan"),
                ("lishui_云和", "Yunhe"),
                ("lishui_庆元", "Qingyuan"),
                ("lishui_景宁", "Jingning"),
                ("lishui_松阳", "Songyang"),
                ("lishui_缙云", "Jinyun"),
                ("lishui_莲都", "Liandu"),
                ("lishui_遂昌", "Suichang"),
                ("lishui_青田", "Qingtian"),
                ("lishui_龙泉", "Longquan"),
            ]:
                latest.loc[latest["dataset"] == raw, "dataset_display"] = disp
            pivot = latest.pivot(index="dataset_display", columns="method", values="online_mse").reset_index()
            pivot.to_csv(table_path, index=False, encoding="utf-8-sig")
            return pivot

    raise FileNotFoundError(f"Missing {table_path} and raw shifted SOTA csv files")


def main() -> None:
    apply_ieee_plot_style()

    df = _load_county_table()
    df = df.rename(columns={"dataset_display": "county"})
    df["county"] = pd.Categorical(df["county"], categories=COUNTY_ORDER, ordered=True)
    df = df.sort_values("county").reset_index(drop=True)

    present_methods = [m for m in METHOD_ORDER if m in df.columns]
    colors = {
        "frozen": "#c6d4e1",
        "gd_full": "#ef8a62",
        "gd_last_layer": "#fdb863",
        "onenet": "#80cdc1",
        "solid": "#92c5de",
        "solidpp": "#4393c3",
        "proceed": "#b2abd2",
        "ours_stage3": "#5aae61",
        "ours_full": "#1b7837",
    }

    fig, axes = plt.subplots(3, 3, figsize=(15.2, 10.2), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        values = [float(row[m]) for m in present_methods]
        x = np.arange(len(present_methods))
        bar_colors = [colors[m] for m in present_methods]
        bars = ax.bar(x, values, color=bar_colors, edgecolor="#4a4a4a", linewidth=0.45)
        best_idx = int(np.argmin(values))
        bars[best_idx].set_linewidth(1.2)
        bars[best_idx].set_edgecolor("#111111")

        ymin = min(values)
        ymax = max(values)
        margin = max((ymax - ymin) * 0.22, 0.004)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(str(row["county"]), pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in present_methods], rotation=45, ha="right")
        ax.set_ylabel("Online MSE")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=1, fontsize=6)

    for ax in axes[len(df) :]:
        ax.axis("off")

    fig.tight_layout(h_pad=2.0, w_pad=1.2)
    fig.savefig(OUT_DIR / "paper_shifted_sota_nine_counties_bar.png", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
