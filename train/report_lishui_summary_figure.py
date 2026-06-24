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

STAGE_ORDER = [
    "baseline",
    "stage1",
    "stage2",
    "stage3",
    "stage1_stage2",
    "stage1_stage3",
    "stage2_stage3",
    "stage1_stage2_stage3",
]
STAGE_LABELS = {
    "baseline": "Frozen",
    "stage1": "Stage I",
    "stage2": "Stage II",
    "stage3": "Stage III",
    "stage1_stage2": "I+II",
    "stage1_stage3": "I+III",
    "stage2_stage3": "II+III",
    "stage1_stage2_stage3": "I+II+III",
}

SHIFT_ORDER = ["frozen", "solidpp", "ours_stage3", "ours_full"]
SHIFT_LABELS = {
    "frozen": "Frozen",
    "solidpp": "SOLID++",
    "ours_stage3": "Ours-III",
    "ours_full": "Ours-Full",
}


def _load_stage_summary() -> pd.DataFrame:
    summary_path = OUT_DIR / "family_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["adapted_online_mse"] = pd.to_numeric(df["adapted_online_mse"])
        return df

    raw_path = OUT_DIR / "family_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing both {summary_path} and {raw_path}")
    df = pd.read_csv(raw_path)
    df["adapted_online_mse"] = pd.to_numeric(df["adapted_online_mse"])
    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "variant"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    summary = latest.groupby("variant", as_index=False)["adapted_online_mse"].mean()
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def _load_shift_summary() -> pd.DataFrame:
    summary_path = OUT_DIR / "final_shifted_tuned_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["online_mse"] = pd.to_numeric(df["online_mse"])
        return df

    raw_path = OUT_DIR / "final_shifted_tuned_by_county.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing both {summary_path} and {raw_path}")
    df = pd.read_csv(raw_path)
    methods = [c for c in df.columns if c != "dataset"]
    summary = pd.DataFrame(
        {
            "method": methods,
            "online_mse": [pd.to_numeric(df[m]).mean() for m in methods],
        }
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def _plot_panel(ax, labels: list[str], values: list[float], title: str, best_idx: int) -> None:
    x = np.arange(len(labels))
    colors = ["#b9c8d8"] * len(labels)
    colors[best_idx] = "#325d88"
    bars = ax.bar(x, values, color=colors, edgecolor="#4a4a4a", linewidth=0.5)
    ax.set_title(title, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Online MSE")

    ymin = min(values)
    ymax = max(values)
    margin = max((ymax - ymin) * 0.18, 0.0008)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=2, fontsize=7)


def main() -> None:
    apply_ieee_plot_style()

    stage_df = _load_stage_summary()
    stage_series = stage_df.set_index("variant")["adapted_online_mse"]
    stage_values = [float(stage_series[k]) for k in STAGE_ORDER if k in stage_series.index]
    stage_labels = [STAGE_LABELS[k] for k in STAGE_ORDER if k in stage_series.index]
    stage_best = int(np.argmin(stage_values))

    shift_df = _load_shift_summary()
    shift_series = shift_df.set_index("method")["online_mse"]
    shift_values = [float(shift_series[k]) for k in SHIFT_ORDER if k in shift_series.index]
    shift_labels = [SHIFT_LABELS[k] for k in SHIFT_ORDER if k in shift_series.index]
    shift_best = int(np.argmin(shift_values))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.1))
    _plot_panel(
        axes[0],
        stage_labels,
        stage_values,
        "Lishui-9 Family Stage Combinations",
        stage_best,
    )
    _plot_panel(
        axes[1],
        shift_labels,
        shift_values,
        "Tuned Shifted-Stream Comparison",
        shift_best,
    )
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "paper_lishui_summary.png", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
