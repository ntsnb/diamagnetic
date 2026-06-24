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


VARIANT_LABELS = {
    "baseline": "Baseline",
    "stage2": "Stage II",
    "stage3": "Stage III",
    "stage1_stage2_stage3": "Full",
}
VARIANT_ORDER = ["baseline", "stage2", "stage3", "stage1_stage2_stage3"]
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        df.sort_values("timestamp")
        .groupby(["backbone", "dataset", "variant"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    out = (
        latest.groupby(["backbone", "variant"], as_index=False)[
            ["offline_test_mse", "offline_test_mae", "adapted_online_mse", "improve_vs_base"]
        ]
        .mean()
    )
    return out


def build_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Eight-Backbone Online Variant Comparison",
        "",
        "This report summarizes the broad backbone comparison with explicit `Baseline`, `Stage II`, `Stage III`, and `Full` variants under the same broad-screening protocol.",
        "",
    ]
    pivot = summary.pivot(index="backbone", columns="variant", values="adapted_online_mse")
    for backbone in BACKBONE_ORDER:
        if backbone not in pivot.index:
            continue
        row = pivot.loc[backbone]
        base = float(row["baseline"])
        s2 = float(row["stage2"])
        s3 = float(row["stage3"])
        full = float(row["stage1_stage2_stage3"])
        lines.extend(
            [
                f"## {BACKBONE_LABELS[backbone]}",
                "",
                f"- Baseline: `{base:.6f}`",
                f"- Stage II: `{s2:.6f}`",
                f"- Stage III: `{s3:.6f}`",
                f"- Full: `{full:.6f}`",
                f"- Stage II gain vs baseline: `{base - s2:.6f}`",
                f"- Stage III gain vs baseline: `{base - s3:.6f}`",
                f"- Full gain vs baseline: `{base - full:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Stage II is typically weaker than Stage III because it relies on the quality of the relevance drift signal and on a stable expert decomposition. When the temporal encoder is weak, the relevance-guided sparse update becomes noisy or too conservative, so the gain is small.",
            "- Stage III is more robust because it is an inference-time correction and does not permanently perturb model parameters.",
            "- The full controller is strongest when Stage II can provide complementary corrections without destabilizing the backbone representation.",
            "",
        ]
    )
    return "\n".join(lines)


def plot(summary: pd.DataFrame, output: Path) -> None:
    apply_ieee_plot_style()
    plot_df = summary.copy()
    plot_df = plot_df[plot_df["variant"].isin(VARIANT_ORDER)].copy()
    plot_df["backbone_label"] = plot_df["backbone"].map(BACKBONE_LABELS)
    plot_df["variant_label"] = plot_df["variant"].map(VARIANT_LABELS)
    plot_df["backbone_label"] = pd.Categorical(
        plot_df["backbone_label"],
        categories=[BACKBONE_LABELS[b] for b in BACKBONE_ORDER if b in plot_df["backbone"].unique()],
        ordered=True,
    )
    plot_df["variant_label"] = pd.Categorical(
        plot_df["variant_label"],
        categories=[VARIANT_LABELS[v] for v in VARIANT_ORDER],
        ordered=True,
    )
    plot_df = plot_df.sort_values(["backbone_label", "variant_label"])

    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    width = 0.19
    xlabels = [BACKBONE_LABELS[b] for b in BACKBONE_ORDER if b in plot_df["backbone"].unique()]
    x = range(len(xlabels))
    colors = {
        "Baseline": "#4C78A8",
        "Stage II": "#9ECAE9",
        "Stage III": "#F58518",
        "Full": "#54A24B",
    }
    for i, variant in enumerate([VARIANT_LABELS[v] for v in VARIANT_ORDER]):
        vals = []
        for label in xlabels:
            sub = plot_df[(plot_df["backbone_label"] == label) & (plot_df["variant_label"] == variant)]
            vals.append(float(sub["adapted_online_mse"].iloc[0]))
        ax.bar([xx + (i - 1.5) * width for xx in x], vals, width=width, label=variant, color=colors[variant])

    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("Mean Online MSE")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/load_benchmarks/backbone_suite_v2.csv")
    parser.add_argument("--summary", default="logs/load_benchmarks/backbone_variants_summary.csv")
    parser.add_argument("--report", default="logs/load_benchmarks/backbone_variants_report.md")
    parser.add_argument("--figure", default="logs/load_benchmarks/paper_backbone_variants_all8.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary = summarize(df)
    summary.to_csv(args.summary, index=False)
    Path(args.report).write_text(build_markdown(summary), encoding="utf-8")
    plot(summary, Path(args.figure))


if __name__ == "__main__":
    main()
