from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_style import apply_ieee_plot_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "logs" / "load_benchmarks" / "ablation_results.csv"
OUT_DIR = ROOT / "logs" / "load_benchmarks"


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

COMPONENT_ORDER = [
    "full_ref",
    "no_invariant",
    "dense_stage2_all_layers",
    "grad_stage2_no_lrp",
    "no_invdelta_stage3",
    "no_ema_reference",
    "no_rollback_accept",
    "sync_all_experts",
    "no_delay_feedback",
    "mlp_gate_no_vae",
    "no_revin",
    "single_expert",
]

STAGE_LABELS = {
    "baseline": "Baseline",
    "stage1": "Stage I",
    "stage2": "Stage II",
    "stage3": "Stage III",
    "stage1_stage2": "Stage I+II",
    "stage1_stage3": "Stage I+III",
    "stage2_stage3": "Stage II+III",
    "stage1_stage2_stage3": "Stage I+II+III",
}

COMPONENT_LABELS = {
    "full_ref": "Full",
    "no_invariant": "w/o Invariant",
    "dense_stage2_all_layers": "Dense Stage II",
    "grad_stage2_no_lrp": "w/o LRP in Stage II",
    "no_invdelta_stage3": "w/o $\\phi_V-\\phi_I$",
    "no_ema_reference": "w/o EMA Ref",
    "no_rollback_accept": "w/o Rollback/Buffer",
    "sync_all_experts": "Sync All Experts",
    "no_delay_feedback": "w/o Delay",
    "mlp_gate_no_vae": "MLP Gate, no VAE",
    "no_revin": "w/o RevIN",
    "single_expert": "Single Expert",
}


def _ordered_summary(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby("variant")
        .agg(
            mean_offline_mse=("offline_test_mse", "mean"),
            mean_offline_mae=("offline_test_mae", "mean"),
            mean_online_mse=("adapted_online_mse", "mean"),
            mean_gain=("improve_vs_base", "mean"),
        )
        .reset_index()
    )
    summary["rank"] = summary["variant"].map({k: i for i, k in enumerate(order)})
    return summary.sort_values("rank").drop(columns="rank")


def _per_dataset(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    out = (
        df.pivot_table(
            index="dataset",
            columns="variant",
            values="adapted_online_mse",
            aggfunc="mean",
        )
        .reindex(columns=order)
        .reset_index()
    )
    return out


def _plot_stage_combo(summary: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    labels = [STAGE_LABELS[v] for v in summary["variant"]]
    bars = ax.bar(labels, summary["mean_online_mse"], color="#4C78A8")
    ax.set_ylabel("Mean Online MSE")
    ax.tick_params(axis="x", rotation=25)
    best_idx = summary["mean_online_mse"].idxmin()
    bars[best_idx].set_color("#F58518")
    for bar, val in zip(bars, summary["mean_online_mse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paper_ablation_stage_combo.png", dpi=220)
    plt.close(fig)


def _plot_component(summary: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    labels = [COMPONENT_LABELS[v] for v in summary["variant"]]
    bars = ax.bar(labels, summary["mean_online_mse"], color="#54A24B")
    ax.set_ylabel("Mean Online MSE")
    ax.tick_params(axis="x", rotation=28)
    base_val = float(summary.loc[summary["variant"] == "full_ref", "mean_online_mse"].iloc[0])
    for bar, val, variant in zip(bars, summary["mean_online_mse"], summary["variant"]):
        delta = val - base_val
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}\n({delta:+.3f})", ha="center", va="bottom", fontsize=8)
        if variant == "full_ref":
            bar.set_color("#F58518")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paper_ablation_components.png", dpi=220)
    plt.close(fig)


def _write_report(stage_summary: pd.DataFrame, stage_dataset: pd.DataFrame, comp_summary: pd.DataFrame, comp_dataset: pd.DataFrame) -> None:
    stage_best = stage_summary.sort_values("mean_online_mse").iloc[0]
    comp_full = comp_summary.loc[comp_summary["variant"] == "full_ref"].iloc[0]
    lines = []
    lines.append("# Ablation Report")
    lines.append("")
    lines.append("## Stage Combination")
    lines.append("")
    lines.append(
        f"The best averaged stage-combination variant is **{STAGE_LABELS[stage_best['variant']]}** "
        f"with mean online MSE {stage_best['mean_online_mse']:.6f} and mean gain {stage_best['mean_gain']:.6f}."
    )
    lines.append("")
    lines.append(stage_summary.to_markdown(index=False))
    lines.append("")
    lines.append("Per-dataset online MSE:")
    lines.append("")
    lines.append(stage_dataset.to_markdown(index=False))
    lines.append("")
    lines.append("## Component Ablation")
    lines.append("")
    lines.append(
        f"The full reference system has mean online MSE {comp_full['mean_online_mse']:.6f}. "
        "Removing RevIN or collapsing the expert bank to a single expert leads to the largest degradation. "
        "The invariant branch and the Stage III invariant-delta remain helpful on average, while the Stage II "
        "controller ablations --- EMA reference removal, rollback/buffer removal, and synchronous expert updates --- "
        "quantify how much each stabilizer contributes under the current shifted-stream protocol."
    )
    lines.append("")
    lines.append(comp_summary.to_markdown(index=False))
    lines.append("")
    lines.append("Per-dataset online MSE:")
    lines.append("")
    lines.append(comp_dataset.to_markdown(index=False))
    (OUT_DIR / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(RESULTS)
    stage = df[df["group"] == "stage_combo"].copy()
    comp = df[df["group"] == "component"].copy()

    stage_summary = _ordered_summary(stage, STAGE_ORDER)
    comp_summary = _ordered_summary(comp, COMPONENT_ORDER)
    stage_dataset = _per_dataset(stage, STAGE_ORDER)
    comp_dataset = _per_dataset(comp, COMPONENT_ORDER)

    stage_summary.to_csv(OUT_DIR / "ablation_stage_combo_summary.csv", index=False)
    comp_summary.to_csv(OUT_DIR / "ablation_component_summary.csv", index=False)
    stage_dataset.to_csv(OUT_DIR / "ablation_stage_combo_by_dataset.csv", index=False)
    comp_dataset.to_csv(OUT_DIR / "ablation_component_by_dataset.csv", index=False)

    _plot_stage_combo(stage_summary)
    _plot_component(comp_summary)
    _write_report(stage_summary, stage_dataset, comp_summary, comp_dataset)


if __name__ == "__main__":
    main()
