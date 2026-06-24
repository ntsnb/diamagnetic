from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_style import apply_ieee_plot_style

LOAD_DIR = ROOT / "logs" / "load_benchmarks"
LISHUI_DIR = ROOT / "logs" / "lishui_benchmarks"


def build_shifted_baselines_with_lishui() -> pd.DataFrame:
    public = pd.read_csv(LOAD_DIR / "shifted_online_baselines.csv")
    public = (
        public.sort_values("timestamp")
        .groupby(["dataset", "method"], as_index=False)
        .tail(1)[["dataset", "method", "online_mse", "shift_score"]]
    )
    lishui_raw = pd.read_csv(LISHUI_DIR / "shifted_sota_methods.csv")
    lishui = (
        lishui_raw[lishui_raw["method"].isin(["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"])]
        .groupby("method", as_index=False)
        .agg(online_mse=("online_mse", "mean"))
    )
    lishui["dataset"] = "lishui9"
    lishui["shift_score"] = float(lishui_raw["shift_score"].mean())
    merged = pd.concat([public, lishui[["dataset", "method", "online_mse", "shift_score"]]], ignore_index=True)
    merged.to_csv(LOAD_DIR / "shifted_online_baselines_with_lishui.csv", index=False)
    return merged


def build_shifted_sota_with_lishui() -> pd.DataFrame:
    public = pd.read_csv(LOAD_DIR / "shifted_sota_methods_summary.csv")
    long = public.melt(id_vars=["dataset"], var_name="label", value_name="online_mse")
    rev = {
        "Frozen": "frozen",
        "GD": "gd_full",
        "Head-GD": "gd_last_layer",
        "OneNet": "onenet",
        "SOLID": "solid",
        "SOLID++": "solidpp",
        "PROCEED": "proceed",
        "Ours-Stage III": "ours_stage3",
        "Ours-Full": "ours_full",
    }
    public_shift = (
        pd.read_csv(LOAD_DIR / "shifted_online_baselines_with_lishui.csv")
        .drop_duplicates("dataset")
        .set_index("dataset")["shift_score"]
    )
    long["method"] = long["label"].map(rev)
    long["shift_score"] = long["dataset"].map(public_shift)
    lishui = pd.read_csv(LISHUI_DIR / "shifted_sota_methods.csv")
    lishui = lishui.groupby("method", as_index=False).agg(online_mse=("online_mse", "mean"))
    lishui["dataset"] = "lishui9"
    lishui["shift_score"] = float(pd.read_csv(LISHUI_DIR / "shifted_sota_methods.csv")["shift_score"].mean())
    merged = pd.concat([long[["dataset", "method", "online_mse", "shift_score"]], lishui], ignore_index=True)
    merged.to_csv(LOAD_DIR / "shifted_sota_methods_with_lishui.csv", index=False)
    return merged


def build_ablations_with_lishui() -> tuple[pd.DataFrame, pd.DataFrame]:
    pub_stage = pd.read_csv(LOAD_DIR / "ablation_stage_combo_summary.csv")[["variant", "mean_online_mse"]]
    pub_stage = pub_stage.rename(columns={"mean_online_mse": "public_online_mse"})
    li_stage = pd.read_csv(LISHUI_DIR / "family_summary.csv")[["variant", "adapted_online_mse"]]
    li_stage = li_stage.rename(columns={"adapted_online_mse": "lishui_online_mse"})
    stage = pub_stage.merge(li_stage, on="variant", how="left")
    stage.to_csv(LOAD_DIR / "ablation_stage_combo_with_lishui.csv", index=False)

    pub_comp = pd.read_csv(LOAD_DIR / "ablation_component_summary.csv")
    pub_comp["variant"] = pub_comp["variant"].map(
        {
            "full_ref": "full",
            "no_invariant": "wo_invariant",
            "dense_stage2_all_layers": "dense_stage2",
            "mlp_gate_no_vae": "mlp_gate_no_vae",
            "no_revin": "wo_revin",
            "single_expert": "single_expert",
        }
    )
    pub_comp = pub_comp.dropna(subset=["variant"])[["variant", "mean_online_mse"]]
    pub_comp = pub_comp.rename(columns={"mean_online_mse": "public_online_mse"})
    li_comp = pd.read_csv(LISHUI_DIR / "component_ablation_summary.csv")[["ablation", "online_mse"]]
    li_comp = li_comp.rename(columns={"ablation": "variant", "online_mse": "lishui_online_mse"})
    comp = pub_comp.merge(li_comp, on="variant", how="outer")
    comp.to_csv(LOAD_DIR / "ablation_component_with_lishui.csv", index=False)
    return stage, comp


def draw_shifted_baselines(df: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    order = ["panama", "tetouan", "spain_energy_weather", "lishui9"]
    labels = ["Panama", "Tetouan", "Spain Energy--Weather", "Lishui-9"]
    methods = ["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"]
    display = {
        "frozen": "Frozen",
        "gd_full": "GD",
        "gd_last_layer": "Head-GD",
        "ours_stage3": "Ours-Stage III",
        "ours_full": "Ours-Full",
    }
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B"]
    x = np.arange(len(order))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    for i, method in enumerate(methods):
        vals = [float(df[(df["dataset"] == d) & (df["method"] == method)]["online_mse"].iloc[0]) for d in order]
        ax.bar(x + (i - 2) * width, vals, width=width, label=display[method], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Online MSE")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(LOAD_DIR / "paper_shifted_baselines.png", dpi=300)
    plt.close(fig)


def draw_shifted_sota(df: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    order = ["panama", "tetouan", "spain_energy_weather", "lishui9"]
    titles = ["Panama", "Tetouan", "Spain Energy--Weather", "Lishui-9"]
    methods = ["frozen", "gd_full", "gd_last_layer", "onenet", "solid", "solidpp", "proceed", "ours_stage3", "ours_full"]
    labels = {
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
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for ax, dataset, title in zip(axes.flat, order, titles):
        sub = df[df["dataset"] == dataset].set_index("method")
        vals = [float(sub.loc[m, "online_mse"]) for m in methods]
        cols = ["#4C78A8" if not m.startswith("ours") else "#F58518" for m in methods]
        ax.bar(range(len(methods)), vals, color=cols)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods], rotation=45, ha="right")
        ax.set_ylabel("Online MSE")
        ax.set_xlabel(title)
    fig.tight_layout()
    fig.savefig(LOAD_DIR / "paper_shifted_sota_methods.png", dpi=300)
    plt.close(fig)


def draw_ablation_stage(stage: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    order = ["baseline", "stage1", "stage2", "stage3", "stage1_stage2_stage3"]
    labels = {
        "baseline": "Baseline",
        "stage1": "Stage I",
        "stage2": "Stage II",
        "stage3": "Stage III",
        "stage1_stage2_stage3": "Full",
    }
    sub = stage[stage["variant"].isin(order)].copy()
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    sub = sub.sort_values("variant")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].bar([labels[v] for v in sub["variant"]], sub["public_online_mse"], color="#4C78A8")
    axes[0].set_title("Public benchmarks")
    axes[0].set_ylabel("Mean online MSE")
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].bar([labels[v] for v in sub["variant"]], sub["lishui_online_mse"], color="#F58518")
    axes[1].set_title("Lishui-9")
    axes[1].tick_params(axis="x", rotation=22)
    fig.tight_layout()
    fig.savefig(LOAD_DIR / "paper_ablation_stage_combo.png", dpi=300)
    plt.close(fig)


def draw_ablation_component(comp: pd.DataFrame) -> None:
    apply_ieee_plot_style()
    order = ["full", "wo_invariant", "dense_stage2", "mlp_gate_no_vae", "wo_revin", "single_expert"]
    labels = {
        "full": "Full",
        "wo_invariant": "w/o Invariant",
        "dense_stage2": "Dense Stage II",
        "mlp_gate_no_vae": "MLP Gate, no VAE",
        "wo_revin": "w/o RevIN",
        "single_expert": "Single Expert",
    }
    sub = comp[comp["variant"].isin(order)].copy()
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    sub = sub.sort_values("variant")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].bar([labels[v] for v in sub["variant"]], sub["public_online_mse"], color="#54A24B")
    axes[0].set_title("Public benchmarks")
    axes[0].set_ylabel("Mean online MSE")
    axes[0].tick_params(axis="x", rotation=28)
    axes[1].bar([labels[v] for v in sub["variant"]], sub["lishui_online_mse"], color="#DD8452")
    axes[1].set_title("Lishui-9")
    axes[1].tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(LOAD_DIR / "paper_ablation_components.png", dpi=300)
    plt.close(fig)


def main() -> None:
    baselines = build_shifted_baselines_with_lishui()
    sota = build_shifted_sota_with_lishui()
    stage, comp = build_ablations_with_lishui()
    draw_shifted_baselines(baselines)
    draw_shifted_sota(sota)
    draw_ablation_stage(stage)
    draw_ablation_component(comp)


if __name__ == "__main__":
    main()
