from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import apply_ieee_plot_style


LOAD_DIR = ROOT / "logs" / "load_benchmarks"
LISHUI_DIR = ROOT / "logs" / "lishui_benchmarks"

PUBLIC_DATASETS = ["panama", "tetouan", "spain_energy_weather"]
PUBLIC_LABELS = {
    "panama": "Panama",
    "tetouan": "Tetouan",
    "spain_energy_weather": "Spain Energy--Weather",
}
LISHUI_LABEL = "Lishui-9 family"
COUNTY_DISPLAY = {
    "lishui_云和": "Yunhe",
    "lishui_庆元": "Qingyuan",
    "lishui_景宁": "Jingning",
    "lishui_松阳": "Songyang",
    "lishui_缙云": "Jinyun",
    "lishui_莲都": "Liandu",
    "lishui_遂昌": "Suichang",
    "lishui_青田": "Qingtian",
    "lishui_龙泉": "Longquan",
    "Jingning": "Jingning",
    "Jinyun": "Jinyun",
    "Liandu": "Liandu",
    "Longquan": "Longquan",
    "Qingtian": "Qingtian",
    "Qingyuan": "Qingyuan",
    "Songyang": "Songyang",
    "Suichang": "Suichang",
    "Yunhe": "Yunhe",
}


def _latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.copy()
    return df.sort_values("timestamp").groupby(keys, as_index=False).tail(1).reset_index(drop=True)


def _lishui_mean_sota() -> pd.DataFrame:
    df = pd.read_csv(LISHUI_DIR / "shifted_sota_methods.csv")
    latest = _latest(df, ["dataset", "method"])
    return latest.groupby("method", as_index=False).agg(
        online_mse=("online_mse", "mean"),
        shift_score=("shift_score", "mean"),
    )


def _plot_dataset_panels(
    dataset_values: dict[str, pd.Series],
    order: list[str],
    labels: dict[str, str],
    outpath: Path,
    ylabel: str = "Online MSE",
    ncols: int = 2,
    rotate: int = 28,
) -> None:
    apply_ieee_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    axes = axes.flatten()
    colors = sns.color_palette("deep", n_colors=len(order))
    for ax, (dataset, values) in zip(axes, dataset_values.items()):
        x = np.arange(len(order))
        y = [float(values.get(k, np.nan)) for k in order]
        ax.bar(x, y, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in order], rotation=rotate, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def build_shifted_baseline_figures() -> None:
    public = pd.read_csv(LOAD_DIR / "shifted_online_baselines.csv")
    public = _latest(public, ["dataset", "method"])
    public = public[public["method"].isin(["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"])]
    family = _lishui_mean_sota()
    family = family[family["method"].isin(["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"])]
    family["dataset"] = LISHUI_LABEL
    merged = pd.concat([public[["dataset", "method", "online_mse"]], family[["dataset", "method", "online_mse"]]], ignore_index=True)
    order = ["frozen", "gd_full", "gd_last_layer", "ours_stage3", "ours_full"]
    labels = {
        "frozen": "Frozen",
        "gd_full": "GD",
        "gd_last_layer": "Head-GD",
        "ours_stage3": "Ours-III",
        "ours_full": "Ours-Full",
    }
    panels = {}
    for dataset in PUBLIC_DATASETS:
        panels[PUBLIC_LABELS[dataset]] = merged[merged["dataset"] == dataset].set_index("method")["online_mse"]
    panels[LISHUI_LABEL] = merged[merged["dataset"] == LISHUI_LABEL].set_index("method")["online_mse"]
    _plot_dataset_panels(panels, order, labels, LOAD_DIR / "paper_shifted_baselines_by_dataset.png")

    county = pd.read_csv(LISHUI_DIR / "shifted_sota_by_county.csv")
    county = county.rename(
        columns={
            "frozen": "Frozen",
            "gd_full": "GD",
            "gd_last_layer": "Head-GD",
            "ours_stage3": "Ours-III",
            "ours_full": "Ours-Full",
        }
    )
    county["dataset_display"] = county["dataset_display"].map(lambda x: COUNTY_DISPLAY.get(x, x))
    heat = county.set_index("dataset_display")[["Frozen", "GD", "Head-GD", "Ours-III", "Ours-Full"]]
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(LISHUI_DIR / "paper_shifted_baselines_by_county.png", dpi=300)
    plt.close(fig)


def build_shifted_sota_figures() -> None:
    public = pd.read_csv(LOAD_DIR / "shifted_sota_methods.csv")
    public = _latest(public, ["dataset", "method"])
    family = _lishui_mean_sota()
    family["dataset"] = LISHUI_LABEL
    merged = pd.concat([public[["dataset", "method", "online_mse"]], family[["dataset", "method", "online_mse"]]], ignore_index=True)
    order = ["frozen", "gd_full", "gd_last_layer", "onenet", "solid", "solidpp", "proceed", "ours_stage3", "ours_full"]
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
    panels = {}
    for dataset in PUBLIC_DATASETS:
        panels[PUBLIC_LABELS[dataset]] = merged[merged["dataset"] == dataset].set_index("method")["online_mse"]
    panels[LISHUI_LABEL] = merged[merged["dataset"] == LISHUI_LABEL].set_index("method")["online_mse"]
    _plot_dataset_panels(panels, order, labels, LOAD_DIR / "paper_shifted_sota_methods_by_dataset.png", rotate=40)

    county = pd.read_csv(LISHUI_DIR / "shifted_sota_by_county.csv")
    county["dataset_display"] = county["dataset_display"].map(lambda x: COUNTY_DISPLAY.get(x, x))
    county = county.set_index("dataset_display")
    county = county.rename(
        columns={
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
    )
    heat = county[["Frozen", "GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED", "Ours-III", "Ours-Full"]]
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(LISHUI_DIR / "paper_shifted_sota_by_county.png", dpi=300)
    plt.close(fig)


def build_stage_combo_figures() -> None:
    public = pd.read_csv(LOAD_DIR / "ablation_stage_combo_by_dataset.csv")
    family = pd.read_csv(LISHUI_DIR / "family_summary.csv").set_index("variant")["adapted_online_mse"]
    order = ["baseline", "stage1", "stage2", "stage3", "stage1_stage2", "stage1_stage3", "stage2_stage3", "stage1_stage2_stage3"]
    labels = {
        "baseline": "Baseline",
        "stage1": "I",
        "stage2": "II",
        "stage3": "III",
        "stage1_stage2": "I+II",
        "stage1_stage3": "I+III",
        "stage2_stage3": "II+III",
        "stage1_stage2_stage3": "I+II+III",
    }
    panels = {
        PUBLIC_LABELS[row["dataset"]]: pd.Series({k: row[k] for k in order if k in row.index})
        for _, row in public.iterrows()
    }
    panels[LISHUI_LABEL] = family
    _plot_dataset_panels(panels, order, labels, LOAD_DIR / "paper_ablation_stage_combo_by_dataset.png", rotate=35)

    county = pd.read_csv(LISHUI_DIR / "family_by_county.csv").rename(
        columns={
            "baseline": "Baseline",
            "stage1": "I",
            "stage2": "II",
            "stage3": "III",
            "stage1_stage2_stage3": "I+II+III",
        }
    )
    county["dataset"] = county["dataset"].map(lambda x: COUNTY_DISPLAY.get(x, x))
    heat = county.set_index("dataset")[["Baseline", "I", "II", "III", "I+II+III"]]
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(LISHUI_DIR / "paper_stage_combo_by_county.png", dpi=300)
    plt.close(fig)


def build_component_figures() -> None:
    public = pd.read_csv(LOAD_DIR / "ablation_component_by_dataset.csv").rename(
        columns={
            "full_ref": "full",
            "no_invariant": "wo_invariant",
            "dense_stage2_all_layers": "dense_stage2",
            "mlp_gate_no_vae": "mlp_gate_no_vae",
            "no_revin": "wo_revin",
            "single_expert": "single_expert",
        }
    )
    family = pd.read_csv(LISHUI_DIR / "component_ablation_summary.csv").rename(columns={"ablation": "variant"})
    family = family.set_index("variant")["online_mse"]
    order = ["full", "wo_invariant", "dense_stage2", "mlp_gate_no_vae", "wo_revin", "single_expert"]
    labels = {
        "full": "Full",
        "wo_invariant": "w/o Inv",
        "dense_stage2": "Dense-II",
        "mlp_gate_no_vae": "MLP gate",
        "wo_revin": "w/o RevIN",
        "single_expert": "1 expert",
    }
    panels = {
        PUBLIC_LABELS[row["dataset"]]: pd.Series({k: row[k] for k in order if k in row.index})
        for _, row in public.iterrows()
    }
    panels[LISHUI_LABEL] = family
    _plot_dataset_panels(panels, order, labels, LOAD_DIR / "paper_ablation_components_by_dataset.png", rotate=30)

    county = pd.read_csv(LISHUI_DIR / "component_ablation.csv")
    latest = _latest(county, ["dataset", "ablation"])
    latest["dataset"] = latest["dataset"].map(lambda x: COUNTY_DISPLAY.get(x, x))
    heat = latest.pivot(index="dataset", columns="ablation", values="online_mse")
    heat = heat[[c for c in order if c in heat.columns]]
    heat = heat.rename(columns=labels)
    apply_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(LISHUI_DIR / "paper_component_ablation_by_county.png", dpi=300)
    plt.close(fig)


def build_shift_score_quantitative_figure() -> None:
    public = pd.read_csv(LOAD_DIR / "shift_score_quantitative_table.csv")
    family = pd.read_csv(LISHUI_DIR / "family_results.csv")
    family = _latest(family, ["dataset", "variant"])
    family = family[family["variant"].isin(["baseline", "stage3", "stage1_stage2_stage3"])]
    family = family.rename(
        columns={
            "adapted_online_mse": "online_mse",
            "variant": "method",
        }
    )
    family["method"] = family["method"].map(
        {"baseline": "Frozen", "stage3": "Stage III", "stage1_stage2_stage3": "Full"}
    )

    apply_ieee_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for ax, dataset in zip(axes.flatten()[:3], PUBLIC_DATASETS):
        sub = public[public["dataset"] == dataset].sort_values("shift_score")
        for method, color in [("Frozen", "#4C78A8"), ("Stage III", "#F58518"), ("Full", "#54A24B")]:
            ax.plot(sub["shift_score"], sub[method], marker="o", label=method, color=color)
        ax.set_title(PUBLIC_LABELS[dataset])
        ax.set_xlabel("Shift score")
        ax.set_ylabel("Online MSE")
    ax = axes.flatten()[3]
    for method, color in [("Frozen", "#4C78A8"), ("Stage III", "#F58518"), ("Full", "#54A24B")]:
        sub = family[family["method"] == method].sort_values("shift_score")
        ax.scatter(sub["shift_score"], sub["online_mse"], s=26, label=method, color=color)
        ax.plot(sub["shift_score"], sub["online_mse"], color=color, alpha=0.55)
    ax.set_title(LISHUI_LABEL)
    ax.set_xlabel("County shift score")
    ax.set_ylabel("Online MSE")
    handles, labels_ = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(LOAD_DIR / "paper_shift_score_quantitative_by_dataset.png", dpi=300)
    plt.close(fig)


def main() -> None:
    build_shifted_baseline_figures()
    build_shifted_sota_figures()
    build_stage_combo_figures()
    build_component_figures()
    build_shift_score_quantitative_figure()


if __name__ == "__main__":
    main()
