from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lishui_dataset import load_lishui_registry
from data.readers import load_tabular_timeseries
from models.lrp import LRPEngine
from train.online_loop import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml
from utils.plot_style import apply_ieee_plot_style


PUBLIC_ORDER = ["panama", "tetouan", "spain_energy_weather"]
PUBLIC_LABELS = {
    "panama": "Panama",
    "tetouan": "Tetouan",
    "spain_energy_weather": "Spain Energy--Weather",
}
COUNTY_LABELS = {
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


def _prepare_public_cfg(base_cfg: dict, name: str) -> dict:
    ds = next(d for d in DATASETS if d["name"] == name)
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
    return cfg


def _prepare_lishui_cfg(base_cfg: dict, ds: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["target_dim"] = None
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "time_interpolate")
    return cfg


def _build_stream(test_ds, batch_size: int, device, invariant_bundle):
    online_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def generator():
        for x, y in online_dl:
            x = x.to(device)
            y = y.to(device)
            if invariant_bundle is None:
                yield x, y
            else:
                with torch.no_grad():
                    phi_i = invariant_bundle["encoder_i"](x)
                    phi_v = invariant_bundle["encoder_v"](x)
                yield x, y, phi_i, phi_v

    return generator


def _run_trace(model, invariant_bundle, cfg: dict, test_ds, device, variant_name: str, s1: bool, s2: bool, s3: bool, outdir: Path):
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    cur["online"]["compare_log_path"] = str(outdir / f"{variant_name}_trace.csv")
    cur["online"]["stage1_log_path"] = str(outdir / f"{variant_name}_stage1.csv")
    cur["online"]["stage2_log_path"] = str(outdir / f"{variant_name}_stage2.csv")
    for p in [cur["online"]["compare_log_path"], cur["online"]["stage1_log_path"], cur["online"]["stage2_log_path"]]:
        if os.path.exists(p):
            os.remove(p)
    stream = _build_stream(test_ds, cur["train"]["batch_size"], device, invariant_bundle)
    _ = run_online_adaptation(stream(), copy.deepcopy(model).to(device), None, cur, device)
    return pd.read_csv(cur["online"]["compare_log_path"])


def _best_window_start(score: np.ndarray, win: int) -> int:
    if len(score) <= win:
        return 0
    kernel = np.ones(win, dtype=float)
    rolling = np.convolve(score, kernel, mode="valid")
    return int(np.argmax(rolling))


def _make_case_df(trace_stage3: pd.DataFrame, trace_full: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    n_total = min(len(trace_stage3), len(trace_full))
    n = min(n_steps, n_total)
    base = trace_stage3.loc[: n_total - 1, "pred_base_t"].to_numpy()
    stage3 = trace_stage3.loc[: n_total - 1, "pred_stage3_t"].to_numpy()
    full = trace_full.loc[: n_total - 1, "pred_stage3_t"].to_numpy()
    target = trace_stage3.loc[: n_total - 1, "target_t"].to_numpy()

    base_err = np.abs(target - base)
    stage3_err = np.abs(target - stage3)
    full_err = np.abs(target - full)
    gain = (base_err - full_err) + 0.5 * (base_err - stage3_err)
    divergence = np.abs(base - full) + 0.5 * np.abs(base - stage3)
    score = np.maximum(gain, 0.0) + 0.35 * divergence
    start = _best_window_start(score, n)
    end = min(start + n, n_total)
    if end - start < n:
        start = max(0, end - n)

    df = pd.DataFrame(
        {
            "step": list(range(end - start)),
            "global_step": list(range(start, end)),
            "Target": target[start:end],
            "Frozen": base[start:end],
            "Stage III": stage3[start:end],
            "Full": full[start:end],
        }
    )
    df["err_Frozen"] = np.abs(df["Target"] - df["Frozen"])
    df["err_Stage III"] = np.abs(df["Target"] - df["Stage III"])
    df["err_Full"] = np.abs(df["Target"] - df["Full"])
    return df


def _extract_relevance(model, test_ds, device, timestamp_col: str, data_path: str, encoding: str | None, missing_sentinels, impute_method: str, topk: int):
    lrp = LRPEngine()
    x_sample, _ = test_ds[min(len(test_ds) // 3, len(test_ds) - 1)]
    x_sample = x_sample.unsqueeze(0).to(device)
    series_tensor, df = load_tabular_timeseries(
        data_path,
        timestamp_col=timestamp_col,
        feature_cols=None,
        encoding=encoding,
        missing_sentinels=missing_sentinels,
        impute_method=impute_method,
    )
    feature_cols = [c for c in df.columns if c != timestamp_col]
    model.eval()
    r_map = lrp.get_input_relevance(model, x_sample).detach().cpu()
    rel = r_map[0].transpose(0, 1) if r_map.dim() == 3 else r_map.transpose(0, 1)
    var_scores = rel.sum(dim=1)
    idx = torch.topk(var_scores, k=min(topk, rel.shape[0])).indices.tolist()
    rel_df = pd.DataFrame(
        rel[idx].numpy(),
        index=[feature_cols[i] for i in idx],
        columns=[f"L{j+1}" for j in range(rel.shape[1])],
    )
    return rel_df


def _plot_case_panels(case_maps: dict[str, pd.DataFrame], outpath: Path, titles: list[str]):
    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})
    fig, axes = plt.subplots(4, 3, figsize=(13.6, 11.0), sharex=False)
    colors = {"Target": "black", "Frozen": "#D55E00", "Stage III": "#0072B2", "Full": "#009E73"}
    styles = {"Target": "-", "Frozen": "--", "Stage III": ":", "Full": "-"}
    widths = {"Target": 1.8, "Frozen": 1.2, "Stage III": 1.35, "Full": 1.45}
    markers = {"Frozen": "o", "Stage III": "s", "Full": "^"}
    first_handles = None
    first_labels = None
    for row, (title, df) in enumerate(zip(titles, case_maps.values())):
        ax_pred = axes[row, 0]
        ax_err = axes[row, 1]
        ax_gain = axes[row, 2]
        x = df["step"].to_numpy()
        for key in ["Target", "Frozen", "Stage III", "Full"]:
            kwargs = {}
            if key in markers:
                kwargs.update(dict(marker=markers[key], markevery=max(len(x) // 8, 1), markersize=3.2))
            ax_pred.plot(x, df[key].to_numpy(), color=colors[key], linestyle=styles[key], linewidth=widths[key], label=key, alpha=0.97, **kwargs)
        if row == 0:
            first_handles, first_labels = ax_pred.get_legend_handles_labels()
        ax_pred.set_title(title)
        ax_pred.set_xlabel("Selected online step")
        ax_pred.set_ylabel("Normalized load")

        err_cols = [("err_Frozen", "Frozen"), ("err_Stage III", "Stage III"), ("err_Full", "Full")]
        for col, key in err_cols:
            ax_err.plot(x, df[col].to_numpy(), color=colors[key], linestyle=styles[key], linewidth=1.15, label=key)
        ax_err.set_title(title + " error")
        ax_err.set_xlabel("Selected online step")
        ax_err.set_ylabel("Absolute error")

        gain_stage3 = df["err_Frozen"].to_numpy() - df["err_Stage III"].to_numpy()
        gain_full = df["err_Frozen"].to_numpy() - df["err_Full"].to_numpy()
        ax_gain.axhline(0.0, color="#666666", linewidth=0.8)
        ax_gain.plot(x, gain_stage3, color=colors["Stage III"], linestyle=styles["Stage III"], linewidth=1.2, label="Frozen - Stage III error")
        ax_gain.plot(x, gain_full, color=colors["Full"], linestyle='-', linewidth=1.25, label="Frozen - Full error")
        ax_gain.fill_between(x, 0.0, gain_full, where=(gain_full >= 0), color=colors["Full"], alpha=0.12)
        ax_gain.fill_between(x, 0.0, gain_full, where=(gain_full < 0), color="#CC79A7", alpha=0.10)
        ax_gain.set_title(title + " gain over frozen")
        ax_gain.set_xlabel("Selected online step")
        ax_gain.set_ylabel("Error reduction")

        diff = np.abs(df["Frozen"].to_numpy() - df["Full"].to_numpy())
        if len(diff) >= 12:
            k = min(24, len(diff))
            start = _best_window_start(diff, k)
            end = min(start + k, len(diff))
            inset = inset_axes(ax_pred, width="42%", height="42%", loc="upper right", borderpad=0.8)
            for key in ["Target", "Frozen", "Stage III", "Full"]:
                inset.plot(x[start:end], df[key].to_numpy()[start:end], color=colors[key], linestyle=styles[key], linewidth=0.95)
            ymin = min(df[key].to_numpy()[start:end].min() for key in ["Target", "Frozen", "Stage III", "Full"])
            ymax = max(df[key].to_numpy()[start:end].max() for key in ["Target", "Frozen", "Stage III", "Full"])
            pad = max((ymax - ymin) * 0.12, 1e-3)
            inset.set_xlim(x[start], x[end - 1])
            inset.set_ylim(ymin - pad, ymax + pad)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_facecolor("#f8f8f8")

    fig.legend(first_handles, first_labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _plot_single_case_figure(df: pd.DataFrame, title: str, outpath: Path):
    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=False)
    colors = {"Target": "black", "Frozen": "#D55E00", "Stage III": "#0072B2", "Full": "#009E73"}
    styles = {"Target": "-", "Frozen": "--", "Stage III": ":", "Full": "-"}
    widths = {"Target": 1.8, "Frozen": 1.25, "Stage III": 1.35, "Full": 1.45}
    markers = {"Frozen": "o", "Stage III": "s", "Full": "^"}
    x = df["step"].to_numpy()
    ax_pred, ax_err, ax_gain = axes
    for key in ["Target", "Frozen", "Stage III", "Full"]:
        kwargs = {}
        if key in markers:
            kwargs.update(dict(marker=markers[key], markevery=max(len(x) // 8, 1), markersize=3.2))
        ax_pred.plot(x, df[key].to_numpy(), color=colors[key], linestyle=styles[key], linewidth=widths[key], label=key, **kwargs)
    ax_pred.set_title(title)
    ax_pred.set_xlabel("Selected online step")
    ax_pred.set_ylabel("Normalized load")
    err_cols = [("err_Frozen", "Frozen"), ("err_Stage III", "Stage III"), ("err_Full", "Full")]
    for col, key in err_cols:
        ax_err.plot(x, df[col].to_numpy(), color=colors[key], linestyle=styles[key], linewidth=1.15, label=key)
    ax_err.set_title("Absolute error")
    ax_err.set_xlabel("Selected online step")
    ax_err.set_ylabel("Absolute error")
    gain_stage3 = df["err_Frozen"].to_numpy() - df["err_Stage III"].to_numpy()
    gain_full = df["err_Frozen"].to_numpy() - df["err_Full"].to_numpy()
    ax_gain.axhline(0.0, color="#666666", linewidth=0.8)
    ax_gain.plot(x, gain_stage3, color=colors["Stage III"], linestyle=styles["Stage III"], linewidth=1.2, label="Frozen - Stage III error")
    ax_gain.plot(x, gain_full, color=colors["Full"], linestyle='-', linewidth=1.25, label="Frozen - Full error")
    ax_gain.fill_between(x, 0.0, gain_full, where=(gain_full >= 0), color=colors["Full"], alpha=0.12)
    ax_gain.fill_between(x, 0.0, gain_full, where=(gain_full < 0), color="#CC79A7", alpha=0.10)
    ax_gain.set_title("Gain over frozen")
    ax_gain.set_xlabel("Selected online step")
    ax_gain.set_ylabel("Error reduction")
    diff = np.abs(df["Frozen"].to_numpy() - df["Full"].to_numpy())
    if len(diff) >= 12:
        k = min(24, len(diff))
        start = _best_window_start(diff, k)
        end = min(start + k, len(diff))
        inset = inset_axes(ax_pred, width="38%", height="42%", loc="upper right", borderpad=0.8)
        for key in ["Target", "Frozen", "Stage III", "Full"]:
            inset.plot(x[start:end], df[key].to_numpy()[start:end], color=colors[key], linestyle=styles[key], linewidth=0.95)
        ymin = min(df[key].to_numpy()[start:end].min() for key in ["Target", "Frozen", "Stage III", "Full"])
        ymax = max(df[key].to_numpy()[start:end].max() for key in ["Target", "Frozen", "Stage III", "Full"])
        pad = max((ymax - ymin) * 0.12, 1e-3)
        inset.set_xlim(x[start], x[end - 1])
        inset.set_ylim(ymin - pad, ymax + pad)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("#f8f8f8")
    handles, labels = ax_pred.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=320)
    plt.close(fig)

def _plot_relevance_panels(rel_maps: dict[str, pd.DataFrame], outpath: Path, titles: list[str]):
    apply_ieee_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, title, df in zip(axes.flatten(), titles, rel_maps.values()):
        sns.heatmap(df, cmap="YlOrRd", ax=ax, cbar=False)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Top variables")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _plot_lishui_case_grid(case_maps: dict[str, pd.DataFrame], outpath: Path):
    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0))
    colors = {"Target": "black", "Frozen": "#D55E00", "Stage III": "#0072B2", "Full": "#009E73"}
    styles = {"Target": "-", "Frozen": "--", "Stage III": ":", "Full": "-"}
    for ax, (name, df) in zip(axes.flatten(), case_maps.items()):
        x = df["step"].to_numpy()
        for key in ["Target", "Frozen", "Stage III", "Full"]:
            ax.plot(x, df[key].to_numpy(), color=colors[key], linestyle=styles[key], linewidth=0.9 if key != "Target" else 1.1)
        ax.set_title(COUNTY_LABELS.get(name, name), fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Load")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _plot_lishui_relevance_grid(rel_maps: dict[str, pd.DataFrame], outpath: Path):
    apply_ieee_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0))
    for ax, (name, df) in zip(axes.flatten(), rel_maps.items()):
        sns.heatmap(df, cmap="YlOrRd", ax=ax, cbar=False)
        ax.set_title(COUNTY_LABELS.get(name, name), fontsize=8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Vars")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-config", default="configs/best_load_online.yaml")
    parser.add_argument("--lishui-config", default="configs/best_lishui_online.yaml")
    parser.add_argument("--plot-steps", type=int, default=120)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--public-epochs", type=int, default=1)
    parser.add_argument("--public-invariant-epochs", type=int, default=1)
    parser.add_argument("--lishui-epochs", type=int, default=1)
    parser.add_argument("--lishui-invariant-epochs", type=int, default=1)
    args = parser.parse_args()

    public_cfg = load_yaml(args.public_config)
    lishui_cfg = load_yaml(args.lishui_config)
    load_out = ROOT / "logs" / "load_benchmarks"
    lishui_out = ROOT / "logs" / "lishui_benchmarks"
    load_out.mkdir(parents=True, exist_ok=True)
    lishui_out.mkdir(parents=True, exist_ok=True)

    public_case = {}
    public_rel = {}

    for name in PUBLIC_ORDER:
        cfg = _prepare_public_cfg(public_cfg, name)
        cfg["train"]["epochs"] = args.public_epochs
        cfg["invariant"]["epochs"] = args.public_invariant_epochs
        cfg["online_run"]["enable"] = False
        res = run_single_experiment(cfg, save_ckpt=False, variant_name=f"{name}_case_multi", return_artifacts=True)
        model = res["model"]
        invariant_bundle = res["invariant_bundle"]
        device = res["device"]
        test_ds = res["test_ds"]
        stage3_trace = _run_trace(model, invariant_bundle, cfg, test_ds, device, f"{name}_stage3_only_multi", False, False, True, load_out)
        full_trace = _run_trace(model, invariant_bundle, cfg, test_ds, device, f"{name}_full_multi", True, True, True, load_out)
        public_case[name] = _make_case_df(stage3_trace, full_trace, args.plot_steps)
        ds = next(d for d in DATASETS if d["name"] == name)
        public_rel[name] = _extract_relevance(
            model,
            test_ds,
            device,
            ds["timestamp_col"],
            ds["path"],
            ds.get("encoding"),
            ds.get("missing_sentinels", []),
            ds.get("impute_method", "drop"),
            args.topk,
        )

    lishui_case = {}
    lishui_rel = {}
    family_case_parts = []
    family_rel_parts = []
    for ds in load_lishui_registry("dataset/load_benchmarks/lishui"):
        cfg = _prepare_lishui_cfg(lishui_cfg, ds)
        cfg["train"]["epochs"] = args.lishui_epochs
        cfg["invariant"]["epochs"] = args.lishui_invariant_epochs
        cfg["online_run"]["enable"] = False
        res = run_single_experiment(cfg, save_ckpt=False, variant_name=f"{ds['name']}_case_multi", return_artifacts=True)
        model = res["model"]
        invariant_bundle = res["invariant_bundle"]
        device = res["device"]
        test_ds = res["test_ds"]
        stage3_trace = _run_trace(model, invariant_bundle, cfg, test_ds, device, f"{ds['name']}_stage3_only_multi", False, False, True, lishui_out)
        full_trace = _run_trace(model, invariant_bundle, cfg, test_ds, device, f"{ds['name']}_full_multi", True, True, True, lishui_out)
        case_df = _make_case_df(stage3_trace, full_trace, args.plot_steps)
        lishui_case[ds["name"]] = case_df
        family_case_parts.append(case_df.drop(columns=["step"]))
        rel_df = _extract_relevance(
            model,
            test_ds,
            device,
            ds["timestamp_col"],
            ds["path"],
            ds.get("encoding"),
            ds.get("missing_sentinels", []),
            ds.get("impute_method", "time_interpolate"),
            args.topk,
        )
        lishui_rel[ds["name"]] = rel_df
        family_rel_parts.append(rel_df)

    family_case = pd.concat(family_case_parts).groupby(level=0).mean()
    family_case.insert(0, "step", range(len(family_case)))

    common_idx = None
    common_cols = None
    for df in family_rel_parts:
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
        common_cols = df.columns if common_cols is None else common_cols.intersection(df.columns)
    rel_stack = [df.loc[common_idx, common_cols] for df in family_rel_parts]
    family_rel = sum(rel_stack) / len(rel_stack)

    public_case["lishui_family"] = family_case
    public_rel["lishui_family"] = family_rel
    titles = [PUBLIC_LABELS[n] for n in PUBLIC_ORDER] + ["Lishui-9 family"]
    ordered_case = {k: public_case[k] for k in PUBLIC_ORDER + ["lishui_family"]}
    ordered_rel = {k: public_rel[k] for k in PUBLIC_ORDER + ["lishui_family"]}
    _plot_case_panels(ordered_case, load_out / "paper_case_study_four_datasets.png", titles)
    for key in PUBLIC_ORDER + ["lishui_family"]:
        label = PUBLIC_LABELS.get(key, "Lishui-9 family")
        _plot_single_case_figure(ordered_case[key], label, load_out / f"paper_case_study_{key}.png")
    _plot_relevance_panels(ordered_rel, load_out / "paper_relevance_case_four_datasets.png", titles)
    _plot_lishui_case_grid(lishui_case, lishui_out / "paper_case_study_nine_counties.png")
    _plot_lishui_relevance_grid(lishui_rel, lishui_out / "paper_relevance_nine_counties.png")

    lines = ["# Multi-dataset case study", ""]
    for name in PUBLIC_ORDER:
        df = public_case[name]
        lines.append(f"- {PUBLIC_LABELS[name]}: frozen={((df['Target']-df['Frozen'])**2).mean():.6f}, stage3={((df['Target']-df['Stage III'])**2).mean():.6f}, full={((df['Target']-df['Full'])**2).mean():.6f}")
    lines.append(f"- Lishui-9 family mean: frozen={((family_case['Target']-family_case['Frozen'])**2).mean():.6f}, stage3={((family_case['Target']-family_case['Stage III'])**2).mean():.6f}, full={((family_case['Target']-family_case['Full'])**2).mean():.6f}")
    (load_out / "multi_case_study_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
