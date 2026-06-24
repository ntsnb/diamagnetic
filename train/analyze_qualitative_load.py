from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.readers import load_tabular_timeseries
from models.lrp import LRPEngine
from train.online_loop import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml
from utils.plot_style import apply_ieee_plot_style


def prepare_dataset_cfg(base_cfg: dict, ds: dict, epochs: int, invariant_epochs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["epochs"] = epochs
    cfg["invariant"]["epochs"] = invariant_epochs
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
    return cfg


def build_stream(test_ds, batch_size: int, device, invariant_bundle):
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


def run_trace(model, invariant_bundle, cfg: dict, test_ds, device, variant_name: str, s1: bool, s2: bool, s3: bool):
    cur = copy.deepcopy(cfg)
    cur["online_run"]["enable"] = True
    cur["online"]["stage1"]["enable"] = s1
    cur["online"]["stage2"]["enable"] = s2
    cur["online"]["stage3"]["enable"] = s3
    cur["online"]["compare_log_path"] = f"logs/load_benchmarks/{variant_name}_trace.csv"
    cur["online"]["stage1_log_path"] = f"logs/load_benchmarks/{variant_name}_stage1.csv"
    cur["online"]["stage2_log_path"] = f"logs/load_benchmarks/{variant_name}_stage2.csv"
    for p in [cur["online"]["compare_log_path"], cur["online"]["stage1_log_path"], cur["online"]["stage2_log_path"]]:
        if os.path.exists(p):
            os.remove(p)
    stream = build_stream(test_ds, cur["train"]["batch_size"], device, invariant_bundle)
    summary = run_online_adaptation(stream(), copy.deepcopy(model).to(device), None, cur, device)
    trace = pd.read_csv(cur["online"]["compare_log_path"])
    return summary, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tuning/load_candidate_i_shifted.yaml")
    parser.add_argument("--dataset", default="spain_energy_weather")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--invariant-epochs", type=int, default=1)
    parser.add_argument("--plot-steps", type=int, default=160)
    args = parser.parse_args()

    apply_ieee_plot_style()
    sns.set_theme(style="whitegrid", font="Times New Roman", rc={"axes.grid": False})
    base_cfg = load_yaml(args.config)
    ds = next(d for d in DATASETS if d["name"] == args.dataset)
    cfg = prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs)
    cfg["online_run"]["enable"] = False
    result = run_single_experiment(cfg, save_ckpt=False, variant_name=f"{args.dataset}_qualitative", return_artifacts=True)

    model = result["model"]
    invariant_bundle = result["invariant_bundle"]
    test_ds = result["test_ds"]
    device = result["device"]

    _, stage3_trace = run_trace(model, invariant_bundle, prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs), test_ds, device, f"{args.dataset}_stage3_only", False, False, True)
    _, full_trace = run_trace(model, invariant_bundle, prepare_dataset_cfg(base_cfg, ds, args.epochs, args.invariant_epochs), test_ds, device, f"{args.dataset}_full", True, True, True)

    n_steps = min(args.plot_steps, len(stage3_trace), len(full_trace))
    plot_df = pd.DataFrame({
        "step": list(range(n_steps)),
        "Target": stage3_trace.loc[: n_steps - 1, "target_t"].to_numpy(),
        "Frozen": stage3_trace.loc[: n_steps - 1, "pred_base_t"].to_numpy(),
        "Stage III": stage3_trace.loc[: n_steps - 1, "pred_stage3_t"].to_numpy(),
        "Full": full_trace.loc[: n_steps - 1, "pred_stage3_t"].to_numpy(),
    })
    plot_long = plot_df.melt("step", var_name="series", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    sns.lineplot(data=plot_long, x="step", y="value", hue="series", ax=ax, linewidth=1.4)
    ax.set_xlabel("Online Step")
    ax.set_ylabel("Normalized Load")
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "load_benchmarks" / f"paper_case_study_{args.dataset}.png", dpi=220)
    plt.close(fig)

    best_step = int((full_trace["base_mse"] - full_trace["stage3_mse"]).fillna(0.0).idxmax())
    sample_idx = min(best_step * cfg["train"]["batch_size"], len(test_ds) - 1)
    x_sample, _ = test_ds[sample_idx]
    x_sample = x_sample.unsqueeze(0).to(device)

    series_tensor, df = load_tabular_timeseries(
        ds["path"],
        timestamp_col=ds["timestamp_col"],
        feature_cols=None,
        encoding=ds.get("encoding"),
        missing_sentinels=ds.get("missing_sentinels"),
        impute_method=ds.get("impute_method", "drop"),
    )
    feature_cols = [c for c in df.columns if c != ds["timestamp_col"]]

    lrp = LRPEngine()
    with torch.no_grad():
        model.eval()
    r_map = lrp.get_input_relevance(model, x_sample).detach().cpu()
    if r_map.dim() == 3:
        rel = r_map[0].transpose(0, 1)
    else:
        rel = r_map.transpose(0, 1)
    var_scores = rel.sum(dim=1)
    topk = min(10, rel.shape[0])
    top_idx = torch.topk(var_scores, k=topk).indices.tolist()
    rel_top = rel[top_idx]
    rel_df = pd.DataFrame(rel_top.numpy(), index=[feature_cols[i] for i in top_idx], columns=[f"lag-{cfg['data']['lookback']-j}" for j in range(cfg["data"]["lookback"])])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(rel_df, cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Top Variables")
    fig.tight_layout()
    fig.savefig(ROOT / "logs" / "load_benchmarks" / f"paper_relevance_case_{args.dataset}.png", dpi=220)
    plt.close(fig)

    summary_lines = [
        f"# Qualitative Analysis on {args.dataset}",
        "",
        f"- Case-study steps plotted: {n_steps}",
        f"- Representative relevance step: {best_step}",
        f"- Representative sample index: {sample_idx}",
        f"- Mean frozen online MSE: {stage3_trace['base_mse'].mean():.6f}",
        f"- Mean Stage III online MSE: {stage3_trace['stage3_mse'].mean():.6f}",
        f"- Mean Full online MSE: {full_trace['stage3_mse'].mean():.6f}",
        "",
        "Top relevance variables:",
        "",
    ]
    for idx in top_idx:
        summary_lines.append(f"- {feature_cols[idx]}")
    (ROOT / "logs" / "load_benchmarks" / f"qualitative_{args.dataset}.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
