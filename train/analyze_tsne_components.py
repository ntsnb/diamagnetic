from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.readers import load_tabular_timeseries
from data.normalization import StandardScaler
from train.run_experiment import run_single_experiment
from train.online_adapt import LabelBuffer, OnlineLoop, StabilityConfig
from models.adaptation import AdaptationGenerator, EMAConfig, OnlineAdaptor, build_param_delta
from models.lrp import LRPEngine
from utils.config import load_yaml
from utils.plot_style import apply_ieee_plot_style


DATASET_SPECS = [
    ("panama", "dataset/load_benchmarks/panama.csv", "datetime", "nat_demand"),
    ("tetouan", "dataset/load_benchmarks/tetouan.csv", "DateTime", "Zone 1 Power Consumption"),
    ("spain_energy_weather", "dataset/load_benchmarks/spain_energy_weather.csv", "time", "actual_load"),
]


def _prepare_cfg(base_cfg: dict, spec: tuple[str, str, str, str], epochs: int) -> dict:
    name, path, timestamp_col, target_col = spec
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["source"] = "file"
    cfg["data"]["path"] = path
    cfg["data"]["timestamp_col"] = timestamp_col
    cfg["data"]["target_col"] = target_col
    cfg["data"]["feature_cols"] = []
    cfg["data"]["normalize"] = True
    cfg["online_run"]["enable"] = False
    cfg["train"]["epochs"] = epochs
    cfg["invariant"]["enable"] = True
    cfg["model"]["temporal_encoder"] = "itransformer"
    return cfg


def _sample_subset(ds, n: int) -> Subset:
    idx = np.linspace(0, len(ds) - 1, num=min(n, len(ds)), dtype=int).tolist()
    return Subset(ds, idx)


def _collect_phi_alpha(model, invariant_bundle, loader, device, limit: int):
    phi_i_list, drift_list, alpha_list = [], [], []
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            phi_i = invariant_bundle["encoder_i"](x)
            phi_v = invariant_bundle["encoder_v"](x)
            drift = phi_v - phi_i
            phi_i_list.append(phi_i.squeeze(0).cpu().numpy())
            drift_list.append(drift.squeeze(0).cpu().numpy())
            alpha_list.append(outputs.alpha.squeeze(0).cpu().numpy())
            count += 1
            if count >= limit:
                break
    return np.asarray(phi_i_list), np.asarray(drift_list), np.asarray(alpha_list)


def _filter_named_tensors(named_tensors: dict[str, torch.Tensor], prefixes: list[str]) -> dict[str, torch.Tensor]:
    if not prefixes:
        return named_tensors
    return {
        name: tensor
        for name, tensor in named_tensors.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _collect_expert_space(model, train_loader, online_loader, device, limit: int):
    expert_points = []
    expert_labels = []
    domain_labels = []
    model.eval()
    with torch.no_grad():
        for loader, domain in ((train_loader, "Train"), (online_loader, "Online")):
            count = 0
            for x, _ in loader:
                x = x.to(device)
                outputs = model(x)
                for expert_idx, emb in enumerate(outputs.expert_embs):
                    expert_points.append(emb.mean(dim=1).squeeze(0).cpu().numpy())
                    expert_labels.append(f"Expert {expert_idx + 1}")
                    domain_labels.append(domain)
                count += 1
                if count >= limit:
                    break
    return np.asarray(expert_points), expert_labels, domain_labels


def _run_stage1_alpha_stream(model, ds, device, cfg, limit: int):
    stage1_cfg = cfg["online"]["stage1"]
    lookback = int(cfg["data"]["lookback"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    gate_bias = torch.nn.Parameter(torch.zeros(len(model.experts), device=device))
    optimizer = torch.optim.Adagrad([gate_bias], lr=stage1_cfg["lr"])
    label_buffer = LabelBuffer(lookback)
    alphas = []
    step = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x, gate_bias=gate_bias)
        alphas.append(outputs.alpha.squeeze(0).detach().cpu().numpy())
        label_buffer.push(outputs.z.detach(), x.detach(), outputs.alpha.detach())
        delayed = label_buffer.pop_with_label(y)
        if delayed is not None and step % max(1, int(stage1_cfg.get("update_interval", 1))) == 0:
            _, x_delayed, alpha_ref, y_delayed = delayed
            gate_snapshot = gate_bias.detach().clone()
            y_hat_delayed = model(x_delayed, gate_bias=gate_bias)
            pred_loss = torch.mean((y_hat_delayed.y_hat - y_delayed) ** 2)
            kl_term = F.kl_div(
                torch.log(y_hat_delayed.alpha.clamp_min(1e-8)),
                alpha_ref,
                reduction="batchmean",
            )
            bias_reg = gate_bias.pow(2).mean()
            loss = pred_loss + stage1_cfg.get("alpha_kl_weight", 0.0) * kl_term + stage1_cfg.get("bias_l2_weight", 0.0) * bias_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if stage1_cfg.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_([gate_bias], stage1_cfg["grad_clip_norm"])
            optimizer.step()
            with torch.no_grad():
                gate_bias.clamp_(-stage1_cfg.get("bias_clip", 0.3), stage1_cfg.get("bias_clip", 0.3))
                if stage1_cfg.get("rollback_on_worse", True):
                    post_pred = model(x_delayed, gate_bias=gate_bias)
                    post_loss = torch.mean((post_pred.y_hat - y_delayed) ** 2).item()
                    if post_loss > pred_loss.item() * (1.0 + stage1_cfg.get("max_loss_increase", 0.0)):
                        gate_bias.copy_(gate_snapshot)
        step += 1
        if len(alphas) >= limit:
            break
    return np.asarray(alphas)


def _run_stage2_drift_stream(model, ds, device, cfg, limit: int):
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    lrp = LRPEngine()
    stability = OnlineLoop(StabilityConfig(cfg["online"]["stability"]["gamma"], cfg["online"]["stability"]["window_size"]))
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    warmup = int(cfg["online"]["stage2"].get("warmup_steps", cfg["online"]["stability"]["window_size"]))
    drift_vecs, labels = [], []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        outputs = model(x)
        r_map = lrp.compute_relevance_map(model, x, method=cfg["online"]["stage2"].get("relevance_method", "rrp"))
        score = stability.update_stability(r_map)
        drift = lrp.compute_drift(x, model, ref_model, method=cfg["online"]["stage2"].get("relevance_method", "rrp"))
        drift = drift / (drift.abs().mean() + 1e-6)
        drift_vecs.append(drift.flatten().detach().cpu().numpy())
        recon_error = torch.mean((outputs.x_recon - x) ** 2).item()
        accepted = (
            step >= warmup
            and score < cfg["online"]["stability"]["gamma"]
            and abs(float(drift.mean().item())) >= cfg["online"]["stage2"].get("min_drift_abs", 0.0)
            and recon_error >= cfg["online"]["stage2"].get("min_recon_error", 0.0)
        )
        labels.append("Selected" if accepted else "Skipped")
        if len(drift_vecs) >= limit:
            break
    return np.asarray(drift_vecs), labels


def _run_stage3_gain_stream(model, invariant_bundle, ds, device, cfg, limit: int):
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    stage3_cfg = cfg["online"]["stage3"]
    bottleneck = int(cfg["online"]["stage2"]["lrp_bottleneck"])
    gen = AdaptationGenerator(drift_dim=bottleneck, rank=bottleneck, max_dim=512).to(device)
    adaptor = OnlineAdaptor(model, EMAConfig(cfg["online"]["stage2"]["ema_eta"], cfg["online"]["stage2"]["ema_K"]))
    prefixes = stage3_cfg.get("target_modules", ["experts", "output_head"])
    drift_vectors, gain_values, labels = [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            outputs = model(x)
            phi_i = invariant_bundle["encoder_i"](x)
            phi_v = invariant_bundle["encoder_v"](x)
            d_v = phi_v - phi_i
            drift_vectors.append(d_v.squeeze(0).cpu().numpy())
            base_loss = torch.mean((outputs.y_hat - y) ** 2)
            recon_error = torch.mean((outputs.x_recon - x) ** 2).item()
            if stage3_cfg.get("gate_by_recon", True) and recon_error < stage3_cfg["min_recon_error"]:
                stage3_y = outputs.y_hat
            else:
                drift_scalar = d_v.mean()
                drift_vec = (drift_scalar / (d_v.abs().mean() + 1e-6)).repeat(bottleneck).to(device)
                delta = build_param_delta(dict(model.named_parameters()), drift_vec, gen)
                delta = _filter_named_tensors(delta, prefixes)
                delta = {
                    k: torch.clamp(v * stage3_cfg["lr_scale"], -stage3_cfg["max_delta"], stage3_cfg["max_delta"])
                    for k, v in delta.items()
                }
                if not delta:
                    stage3_y = outputs.y_hat
                else:
                    candidate = adaptor.stage3_predict(x, delta)
                    pred_shift = torch.mean(torch.abs(candidate.y_hat - outputs.y_hat)).item()
                    if pred_shift > stage3_cfg["max_pred_shift"]:
                        stage3_y = outputs.y_hat
                    else:
                        recon_floor = max(stage3_cfg["min_recon_error"], 1e-6)
                        recon_ratio = max(0.0, recon_error - recon_floor) / recon_floor
                        recon_ratio = min(recon_ratio, 1.0)
                        shift_ratio = min(1.0, pred_shift / max(stage3_cfg["max_pred_shift"], 1e-6))
                        blend = stage3_cfg["blend_alpha"] * (1.0 + 0.5 * recon_ratio) * (1.0 - 0.35 * shift_ratio)
                        blend = max(stage3_cfg.get("blend_alpha_min", blend), min(blend, stage3_cfg.get("blend_alpha_max", blend)))
                        stage3_y = (1.0 - blend) * outputs.y_hat + blend * candidate.y_hat
            stage3_loss = torch.mean((stage3_y - y) ** 2)
            gain = float((base_loss - stage3_loss).item())
            gain_values.append(gain)
        if len(drift_vectors) >= limit:
            break

    gain_arr = np.asarray(gain_values)
    pos = gain_arr[gain_arr > 0]
    high_thresh = float(np.median(pos)) if len(pos) > 0 else 0.0
    for gain in gain_arr:
        if gain <= 0:
            labels.append("No gain")
        elif gain >= high_thresh:
            labels.append("Strong gain")
        else:
            labels.append("Mild gain")
    return np.asarray(drift_vectors), gain_arr, labels


def _tsne_embed(arrays: list[np.ndarray], random_state: int = 42):
    all_x = np.concatenate(arrays, axis=0)
    perplexity = max(5, min(30, all_x.shape[0] // 6))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(all_x)
    outs = []
    start = 0
    for arr in arrays:
        outs.append(emb[start : start + len(arr)])
        start += len(arr)
    return outs


def _centroid_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _plot_phi_drift(results: dict, output: Path):
    apply_ieee_plot_style()
    fig, axes = plt.subplots(len(results), 2, figsize=(9.8, 9.2))
    if len(results) == 1:
        axes = np.array([axes])
    colors = {"Train": "#4C78A8", "Online": "#E45756"}
    for row, (dataset, item) in enumerate(results.items()):
        for col, key in enumerate(["phi", "drift"]):
            ax = axes[row, col]
            emb_train, emb_online = item[f"{key}_emb"]
            ax.scatter(emb_train[:, 0], emb_train[:, 1], s=8, alpha=0.65, c=colors["Train"], label="Train")
            ax.scatter(emb_online[:, 0], emb_online[:, 1], s=8, alpha=0.65, c=colors["Online"], label="Online")
            if row == 0:
                ax.set_title(r"$\phi_I$ space" if key == "phi" else r"$\phi_V-\phi_I$ space")
            ax.set_xlabel(dataset.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel("t-SNE dim 2")
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_stagewise(results: dict, output: Path):
    apply_ieee_plot_style()
    fig, axes = plt.subplots(len(results), 2, figsize=(9.8, 9.2))
    if len(results) == 1:
        axes = np.array([axes])
    alpha_colors = {"Train": "#4C78A8", "Frozen online": "#9ECAE9", "Stage I online": "#F58518"}
    drift_colors = {"Selected": "#54A24B", "Skipped": "#B0B0B0"}
    for row, (dataset, item) in enumerate(results.items()):
        ax = axes[row, 0]
        alpha_train, alpha_frozen, alpha_stage1 = item["alpha_emb"]
        ax.scatter(alpha_train[:, 0], alpha_train[:, 1], s=8, alpha=0.65, c=alpha_colors["Train"], label="Train")
        ax.scatter(alpha_frozen[:, 0], alpha_frozen[:, 1], s=8, alpha=0.65, c=alpha_colors["Frozen online"], label="Frozen online")
        ax.scatter(alpha_stage1[:, 0], alpha_stage1[:, 1], s=8, alpha=0.65, c=alpha_colors["Stage I online"], label="Stage I online")
        if row == 0:
            ax.set_title(r"Stage I gate space ($\alpha$)")
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel(dataset.replace("_", " ").title())
        ax.set_ylabel("t-SNE dim 2")

        ax = axes[row, 1]
        drift_emb = item["stage2_emb"]
        labels = np.asarray(item["stage2_labels"])
        for lab in ["Selected", "Skipped"]:
            sel = labels == lab
            ax.scatter(drift_emb[sel, 0], drift_emb[sel, 1], s=8, alpha=0.65, c=drift_colors[lab], label=lab)
        if row == 0:
            ax.set_title("Stage II relevance-drift space")
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel(dataset.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_stage3_experts(results: dict, output: Path):
    apply_ieee_plot_style()
    fig, axes = plt.subplots(len(results), 2, figsize=(9.8, 9.2))
    if len(results) == 1:
        axes = np.array([axes])
    stage3_colors = {"Strong gain": "#54A24B", "Mild gain": "#F58518", "No gain": "#B0B0B0"}
    expert_palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    for row, (dataset, item) in enumerate(results.items()):
        ax = axes[row, 0]
        emb = item["stage3_emb"]
        labels = np.asarray(item["stage3_labels"])
        for lab in ["Strong gain", "Mild gain", "No gain"]:
            sel = labels == lab
            if sel.any():
                ax.scatter(emb[sel, 0], emb[sel, 1], s=8, alpha=0.7, c=stage3_colors[lab], label=lab)
        if row == 0:
            ax.set_title(r"Stage III drift space")
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel(dataset.replace("_", " ").title())
        ax.set_ylabel("t-SNE dim 2")

        ax = axes[row, 1]
        emb = item["expert_emb"]
        labels = np.asarray(item["expert_labels"])
        for idx, lab in enumerate(sorted(set(labels))):
            sel = labels == lab
            ax.scatter(emb[sel, 0], emb[sel, 1], s=8, alpha=0.65, c=expert_palette[idx % len(expert_palette)], label=lab)
        if row == 0:
            ax.set_title("Expert mechanism space")
            ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.set_xlabel(dataset.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _mean_pairwise_centroid_distance(points: np.ndarray, labels: list[str]) -> float:
    labels_arr = np.asarray(labels)
    uniq = sorted(set(labels))
    centroids = [points[labels_arr == lab].mean(axis=0) for lab in uniq]
    if len(centroids) < 2:
        return 0.0
    vals = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            vals.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_load_online.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--outdir", default="logs/load_benchmarks/tsne")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)

    summary_rows = []
    results = {}

    for spec in DATASET_SPECS:
        dataset = spec[0]
        dataset_cfg = _prepare_cfg(cfg, spec, args.epochs)
        run = run_single_experiment(dataset_cfg, variant_name=f"tsne_{dataset}", return_artifacts=True)
        model = run["model"]
        device = run["device"]
        invariant_bundle = run["invariant_bundle"]

        train_loader = DataLoader(_sample_subset(run["train_ds"], args.samples), batch_size=1, shuffle=False)
        online_loader = DataLoader(_sample_subset(run["test_ds"], args.samples), batch_size=1, shuffle=False)

        phi_train, drift_train, alpha_train = _collect_phi_alpha(model, invariant_bundle, train_loader, device, args.samples)
        phi_online, drift_online, alpha_online = _collect_phi_alpha(model, invariant_bundle, online_loader, device, args.samples)
        alpha_stage1 = _run_stage1_alpha_stream(model, _sample_subset(run["test_ds"], args.samples), device, dataset_cfg, args.samples)
        stage2_drift, stage2_labels = _run_stage2_drift_stream(model, _sample_subset(run["test_ds"], min(args.samples, 160)), device, dataset_cfg, min(args.samples, 160))
        stage3_drift, stage3_gain, stage3_labels = _run_stage3_gain_stream(
            model, invariant_bundle, _sample_subset(run["test_ds"], min(args.samples, 160)), device, dataset_cfg, min(args.samples, 160)
        )
        expert_points, expert_labels, expert_domains = _collect_expert_space(
            model, train_loader, online_loader, device, min(args.samples, 120)
        )

        phi_emb = _tsne_embed([phi_train, phi_online])
        drift_emb = _tsne_embed([drift_train, drift_online])
        alpha_emb = _tsne_embed([alpha_train, alpha_online, alpha_stage1])
        stage2_emb = _tsne_embed([stage2_drift])[0]
        stage3_emb = _tsne_embed([stage3_drift])[0]
        expert_emb = _tsne_embed([expert_points])[0]

        results[dataset] = {
            "phi_emb": phi_emb,
            "drift_emb": drift_emb,
            "alpha_emb": alpha_emb,
            "stage2_emb": stage2_emb,
            "stage2_labels": stage2_labels,
            "stage3_emb": stage3_emb,
            "stage3_labels": stage3_labels,
            "expert_emb": expert_emb,
            "expert_labels": expert_labels,
        }

        summary_rows.extend(
            [
                {
                    "dataset": dataset,
                    "metric": "phi_centroid_gap_train_online",
                    "value": _centroid_gap(phi_train, phi_online),
                },
                {
                    "dataset": dataset,
                    "metric": "drift_centroid_gap_train_online",
                    "value": _centroid_gap(drift_train, drift_online),
                },
                {
                    "dataset": dataset,
                    "metric": "alpha_gap_train_vs_frozen_online",
                    "value": _centroid_gap(alpha_train, alpha_online),
                },
                {
                    "dataset": dataset,
                    "metric": "alpha_gap_train_vs_stage1_online",
                    "value": _centroid_gap(alpha_train, alpha_stage1),
                },
                {
                    "dataset": dataset,
                    "metric": "stage2_selected_ratio",
                    "value": float(np.mean(np.asarray(stage2_labels) == "Selected")),
                },
                {
                    "dataset": dataset,
                    "metric": "stage3_mean_gain",
                    "value": float(np.mean(stage3_gain)),
                },
                {
                    "dataset": dataset,
                    "metric": "stage3_positive_gain_ratio",
                    "value": float(np.mean(stage3_gain > 0)),
                },
                {
                    "dataset": dataset,
                    "metric": "expert_centroid_distance",
                    "value": _mean_pairwise_centroid_distance(expert_points, expert_labels),
                },
            ]
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "tsne_summary.csv", index=False)
    _plot_phi_drift(results, outdir / "paper_tsne_phi_drift.png")
    _plot_stagewise(results, outdir / "paper_tsne_stagewise.png")
    _plot_stage3_experts(results, outdir / "paper_tsne_stage3_experts.png")

    report_lines = [
        "# t-SNE Component and Stage Visualization",
        "",
        "This report visualizes the representation spaces of the invariant module, the variant drift signal, the Stage~I gate recalibration, and the Stage~II controller on the three main load datasets.",
        "",
    ]
    for dataset in summary["dataset"].unique():
        sub = summary[summary["dataset"] == dataset]
        report_lines.append(f"## {dataset}")
        for _, row in sub.iterrows():
            report_lines.append(f"- `{row['metric']}`: `{row['value']:.6f}`")
        report_lines.append("")
    (outdir / "tsne_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
