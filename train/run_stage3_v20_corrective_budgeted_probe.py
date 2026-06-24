from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v20 import FactorMechanismV20Config, build_factor_mechanism_v20  # noqa: E402
from train.offline_factor_v20 import (  # noqa: E402
    FactorMechanismV20LossConfig,
    evaluate_factor_v20,
    train_offline_factor_v20,
)
from train.online_loop_stage3_v20 import run_online_adaptation_v20  # noqa: E402
from train.run_stage3_v18_structured_env_release_probe import load_yaml  # noqa: E402
from train.stage3_generator_training_v20 import (  # noqa: E402
    Stage3GeneratorCandidateV20,
    collect_stage3_v20_risk_gate_examples,
    iter_v20_representation_stream,
    train_stage3_generator_v20,
)
from train.stage3_v20_artifacts import load_stage3_v20_artifact_bundle  # noqa: E402
import train.run_stage3_v18_structured_env_release_probe as v18_probe  # noqa: E402


PROTOCOL_VERSION = "v20_corrective_budgeted_probe"


def validate_v20_config_contract(cfg: dict[str, Any]) -> None:
    stage3_cfg = cfg.get("stage3_v2", {})
    online_cfg = cfg.get("online", {}).get("stage3", {})
    if str(stage3_cfg.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError("V20 config requires stage3_v2.protocol_version=v20_corrective_budgeted_probe")
    if str(online_cfg.get("version", "")) != PROTOCOL_VERSION:
        raise ValueError("V20 config requires online.stage3.version=v20_corrective_budgeted_probe")
    if str(online_cfg.get("signal_mode", "")) != "fluctuation_delta":
        raise ValueError("V20 config requires online.stage3.signal_mode=fluctuation_delta")
    roots = (
        str(stage3_cfg.get("log_root", "")),
        str(stage3_cfg.get("checkpoint_root", "")),
        str(stage3_cfg.get("results_root", "")),
    )
    if not all("s3v20expert" in root for root in roots):
        raise ValueError("V20 config must use s3v20expert output roots")


def _build_v20_model(cfg: dict[str, Any], input_dim: int, target_dim: int):
    model_cfg = cfg.get("model_v20", cfg.get("model", {}))
    return build_factor_mechanism_v20(
        FactorMechanismV20Config(
            input_dim=input_dim,
            seq_len=int(cfg["data"]["lookback"]),
            out_dim=int(cfg["data"]["horizon"]),
            d_model=int(model_cfg.get("d_model", 128)),
            n_experts=int(model_cfg.get("n_experts", 4)),
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            repr_dim=int(model_cfg.get("repr_dim", 64)),
            expert_hidden_dim=int(model_cfg.get("expert_hidden_dim", model_cfg.get("d_model", 128))),
            history_hidden_dim=int(
                model_cfg.get("history_hidden_dim", model_cfg.get("expert_hidden_dim", model_cfg.get("d_model", 128)))
            ),
            release_hidden_dim=int(
                model_cfg.get("release_hidden_dim", model_cfg.get("expert_hidden_dim", model_cfg.get("d_model", 128)))
            ),
            env_count=int(model_cfg.get("env_count", 4)),
            router_env_hidden_dim=int(model_cfg.get("router_env_hidden_dim", 64)),
            temporal_encoder=str(model_cfg.get("temporal_encoder", "transformer")),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_revin=bool(model_cfg.get("use_revin", False)),
            target_index=int(model_cfg.get("target_index", 0)),
            use_vae=bool(model_cfg.get("use_vae", True)),
            vae_eval_mode=str(model_cfg.get("vae_eval_mode", "mean")),
            zero_init_variant=bool(model_cfg.get("zero_init_variant", True)),
            use_projected_residual=bool(model_cfg.get("use_projected_residual", True)),
            projection_eps=float(model_cfg.get("projection_eps", 1e-8)),
        )
    )


def _build_v20_loss_cfg(cfg: dict[str, Any]) -> FactorMechanismV20LossConfig:
    inv_cfg = cfg.get("invariant", {})
    train_cfg = cfg.get("train", {})
    rep_cfg = FactorMechanismV20LossConfig().representation
    rep_cfg.lambda_forecast = float(inv_cfg.get("lambda_forecast", rep_cfg.lambda_forecast))
    rep_cfg.lambda_inv_forecast = float(inv_cfg.get("lambda_inv_forecast", rep_cfg.lambda_inv_forecast))
    rep_cfg.lambda_residual = float(inv_cfg.get("lambda_residual", rep_cfg.lambda_residual))
    rep_cfg.lambda_orth = float(inv_cfg.get("lambda_orth", rep_cfg.lambda_orth))
    rep_cfg.lambda_smooth = float(inv_cfg.get("lambda_smooth", rep_cfg.lambda_smooth))
    rep_cfg.lambda_norm = float(inv_cfg.get("lambda_norm", rep_cfg.lambda_norm))
    rep_cfg.lambda_hsic = float(inv_cfg.get("lambda_hsic", rep_cfg.lambda_hsic))
    rep_cfg.lambda_barlow = float(inv_cfg.get("lambda_barlow", rep_cfg.lambda_barlow))
    rep_cfg.lambda_temporal_ranking = float(inv_cfg.get("lambda_temporal_ranking", rep_cfg.lambda_temporal_ranking))
    rep_cfg.temporal_ranking_margin = float(inv_cfg.get("temporal_ranking_margin", rep_cfg.temporal_ranking_margin))
    rep_cfg.lambda_deployed_temporal_ranking = float(
        inv_cfg.get("lambda_deployed_temporal_ranking", rep_cfg.lambda_deployed_temporal_ranking)
    )
    rep_cfg.deployed_temporal_ranking_margin = float(
        inv_cfg.get("deployed_temporal_ranking_margin", rep_cfg.deployed_temporal_ranking_margin)
    )
    rep_cfg.lambda_temporal_contrastive = float(
        inv_cfg.get("lambda_temporal_contrastive", rep_cfg.lambda_temporal_contrastive)
    )
    rep_cfg.temporal_contrastive_temperature = float(
        inv_cfg.get("temporal_contrastive_temperature", rep_cfg.temporal_contrastive_temperature)
    )
    rep_cfg.lambda_vicreg_variance = float(inv_cfg.get("lambda_vicreg_variance", rep_cfg.lambda_vicreg_variance))
    rep_cfg.lambda_vicreg_covariance = float(inv_cfg.get("lambda_vicreg_covariance", rep_cfg.lambda_vicreg_covariance))
    rep_cfg.lambda_phi_vicreg_variance = float(
        inv_cfg.get("lambda_phi_vicreg_variance", rep_cfg.lambda_phi_vicreg_variance)
    )
    rep_cfg.lambda_phi_vicreg_covariance = float(
        inv_cfg.get("lambda_phi_vicreg_covariance", rep_cfg.lambda_phi_vicreg_covariance)
    )
    rep_cfg.vicreg_variance_target = float(inv_cfg.get("vicreg_variance_target", rep_cfg.vicreg_variance_target))
    return FactorMechanismV20LossConfig(
        beta_vae=float(train_cfg.get("beta_vae", 0.05)),
        lambda_load_balance=float(inv_cfg.get("lambda_load_balance", 0.05)),
        lambda_env_balance=float(inv_cfg.get("lambda_env_balance", 0.01)),
        lambda_expert_diversity=float(inv_cfg.get("lambda_expert_diversity", 0.02)),
        lambda_local_inv_forecast=float(inv_cfg.get("lambda_local_inv_forecast", 0.2)),
        lambda_local_history_proxy=float(inv_cfg.get("lambda_local_history_proxy", 0.12)),
        lambda_local_release_increment=float(inv_cfg.get("lambda_local_release_increment", 0.35)),
        lambda_env_temporal=float(inv_cfg.get("lambda_env_temporal", 0.05)),
        lambda_router_stability=float(inv_cfg.get("lambda_router_stability", 0.1)),
        lambda_anchor_preservation=float(inv_cfg.get("lambda_anchor_preservation", 0.1)),
        lambda_env_correction_sparsity=float(inv_cfg.get("lambda_env_correction_sparsity", 0.02)),
        lambda_release_confidence=float(inv_cfg.get("lambda_release_confidence", 0.1)),
        lambda_anchor_diversity=float(inv_cfg.get("lambda_anchor_diversity", 0.02)),
        lambda_release_temporal=float(inv_cfg.get("lambda_release_temporal", 0.05)),
        lambda_conditional_advantage=float(inv_cfg.get("lambda_conditional_advantage", 0.2)),
        conditional_advantage_margin=float(inv_cfg.get("conditional_advantage_margin", 0.001)),
        lambda_release_null=float(inv_cfg.get("lambda_release_null", 0.05)),
        lambda_counterfactual_shuffle=float(inv_cfg.get("lambda_counterfactual_shuffle", 0.1)),
        counterfactual_shuffle_margin=float(inv_cfg.get("counterfactual_shuffle_margin", 0.001)),
        lambda_budget_alignment=float(inv_cfg.get("lambda_budget_alignment", 0.15)),
        lambda_history_budget_cap=float(inv_cfg.get("lambda_history_budget_cap", 0.08)),
        representation=rep_cfg,
    )


def _train_base_model_v20(cfg: dict[str, Any], model, train_iter, val_iter, device: torch.device) -> list[dict[str, float]]:
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 5e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = v18_probe.build_offline_scheduler(cfg, opt)
    loss_cfg = _build_v20_loss_cfg(cfg)
    rows: list[dict[str, float]] = []
    epochs = int(cfg["train"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        epoch_rows = train_offline_factor_v20(train_iter, model, opt, loss_cfg, device)
        if scheduler is not None:
            scheduler.step()
        val_mse, val_mae = evaluate_factor_v20(val_iter, model, device)
        for row in epoch_rows:
            row["epoch"] = float(epoch)
            row["val_mse"] = float(val_mse)
            row["val_mae"] = float(val_mae)
        rows.extend(epoch_rows)
    return rows


def _collect_split_tensors_v20(
    *,
    dataset,
    batch_size: int,
    device: torch.device,
    model,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_parts: list[torch.Tensor] = []
    y_base_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x)
        raw = out.pooled_budgeted_release
        raw_parts.append(raw.detach().cpu().reshape(raw.shape[0], -1))
        y_base_parts.append(out.y_hat.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not raw_parts:
        raise ValueError("V20 collected no split samples")
    return torch.cat(raw_parts, dim=0), torch.cat(y_base_parts, dim=0), torch.cat(y_parts, dim=0)


def _rename_v18_outputs_to_v20(out_dir: Path) -> None:
    manifest_v18 = out_dir / "run_manifest_v18.json"
    if manifest_v18.exists():
        manifest = json.loads(manifest_v18.read_text(encoding="utf-8"))
        manifest["protocol_version"] = PROTOCOL_VERSION
        manifest["v20_model_native_drift_source"] = "pooled_budgeted_release"
        manifest.pop("v18_model_native_drift_source", None)
        (out_dir / "run_manifest_v20.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_v18.unlink()

    for path in list(out_dir.iterdir()):
        if "v18" in path.name:
            target = path.with_name(path.name.replace("v18", "v20"))
            if target.exists():
                continue
            path.rename(target)


def run_stage3_v20_corrective_budgeted_probe(args: argparse.Namespace) -> Path:
    v18_probe.PROTOCOL_VERSION = PROTOCOL_VERSION
    v18_probe.validate_v18_config_contract = validate_v20_config_contract
    v18_probe._build_v18_model = _build_v20_model
    v18_probe._build_v18_loss_cfg = _build_v20_loss_cfg
    v18_probe._train_base_model_v18 = _train_base_model_v20
    v18_probe._collect_split_tensors_v18 = _collect_split_tensors_v20
    v18_probe.run_online_adaptation_v18 = run_online_adaptation_v20
    v18_probe.train_stage3_generator_v18 = train_stage3_generator_v20
    v18_probe.collect_stage3_v18_risk_gate_examples = collect_stage3_v20_risk_gate_examples
    v18_probe.iter_v18_representation_stream = iter_v20_representation_stream
    v18_probe.load_stage3_v18_artifact_bundle = load_stage3_v20_artifact_bundle
    v18_probe.Stage3GeneratorCandidateV18 = Stage3GeneratorCandidateV20

    out_dir = v18_probe.run_stage3_v18_structured_env_release_probe(args)
    _rename_v18_outputs_to_v20(out_dir)
    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V20 corrective-budgeted residual-target probe.")
    parser.add_argument("--config", type=str, default="configs/stage3_v20_corrective_budgeted_probe.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--validation-windows", type=int, default=None)
    parser.add_argument("--out-root", type=str, default="logs/s3v20expert")
    parser.add_argument("--checkpoint-root", type=str, default="ckpt/s3v20expert")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--feature-modes", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = run_stage3_v20_corrective_budgeted_probe(args)
    print(out_dir)


if __name__ == "__main__":
    main()

