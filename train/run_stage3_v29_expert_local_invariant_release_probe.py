from __future__ import annotations

import argparse
import csv
import hashlib
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

from models.factor_mechanism_v29 import FactorMechanismV29Config, build_factor_mechanism_v29  # noqa: E402
from train.offline_factor_v29 import (  # noqa: E402
    FactorMechanismV29LossConfig,
    evaluate_factor_v29,
    train_offline_factor_v29,
)
from train.online_loop_stage3_v29 import run_online_adaptation_v29  # noqa: E402
from train.run_stage3_v18_structured_env_release_probe import load_yaml  # noqa: E402,F401
from train.run_stage3_v26_multiscale_change_probe import _build_v26_loss_cfg  # noqa: E402
from train.stage3_generator_training_v29 import (  # noqa: E402
    Stage3GeneratorCandidateV29,
    collect_stage3_v29_risk_gate_examples,
    iter_v29_representation_stream,
    train_stage3_generator_v29,
)
from train.stage3_v29_artifacts import load_stage3_v29_artifact_bundle  # noqa: E402
from train.stage3_v29_online_gate_features import build_online_gate_dynamic_feature_row  # noqa: E402
import train.run_stage3_v18_structured_env_release_probe as v18_probe  # noqa: E402


PROTOCOL_VERSION = "v29_expert_local_invariant_release_probe"


def validate_v29_config_contract(cfg: dict[str, Any]) -> None:
    stage3_cfg = cfg.get("stage3_v2", {})
    online_cfg = cfg.get("online", {}).get("stage3", {})
    if str(stage3_cfg.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError("V29 config requires stage3_v2.protocol_version=v29_expert_local_invariant_release_probe")
    if str(online_cfg.get("version", "")) != PROTOCOL_VERSION:
        raise ValueError("V29 config requires online.stage3.version=v29_expert_local_invariant_release_probe")
    if str(online_cfg.get("drift_source", "")) != "pooled_servable_change":
        raise ValueError("V29 config requires online.stage3.drift_source=pooled_servable_change")
    if str(online_cfg.get("signal_mode", "")) != "fluctuation_delta":
        raise ValueError("V29 config requires online.stage3.signal_mode=fluctuation_delta")
    roots = (
        str(stage3_cfg.get("log_root", "")),
        str(stage3_cfg.get("checkpoint_root", "")),
        str(stage3_cfg.get("results_root", "")),
    )
    if not all("s3v29expert" in root for root in roots):
        raise ValueError("V29 config must use s3v29expert output roots")


def _validate_v29_runtime_config_contract(cfg: dict[str, Any]) -> None:
    stage3_cfg = cfg.get("stage3_v2", {})
    online_cfg = cfg.get("online", {}).get("stage3", {})
    if str(stage3_cfg.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError("V29 runtime config requires stage3_v2.protocol_version=v29_expert_local_invariant_release_probe")
    if str(online_cfg.get("version", "")) != PROTOCOL_VERSION:
        raise ValueError("V29 runtime config requires online.stage3.version=v29_expert_local_invariant_release_probe")
    if str(online_cfg.get("signal_mode", "")) != "fluctuation_delta":
        raise ValueError("V29 runtime config requires online.stage3.signal_mode=fluctuation_delta")
    roots = (
        str(stage3_cfg.get("log_root", "")),
        str(stage3_cfg.get("checkpoint_root", "")),
        str(stage3_cfg.get("results_root", "")),
    )
    if not all("s3v29expert" in root for root in roots):
        raise ValueError("V29 runtime config must use s3v29expert output roots")


def _build_v29_model(cfg: dict[str, Any], input_dim: int, target_dim: int):
    model_cfg = cfg.get("model_v29", cfg.get("model", {}))
    return build_factor_mechanism_v29(
        FactorMechanismV29Config(
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
            change_memory_hidden_dim=int(
                model_cfg.get(
                    "change_memory_hidden_dim",
                    model_cfg.get("history_hidden_dim", model_cfg.get("expert_hidden_dim", model_cfg.get("d_model", 128))),
                )
            ),
            env_count=int(model_cfg.get("env_count", 4)),
            router_env_hidden_dim=int(model_cfg.get("router_env_hidden_dim", 64)),
            invariant_release_hidden_dim=int(
                model_cfg.get("invariant_release_hidden_dim", model_cfg.get("d_model", 128))
            ),
            router_invariant_hidden_dim=int(model_cfg.get("router_invariant_hidden_dim", 64)),
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


def _build_v29_loss_cfg(cfg: dict[str, Any]) -> FactorMechanismV29LossConfig:
    inv_cfg = cfg.get("invariant", {})
    base = _build_v26_loss_cfg(cfg)
    return FactorMechanismV29LossConfig(
        **base.__dict__,
        lambda_support_alignment=float(inv_cfg.get("lambda_support_alignment", 0.18)),
        lambda_suppression_alignment=float(inv_cfg.get("lambda_suppression_alignment", 0.16)),
        lambda_support_conditioned_advantage=float(inv_cfg.get("lambda_support_conditioned_advantage", 0.12)),
        support_advantage_margin=float(inv_cfg.get("support_advantage_margin", 0.001)),
        lambda_suppression_gate_sparsity=float(inv_cfg.get("lambda_suppression_gate_sparsity", 0.02)),
        lambda_local_env_consistency=float(inv_cfg.get("lambda_local_env_consistency", 0.08)),
        lambda_invariant_correction_budget=float(inv_cfg.get("lambda_invariant_correction_budget", 0.04)),
        lambda_residual_release_alignment=float(inv_cfg.get("lambda_residual_release_alignment", 0.12)),
        lambda_residual_release_advantage=float(inv_cfg.get("lambda_residual_release_advantage", 0.12)),
        residual_release_advantage_margin=float(inv_cfg.get("residual_release_advantage_margin", 0.001)),
        lambda_expert_release_orthogonal=float(inv_cfg.get("lambda_expert_release_orthogonal", 0.04)),
        lambda_residual_state_noncollapse=float(inv_cfg.get("lambda_residual_state_noncollapse", 0.03)),
        residual_state_std_floor=float(inv_cfg.get("residual_state_std_floor", 0.05)),
    )


def _train_base_model_v29(cfg: dict[str, Any], model, train_iter, val_iter, device: torch.device) -> list[dict[str, float]]:
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 5e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = v18_probe.build_offline_scheduler(cfg, opt)
    loss_cfg = _build_v29_loss_cfg(cfg)
    rows: list[dict[str, float]] = []
    epochs = int(cfg["train"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        epoch_rows = train_offline_factor_v29(train_iter, model, opt, loss_cfg, device)
        if scheduler is not None:
            scheduler.step()
        val_mse, val_mae = evaluate_factor_v29(val_iter, model, device)
        for row in epoch_rows:
            row["epoch"] = float(epoch)
            row["val_mse"] = float(val_mse)
            row["val_mae"] = float(val_mae)
        rows.extend(epoch_rows)
    return rows


def _collect_split_tensors_v29(
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
        raw = out.pooled_servable_change
        raw_parts.append(raw.detach().cpu().reshape(raw.shape[0], -1))
        y_base_parts.append(out.y_hat.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not raw_parts:
        raise ValueError("V29 collected no split samples")
    return torch.cat(raw_parts, dim=0), torch.cat(y_base_parts, dim=0), torch.cat(y_parts, dim=0)


def collect_stage3_v29_online_gate_dynamic_feature_rows(
    *,
    dataset,
    batch_size: int,
    device: torch.device,
    model,
    max_batches: int | None,
    split: str,
    target_name: str = "raw_residual_vector",
    window_size: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for batch_idx, (x, _y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        sample_start = batch_idx * batch_size
        sample_end = sample_start + int(x.shape[0])
        if window_size and window_size > 0:
            validation_window_id = str(sample_start // int(window_size))
        else:
            validation_window_id = str(batch_idx)
        with torch.no_grad():
            out = model(x)
        rows.append(
            build_online_gate_dynamic_feature_row(
                out,
                split=split,
                batch_index=batch_idx,
                sample_start=sample_start,
                sample_end=sample_end,
                validation_window_id=validation_window_id,
                target_name=target_name,
            )
        )
    if not rows:
        raise ValueError("V29 collected no online gate dynamic feature rows")
    return rows


def write_stage3_v29_online_gate_dynamic_features(
    *,
    out_dir: Path,
    cfg: dict[str, Any],
    model,
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    del train_ds, test_ds
    targets = v18_probe._resolve_v15_targets(cfg, getattr(args, "targets", None))
    target_name = str(targets[0]) if targets else "raw_residual_vector"
    max_batches = getattr(args, "max_eval_batches", None)
    validation_windows = int(getattr(args, "validation_windows", None) or cfg.get("stage3_v2", {}).get("validation_windows", 1))
    window_size: int | None = None
    if validation_windows > 0 and len(val_ds) > 0:
        window_size = max(len(val_ds) // validation_windows, 1)
    rows = collect_stage3_v29_online_gate_dynamic_feature_rows(
        dataset=val_ds,
        batch_size=batch_size,
        device=device,
        model=model,
        max_batches=max_batches,
        split="validation",
        target_name=target_name,
        window_size=window_size,
    )
    v18_probe._append_csv(out_dir / "stage3_v29_online_gate_dynamic_features.csv", rows)
    return rows


def _rename_v18_outputs_to_v29(out_dir: Path) -> None:
    resolved_config_path = out_dir / "resolved_config.json"
    resolved_config_snapshot: dict[str, str] | None = None
    if resolved_config_path.exists():
        resolved_cfg = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        resolved_cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = "pooled_servable_change"
        resolved_config_path.write_text(json.dumps(resolved_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        digest = hashlib.sha256(resolved_config_path.read_bytes()).hexdigest()
        resolved_config_snapshot = {
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_sha256": digest,
            "resolved_config_sha256_from_disk": digest,
        }

    dynamic_features_path = out_dir / "stage3_v29_online_gate_dynamic_features.csv"
    dynamic_features_snapshot: dict[str, Any] | None = None
    if dynamic_features_path.exists():
        with dynamic_features_path.open(newline="", encoding="utf-8") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        digest = hashlib.sha256(dynamic_features_path.read_bytes()).hexdigest()
        dynamic_features_snapshot = {
            "path": str(dynamic_features_path),
            "sha256": digest,
            "sha256_from_disk": digest,
            "row_count": row_count,
            "scope": "validation_model_forward_pre_outcome",
            "leakage_contract": "no_ground_truth_gain_counterfactual_or_oracle_label_inputs",
            "producer": "write_stage3_v29_online_gate_dynamic_features",
        }

    manifest_v18 = out_dir / "run_manifest_v18.json"
    if manifest_v18.exists():
        manifest = json.loads(manifest_v18.read_text(encoding="utf-8"))
        manifest["protocol_version"] = PROTOCOL_VERSION
        manifest["v29_model_native_drift_source"] = "pooled_servable_change"
        manifest["v29_native_state"] = "expert_local_invariant_release"
        if resolved_config_snapshot is not None:
            manifest["resolved_config_json"] = resolved_config_snapshot
        if dynamic_features_snapshot is not None:
            manifest["stage3_v29_online_gate_dynamic_features_csv"] = dynamic_features_snapshot
        manifest.pop("v18_model_native_drift_source", None)
        (out_dir / "run_manifest_v29.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_v18.unlink()

    for path in list(out_dir.iterdir()):
        if "v18" in path.name:
            target = path.with_name(path.name.replace("v18", "v29"))
            if target.exists():
                continue
            path.rename(target)


def run_stage3_v29_expert_local_invariant_release_probe(args: argparse.Namespace) -> Path:
    validate_v29_config_contract(load_yaml(args.config))
    v18_probe.PROTOCOL_VERSION = PROTOCOL_VERSION
    v18_probe.validate_v18_config_contract = _validate_v29_runtime_config_contract
    v18_probe._build_v18_model = _build_v29_model
    v18_probe._build_v18_loss_cfg = _build_v29_loss_cfg
    v18_probe._train_base_model_v18 = _train_base_model_v29
    v18_probe._collect_split_tensors_v18 = _collect_split_tensors_v29
    v18_probe.run_online_adaptation_v18 = run_online_adaptation_v29
    v18_probe.train_stage3_generator_v18 = train_stage3_generator_v29
    v18_probe.collect_stage3_v18_risk_gate_examples = collect_stage3_v29_risk_gate_examples
    v18_probe.iter_v18_representation_stream = iter_v29_representation_stream
    v18_probe.load_stage3_v18_artifact_bundle = load_stage3_v29_artifact_bundle
    v18_probe.Stage3GeneratorCandidateV18 = Stage3GeneratorCandidateV29
    v18_probe.write_online_gate_dynamic_features = write_stage3_v29_online_gate_dynamic_features

    out_dir = v18_probe.run_stage3_v18_structured_env_release_probe(args)
    _rename_v18_outputs_to_v29(out_dir)
    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V29 expert-local invariant release residual-target probe.")
    parser.add_argument("--config", type=str, default="configs/stage3_v29_expert_local_invariant_release_probe.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--validation-windows", type=int, default=None)
    parser.add_argument("--out-root", type=str, default="logs/s3v29expert")
    parser.add_argument("--checkpoint-root", type=str, default="ckpt/s3v29expert")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--feature-modes", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_stage3_v29_expert_local_invariant_release_probe(args)


if __name__ == "__main__":
    main()
