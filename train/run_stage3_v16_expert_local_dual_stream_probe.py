from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v16 import FactorMechanismV16Config, build_factor_mechanism_v16  # noqa: E402
from models.stage3_residual_teacher_v14 import (  # noqa: E402
    TrainOnlyStabilityFluctuationResidualTeacherV14,
    build_stage3_v14_context_features,
    save_stage3_v14_teacher_checkpoint,
)
from train.offline_factor_v16 import (  # noqa: E402
    FactorMechanismV16LossConfig,
    evaluate_factor_v16,
    train_offline_factor_v16,
)
from train.online_loop_stage3_v16 import run_online_adaptation_v16  # noqa: E402
from train.run_experiment import build_offline_scheduler  # noqa: E402
from train.run_stage3_v12c_sensitivity_audit import (  # noqa: E402
    _append_csv,
    _apply_dataset_override,
    _build_dataloaders,
    _git_status_short,
    _limit_batches,
    _load_series_and_features,
    _ordered_loader,
    _prepare_fresh_run_dirs,
    _sha256_file,
    _stage3_drift_polarity,
    _teacher_diagnostic_aggregate,
    _write_resolved_config_snapshot,
    load_yaml,
    set_seed,
)
from train.run_stage3_v14_stability_teacher_probe import (  # noqa: E402
    DEFAULT_MODES,
    _build_history_builder,
    _raw_previous_tail,
    _residual_previous_tail,
    _teacher_from_cfg,
    _validate_v14_gate_modes,
)
from train.run_stage3_v15_residual_target_predictability_probe import (  # noqa: E402
    DEFAULT_V15_FEATURE_MODES,
    PROTOCOL_VERSION as V15_PROTOCOL_VERSION,
    _partial_gain_rows,
    _resolve_v15_targets,
    _rolling_rows_for_split,
    _rows_for_split,
    _summary_rows_for_target_feature,
    _v15_target_gate_rows,
    build_v15_target_tensor,
    v15_directional_stop_reasons,
    v15_target_to_teacher_tensors,
)
from train.stage3_eval_policy import assert_deterministic_eval_model  # noqa: E402
from train.stage3_generator_training import (  # noqa: E402
    load_stage3_generator_checkpoint,
    save_stage3_generator_checkpoint,
)
from train.stage3_generator_training_v16 import (  # noqa: E402
    Stage3GeneratorCandidateV16,
    collect_stage3_v16_risk_gate_examples,
    iter_v16_representation_stream,
    train_stage3_generator_v16,
)
from train.stage3_risk_gate_v2 import (  # noqa: E402
    Stage3RiskGateBundle,
    save_stage3_risk_gate_checkpoint,
    train_stage3_risk_gate,
)
from train.stage3_v13_signal_contract import Stage3SignalBuilderV13  # noqa: E402
from train.stage3_v16_artifacts import load_stage3_v16_artifact_bundle  # noqa: E402


PROTOCOL_VERSION = "v16_expert_local_dual_stream_probe"


def validate_v16_config_contract(cfg: dict[str, Any]) -> None:
    stage3_cfg = cfg.get("stage3_v2", {})
    online_cfg = cfg.get("online", {}).get("stage3", {})
    if str(stage3_cfg.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError("V16 config requires stage3_v2.protocol_version=v16_expert_local_dual_stream_probe")
    if str(online_cfg.get("version", "")) != PROTOCOL_VERSION:
        raise ValueError("V16 config requires online.stage3.version=v16_expert_local_dual_stream_probe")
    if str(online_cfg.get("drift_source", "")) != "delta_phi_v":
        raise ValueError("V16 config requires online.stage3.drift_source=delta_phi_v")
    if str(online_cfg.get("signal_mode", "")) != "fluctuation_delta":
        raise ValueError("V16 config requires online.stage3.signal_mode=fluctuation_delta")
    roots = (
        str(stage3_cfg.get("log_root", "")),
        str(stage3_cfg.get("checkpoint_root", "")),
        str(stage3_cfg.get("results_root", "")),
    )
    if not all("s3v16expert" in root for root in roots):
        raise ValueError("V16 config must use s3v16expert output roots")


def _build_v16_model(cfg: dict[str, Any], input_dim: int, target_dim: int):
    model_cfg = cfg.get("model_v16", cfg.get("model", {}))
    return build_factor_mechanism_v16(
        FactorMechanismV16Config(
            input_dim=input_dim,
            seq_len=int(cfg["data"]["lookback"]),
            out_dim=int(cfg["data"]["horizon"]),
            d_model=int(model_cfg.get("d_model", 128)),
            n_experts=int(model_cfg.get("n_experts", 4)),
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            repr_dim=int(model_cfg.get("repr_dim", 64)),
            expert_hidden_dim=int(model_cfg.get("expert_hidden_dim", model_cfg.get("d_model", 128))),
            env_count=int(model_cfg.get("env_count", 4)),
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


def _build_v16_loss_cfg(cfg: dict[str, Any]) -> FactorMechanismV16LossConfig:
    inv_cfg = cfg.get("invariant", {})
    train_cfg = cfg.get("train", {})
    rep_cfg = FactorMechanismV16LossConfig().representation
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
    rep_cfg.lambda_vicreg_covariance = float(
        inv_cfg.get("lambda_vicreg_covariance", rep_cfg.lambda_vicreg_covariance)
    )
    rep_cfg.lambda_phi_vicreg_variance = float(
        inv_cfg.get("lambda_phi_vicreg_variance", rep_cfg.lambda_phi_vicreg_variance)
    )
    rep_cfg.lambda_phi_vicreg_covariance = float(
        inv_cfg.get("lambda_phi_vicreg_covariance", rep_cfg.lambda_phi_vicreg_covariance)
    )
    rep_cfg.vicreg_variance_target = float(inv_cfg.get("vicreg_variance_target", rep_cfg.vicreg_variance_target))
    return FactorMechanismV16LossConfig(
        beta_vae=float(train_cfg.get("beta_vae", 0.1)),
        lambda_load_balance=float(inv_cfg.get("lambda_load_balance", 0.05)),
        lambda_env_balance=float(inv_cfg.get("lambda_env_balance", 0.01)),
        lambda_expert_diversity=float(inv_cfg.get("lambda_expert_diversity", 0.01)),
        representation=rep_cfg,
    )


def _train_base_model_v16(
    cfg: dict[str, Any],
    model,
    train_dl,
    val_dl,
    device: torch.device,
) -> list[dict[str, Any]]:
    train_cfg = cfg.get("train", {})
    loss_cfg = _build_v16_loss_cfg(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_offline_scheduler(cfg, opt)
    rows: list[dict[str, Any]] = []
    for epoch in range(int(train_cfg.get("epochs", 1))):
        batch_rows = train_offline_factor_v16(train_dl, model, opt, loss_cfg, device)
        train_mse, train_mae = evaluate_factor_v16(train_dl, model, device)
        val_mse, val_mae = evaluate_factor_v16(val_dl, model, device)
        aggregate = {
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae,
        }
        if batch_rows:
            numeric_keys = [key for key in batch_rows[0].keys() if key != "batch_index"]
            for key in numeric_keys:
                aggregate[key] = float(sum(float(row[key]) for row in batch_rows) / len(batch_rows))
        rows.append(aggregate)
        if scheduler is not None:
            scheduler.step()
    return rows


def _collect_split_tensors_v16(
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
        raw = out.pooled_delta_phi_v
        raw_parts.append(raw.detach().cpu().reshape(raw.shape[0], -1))
        y_base_parts.append(out.y_hat.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not raw_parts:
        raise ValueError("V16 collected no split samples")
    return torch.cat(raw_parts, dim=0), torch.cat(y_base_parts, dim=0), torch.cat(y_parts, dim=0)


def _generator_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("stage3_v2", {}).get("generator", {}).get("enabled", True))


def _online_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("online_run", {}).get("enable", True)) and bool(cfg.get("online", {}).get("stage3", {}).get("enable", True))


def _select_v16_generator_candidate(
    *,
    cfg: dict[str, Any],
    model,
    val_ds,
    device: torch.device,
    candidates: list[Stage3GeneratorCandidateV16],
    max_batches: int | None,
    out_dir: Path,
    dataset_label: str,
    run_id: str,
    seed: int,
) -> tuple[Path | None, list[dict[str, Any]]]:
    if not candidates:
        return None, []
    selection_cfg = cfg.get("stage3_v2", {}).get("generator", {}).get("checkpoint_selection", {})
    eval_cfg = deepcopy(cfg)
    eval_cfg.setdefault("online", {}).setdefault("stage3", {})["risk_gate_enable"] = False
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_path: Path | None = None
    for candidate in candidates:
        generator, metadata = load_stage3_generator_checkpoint(candidate.checkpoint_path, model=model)
        generator = generator.to(device)
        metrics = run_online_adaptation_v16(
            stream=iter_v16_representation_stream(
                val_ds,
                batch_size=int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
                device=device,
                model=model,
                max_batches=max_batches,
            ),
            model=model,
            generator=generator,
            risk_gate=None,
            cfg=eval_cfg,
            device=device,
            return_step_rows=False,
            trace_context={"dataset": dataset_label, "run_id": run_id, "seed": seed, "selection_stage": "pre_risk_gate"},
        )
        row = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": seed,
            "candidate_epoch": candidate.epoch,
            "candidate_checkpoint": str(candidate.checkpoint_path),
            "candidate_last_train_loss": candidate.last_train_loss,
            "generator_protocol_version": metadata.get("protocol_version", ""),
            **metrics,
        }
        rows.append(row)
        if best_row is None:
            best_row = row
            best_path = candidate.checkpoint_path
            continue
        current = (float(row["online_stage3_mse"]), int(row["candidate_epoch"]))
        best = (float(best_row["online_stage3_mse"]), int(best_row["candidate_epoch"]))
        if current < best:
            best_row = row
            best_path = candidate.checkpoint_path
    if best_row is not None:
        best_row["selected"] = 1.0
        for row in rows:
            row.setdefault("selected", 0.0)
        abstain = bool(selection_cfg.get("abstain_if_no_pass", False))
        if abstain and float(best_row["online_stage3_mse"]) >= float(best_row["online_base_mse"]):
            return None, rows
    _append_csv(out_dir / "generator_candidate_selection_v16.csv", rows)
    return best_path, rows


def _select_v16_risk_gate_threshold(
    *,
    cfg: dict[str, Any],
    model,
    generator,
    risk_gate: Stage3RiskGateBundle,
    val_ds,
    device: torch.device,
    max_batches: int | None,
    dataset_label: str,
    run_id: str,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
    thresholds = gate_cfg.get("threshold_grid") or [cfg["online"]["stage3"].get("risk_gate_threshold", 0.5)]
    rows: list[dict[str, Any]] = []
    best_threshold = float(thresholds[0])
    best_metric = float("inf")
    for threshold in thresholds:
        eval_cfg = deepcopy(cfg)
        eval_cfg["online"]["stage3"]["risk_gate_enable"] = True
        eval_cfg["online"]["stage3"]["risk_gate_threshold"] = float(threshold)
        metrics = run_online_adaptation_v16(
            stream=iter_v16_representation_stream(
                val_ds,
                batch_size=int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
                device=device,
                model=model,
                max_batches=max_batches,
            ),
            model=model,
            generator=generator,
            risk_gate=risk_gate,
            cfg=eval_cfg,
            device=device,
            return_step_rows=False,
            trace_context={"dataset": dataset_label, "run_id": run_id, "seed": seed, "selection_stage": "risk_gate_threshold"},
        )
        row = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": seed,
            "risk_gate_threshold": float(threshold),
            **metrics,
        }
        rows.append(row)
        if float(metrics["online_stage3_mse"]) < best_metric:
            best_metric = float(metrics["online_stage3_mse"])
            best_threshold = float(threshold)
    return best_threshold, rows


def run_stage3_v16_expert_local_dual_stream_probe(args: argparse.Namespace) -> Path:
    cfg = _apply_dataset_override(load_yaml(args.config), args.dataset)
    cfg.setdefault("stage3_v2", {})["protocol_version"] = PROTOCOL_VERSION
    cfg.setdefault("online", {}).setdefault("stage3", {})["version"] = PROTOCOL_VERSION
    cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = "delta_phi_v"
    cfg.setdefault("online", {}).setdefault("stage3", {})["signal_mode"] = "fluctuation_delta"
    cfg.setdefault("online", {}).setdefault("stage3", {})["canonicalize_delta"] = False
    cfg.setdefault("online", {}).setdefault("stage3", {})["calibrate_delta"] = False
    cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})["diagnostic_include_offline_counterfactuals"] = True
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    cfg["seed"] = args.seed
    validate_v16_config_contract(cfg)
    set_seed(args.seed)

    run_id = args.run_id or f"stage3_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_label = args.dataset or Path(cfg["data"]["path"]).stem
    out_dir = Path(args.out_root) / run_id / dataset_label / f"seed_{args.seed}"
    ckpt_dir = Path(args.checkpoint_root) / dataset_label / run_id / f"seed_{args.seed}"
    _prepare_fresh_run_dirs(out_dir=out_dir, ckpt_dir=ckpt_dir, overwrite=bool(getattr(args, "overwrite", False)))
    resolved_config_snapshot = _write_resolved_config_snapshot(cfg=cfg, out_dir=out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    series, _, target_dim = _load_series_and_features(cfg)
    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, split_info = _build_dataloaders(cfg, series, target_dim)
    train_iter = _limit_batches(train_dl, args.max_train_batches)
    val_iter = _limit_batches(val_dl, args.max_eval_batches)
    test_iter = _limit_batches(test_dl, args.max_eval_batches)
    train_eval_iter = _limit_batches(
        _ordered_loader(train_ds, cfg["train"]["batch_size"], drop_last=True),
        args.max_eval_batches,
    )

    model = _build_v16_model(cfg, series.shape[1], target_dim).to(device)
    assert_deterministic_eval_model(model, context="Stage III V16 expert-local dual-stream probe")

    offline_rows = _train_base_model_v16(cfg, model, train_iter, val_iter, device)
    train_mse, train_mae = evaluate_factor_v16(train_eval_iter, model, device)
    val_mse, val_mae = evaluate_factor_v16(val_iter, model, device)
    test_mse, test_mae = evaluate_factor_v16(test_iter, model, device)
    torch.save(model.state_dict(), ckpt_dir / "base_model_v16.pt")
    _append_csv(out_dir / "offline_metrics_v16.csv", offline_rows)

    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    if batch_size != 1:
        raise ValueError("V16 residual-target diagnostic requires stage3_v2.online_batch_size=1")
    train_raw, train_y_base, train_y = _collect_split_tensors_v16(
        dataset=train_ds,
        batch_size=batch_size,
        device=device,
        model=model,
        max_batches=args.max_generator_batches,
    )
    val_raw, val_y_base, val_y = _collect_split_tensors_v16(
        dataset=val_ds,
        batch_size=batch_size,
        device=device,
        model=model,
        max_batches=args.max_eval_batches,
    )
    test_raw, test_y_base, test_y = _collect_split_tensors_v16(
        dataset=test_ds,
        batch_size=batch_size,
        device=device,
        model=model,
        max_batches=args.max_eval_batches,
    )

    teacher_cfg = cfg.get("stage3_v2", {}).get("residual_teacher", {})
    history_cfg = cfg.get("stage3_v2", {}).get("v12d_history", {})
    history_builder = _build_history_builder(
        output_shape=tuple(train_y.shape[1:]),
        history_cfg=history_cfg,
    ).fit(y_base_train=train_y_base, y_train=train_y, split="train")
    history_lags = tuple(int(value) for value in getattr(history_builder, "lags"))
    feedback_delay = int(getattr(history_builder, "label_delay", getattr(history_builder, "feedback_delay", 0)))
    signal_builder = Stage3SignalBuilderV13(
        drift_dim=train_raw.shape[1],
        mode=str(teacher_cfg.get("signal_mode", cfg["online"]["stage3"].get("signal_mode", "fluctuation_delta"))),
        normalize=bool(teacher_cfg.get("normalize_vector", cfg["online"]["stage3"].get("normalize_vector", True))),
        robust_scale_quantile=float(teacher_cfg.get("robust_scale_quantile", 0.75)),
        eps=float(teacher_cfg.get("signal_eps", 1e-6)),
        polarity=_stage3_drift_polarity(cfg),
        random_seed=int(teacher_cfg.get("counterfactual_seed", 1729)),
    ).fit(train_raw, split="train")
    train_signal_result = signal_builder.transform(train_raw, counterfactual_mode="real")
    train_signal = train_signal_result.signal
    train_history = history_builder.transform(y_base=train_y_base, y_true=train_y).features
    targets = _resolve_v15_targets(cfg, args.targets)
    feature_modes = args.feature_modes or cfg.get("stage3_v2", {}).get("v15_feature_modes", list(DEFAULT_V15_FEATURE_MODES))
    feature_modes = [str(value) for value in feature_modes]

    teachers: dict[tuple[str, str], TrainOnlyStabilityFluctuationResidualTeacherV14] = {}
    teacher_diagnostic_rows: list[dict[str, Any]] = []
    for target_name in targets:
        target_tensor = build_v15_target_tensor(target_name, y_base=train_y_base, y_true=train_y)
        y_base_target, y_true_target = v15_target_to_teacher_tensors(target_tensor)
        for feature_mode in feature_modes:
            feature_slug = feature_mode.replace("_", "")
            context = build_stage3_v14_context_features(
                fluctuation_signal=train_signal,
                history_features=train_history,
                level_offset=train_signal_result.level_offset,
                mode=feature_mode,
            )
            teacher = _teacher_from_cfg(
                teacher_cfg=teacher_cfg,
                feature_mode=feature_mode,
                feature_dim=context.feature_dim,
                output_shape=tuple(y_true_target.shape[1:]),
                history_lags=history_lags,
                feedback_delay=feedback_delay,
                history_feature_dim=context.history_dim,
                drift_signal_dim=context.fluctuation_dim if "level_offset" not in feature_mode else context.level_offset_dim,
            ).fit(
                drift_train=context.features,
                y_base_train=y_base_target,
                y_train=y_true_target,
                split="train",
                sample_ids=torch.arange(context.features.shape[0], dtype=torch.long),
            )
            teachers[(target_name, feature_mode)] = teacher
            save_stage3_v14_teacher_checkpoint(
                path=ckpt_dir / f"teacher_v16_{target_name}_{feature_slug}.pt",
                teacher=teacher,
                metadata={
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "target_name": target_name,
                    "feature_mode": feature_mode,
                    "v16_protocol_version": PROTOCOL_VERSION,
                },
            )
            teacher_diagnostic_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "target_name": target_name,
                    "feature_mode": feature_mode,
                    **teacher.diagnostic_summary(),
                }
            )
    _append_csv(out_dir / "residual_teacher_v16_diagnostics.csv", teacher_diagnostic_rows)
    _append_csv(out_dir / "stage3_v16_signal_contract_diagnostics.csv", [signal_builder.diagnostic_summary()])
    _append_csv(out_dir / "stage3_v16_history_contract_diagnostics.csv", [history_builder.diagnostic_summary()])

    modes = args.modes or cfg.get("stage3_v2", {}).get("sensitivity_modes") or list(DEFAULT_MODES)
    modes = [str(mode) for mode in modes]
    _validate_v14_gate_modes(modes)
    train_raw_tail = train_raw[-1:]
    train_residual_tail = history_builder.residual_tail(y_base=train_y_base, y_true=train_y)
    validation_rows = _rows_for_split(
        cfg=cfg,
        split="validation",
        raw_drift=val_raw,
        y_base=val_y_base,
        y_true=val_y,
        previous_raw_tail=train_raw_tail,
        previous_residual_tail=train_residual_tail,
        teachers=teachers,
        signal_builder=signal_builder,
        history_builder=history_builder,
        modes=modes,
        targets=targets,
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
    )
    validation_summary_rows = [
        {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "validation", **row}
        for row in _summary_rows_for_target_feature(validation_rows)
    ]
    _append_csv(out_dir / "teacher_only_validation_summary_v16.csv", validation_rows)
    _append_csv(out_dir / "teacher_only_validation_diagnostic_summary_v16.csv", validation_summary_rows)
    _append_csv(
        out_dir / "teacher_only_validation_partial_gain_vs_history_v16.csv",
        _partial_gain_rows(
            split="validation",
            summary_rows=validation_summary_rows,
            run_id=run_id,
            dataset_label=dataset_label,
            seed=args.seed,
        ),
    )

    validation_windows = int(args.validation_windows or cfg.get("stage3_v2", {}).get("validation_windows", 1))
    rolling_rows: list[dict[str, Any]] = []
    rolling_window_rows: list[dict[str, Any]] = []
    rolling_aggregate_rows: list[dict[str, Any]] = []
    rolling_partial_rows: list[dict[str, Any]] = []
    if validation_windows > 1:
        rolling_rows, rolling_window_rows, rolling_aggregate_rows, rolling_partial_rows = _rolling_rows_for_split(
            cfg=cfg,
            raw_drift=val_raw,
            y_base=val_y_base,
            y_true=val_y,
            previous_raw_tail=train_raw_tail,
            previous_residual_tail=train_residual_tail,
            teachers=teachers,
            signal_builder=signal_builder,
            history_builder=history_builder,
            modes=modes,
            targets=targets,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            window_count=validation_windows,
        )
        _append_csv(out_dir / "teacher_only_rolling_validation_summary_v16.csv", rolling_rows)
        _append_csv(out_dir / "teacher_only_rolling_validation_window_summary_v16.csv", rolling_window_rows)
        _append_csv(out_dir / "teacher_only_rolling_validation_aggregate_v16.csv", rolling_aggregate_rows)
        _append_csv(out_dir / "teacher_only_rolling_partial_gain_vs_history_v16.csv", rolling_partial_rows)

    val_raw_tail = val_raw[-1:] if val_raw.numel() else train_raw_tail
    val_residual_tail = history_builder.residual_tail(y_base=val_y_base, y_true=val_y)
    test_rows = _rows_for_split(
        cfg=cfg,
        split="test",
        raw_drift=test_raw,
        y_base=test_y_base,
        y_true=test_y,
        previous_raw_tail=val_raw_tail,
        previous_residual_tail=val_residual_tail,
        teachers=teachers,
        signal_builder=signal_builder,
        history_builder=history_builder,
        modes=modes,
        targets=targets,
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
    )
    test_summary_rows = [
        {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "test", **row}
        for row in _summary_rows_for_target_feature(test_rows)
    ]
    _append_csv(out_dir / "teacher_only_test_summary_v16.csv", test_rows)
    _append_csv(out_dir / "teacher_only_test_diagnostic_summary_v16.csv", test_summary_rows)

    gate_rows = _v15_target_gate_rows(
        cfg=cfg,
        targets=targets,
        rolling_aggregate_rows=rolling_aggregate_rows,
        rolling_partial_rows=rolling_partial_rows,
    )
    _append_csv(out_dir / "teacher_only_v16_target_gate.csv", gate_rows)
    stop_reasons = v15_directional_stop_reasons(gate_rows)
    generator_checkpoint_sha256 = ""
    risk_gate_checkpoint_sha256 = ""
    generator_selection_rows: list[dict[str, Any]] = []
    risk_gate_metrics: dict[str, Any] = {}
    risk_gate_threshold_rows: list[dict[str, Any]] = []
    online_summary_rows: list[dict[str, Any]] = []
    selected_risk_gate_threshold = cfg["online"]["stage3"].get("risk_gate_threshold")
    if stop_reasons:
        (out_dir / "teacher_only_abort.json").write_text(
            json.dumps({"stop_reasons": stop_reasons, "protocol_version": PROTOCOL_VERSION}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    elif _generator_enabled(cfg):
        candidate_dir = ckpt_dir / "generator_candidates_v16"
        generator, generator_rows, candidates = train_stage3_generator_v16(
            cfg=cfg,
            model=model,
            train_ds=train_ds,
            device=device,
            max_batches=args.max_generator_batches,
            candidate_checkpoint_dir=candidate_dir,
            candidate_metadata={
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": args.seed,
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        _append_csv(out_dir / "generator_train_v16.csv", generator_rows)
        selected_candidate_path, generator_selection_rows = _select_v16_generator_candidate(
            cfg=cfg,
            model=model,
            val_ds=val_ds,
            device=device,
            candidates=candidates,
            max_batches=args.max_eval_batches,
            out_dir=out_dir,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
        )
        if selected_candidate_path is None:
            stop_reasons = ["generator_candidate_selection_failed"]
            (out_dir / "stage3_v16_abort.json").write_text(
                json.dumps({"stop_reasons": stop_reasons, "protocol_version": PROTOCOL_VERSION}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            selected_generator, selected_generator_metadata = load_stage3_generator_checkpoint(selected_candidate_path, model=model)
            selected_generator = selected_generator.to(device)
            final_generator_ckpt = ckpt_dir / "generator_v16_selected.pt"
            save_stage3_generator_checkpoint(
                path=final_generator_ckpt,
                generator=selected_generator,
                metadata={
                    **selected_generator_metadata,
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "protocol_version": PROTOCOL_VERSION,
                    "selection_stage": "pre_risk_gate_validation",
                    "selected_candidate_path": str(selected_candidate_path),
                },
                overwrite=True,
            )

            risk_gate_ckpt: Path | None = None
            if bool(cfg["online"]["stage3"].get("risk_gate_enable", False)):
                risk_features, risk_labels, risk_rows = collect_stage3_v16_risk_gate_examples(
                    cfg=cfg,
                    model=model,
                    generator=selected_generator,
                    train_ds=train_ds,
                    device=device,
                    max_batches=args.max_generator_batches,
                )
                gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
                risk_gate, standardizer, risk_gate_metrics = train_stage3_risk_gate(
                    risk_features,
                    risk_labels,
                    hidden_dim=int(gate_cfg.get("hidden_dim", 64)),
                    dropout=float(gate_cfg.get("dropout", 0.0)),
                    epochs=int(gate_cfg.get("epochs", 120)),
                    lr=float(gate_cfg.get("lr", 1e-3)),
                    positive_weight=gate_cfg.get("positive_weight"),
                    model_type=str(gate_cfg.get("model_type", "mlp")),
                )
                risk_gate_ckpt = ckpt_dir / "risk_gate_v16.pt"
                save_stage3_risk_gate_checkpoint(
                    path=risk_gate_ckpt,
                    gate=risk_gate,
                    standardizer=standardizer,
                    metadata={
                        "run_id": run_id,
                        "dataset": dataset_label,
                        "seed": args.seed,
                        "protocol_version": PROTOCOL_VERSION,
                    },
                    overwrite=True,
                )
                _append_csv(out_dir / "risk_gate_train_examples_v16.csv", risk_rows)
                _append_csv(out_dir / "risk_gate_train_metrics_v16.csv", [{"run_id": run_id, "dataset": dataset_label, "seed": args.seed, **risk_gate_metrics}])

            artifact_bundle = load_stage3_v16_artifact_bundle(
                generator_checkpoint=final_generator_ckpt,
                model=model,
                risk_gate_checkpoint=risk_gate_ckpt,
            )
            selected_generator = artifact_bundle.generator.to(device)
            selected_risk_gate = artifact_bundle.risk_gate.to(device) if artifact_bundle.risk_gate is not None else None
            if (
                selected_risk_gate is not None
                and bool(cfg.get("stage3_v2", {}).get("risk_gate", {}).get("calibrate_threshold", True))
            ):
                selected_risk_gate_threshold, risk_gate_threshold_rows = _select_v16_risk_gate_threshold(
                    cfg=cfg,
                    model=model,
                    generator=selected_generator,
                    risk_gate=selected_risk_gate,
                    val_ds=val_ds,
                    device=device,
                    max_batches=args.max_eval_batches,
                    dataset_label=dataset_label,
                    run_id=run_id,
                    seed=args.seed,
                )
                cfg["online"]["stage3"]["risk_gate_threshold"] = selected_risk_gate_threshold
                _append_csv(out_dir / "risk_gate_threshold_selection_v16.csv", risk_gate_threshold_rows)

            if _online_enabled(cfg):
                write_step_metrics = bool(cfg.get("stage3_v2", {}).get("write_step_metrics", False))
                val_metrics = run_online_adaptation_v16(
                    stream=iter_v16_representation_stream(
                        val_ds,
                        batch_size=batch_size,
                        device=device,
                        model=model,
                        max_batches=args.max_eval_batches,
                    ),
                    model=model,
                    generator=selected_generator,
                    risk_gate=selected_risk_gate,
                    cfg=cfg,
                    device=device,
                    return_step_rows=write_step_metrics,
                    trace_context={"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "validation"},
                )
                test_metrics = run_online_adaptation_v16(
                    stream=iter_v16_representation_stream(
                        test_ds,
                        batch_size=batch_size,
                        device=device,
                        model=model,
                        max_batches=args.max_eval_batches,
                    ),
                    model=model,
                    generator=selected_generator,
                    risk_gate=selected_risk_gate,
                    cfg=cfg,
                    device=device,
                    return_step_rows=write_step_metrics,
                    trace_context={"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "test"},
                )
                if write_step_metrics and "step_rows" in val_metrics:
                    _append_csv(out_dir / "online_validation_step_metrics_v16.csv", val_metrics.pop("step_rows"))
                if write_step_metrics and "step_rows" in test_metrics:
                    _append_csv(out_dir / "online_test_step_metrics_v16.csv", test_metrics.pop("step_rows"))
                online_summary_rows = [
                    {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "validation", **val_metrics},
                    {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "test", **test_metrics},
                ]
                _append_csv(out_dir / "online_summary_v16.csv", online_summary_rows)

            generator_checkpoint_sha256 = _sha256_file(final_generator_ckpt)
            risk_gate_checkpoint_sha256 = _sha256_file(risk_gate_ckpt) if risk_gate_ckpt is not None else ""

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": args.seed,
        "dataset": dataset_label,
        "run_id": run_id,
        "git_status_short": _git_status_short(),
        "resolved_config_json": resolved_config_snapshot,
        "base_checkpoint_sha256": _sha256_file(ckpt_dir / "base_model_v16.pt"),
        "base_train_mse": train_mse,
        "base_train_mae": train_mae,
        "base_val_mse": val_mse,
        "base_val_mae": val_mae,
        "base_test_mse": test_mse,
        "base_test_mae": test_mae,
        "split_info": split_info,
        "v15_reference_protocol": V15_PROTOCOL_VERSION,
        "v16_model_native_drift_source": "pooled_delta_phi_v",
        "stage3_stop_reasons": stop_reasons,
        "stage3_generator_enabled": _generator_enabled(cfg),
        "stage3_generator_candidate_count": len(generator_selection_rows),
        "stage3_generator_checkpoint_sha256": generator_checkpoint_sha256,
        "stage3_risk_gate_checkpoint_sha256": risk_gate_checkpoint_sha256,
        "stage3_risk_gate_threshold": selected_risk_gate_threshold,
        "stage3_risk_gate_train_metrics": risk_gate_metrics,
        "stage3_online_summary_rows": online_summary_rows,
    }
    (out_dir / "run_manifest_v16.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V16 expert-local dual-stream residual-target probe.")
    parser.add_argument("--config", type=str, default="configs/stage3_v16_expert_local_dual_stream_probe.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--validation-windows", type=int, default=None)
    parser.add_argument("--out-root", type=str, default="logs/s3v16expert")
    parser.add_argument("--checkpoint-root", type=str, default="ckpt/s3v16expert")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--feature-modes", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = run_stage3_v16_expert_local_dual_stream_probe(args)
    print(str(out_dir))


if __name__ == "__main__":
    main()
