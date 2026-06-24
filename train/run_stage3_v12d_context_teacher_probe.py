from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.stage3_residual_teacher_v12d import (  # noqa: E402
    TrainOnlyContextOOFResidualTeacherV12d,
    save_stage3_v12d_teacher_checkpoint,
)
from train.run_stage3_v12c_sensitivity_audit import (  # noqa: E402
    _alignment_diagnostic_rows,
    _append_csv,
    _apply_dataset_override,
    _build_dataloaders,
    _build_model,
    _extract_prediction,
    _fit_delta_standardizer,
    _fit_residual_canonicalizer,
    _git_status_short,
    _limit_batches,
    _load_series_and_features,
    _ordered_loader,
    _prepare_fresh_run_dirs,
    _sha256_file,
    _stage3_drift_polarity,
    _stage3_drift_source,
    _teacher_diagnostic_aggregate,
    _teacher_gate_mode_rows,
    _train_aligned_model,
    _train_base_model,
    _write_resolved_config_snapshot,
    evaluate_factor,
    load_yaml,
    set_seed,
)
from train.run_stage3_v12c_sensitivity_audit import _aligned_stream  # noqa: E402
from train.stage3_eval_policy import assert_deterministic_eval_model  # noqa: E402
from train.stage3_v12c_signal_contract import Stage3TeacherSignalBuilderV12c  # noqa: E402
from train.stage3_v12d_context_features import (  # noqa: E402
    CausalResidualHistoryFeatureBuilderV12d,
    build_stage3_v12d_context_features,
)
from train.stage3_v12e_label_availability import PerHorizonLabelAvailabilityHistoryBuilderV12e  # noqa: E402
from train.stage3_v12e_conditional_permutation import (  # noqa: E402
    build_history_norm_strata,
    conditional_permute_drift,
    conditional_random_matched_norm,
)
from train.stage3_v9_teacher_diagnostics import (  # noqa: E402
    compute_teacher_correction_metrics,
    summarize_teacher_mode_rows,
)


DEFAULT_MODES = ("real", "zero", "shuffled", "sign_flipped", "amplified_2x", "random_matched_norm")
DEFAULT_FEATURE_MODES = ("drift_only", "history_only", "drift_plus_history")
FEATURE_MODE_SLUGS = {"drift_only": "d", "history_only": "h", "drift_plus_history": "dh"}
CONDITIONAL_DRIFT_MODES = {"conditional_shuffled", "conditional_permuted_drift", "conditional_random_matched_norm"}


def _collect_split_tensors(
    *,
    cfg: dict[str, Any],
    dataset,
    batch_size: int,
    device: torch.device,
    aligned_model,
    model,
    max_batches: int | None,
    delta_standardizer,
    residual_canonicalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_parts: list[torch.Tensor] = []
    y_base_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    model.eval()
    stream = _aligned_stream(
        dataset,
        batch_size,
        device,
        aligned_model,
        max_batches=max_batches,
        drift_source=_stage3_drift_source(cfg),
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )
    for x, y, phi_i, phi_v in stream:
        with torch.no_grad():
            y_base = _extract_prediction(model(x))
        raw_parts.append((phi_v - phi_i).detach().cpu().reshape(1, -1))
        y_base_parts.append(y_base.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not raw_parts:
        raise ValueError("V12d collected no split samples")
    return torch.cat(raw_parts, dim=0), torch.cat(y_base_parts, dim=0), torch.cat(y_parts, dim=0)


def _raw_previous_tail(raw: torch.Tensor, fallback_tail: torch.Tensor | None, start: int) -> torch.Tensor | None:
    if start > 0:
        return raw[start - 1 : start]
    if fallback_tail is None or fallback_tail.numel() == 0:
        return None
    return fallback_tail.reshape(-1, raw.shape[1])[-1:]


def _residual_previous_tail(
    *,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    fallback_tail: torch.Tensor | None,
    start: int,
    required_tail_length: int,
) -> torch.Tensor | None:
    if required_tail_length <= 0:
        return None
    residual = (y_true.detach().float().reshape(y_true.shape[0], -1) - y_base.detach().float().reshape(y_base.shape[0], -1))
    current_tail = residual[max(0, start - required_tail_length) : start]
    if fallback_tail is not None and current_tail.shape[0] < required_tail_length:
        prepend = fallback_tail.detach().float().reshape(-1, residual.shape[1])
        needed = required_tail_length - current_tail.shape[0]
        current_tail = torch.cat([prepend[-needed:], current_tail], dim=0)
    if current_tail.numel() == 0:
        return None
    return current_tail[-required_tail_length:]


def _window_slices(sample_count: int, window_count: int) -> list[slice]:
    if window_count <= 1:
        return [slice(0, sample_count)]
    count = min(window_count, sample_count)
    base = sample_count // count
    rem = sample_count % count
    slices: list[slice] = []
    cursor = 0
    for idx in range(count):
        size = base + (1 if idx < rem else 0)
        slices.append(slice(cursor, cursor + size))
        cursor += size
    return slices


def _teacher_from_cfg(
    *,
    teacher_cfg: dict[str, Any],
    feature_mode: str,
    feature_dim: int,
    output_shape: tuple[int, ...],
    history_lags: tuple[int, ...],
    feedback_delay: int,
    history_feature_dim: int,
    drift_signal_dim: int,
) -> TrainOnlyContextOOFResidualTeacherV12d:
    return TrainOnlyContextOOFResidualTeacherV12d(
        drift_dim=feature_dim,
        output_shape=output_shape,
        context_feature_mode=feature_mode,
        history_lags=history_lags,
        feedback_delay=feedback_delay,
        history_feature_dim=history_feature_dim,
        drift_signal_dim=drift_signal_dim,
        ridge_grid=teacher_cfg.get("ridge_grid", [teacher_cfg.get("ridge", 1e-3)]),
        alpha_grid=teacher_cfg.get("alpha_grid", teacher_cfg.get("shrinkage_grid")),
        fold_count=int(teacher_cfg.get("fold_count", teacher_cfg.get("calibration_block_count", 5))),
        min_prefix_samples=int(teacher_cfg.get("min_prefix_samples", 32)),
        purge_gap=int(teacher_cfg.get("purge_gap", 0)),
        min_block_mse_gain=float(teacher_cfg.get("min_block_mse_gain", 0.0)),
        min_block_mae_gain=float(teacher_cfg.get("min_block_mae_gain", 0.0)),
        min_block_sign_agreement=float(teacher_cfg.get("min_block_sign_agreement", 0.5)),
        min_block_pass_fraction=float(teacher_cfg.get("min_block_pass_fraction", 0.6)),
        min_output_gain=float(teacher_cfg.get("min_output_gain", 0.0)),
        calibration_mae_weight=float(teacher_cfg.get("calibration_mae_weight", 0.0)),
        allow_negative_alpha=bool(teacher_cfg.get("allow_negative_alpha", True)),
        reliability_gate_enabled=bool(teacher_cfg.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(teacher_cfg.get("reliability_gate_quantile", 0.95)),
        min_reliable_fraction=float(teacher_cfg.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(teacher_cfg.get("disable_if_unreliable", False)),
        selected_policy=str(teacher_cfg.get("selected_policy", "relaxed")),
        claim_policy=str(teacher_cfg.get("claim_policy", "strict")),
    )


def _feature_mode_slug(feature_mode: str) -> str:
    return FEATURE_MODE_SLUGS.get(feature_mode, feature_mode.replace("_", "")[:12])


def _history_protocol(history_cfg: dict[str, Any]) -> str:
    return str(history_cfg.get("protocol", "v12d_sample_delay")).lower()


def _build_history_builder(
    *,
    output_shape: tuple[int, ...],
    history_cfg: dict[str, Any],
):
    protocol = _history_protocol(history_cfg)
    history_lags = tuple(int(value) for value in history_cfg.get("lags", [1, 2, 3, 6, 12, 24]))
    common_kwargs = {
        "output_shape": output_shape,
        "lags": history_lags,
        "include_availability_mask": bool(history_cfg.get("include_availability_mask", True)),
        "standardize": bool(history_cfg.get("standardize", True)),
        "robust_scale_quantile": float(history_cfg.get("robust_scale_quantile", 0.75)),
        "eps": float(history_cfg.get("eps", 1e-6)),
    }
    if protocol in {"v12e", "v12e_per_horizon", "v12e_per_horizon_label_availability"}:
        return PerHorizonLabelAvailabilityHistoryBuilderV12e(
            **common_kwargs,
            horizon_offsets=history_cfg.get("horizon_offsets"),
            label_delay=int(history_cfg.get("label_delay", history_cfg.get("feedback_delay", 0))),
        )
    if protocol in {"v12d", "v12d_sample_delay", "sample_delay"}:
        return CausalResidualHistoryFeatureBuilderV12d(
            **common_kwargs,
            feedback_delay=int(history_cfg.get("feedback_delay", 0)),
        )
    raise ValueError(f"unsupported V12d/V12e history protocol: {protocol}")


def _rows_for_split(
    *,
    cfg: dict[str, Any],
    split: str,
    raw_drift: torch.Tensor,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_raw_tail: torch.Tensor | None,
    previous_residual_tail: torch.Tensor | None,
    teachers: dict[str, TrainOnlyContextOOFResidualTeacherV12d],
    signal_builder: Stage3TeacherSignalBuilderV12c,
    history_builder: CausalResidualHistoryFeatureBuilderV12d,
    modes: list[str],
    dataset_label: str,
    run_id: str,
    seed: int,
    window_id: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> list[dict[str, Any]]:
    history_result = history_builder.transform(
        y_base=y_base,
        y_true=y_true,
        previous_residual_tail=previous_residual_tail,
    )
    conditional_cfg = cfg.get("stage3_v2", {}).get("v12e_conditional_permutation", {})
    conditional_bin_count = int(conditional_cfg.get("history_norm_bin_count", 4))
    conditional_seed = int(conditional_cfg.get("seed", 1729)) + int(seed)
    strata = build_history_norm_strata(
        history_features=history_result.features,
        bin_count=conditional_bin_count,
    )
    rows: list[dict[str, Any]] = []
    for mode in modes:
        signal_mode = "real" if mode in CONDITIONAL_DRIFT_MODES else mode
        signal_result = signal_builder.transform(
            raw_drift,
            previous_tail=previous_raw_tail,
            counterfactual_mode=signal_mode,
        )
        drift_signal = signal_result.signal
        conditional_protocol = ""
        if mode in {"conditional_shuffled", "conditional_permuted_drift"}:
            drift_signal = conditional_permute_drift(
                drift_signal=drift_signal,
                strata=strata,
                seed=conditional_seed,
            )
            conditional_protocol = "history_norm_stratified_row_permutation"
        elif mode == "conditional_random_matched_norm":
            drift_signal = conditional_random_matched_norm(
                drift_signal=drift_signal,
                strata=strata,
                seed=conditional_seed,
            )
            conditional_protocol = "history_norm_stratified_random_matched_norm"
        for feature_mode, teacher in teachers.items():
            context = build_stage3_v12d_context_features(
                drift_signal=drift_signal,
                history_features=history_result.features,
                mode=feature_mode,  # type: ignore[arg-type]
            )
            with torch.no_grad():
                correction = teacher.predict_correction(context.features).detach().cpu()
            metrics = compute_teacher_correction_metrics(
                y_base=y_base,
                y_true=y_true,
                correction=correction,
            )
            row: dict[str, Any] = {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": seed,
                "split": split,
                "feature_mode": feature_mode,
                "mode": mode,
                "intervention_online_causal": mode in {"real", "zero", "sign_flipped", "amplified_2x", "amplified_5x"},
                "intervention_protocol": "streaming_causal" if mode in {"real", "zero", "sign_flipped", "amplified_2x", "amplified_5x"} else "offline_diagnostic",
                "counterfactual_intervention_space": cfg.get("stage3_v2", {}).get("counterfactual_intervention_space", "generator_input"),
                "teacher_signal_contract": signal_builder.protocol_version,
                "teacher_history_contract": history_builder.protocol_version,
                "teacher_conditional_permutation_protocol": conditional_protocol,
                "teacher_conditional_permutation_bin_count": conditional_bin_count if conditional_protocol else "",
                "teacher_context_protocol": TrainOnlyContextOOFResidualTeacherV12d.protocol_version,
                "teacher_fit_split": teacher.fit_split,
                "teacher_fit_sample_count": teacher.fit_sample_count,
                "teacher_train_residual_r2": teacher.teacher_residual_r2,
                "teacher_shrinkage_alpha": teacher.shrinkage_alpha,
                "teacher_alpha_source": teacher.selected_alpha_source,
                "teacher_reliability_gate_active": teacher.reliability_gate_active,
                "teacher_context_feature_dim": context.feature_dim,
                "teacher_context_drift_dim": context.drift_dim,
                "teacher_context_history_dim": context.history_dim,
                "teacher_drift_norm_mean": float(torch.linalg.vector_norm(drift_signal, dim=1).mean().item()),
                "teacher_history_feature_norm_mean": float(torch.linalg.vector_norm(history_result.features, dim=1).mean().item()),
                **metrics,
            }
            if window_id is not None:
                row.update(
                    {
                        "validation_window_id": window_id,
                        "validation_window_start_sample": window_start,
                        "validation_window_end_sample": window_end,
                    }
                )
            rows.append(row)
    return rows


def _summary_rows_for_feature_modes(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_modes = sorted({str(row["feature_mode"]) for row in rows})
    summaries: list[dict[str, Any]] = []
    for feature_mode in feature_modes:
        mode_rows = [row for row in rows if str(row["feature_mode"]) == feature_mode]
        summaries.append({"feature_mode": feature_mode, **summarize_teacher_mode_rows(_teacher_gate_mode_rows(cfg=cfg, rows=mode_rows))})
    return summaries


def _partial_gain_row(*, split: str, summary_rows: list[dict[str, Any]], run_id: str, dataset_label: str, seed: int) -> dict[str, Any]:
    by_feature = {str(row["feature_mode"]): row for row in summary_rows}
    history = by_feature.get("history_only", {})
    both = by_feature.get("drift_plus_history", {})

    def value(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    h_mse = value(history, "teacher_real_mse_gain")
    dh_mse = value(both, "teacher_real_mse_gain")
    h_mae = value(history, "teacher_real_mae_gain")
    dh_mae = value(both, "teacher_real_mae_gain")
    h_sign = value(history, "teacher_real_sign_agreement_rate")
    dh_sign = value(both, "teacher_real_sign_agreement_rate")
    return {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": seed,
        "split": split,
        "history_only_real_mse_gain": h_mse,
        "drift_plus_history_real_mse_gain": dh_mse,
        "partial_mse_gain_vs_history": dh_mse - h_mse if h_mse == h_mse and dh_mse == dh_mse else float("nan"),
        "history_only_real_mae_gain": h_mae,
        "drift_plus_history_real_mae_gain": dh_mae,
        "partial_mae_gain_vs_history": dh_mae - h_mae if h_mae == h_mae and dh_mae == dh_mae else float("nan"),
        "history_only_sign_agreement": h_sign,
        "drift_plus_history_sign_agreement": dh_sign,
        "partial_sign_agreement_vs_history": dh_sign - h_sign if h_sign == h_sign and dh_sign == dh_sign else float("nan"),
    }


def _rolling_rows_for_split(
    *,
    cfg: dict[str, Any],
    raw_drift: torch.Tensor,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_raw_tail: torch.Tensor | None,
    previous_residual_tail: torch.Tensor | None,
    teachers: dict[str, TrainOnlyContextOOFResidualTeacherV12d],
    signal_builder: Stage3TeacherSignalBuilderV12c,
    history_builder: CausalResidualHistoryFeatureBuilderV12d,
    modes: list[str],
    dataset_label: str,
    run_id: str,
    seed: int,
    window_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    residual_tail_len = history_builder.required_tail_length
    for window_id, window_slice in enumerate(_window_slices(raw_drift.shape[0], window_count)):
        start = int(window_slice.start or 0)
        stop = int(window_slice.stop or raw_drift.shape[0])
        raw_tail = _raw_previous_tail(raw_drift, previous_raw_tail, start)
        residual_tail = _residual_previous_tail(
            y_base=y_base,
            y_true=y_true,
            fallback_tail=previous_residual_tail,
            start=start,
            required_tail_length=residual_tail_len,
        )
        rows = _rows_for_split(
            cfg=cfg,
            split=f"validation_window_{window_id}",
            raw_drift=raw_drift[window_slice],
            y_base=y_base[window_slice],
            y_true=y_true[window_slice],
            previous_raw_tail=raw_tail,
            previous_residual_tail=residual_tail,
            teachers=teachers,
            signal_builder=signal_builder,
            history_builder=history_builder,
            modes=modes,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            window_id=window_id,
            window_start=start,
            window_end=stop,
        )
        summaries = _summary_rows_for_feature_modes(cfg, rows)
        for summary in summaries:
            summary.update(
                {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": seed,
                    "split": f"validation_window_{window_id}",
                    "validation_window_id": window_id,
                    "validation_window_start_sample": start,
                    "validation_window_end_sample": stop,
                    "validation_window_count": window_count,
                }
            )
        summary_rows.extend(summaries)
        partial = _partial_gain_row(
            split=f"validation_window_{window_id}",
            summary_rows=summaries,
            run_id=run_id,
            dataset_label=dataset_label,
            seed=seed,
        )
        partial.update({"validation_window_id": window_id, "validation_window_start_sample": start, "validation_window_end_sample": stop})
        partial_rows.append(partial)
        all_rows.extend(rows)
    for feature_mode in sorted({str(row["feature_mode"]) for row in summary_rows}):
        mode_rows = [row for row in summary_rows if str(row["feature_mode"]) == feature_mode]
        aggregate = _teacher_diagnostic_aggregate(
            cfg=cfg,
            run_id=run_id,
            dataset_label=dataset_label,
            seed=seed,
            split=f"validation_rolling_{feature_mode}",
            window_rows=mode_rows,
        )
        aggregate["feature_mode"] = feature_mode
        aggregate_rows.append(aggregate)
    return all_rows, summary_rows, aggregate_rows, partial_rows


def run_stage3_v12d_context_teacher_probe(args: argparse.Namespace) -> Path:
    cfg = _apply_dataset_override(load_yaml(args.config), args.dataset)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.invariant_epochs is not None:
        cfg["invariant"]["epochs"] = args.invariant_epochs
    if args.drift_source is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = args.drift_source
    if args.drift_polarity is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_polarity"] = args.drift_polarity
    if args.signal_mode is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["signal_mode"] = args.signal_mode
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})["signal_mode"] = args.signal_mode
    if args.normalize_vector is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["normalize_vector"] = args.normalize_vector
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})["normalize_vector"] = args.normalize_vector
    history_override = cfg.setdefault("stage3_v2", {}).setdefault("v12d_history", {})
    if getattr(args, "history_protocol", None) is not None:
        history_override["protocol"] = args.history_protocol
        if str(args.history_protocol).lower().startswith("v12e"):
            cfg.setdefault("stage3_v2", {})["protocol_version"] = "v12e_context_teacher_probe"
            cfg.setdefault("online", {}).setdefault("stage3", {})["version"] = "v12e_context_teacher_probe"
    if getattr(args, "history_label_delay", None) is not None:
        history_override["label_delay"] = args.history_label_delay
    if getattr(args, "history_feedback_delay", None) is not None:
        history_override["feedback_delay"] = args.history_feedback_delay
    if getattr(args, "history_lags", None) is not None:
        history_override["lags"] = args.history_lags
    if getattr(args, "history_horizon_offsets", None) is not None:
        history_override["horizon_offsets"] = args.history_horizon_offsets
    if getattr(args, "diagnostic_include_offline_counterfactuals", None) is not None:
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})[
            "diagnostic_include_offline_counterfactuals"
        ] = args.diagnostic_include_offline_counterfactuals
    cfg["seed"] = args.seed
    cfg.setdefault("stage3_v2", {})["protocol_version"] = "v12d_context_teacher_probe"
    cfg.setdefault("online", {}).setdefault("stage3", {})["version"] = "v12d_context_teacher_probe"
    set_seed(args.seed)

    run_id = args.run_id or f"stage3_v12d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    ordered_train_iter = _limit_batches(
        _ordered_loader(train_ds, cfg["train"]["batch_size"], drop_last=True),
        args.max_train_batches,
    )
    model = _build_model(cfg, series.shape[1], target_dim).to(device)
    assert_deterministic_eval_model(model, context="Stage III V12d context teacher probe")

    offline_rows = _train_base_model(cfg, model, train_iter, val_iter, device)
    train_eval_iter = _limit_batches(DataLoader(train_ds, batch_size=cfg["train"]["batch_size"]), args.max_eval_batches)
    train_mse, train_mae = evaluate_factor(train_eval_iter, model, device)
    val_mse, val_mae = evaluate_factor(val_iter, model, device)
    test_mse, test_mae = evaluate_factor(test_iter, model, device)
    torch.save(model.state_dict(), ckpt_dir / "base_model.pt")
    _append_csv(out_dir / "offline_metrics.csv", offline_rows)

    aligned_model, aligned_rows = _train_aligned_model(cfg, ordered_train_iter, val_iter, series.shape[1], device)
    torch.save(aligned_model.state_dict(), ckpt_dir / "aligned_representation.pt")
    _append_csv(out_dir / "aligned_representation_metrics.csv", aligned_rows)

    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    if batch_size != 1:
        raise ValueError("V12d context teacher probe requires stage3_v2.online_batch_size=1")
    residual_canonicalizer = _fit_residual_canonicalizer(
        cfg,
        train_ds,
        batch_size,
        device,
        aligned_model,
        base_model=model,
        max_batches=args.max_generator_batches,
    )
    residual_canonicalizer_ckpt = None
    if residual_canonicalizer is not None:
        residual_canonicalizer_ckpt = ckpt_dir / "residual_drift_canonicalizer.pt"
        torch.save(residual_canonicalizer.state_dict(), residual_canonicalizer_ckpt)
    alignment_rows = []
    for split_name, split_ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        alignment_rows.extend(
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": args.seed,
                **row,
            }
            for row in _alignment_diagnostic_rows(
                dataset=split_ds,
                split=split_name,
                batch_size=batch_size,
                device=device,
                aligned_model=aligned_model,
                max_batches=args.max_generator_batches if split_name == "train" else args.max_eval_batches,
                residual_canonicalizer=residual_canonicalizer,
            )
        )
    _append_csv(out_dir / "alignment_diagnostics.csv", alignment_rows)

    delta_standardizer = _fit_delta_standardizer(
        cfg,
        train_ds,
        batch_size,
        device,
        aligned_model,
        max_batches=args.max_generator_batches,
        residual_canonicalizer=residual_canonicalizer,
    )
    delta_standardizer_ckpt = None
    if delta_standardizer is not None:
        delta_standardizer_ckpt = ckpt_dir / "delta_standardizer.pt"
        torch.save(delta_standardizer.state_dict(), delta_standardizer_ckpt)

    train_raw, train_y_base, train_y = _collect_split_tensors(
        cfg=cfg,
        dataset=train_ds,
        batch_size=batch_size,
        device=device,
        aligned_model=aligned_model,
        model=model,
        max_batches=args.max_generator_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )
    val_raw, val_y_base, val_y = _collect_split_tensors(
        cfg=cfg,
        dataset=val_ds,
        batch_size=batch_size,
        device=device,
        aligned_model=aligned_model,
        model=model,
        max_batches=args.max_eval_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )
    test_raw, test_y_base, test_y = _collect_split_tensors(
        cfg=cfg,
        dataset=test_ds,
        batch_size=batch_size,
        device=device,
        aligned_model=aligned_model,
        model=model,
        max_batches=args.max_eval_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )

    teacher_cfg = cfg.get("stage3_v2", {}).get("residual_teacher", {})
    history_cfg = cfg.get("stage3_v2", {}).get("v12d_history", {})
    history_builder = _build_history_builder(
        output_shape=tuple(train_y.shape[1:]),
        history_cfg=history_cfg,
    ).fit(y_base_train=train_y_base, y_train=train_y, split="train")
    history_lags = tuple(int(value) for value in getattr(history_builder, "lags"))
    feedback_delay = int(getattr(history_builder, "label_delay", getattr(history_builder, "feedback_delay", 0)))
    signal_builder = Stage3TeacherSignalBuilderV12c(
        drift_dim=train_raw.shape[1],
        mode=str(teacher_cfg.get("signal_mode", cfg["online"]["stage3"].get("signal_mode", "delta_vector"))),
        normalize=bool(teacher_cfg.get("normalize_vector", cfg["online"]["stage3"].get("normalize_vector", True))),
        robust_scale_quantile=float(teacher_cfg.get("robust_scale_quantile", 0.75)),
        eps=float(teacher_cfg.get("signal_eps", 1e-6)),
        polarity=_stage3_drift_polarity(cfg),
        random_seed=int(teacher_cfg.get("counterfactual_seed", 1729)),
    ).fit(train_raw, split="train")
    train_signal = signal_builder.transform(train_raw, counterfactual_mode="real").signal
    train_history = history_builder.transform(y_base=train_y_base, y_true=train_y).features
    feature_modes = args.feature_modes or cfg.get("stage3_v2", {}).get("v12d_feature_modes", list(DEFAULT_FEATURE_MODES))

    teachers: dict[str, TrainOnlyContextOOFResidualTeacherV12d] = {}
    teacher_diagnostic_rows: list[dict[str, Any]] = []
    for feature_mode in feature_modes:
        feature_slug = _feature_mode_slug(feature_mode)
        context = build_stage3_v12d_context_features(
            drift_signal=train_signal,
            history_features=train_history,
            mode=feature_mode,
        )
        teacher = _teacher_from_cfg(
            teacher_cfg=teacher_cfg,
            feature_mode=feature_mode,
            feature_dim=context.feature_dim,
            output_shape=tuple(train_y.shape[1:]),
            history_lags=history_lags,
            feedback_delay=feedback_delay,
            history_feature_dim=context.history_dim,
            drift_signal_dim=context.drift_dim,
        ).fit(
            drift_train=context.features,
            y_base_train=train_y_base,
            y_train=train_y,
            split="train",
            sample_ids=torch.arange(context.features.shape[0], dtype=torch.long),
        )
        teacher.stage3_signal_builder_v12c = signal_builder
        teacher.stage3_history_builder_v12d = history_builder
        teachers[feature_mode] = teacher
        save_stage3_v12d_teacher_checkpoint(
            path=ckpt_dir / f"teacher_v12d_{feature_slug}.pt",
            teacher=teacher,
            metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "feature_mode": feature_mode},
        )
        teacher_diagnostic_rows.append({"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "feature_mode": feature_mode, **teacher.diagnostic_summary()})
        _append_csv(out_dir / f"v12d_alpha_blocks_{feature_slug}.csv", teacher.alpha_block_diagnostic_rows())
        _append_csv(out_dir / f"v12d_ridge_{feature_slug}.csv", teacher.ridge_diagnostic_rows())
    _append_csv(out_dir / "residual_teacher_v12d_diagnostics.csv", teacher_diagnostic_rows)
    _append_csv(out_dir / "stage3_v12d_signal_contract_diagnostics.csv", [signal_builder.diagnostic_summary()])
    _append_csv(out_dir / "stage3_v12d_history_contract_diagnostics.csv", [history_builder.diagnostic_summary()])

    modes = args.modes or cfg.get("stage3_v2", {}).get("sensitivity_modes") or list(DEFAULT_MODES)
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
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
    )
    validation_summary_rows = [
        {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "validation", **row}
        for row in _summary_rows_for_feature_modes(cfg, validation_rows)
    ]
    _append_csv(out_dir / "teacher_only_validation_summary.csv", validation_rows)
    _append_csv(out_dir / "teacher_only_validation_diagnostic_summary.csv", validation_summary_rows)
    _append_csv(
        out_dir / "teacher_only_validation_partial_gain_vs_history.csv",
        [_partial_gain_row(split="validation", summary_rows=validation_summary_rows, run_id=run_id, dataset_label=dataset_label, seed=args.seed)],
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
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            window_count=validation_windows,
        )
        _append_csv(out_dir / "teacher_only_rolling_validation_summary.csv", rolling_rows)
        _append_csv(out_dir / "teacher_only_rolling_validation_window_summary.csv", rolling_window_rows)
        _append_csv(out_dir / "teacher_only_rolling_validation_aggregate.csv", rolling_aggregate_rows)
        _append_csv(out_dir / "teacher_only_rolling_partial_gain_vs_history.csv", rolling_partial_rows)

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
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
    )
    test_summary_rows = [
        {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "test", **row}
        for row in _summary_rows_for_feature_modes(cfg, test_rows)
    ]
    _append_csv(out_dir / "teacher_only_test_summary.csv", test_rows)
    _append_csv(out_dir / "teacher_only_test_diagnostic_summary.csv", test_summary_rows)
    _append_csv(
        out_dir / "teacher_only_test_partial_gain_vs_history.csv",
        [_partial_gain_row(split="test", summary_rows=test_summary_rows, run_id=run_id, dataset_label=dataset_label, seed=args.seed)],
    )

    abort_gate = False
    if rolling_aggregate_rows:
        by_feature = {str(row["feature_mode"]): row for row in rolling_aggregate_rows}
        drift_plus_history_gate = by_feature.get("drift_plus_history", {})
        abort_gate = not bool(drift_plus_history_gate.get("teacher_diagnostic_gate_pass", False))
    if bool(teacher_cfg.get("abort_if_teacher_diagnostic_fails", True)) and abort_gate:
        (out_dir / "teacher_only_abort.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "reason": "v12d_drift_plus_history_teacher_gate_failed",
                    "teacher_only_rolling_validation_aggregate": rolling_aggregate_rows,
                    "partial_gain_vs_history": rolling_partial_rows,
                    "next_step": "Do not train a generator until drift+history beats history-only under prospective gates.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        (out_dir / "teacher_only_complete.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "reason": "v12d_teacher_only_probe_completed",
                    "teacher_only_rolling_validation_aggregate": rolling_aggregate_rows,
                    "partial_gain_vs_history": rolling_partial_rows,
                    "next_step": "Review V12d partial gains before deciding whether generator training is scientifically justified.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    manifest = {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": args.seed,
        "config": args.config,
        "config_sha256": _sha256_file(args.config),
        **resolved_config_snapshot,
        "data_path": cfg["data"]["path"],
        "data_sha256": _sha256_file(cfg["data"]["path"]) if cfg["data"]["source"] != "synthetic" else "synthetic",
        "output_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "base_model_checkpoint_sha256": _sha256_file(ckpt_dir / "base_model.pt"),
        "aligned_representation_checkpoint_sha256": _sha256_file(ckpt_dir / "aligned_representation.pt"),
        "residual_canonicalizer_checkpoint_sha256": (
            _sha256_file(residual_canonicalizer_ckpt) if residual_canonicalizer_ckpt is not None else ""
        ),
        "delta_standardizer_checkpoint_sha256": (
            _sha256_file(delta_standardizer_ckpt) if delta_standardizer_ckpt is not None else ""
        ),
        "split_info": split_info,
        "offline_train_mse": train_mse,
        "offline_train_mae": train_mae,
        "offline_val_mse": val_mse,
        "offline_val_mae": val_mae,
        "offline_test_mse": test_mse,
        "offline_test_mae": test_mae,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_status_short": _git_status_short(),
        "stage3_protocol_version": cfg.get("stage3_v2", {}).get("protocol_version", "v12d_context_teacher_probe"),
        "stage3_drift_source": _stage3_drift_source(cfg),
        "stage3_drift_polarity": _stage3_drift_polarity(cfg),
        "stage3_signal_mode": cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
        "stage3_normalize_vector": bool(cfg["online"]["stage3"].get("normalize_vector", True)),
        "v12d_feature_modes": list(feature_modes),
        "v12d_history_lags": list(history_lags),
        "v12d_feedback_delay": feedback_delay,
        "history_protocol": getattr(history_builder, "protocol_version", _history_protocol(history_cfg)),
        "history_horizon_offsets": list(getattr(history_builder, "horizon_offsets", [])),
        "history_label_delay": int(getattr(history_builder, "label_delay", feedback_delay)),
        "v12d_label_feedback_status": "diagnostic_immediate_previous_label_available",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V12d teacher-only drift/history context probe.")
    parser.add_argument("--config", default="configs/stage3_v12d_context_teacher_probe.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--invariant-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=480)
    parser.add_argument("--max-eval-batches", type=int, default=480)
    parser.add_argument("--validation-windows", type=int, default=5)
    parser.add_argument("--drift-source", choices=["phi_gap", "delta_phi_v"], default=None)
    parser.add_argument("--drift-polarity", type=float, default=None)
    parser.add_argument("--signal-mode", choices=["level_vector", "delta_vector"], default=None)
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-modes", nargs="*", default=None, choices=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--history-protocol", default=None)
    parser.add_argument("--history-label-delay", type=int, default=None)
    parser.add_argument("--history-feedback-delay", type=int, default=None)
    parser.add_argument("--history-lags", nargs="*", type=int, default=None)
    parser.add_argument("--history-horizon-offsets", nargs="*", type=int, default=None)
    parser.add_argument("--diagnostic-include-offline-counterfactuals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out-root", default="logs/s3v12dctx")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v12dctx")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_dir = run_stage3_v12d_context_teacher_probe(args)
    print(out_dir)


if __name__ == "__main__":
    main()
