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

from models.stage3_residual_teacher_v14 import (  # noqa: E402
    TrainOnlyStabilityFluctuationResidualTeacherV14,
    build_stage3_v14_context_features,
    save_stage3_v14_teacher_checkpoint,
)
from train.run_stage3_v12c_sensitivity_audit import (  # noqa: E402
    _alignment_diagnostic_rows,
    _append_csv,
    _apply_dataset_override,
    _build_dataloaders,
    _build_model,
    _extract_prediction,
    _git_status_short,
    _limit_batches,
    _load_series_and_features,
    _ordered_loader,
    _prepare_fresh_run_dirs,
    _sha256_file,
    _stage3_drift_polarity,
    _stage3_drift_source,
    _teacher_diagnostic_aggregate,
    _train_aligned_model,
    _train_base_model,
    _write_resolved_config_snapshot,
    evaluate_factor,
    load_yaml,
    set_seed,
)
from train.run_stage3_v14_stability_teacher_probe import (  # noqa: E402
    CONDITIONAL_FLUCTUATION_MODES,
    DEFAULT_MODES,
    _build_history_builder,
    _collect_split_tensors,
    _raw_previous_tail,
    _residual_previous_tail,
    _teacher_from_cfg,
    _validate_v14_gate_modes,
)
from train.stage3_eval_policy import assert_deterministic_eval_model  # noqa: E402
from train.stage3_v12e_conditional_permutation import (  # noqa: E402
    build_history_norm_strata,
    conditional_permute_drift,
    conditional_random_matched_norm,
)
from train.stage3_v13_signal_contract import Stage3SignalBuilderV13  # noqa: E402
from train.stage3_v9_teacher_diagnostics import (  # noqa: E402
    compute_teacher_correction_metrics,
    summarize_teacher_mode_rows,
)


PRIMARY_V15_TARGETS = ("raw_residual_vector", "residual_change_vector")
SECONDARY_V15_TARGETS = ("residual_sign",)
V15_TARGETS = PRIMARY_V15_TARGETS + SECONDARY_V15_TARGETS
DEFAULT_V15_FEATURE_MODES = (
    "history_only",
    "fluctuation_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
)
V15_FEATURE_SLUGS = {
    "history_only": "h",
    "fluctuation_only": "f",
    "fluctuation_plus_history": "fh",
    "level_offset_only": "l",
    "level_offset_plus_history": "lh",
}
V15_TARGET_SLUGS = {
    "raw_residual_vector": "raw",
    "residual_change_vector": "dres",
    "residual_sign": "sign",
}
PROTOCOL_VERSION = "v15_residual_target_predictability_probe"


def _as_residual_matrix(*, y_base: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_base_2d = y_base.detach().float().reshape(y_base.shape[0], -1)
    y_true_2d = y_true.detach().float().reshape(y_true.shape[0], -1)
    if y_base_2d.shape != y_true_2d.shape:
        raise ValueError("y_base and y_true must have the same flattened shape")
    return y_true_2d - y_base_2d


def build_v15_target_tensor(
    target_name: str,
    *,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_residual_tail: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = _as_residual_matrix(y_base=y_base, y_true=y_true)
    if target_name == "raw_residual_vector":
        return residual.detach().float()
    if target_name == "residual_change_vector":
        if previous_residual_tail is None or previous_residual_tail.numel() == 0:
            previous = torch.cat([residual[:1], residual[:-1]], dim=0)
        else:
            tail = previous_residual_tail.detach().float().reshape(-1, residual.shape[1]).to(
                device=residual.device,
                dtype=residual.dtype,
            )
            if tail.shape[1] != residual.shape[1]:
                raise ValueError("previous_residual_tail feature dimension must match residual dimension")
            previous = torch.cat([tail[-1:], residual[:-1]], dim=0)
        return (residual - previous).detach().float()
    if target_name == "residual_sign":
        return torch.sign(residual).detach().float()
    raise ValueError(f"unsupported V15 target: {target_name}")


def v15_target_to_teacher_tensors(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_2d = target.detach().float().reshape(target.shape[0], -1)
    return torch.zeros_like(target_2d), target_2d


def _target_role(target_name: str) -> str:
    if target_name in PRIMARY_V15_TARGETS:
        return "primary"
    if target_name in SECONDARY_V15_TARGETS:
        return "secondary"
    raise ValueError(f"unsupported V15 target: {target_name}")


def _feature_slug(feature_mode: str) -> str:
    return V15_FEATURE_SLUGS.get(feature_mode, feature_mode.replace("_", "")[:12])


def _target_slug(target_name: str) -> str:
    return V15_TARGET_SLUGS.get(target_name, target_name.replace("_", "")[:12])


def _float_value(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def validate_v15_config_contract(cfg: dict[str, Any]) -> None:
    stage3_cfg = cfg.get("stage3_v2", {})
    online_cfg = cfg.get("online", {}).get("stage3", {})
    if str(stage3_cfg.get("protocol_version", "")) != PROTOCOL_VERSION:
        raise ValueError("V15 config requires stage3_v2.protocol_version=v15_residual_target_predictability_probe")
    if str(online_cfg.get("version", "")) != PROTOCOL_VERSION:
        raise ValueError("V15 config requires online.stage3.version=v15_residual_target_predictability_probe")
    if str(online_cfg.get("drift_source", "")) != "delta_phi_v":
        raise ValueError("V15 config requires online.stage3.drift_source=delta_phi_v")
    if str(online_cfg.get("signal_mode", "")) != "fluctuation_delta":
        raise ValueError("V15 config requires online.stage3.signal_mode=fluctuation_delta")
    if bool(online_cfg.get("canonicalize_delta", False)):
        raise ValueError("V15 main path must not use delta canonicalization")
    if bool(online_cfg.get("calibrate_delta", False)):
        raise ValueError("V15 main path must not use delta calibration")
    roots = (
        str(stage3_cfg.get("log_root", "")),
        str(stage3_cfg.get("checkpoint_root", "")),
        str(stage3_cfg.get("results_root", "")),
    )
    if not all("s3v15target" in root for root in roots):
        raise ValueError("V15 config must use s3v15target output roots")


def v15_directional_stop_reasons(target_gate_rows: list[dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    primary_rows = [row for row in target_gate_rows if str(row.get("target_role", "")) == "primary"]
    passed = [row for row in primary_rows if bool(row.get("primary_gate_pass", False))]
    if not passed:
        reasons.extend(["do_not_train_generator", "no_primary_target_passed"])
    for row in primary_rows:
        if str(row.get("only_auxiliary_level_offset_passed", "False")) == "True":
            reasons.append("only_auxiliary_level_offset_passed")
    return sorted(set(reasons))


def _summary_rows_for_target_feature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    targets = sorted({str(row["target_name"]) for row in rows})
    for target_name in targets:
        target_rows = [row for row in rows if str(row["target_name"]) == target_name]
        feature_modes = sorted({str(row["feature_mode"]) for row in target_rows})
        for feature_mode in feature_modes:
            mode_rows = [row for row in target_rows if str(row["feature_mode"]) == feature_mode]
            first = mode_rows[0]
            summaries.append(
                {
                    "target_name": target_name,
                    "target_role": first["target_role"],
                    "feature_mode": feature_mode,
                    **summarize_teacher_mode_rows(mode_rows),
                }
            )
    return summaries


def _partial_gain_rows(
    *,
    split: str,
    summary_rows: list[dict[str, Any]],
    run_id: str,
    dataset_label: str,
    seed: int,
    window_id: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = sorted({str(row["target_name"]) for row in summary_rows})
    for target_name in targets:
        target_rows = [row for row in summary_rows if str(row["target_name"]) == target_name]
        by_feature = {str(row["feature_mode"]): row for row in target_rows}
        history = by_feature.get("history_only", {})
        both = by_feature.get("fluctuation_plus_history", {})
        h_mse = _float_value(history, "teacher_real_mse_gain")
        fh_mse = _float_value(both, "teacher_real_mse_gain")
        h_mae = _float_value(history, "teacher_real_mae_gain")
        fh_mae = _float_value(both, "teacher_real_mae_gain")
        h_sign = _float_value(history, "teacher_real_sign_agreement_rate")
        fh_sign = _float_value(both, "teacher_real_sign_agreement_rate")
        row: dict[str, Any] = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": seed,
            "split": split,
            "target_name": target_name,
            "target_role": _target_role(target_name),
            "history_only_real_mse_gain": h_mse,
            "fluctuation_plus_history_real_mse_gain": fh_mse,
            "partial_mse_gain_vs_history": fh_mse - h_mse if h_mse == h_mse and fh_mse == fh_mse else float("nan"),
            "history_only_real_mae_gain": h_mae,
            "fluctuation_plus_history_real_mae_gain": fh_mae,
            "partial_mae_gain_vs_history": fh_mae - h_mae if h_mae == h_mae and fh_mae == fh_mae else float("nan"),
            "history_only_sign_agreement": h_sign,
            "fluctuation_plus_history_sign_agreement": fh_sign,
            "partial_sign_agreement_vs_history": fh_sign - h_sign if h_sign == h_sign and fh_sign == fh_sign else float("nan"),
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


def _rows_for_split(
    *,
    cfg: dict[str, Any],
    split: str,
    raw_drift: torch.Tensor,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_raw_tail: torch.Tensor | None,
    previous_residual_tail: torch.Tensor | None,
    teachers: dict[tuple[str, str], TrainOnlyStabilityFluctuationResidualTeacherV14],
    signal_builder: Stage3SignalBuilderV13,
    history_builder,
    modes: list[str],
    targets: list[str],
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
    target_tensors = {
        target_name: build_v15_target_tensor(
            target_name,
            y_base=y_base,
            y_true=y_true,
            previous_residual_tail=previous_residual_tail,
        )
        for target_name in targets
    }
    rows: list[dict[str, Any]] = []
    for mode in modes:
        signal_mode = "real" if mode in CONDITIONAL_FLUCTUATION_MODES else mode
        signal_result = signal_builder.transform(
            raw_drift,
            previous_level_tail=previous_raw_tail,
            counterfactual_mode=signal_mode,
        )
        fluctuation_signal = signal_result.signal
        conditional_protocol = ""
        if mode == "conditional_shuffled":
            fluctuation_signal = conditional_permute_drift(
                drift_signal=fluctuation_signal,
                strata=strata,
                seed=conditional_seed,
            )
            conditional_protocol = "history_norm_stratified_row_permutation"
        elif mode == "conditional_random_matched_norm":
            fluctuation_signal = conditional_random_matched_norm(
                drift_signal=fluctuation_signal,
                strata=strata,
                seed=conditional_seed,
            )
            conditional_protocol = "history_norm_stratified_random_matched_norm"
        for target_name, target_tensor in target_tensors.items():
            y_base_target, y_true_target = v15_target_to_teacher_tensors(target_tensor)
            for (teacher_target, feature_mode), teacher in teachers.items():
                if teacher_target != target_name:
                    continue
                context = build_stage3_v14_context_features(
                    fluctuation_signal=fluctuation_signal,
                    history_features=history_result.features,
                    level_offset=signal_result.level_offset,
                    mode=feature_mode,
                )
                with torch.no_grad():
                    correction = teacher.predict_correction(context.features).detach().cpu()
                metrics = compute_teacher_correction_metrics(
                    y_base=y_base_target,
                    y_true=y_true_target,
                    correction=correction,
                )
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": seed,
                    "split": split,
                    "target_name": target_name,
                    "target_role": _target_role(target_name),
                    "feature_mode": feature_mode,
                    "mode": mode,
                    "intervention_online_causal": mode
                    in {"real", "zero", "sign_flipped", "amplified_2x", "amplified_5x"},
                    "intervention_protocol": "streaming_causal"
                    if mode in {"real", "zero", "sign_flipped", "amplified_2x", "amplified_5x"}
                    else "offline_diagnostic",
                    "counterfactual_intervention_space": cfg.get("stage3_v2", {}).get(
                        "counterfactual_intervention_space",
                        "generator_input",
                    ),
                    "teacher_signal_contract": signal_builder.protocol_version,
                    "teacher_history_contract": history_builder.protocol_version,
                    "teacher_conditional_permutation_protocol": conditional_protocol,
                    "teacher_conditional_permutation_bin_count": conditional_bin_count if conditional_protocol else "",
                    "teacher_context_protocol": TrainOnlyStabilityFluctuationResidualTeacherV14.protocol_version,
                    "teacher_fit_split": teacher.fit_split,
                    "teacher_fit_sample_count": teacher.fit_sample_count,
                    "teacher_train_residual_r2": teacher.teacher_residual_r2,
                    "teacher_shrinkage_alpha": teacher.shrinkage_alpha,
                    "teacher_alpha_source": teacher.selected_alpha_source,
                    "teacher_reliability_gate_active": teacher.reliability_gate_active,
                    "teacher_context_feature_dim": context.feature_dim,
                    "teacher_context_fluctuation_dim": context.fluctuation_dim,
                    "teacher_context_history_dim": context.history_dim,
                    "teacher_context_level_offset_dim": context.level_offset_dim,
                    "teacher_fluctuation_norm_mean": float(torch.linalg.vector_norm(fluctuation_signal, dim=1).mean().item()),
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


def _rolling_rows_for_split(
    *,
    cfg: dict[str, Any],
    raw_drift: torch.Tensor,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_raw_tail: torch.Tensor | None,
    previous_residual_tail: torch.Tensor | None,
    teachers: dict[tuple[str, str], TrainOnlyStabilityFluctuationResidualTeacherV14],
    signal_builder: Stage3SignalBuilderV13,
    history_builder,
    modes: list[str],
    targets: list[str],
    dataset_label: str,
    run_id: str,
    seed: int,
    window_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    residual_tail_len = max(int(getattr(history_builder, "required_tail_length", 1)), 1)
    for window_id, window_slice in enumerate(_window_slices(raw_drift.shape[0], window_count)):
        start = int(window_slice.start or 0)
        stop = int(window_slice.stop or raw_drift.shape[0])
        window_previous_raw = _raw_previous_tail(raw_drift, previous_raw_tail, start)
        window_previous_residual = _residual_previous_tail(
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
            previous_raw_tail=window_previous_raw,
            previous_residual_tail=window_previous_residual,
            teachers=teachers,
            signal_builder=signal_builder,
            history_builder=history_builder,
            modes=modes,
            targets=targets,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            window_id=window_id,
            window_start=start,
            window_end=stop,
        )
        all_rows.extend(rows)
        window_summaries = [
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": seed,
                "split": f"validation_window_{window_id}",
                "validation_window_id": window_id,
                "validation_window_start_sample": start,
                "validation_window_end_sample": stop,
                **row,
            }
            for row in _summary_rows_for_target_feature(rows)
        ]
        summary_rows.extend(window_summaries)
        partial_rows.extend(
            _partial_gain_rows(
                split=f"validation_window_{window_id}",
                summary_rows=window_summaries,
                run_id=run_id,
                dataset_label=dataset_label,
                seed=seed,
                window_id=window_id,
                window_start=start,
                window_end=stop,
            )
        )
    for target_name in targets:
        for feature_mode in sorted({key[1] for key in teachers if key[0] == target_name}):
            matching = [
                row
                for row in summary_rows
                if str(row["target_name"]) == target_name and str(row["feature_mode"]) == feature_mode
            ]
            if not matching:
                continue
            aggregate = _teacher_diagnostic_aggregate(
                cfg=cfg,
                run_id=run_id,
                dataset_label=dataset_label,
                seed=seed,
                split=f"validation_rolling_{target_name}_{feature_mode}",
                window_rows=matching,
            )
            aggregate["target_name"] = target_name
            aggregate["target_role"] = _target_role(target_name)
            aggregate["feature_mode"] = feature_mode
            aggregate_rows.append(aggregate)
    return all_rows, summary_rows, aggregate_rows, partial_rows


def _v15_target_gate_rows(
    *,
    cfg: dict[str, Any],
    targets: list[str],
    rolling_aggregate_rows: list[dict[str, Any]],
    rolling_partial_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    v15_cfg = cfg.get("stage3_v2", {}).get("v15_gate", {})
    min_positive_windows = int(v15_cfg.get("min_positive_windows", 4))
    min_partial_mse = float(v15_cfg.get("min_partial_mse_gain_vs_history", 0.0))
    min_partial_mae = float(v15_cfg.get("min_partial_mae_gain_vs_history", 0.0))
    max_harmful_mse = float(v15_cfg.get("max_harmful_partial_mse_gain", -0.05))
    max_harmful_mae = float(v15_cfg.get("max_harmful_partial_mae_gain", -0.05))
    min_cf_margin = float(
        cfg.get("stage3_v2", {})
        .get("residual_teacher", {})
        .get("diagnostic_counterfactual_margin", cfg.get("stage3_v2", {}).get("claim_gate", {}).get("counterfactual_margin", 0.0))
    )
    aggregate_by_key = {
        (str(row.get("target_name", "")), str(row.get("feature_mode", ""))): row for row in rolling_aggregate_rows
    }
    gate_rows: list[dict[str, Any]] = []
    for target_name in targets:
        partials = [row for row in rolling_partial_rows if str(row.get("target_name", "")) == target_name]
        positive_rows = [
            row
            for row in partials
            if _float_value(row, "partial_mse_gain_vs_history", -1.0) > min_partial_mse
            and _float_value(row, "partial_mae_gain_vs_history", -1.0) > min_partial_mae
        ]
        min_mse = min((_float_value(row, "partial_mse_gain_vs_history") for row in partials), default=float("nan"))
        min_mae = min((_float_value(row, "partial_mae_gain_vs_history") for row in partials), default=float("nan"))
        main_agg = aggregate_by_key.get((target_name, "fluctuation_plus_history"), {})
        main_margin = _float_value(main_agg, "teacher_worst_real_vs_best_counterfactual_mse_gain_margin", 0.0)
        counterfactual_pass = main_margin > min_cf_margin
        stability_pass = len(positive_rows) >= min_positive_windows
        harmful_pass = (min_mse == min_mse and min_mse >= max_harmful_mse) and (min_mae == min_mae and min_mae >= max_harmful_mae)
        primary_gate_pass = (
            _target_role(target_name) == "primary"
            and stability_pass
            and harmful_pass
            and counterfactual_pass
        )
        level_agg = aggregate_by_key.get((target_name, "level_offset_plus_history"), {})
        level_mse = _float_value(level_agg, "teacher_worst_real_mse_gain", 0.0)
        main_mse = _float_value(main_agg, "teacher_worst_real_mse_gain", 0.0)
        only_auxiliary_level_offset_passed = (
            level_mse > 0.0 and (main_mse != main_mse or main_mse <= 0.0) and not primary_gate_pass
        )
        gate_rows.append(
            {
                "target_name": target_name,
                "target_role": _target_role(target_name),
                "primary_gate_pass": primary_gate_pass,
                "positive_partial_window_count": len(positive_rows),
                "required_positive_partial_window_count": min_positive_windows,
                "window_count": len(partials),
                "min_partial_mse_gain_vs_history": min_mse,
                "min_partial_mae_gain_vs_history": min_mae,
                "max_harmful_partial_mse_gain_allowed": max_harmful_mse,
                "max_harmful_partial_mae_gain_allowed": max_harmful_mae,
                "counterfactual_margin": main_margin,
                "counterfactual_margin_required": min_cf_margin,
                "counterfactual_pass": counterfactual_pass,
                "only_auxiliary_level_offset_passed": only_auxiliary_level_offset_passed,
            }
        )
    return gate_rows


def _resolve_v15_targets(cfg: dict[str, Any], args_targets: list[str] | None) -> list[str]:
    if args_targets:
        targets = [str(value) for value in args_targets]
    else:
        target_cfg = cfg.get("stage3_v2", {}).get("v15_targets", {})
        targets = [str(value) for value in target_cfg.get("primary", PRIMARY_V15_TARGETS)]
        targets.extend(str(value) for value in target_cfg.get("secondary", SECONDARY_V15_TARGETS))
    invalid = [target for target in targets if target not in V15_TARGETS]
    if invalid:
        raise ValueError(f"unsupported V15 targets: {', '.join(invalid)}")
    return list(dict.fromkeys(targets))


def run_stage3_v15_residual_target_predictability_probe(args: argparse.Namespace) -> Path:
    cfg = _apply_dataset_override(load_yaml(args.config), args.dataset)
    cfg.setdefault("stage3_v2", {})["protocol_version"] = PROTOCOL_VERSION
    cfg.setdefault("online", {}).setdefault("stage3", {})["version"] = PROTOCOL_VERSION
    cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = "delta_phi_v"
    cfg.setdefault("online", {}).setdefault("stage3", {})["signal_mode"] = "fluctuation_delta"
    cfg.setdefault("online", {}).setdefault("stage3", {})["canonicalize_delta"] = False
    cfg.setdefault("online", {}).setdefault("stage3", {})["calibrate_delta"] = False
    cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})[
        "diagnostic_include_offline_counterfactuals"
    ] = True
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.invariant_epochs is not None:
        cfg["invariant"]["epochs"] = args.invariant_epochs
    if args.drift_source is not None:
        if args.drift_source != "delta_phi_v":
            raise ValueError("V15 requires --drift-source delta_phi_v")
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = args.drift_source
    if args.drift_polarity is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_polarity"] = args.drift_polarity
    if args.signal_mode is not None:
        if args.signal_mode != "fluctuation_delta":
            raise ValueError("V15 requires --signal-mode fluctuation_delta")
        cfg.setdefault("online", {}).setdefault("stage3", {})["signal_mode"] = args.signal_mode
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})["signal_mode"] = args.signal_mode
    if args.normalize_vector is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["normalize_vector"] = args.normalize_vector
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})["normalize_vector"] = args.normalize_vector
    history_override = cfg.setdefault("stage3_v2", {}).setdefault("v12d_history", {})
    if getattr(args, "history_protocol", None) is not None:
        history_override["protocol"] = args.history_protocol
    if getattr(args, "history_label_delay", None) is not None:
        history_override["label_delay"] = args.history_label_delay
    if getattr(args, "history_feedback_delay", None) is not None:
        history_override["feedback_delay"] = args.history_feedback_delay
    if getattr(args, "history_lags", None) is not None:
        history_override["lags"] = args.history_lags
    if getattr(args, "history_horizon_offsets", None) is not None:
        history_override["horizon_offsets"] = args.history_horizon_offsets
    if getattr(args, "diagnostic_include_offline_counterfactuals", None) is not None:
        if args.diagnostic_include_offline_counterfactuals is False:
            raise ValueError("V15 requires offline and conditional counterfactuals to participate in the diagnostic")
        cfg.setdefault("stage3_v2", {}).setdefault("residual_teacher", {})[
            "diagnostic_include_offline_counterfactuals"
        ] = args.diagnostic_include_offline_counterfactuals
    cfg["seed"] = args.seed
    validate_v15_config_contract(cfg)
    set_seed(args.seed)

    run_id = args.run_id or f"stage3_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    assert_deterministic_eval_model(model, context="Stage III V15 residual-target predictability probe")

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
        raise ValueError("V15 residual-target diagnostic requires stage3_v2.online_batch_size=1")
    residual_canonicalizer = None
    delta_standardizer = None
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
            feature_slug = _feature_slug(feature_mode)
            target_slug = _target_slug(target_name)
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
            teacher.stage3_signal_builder_v15 = signal_builder
            teacher.stage3_history_builder_v15 = history_builder
            teachers[(target_name, feature_mode)] = teacher
            save_stage3_v14_teacher_checkpoint(
                path=ckpt_dir / f"teacher_v15_{target_slug}_{feature_slug}.pt",
                teacher=teacher,
                metadata={
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "target_name": target_name,
                    "target_role": _target_role(target_name),
                    "feature_mode": feature_mode,
                    "v15_protocol_version": PROTOCOL_VERSION,
                },
            )
            teacher_diagnostic_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "target_name": target_name,
                    "target_role": _target_role(target_name),
                    "feature_mode": feature_mode,
                    **teacher.diagnostic_summary(),
                }
            )
            _append_csv(out_dir / f"v15_alpha_blocks_{target_slug}_{feature_slug}.csv", teacher.alpha_block_diagnostic_rows())
            _append_csv(out_dir / f"v15_ridge_{target_slug}_{feature_slug}.csv", teacher.ridge_diagnostic_rows())
    _append_csv(out_dir / "residual_teacher_v15_diagnostics.csv", teacher_diagnostic_rows)
    _append_csv(out_dir / "stage3_v15_signal_contract_diagnostics.csv", [signal_builder.diagnostic_summary()])
    _append_csv(out_dir / "stage3_v15_history_contract_diagnostics.csv", [history_builder.diagnostic_summary()])

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
    _append_csv(out_dir / "teacher_only_validation_summary.csv", validation_rows)
    _append_csv(out_dir / "teacher_only_validation_diagnostic_summary.csv", validation_summary_rows)
    _append_csv(
        out_dir / "teacher_only_validation_partial_gain_vs_history.csv",
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
        targets=targets,
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
    )
    test_summary_rows = [
        {"run_id": run_id, "dataset": dataset_label, "seed": args.seed, "split": "test", **row}
        for row in _summary_rows_for_target_feature(test_rows)
    ]
    _append_csv(out_dir / "teacher_only_test_summary.csv", test_rows)
    _append_csv(out_dir / "teacher_only_test_diagnostic_summary.csv", test_summary_rows)
    _append_csv(
        out_dir / "teacher_only_test_partial_gain_vs_history.csv",
        _partial_gain_rows(
            split="test",
            summary_rows=test_summary_rows,
            run_id=run_id,
            dataset_label=dataset_label,
            seed=args.seed,
        ),
    )

    target_gate_rows = _v15_target_gate_rows(
        cfg=cfg,
        targets=targets,
        rolling_aggregate_rows=rolling_aggregate_rows,
        rolling_partial_rows=rolling_partial_rows,
    )
    _append_csv(out_dir / "teacher_only_v15_target_gate.csv", target_gate_rows)
    stop_reasons = v15_directional_stop_reasons(target_gate_rows)
    payload = {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": args.seed,
        "directional_stop_reasons": stop_reasons,
        "target_gate_rows": target_gate_rows,
        "teacher_only_rolling_validation_aggregate": rolling_aggregate_rows,
        "partial_gain_vs_history": rolling_partial_rows,
        "generator_training_allowed": not stop_reasons,
        "next_step": (
            "Do not train a generator or redesign representations unless a primary V15 target passes."
            if stop_reasons
            else "Review primary target evidence before considering representation retraining; generator remains a separate later gate."
        ),
    }
    if stop_reasons:
        payload["reason"] = "v15_no_primary_residual_target_passed"
        (out_dir / "teacher_only_abort.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        payload["reason"] = "v15_primary_residual_target_predictability_gate_passed"
        (out_dir / "teacher_only_complete.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
        "stage3_protocol_version": PROTOCOL_VERSION,
        "stage3_drift_source": _stage3_drift_source(cfg),
        "stage3_drift_polarity": _stage3_drift_polarity(cfg),
        "stage3_signal_mode": cfg["online"]["stage3"].get("signal_mode", "fluctuation_delta"),
        "stage3_normalize_vector": bool(cfg["online"]["stage3"].get("normalize_vector", True)),
        "v15_targets": list(targets),
        "v15_primary_targets": list(PRIMARY_V15_TARGETS),
        "v15_secondary_targets": list(SECONDARY_V15_TARGETS),
        "v15_feature_modes": list(feature_modes),
        "v15_history_lags": list(history_lags),
        "v15_feedback_delay": feedback_delay,
        "history_protocol": getattr(history_builder, "protocol_version", ""),
        "history_horizon_offsets": list(getattr(history_builder, "horizon_offsets", [])),
        "history_label_delay": int(getattr(history_builder, "label_delay", feedback_delay)),
        "generator_training_allowed": not stop_reasons,
        "generator_artifact_count": len(list(ckpt_dir.glob("generator*.pt"))),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V15 residual-target predictability teacher-only diagnostic.")
    parser.add_argument("--config", default="configs/stage3_v15_residual_target_predictability_probe.yaml")
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
    parser.add_argument("--signal-mode", choices=["fluctuation_delta"], default=None)
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-modes", nargs="*", default=None, choices=list(DEFAULT_V15_FEATURE_MODES))
    parser.add_argument("--targets", nargs="*", default=None, choices=list(V15_TARGETS))
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--history-protocol", default=None)
    parser.add_argument("--history-label-delay", type=int, default=None)
    parser.add_argument("--history-feedback-delay", type=int, default=None)
    parser.add_argument("--history-lags", nargs="*", type=int, default=None)
    parser.add_argument("--history-horizon-offsets", nargs="*", type=int, default=None)
    parser.add_argument("--diagnostic-include-offline-counterfactuals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out-root", default="logs/s3v15target")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v15target")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_dir = run_stage3_v15_residual_target_predictability_probe(args)
    print(out_dir)


if __name__ == "__main__":
    main()
