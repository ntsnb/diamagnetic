from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "teacher_only_v29_target_gate.csv",
    "teacher_only_validation_diagnostic_summary_v29.csv",
    "teacher_only_validation_partial_gain_vs_history_v29.csv",
    "teacher_only_rolling_validation_window_summary_v29.csv",
    "teacher_only_rolling_partial_gain_vs_history_v29.csv",
)

BANNED_ONLINE_FEATURE_NAME_TOKENS = (
    "gain",
    "counterfactual",
    "label",
    "partial",
    "teacher_real",
    "outcome",
    "best",
    "target_y",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _coerce_csv_scalar(value: object) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer() and "." not in text and "e" not in lowered:
        return int(numeric)
    return numeric


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.12g}"


def _finite_float(value: object, default: float = -float("inf")) -> float:
    parsed = _to_float(value, default)
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return default
    return parsed


def _is_finite_value(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first(rows: list[dict[str, str]], **matches: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in matches.items()):
            return row
    return {}


def _target_rows(rows: list[dict[str, str]], *, target_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("target_name") == target_name]


def _read_first_optional_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _missing_required_files(run_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (run_dir / name).exists()]


def _stop_reasons(run_dir: Path) -> list[str]:
    payload = _read_json(run_dir / "teacher_only_abort.json")
    reasons = payload.get("stop_reasons", [])
    if isinstance(reasons, list):
        return [str(reason) for reason in reasons]
    return [str(reasons)] if reasons else []


def _discover_run_dirs(roots: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("teacher_only_v29_target_gate.csv"):
            discovered.append(path.parent)
    return discovered


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def summarize_run(
    run_dir: Path,
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    missing = _missing_required_files(run_dir)
    if missing:
        return None, {
            "run_dir": str(run_dir),
            "missing_files": ";".join(missing),
            "skip_reason": "missing_required_v29_teacher_files",
        }

    gate_rows = _read_csv(run_dir / "teacher_only_v29_target_gate.csv")
    validation_rows = _read_csv(run_dir / "teacher_only_validation_diagnostic_summary_v29.csv")
    validation_partial_rows = _read_csv(run_dir / "teacher_only_validation_partial_gain_vs_history_v29.csv")
    rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv")
    rolling_partial_rows = _read_csv(run_dir / "teacher_only_rolling_partial_gain_vs_history_v29.csv")

    gate = _first(gate_rows, target_name=target_name)
    if not gate:
        return None, {
            "run_dir": str(run_dir),
            "missing_files": "",
            "skip_reason": f"target_not_found:{target_name}",
        }

    target_role = str(gate.get("target_role", ""))
    validation = _first(validation_rows, target_name=target_name, feature_mode=feature_mode)
    validation_history = _first(validation_rows, target_name=target_name, feature_mode="history_only")
    validation_partial = _first(validation_partial_rows, target_name=target_name)
    rolling_target = _target_rows(rolling_rows, target_name=target_name)
    rolling_feature = [row for row in rolling_target if row.get("feature_mode") == feature_mode]
    rolling_partial_target = _target_rows(rolling_partial_rows, target_name=target_name)
    window0 = _first(
        rolling_rows,
        target_name=target_name,
        feature_mode=feature_mode,
        validation_window_id="0",
    )
    window0_history = _first(
        rolling_rows,
        target_name=target_name,
        feature_mode="history_only",
        validation_window_id="0",
    )
    window0_partial = _first(rolling_partial_rows, target_name=target_name, validation_window_id="0")

    stop_reasons = _stop_reasons(run_dir)
    target_gate_pass = _truthy(gate.get("primary_gate_pass"))
    any_primary_gate_pass = any(
        row.get("target_role") == "primary" and _truthy(row.get("primary_gate_pass")) for row in gate_rows
    )
    teacher_abort_written = (run_dir / "teacher_only_abort.json").exists()
    generator_unlocked = any_primary_gate_pass and not teacher_abort_written

    positive_partial_windows = _to_int(gate.get("positive_partial_window_count"))
    required_partial_windows = _to_int(gate.get("required_positive_partial_window_count"))
    min_partial_mse = _to_float(gate.get("min_partial_mse_gain_vs_history"))
    min_partial_mae = _to_float(gate.get("min_partial_mae_gain_vs_history"))
    counterfactual_margin = _to_float(gate.get("counterfactual_margin"))
    validation_partial_mse = _to_float(validation_partial.get("partial_mse_gain_vs_history"))
    validation_partial_mae = _to_float(validation_partial.get("partial_mae_gain_vs_history"))
    validation_partial_sign = _to_float(validation_partial.get("partial_sign_agreement_vs_history"))
    window0_mse = _to_float(window0.get("teacher_real_mse_gain"))
    window0_mae = _to_float(window0.get("teacher_real_mae_gain"))
    window0_partial_mse = _to_float(window0_partial.get("partial_mse_gain_vs_history"))
    window0_partial_mae = _to_float(window0_partial.get("partial_mae_gain_vs_history"))
    window0_counterfactual_margin = _to_float(window0.get("teacher_real_vs_best_counterfactual_mse_gain_margin"))

    worst_window_mse = min(
        (_to_float(row.get("teacher_real_mse_gain")) for row in rolling_feature),
        default=float("nan"),
    )
    worst_window_counterfactual_margin = min(
        (_to_float(row.get("teacher_real_vs_best_counterfactual_mse_gain_margin")) for row in rolling_feature),
        default=float("nan"),
    )
    worst_window_partial_mse = min(
        (_to_float(row.get("partial_mse_gain_vs_history")) for row in rolling_partial_target),
        default=float("nan"),
    )

    selector_metric_complete = all(
        _is_finite_value(value)
        for value in (
            min_partial_mse,
            min_partial_mae,
            counterfactual_margin,
            validation_partial_mse,
            validation_partial_mae,
            window0_mse,
            window0_partial_mse,
            window0_counterfactual_margin,
        )
    )

    if not selector_metric_complete:
        selection_label = "incomplete_diagnostic"
    elif target_gate_pass and generator_unlocked:
        selection_label = "generator_unlocked_candidate"
    elif target_gate_pass:
        selection_label = "teacher_gate_candidate"
    elif window0_mse >= 0.0 and counterfactual_margin >= -0.05:
        selection_label = "gate_near_diagnostic"
    else:
        selection_label = "diagnostic_only"

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": validation.get("run_id") or gate.get("run_id") or run_dir.parents[2].name,
        "dataset": validation.get("dataset") or run_dir.parents[1].name,
        "seed": validation.get("seed") or run_dir.name.replace("seed_", ""),
        "target_name": target_name,
        "target_role": target_role,
        "feature_mode": feature_mode,
        "selection_label": selection_label,
        "selector_metric_complete": selector_metric_complete,
        "target_gate_pass": target_gate_pass,
        "any_primary_gate_pass": any_primary_gate_pass,
        "teacher_abort_written": teacher_abort_written,
        "generator_unlocked": generator_unlocked,
        "stop_reasons": ";".join(stop_reasons),
        "positive_partial_window_count": positive_partial_windows,
        "required_positive_partial_window_count": required_partial_windows,
        "window_count": _to_int(gate.get("window_count")),
        "min_partial_mse_gain_vs_history": min_partial_mse,
        "min_partial_mae_gain_vs_history": min_partial_mae,
        "counterfactual_margin": counterfactual_margin,
        "counterfactual_pass": _truthy(gate.get("counterfactual_pass")),
        "validation_fh_mse_gain": _to_float(validation.get("teacher_real_mse_gain")),
        "validation_fh_mae_gain": _to_float(validation.get("teacher_real_mae_gain")),
        "validation_fh_sign_agreement": _to_float(validation.get("teacher_real_sign_agreement_rate")),
        "validation_history_mse_gain": _to_float(validation_history.get("teacher_real_mse_gain")),
        "validation_history_mae_gain": _to_float(validation_history.get("teacher_real_mae_gain")),
        "validation_partial_mse_gain_vs_history": validation_partial_mse,
        "validation_partial_mae_gain_vs_history": validation_partial_mae,
        "validation_partial_sign_agreement_vs_history": validation_partial_sign,
        "window0_fh_mse_gain": window0_mse,
        "window0_fh_mae_gain": window0_mae,
        "window0_fh_sign_agreement": _to_float(window0.get("teacher_real_sign_agreement_rate")),
        "window0_history_mse_gain": _to_float(window0_history.get("teacher_real_mse_gain")),
        "window0_history_mae_gain": _to_float(window0_history.get("teacher_real_mae_gain")),
        "window0_partial_mse_gain_vs_history": window0_partial_mse,
        "window0_partial_mae_gain_vs_history": window0_partial_mae,
        "window0_counterfactual_margin": window0_counterfactual_margin,
        "window0_best_counterfactual_mode": window0.get("teacher_best_counterfactual_mode", ""),
        "window0_real_is_best_mode": _truthy(window0.get("teacher_real_is_best_mode")),
        "worst_window_fh_mse_gain": worst_window_mse,
        "worst_window_counterfactual_margin": worst_window_counterfactual_margin,
        "worst_window_partial_mse_gain_vs_history": worst_window_partial_mse,
    }
    row["ranking_tuple"] = _ranking_tuple_text(row)
    return row, None


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row["target_gate_pass"]),
        bool(row["generator_unlocked"]),
        bool(row.get("selector_metric_complete")),
        _finite_float(row["window0_fh_mse_gain"]) >= 0.0,
        _finite_float(row["counterfactual_margin"]),
        _to_int(row["positive_partial_window_count"]),
        _finite_float(row["min_partial_mse_gain_vs_history"]),
        _finite_float(row["min_partial_mae_gain_vs_history"]),
        _finite_float(row["validation_partial_mse_gain_vs_history"]),
        _finite_float(row["validation_partial_mae_gain_vs_history"]),
        _finite_float(row["validation_fh_mse_gain"]),
    )


def _ranking_tuple_text(row: dict[str, Any]) -> str:
    key = _ranking_key(row)
    return (
        "gate={};unlock={};complete={};w0_nonharm={};cf={};pos_windows={};min_pmse={};"
        "min_pmae={};val_pmse={};val_pmae={};val_fh_mse={}"
    ).format(
        int(bool(key[0])),
        int(bool(key[1])),
        int(bool(key[2])),
        int(bool(key[3])),
        _format_float(float(key[4])),
        key[5],
        _format_float(float(key[6])),
        _format_float(float(key[7])),
        _format_float(float(key[8])),
        _format_float(float(key[9])),
        _format_float(float(key[10])),
    )


def select_runs(
    run_dirs: list[Path],
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run_dir in _dedupe_paths(run_dirs):
        row, skipped_row = summarize_run(run_dir, target_name=target_name, feature_mode=feature_mode)
        if row is not None:
            candidates.append(row)
        if skipped_row is not None:
            skipped.append(skipped_row)
    candidates.sort(key=_ranking_key, reverse=True)
    for idx, row in enumerate(candidates, start=1):
        row["strict_rank"] = idx
    return candidates, skipped


def _require_complete_selector_metrics(
    candidates: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    filtered_skipped = list(skipped)
    for row in candidates:
        if bool(row.get("selector_metric_complete", True)):
            complete.append(row)
            continue
        filtered_skipped.append(
            {
                "run_dir": str(row.get("run_dir", "")),
                "run_id": str(row.get("run_id", "")),
                "missing_files": "",
                "skip_reason": "selector_metric_incomplete",
                "selection_label": str(row.get("selection_label", "")),
            }
        )
    for idx, row in enumerate(complete, start=1):
        row["strict_rank"] = idx
    return complete, filtered_skipped


def build_window_breakdown(
    candidates: list[dict[str, Any]],
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv")
        partial_rows = _read_csv(run_dir / "teacher_only_rolling_partial_gain_vs_history_v29.csv")
        for partial in _target_rows(partial_rows, target_name=target_name):
            window_id = str(partial.get("validation_window_id", ""))
            fh = _first(
                rolling_rows,
                target_name=target_name,
                feature_mode=feature_mode,
                validation_window_id=window_id,
            )
            history = _first(
                rolling_rows,
                target_name=target_name,
                feature_mode="history_only",
                validation_window_id=window_id,
            )
            partial_mse = _to_float(partial.get("partial_mse_gain_vs_history"))
            partial_mae = _to_float(partial.get("partial_mae_gain_vs_history"))
            cf_margin = _to_float(fh.get("teacher_real_vs_best_counterfactual_mse_gain_margin"))
            fh_mse = _to_float(fh.get("teacher_real_mse_gain"))
            failure_reasons: list[str] = []
            if fh_mse < 0.0:
                failure_reasons.append("fh_mse_harmful")
            if partial_mse < 0.0:
                failure_reasons.append("partial_mse_negative")
            if partial_mae < 0.0:
                failure_reasons.append("partial_mae_negative")
            if partial_mse < -0.05:
                failure_reasons.append("partial_mse_below_gate_floor")
            if partial_mae < -0.05:
                failure_reasons.append("partial_mae_below_gate_floor")
            if cf_margin < 0.0:
                failure_reasons.append("counterfactual_negative")

            rows.append(
                {
                    "strict_rank": candidate.get("strict_rank", ""),
                    "run_id": candidate.get("run_id", ""),
                    "selection_label": candidate.get("selection_label", ""),
                    "target_name": target_name,
                    "feature_mode": feature_mode,
                    "validation_window_id": window_id,
                    "fh_mse_gain": fh_mse,
                    "history_mse_gain": _to_float(history.get("teacher_real_mse_gain")),
                    "partial_mse_gain_vs_history": partial_mse,
                    "fh_mae_gain": _to_float(fh.get("teacher_real_mae_gain")),
                    "history_mae_gain": _to_float(history.get("teacher_real_mae_gain")),
                    "partial_mae_gain_vs_history": partial_mae,
                    "fh_sign_agreement": _to_float(fh.get("teacher_real_sign_agreement_rate")),
                    "history_sign_agreement": _to_float(history.get("teacher_real_sign_agreement_rate")),
                    "partial_sign_agreement_vs_history": _to_float(
                        partial.get("partial_sign_agreement_vs_history")
                    ),
                    "best_counterfactual_mode": fh.get("teacher_best_counterfactual_mode", ""),
                    "counterfactual_margin": cf_margin,
                    "real_is_best_mode": _truthy(fh.get("teacher_real_is_best_mode")),
                    "failure_reasons": ";".join(failure_reasons),
                }
            )
    rows.sort(key=lambda row: (_to_int(row["strict_rank"]), _to_int(row["validation_window_id"])))
    return rows


def _available_feature_rows(
    rolling_rows: list[dict[str, str]],
    *,
    target_name: str,
    window_id: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rolling_rows
        if row.get("target_name") == target_name and str(row.get("validation_window_id", "")) == window_id
    ]


def _select_policy_row(
    rows: list[dict[str, str]],
    *,
    feature_mode: str,
    policy_name: str,
) -> dict[str, str]:
    history = _first(rows, feature_mode="history_only")
    fh = _first(rows, feature_mode=feature_mode)
    if policy_name == "fh_only":
        return fh
    if policy_name == "history_protected_oracle":
        if not fh:
            return history
        if not history:
            return fh
        return history if _to_float(history.get("teacher_real_mse_gain")) > _to_float(fh.get("teacher_real_mse_gain")) else fh
    if policy_name == "best_available_feature_oracle":
        return max(rows, key=lambda row: _to_float(row.get("teacher_real_mse_gain"), -float("inf")), default={})
    if policy_name == "cf_safe_best_feature_oracle":
        cf_safe = [
            row
            for row in rows
            if _to_float(row.get("teacher_real_vs_best_counterfactual_mse_gain_margin"), -float("inf")) >= 0.0
        ]
        return max(cf_safe or [history], key=lambda row: _to_float(row.get("teacher_real_mse_gain"), -float("inf")), default={})
    raise ValueError(f"unknown feature-mode policy: {policy_name}")


def build_feature_mode_policy_diagnostics(
    candidates: list[dict[str, Any]],
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> list[dict[str, Any]]:
    policy_names = (
        "fh_only",
        "history_protected_oracle",
        "best_available_feature_oracle",
        "cf_safe_best_feature_oracle",
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv")
        window_ids = sorted(
            {
                str(row.get("validation_window_id", ""))
                for row in rolling_rows
                if row.get("target_name") == target_name and str(row.get("validation_window_id", "")) != ""
            },
            key=lambda value: _to_int(value),
        )
        required_windows = _to_int(candidate.get("required_positive_partial_window_count"))
        for policy_name in policy_names:
            selected_modes: list[str] = []
            partial_mse_values: list[float] = []
            partial_mae_values: list[float] = []
            cf_margin_values: list[float] = []
            selected_mse_values: list[float] = []
            selected_mae_values: list[float] = []
            positive_window_count = 0
            for window_id in window_ids:
                feature_rows = _available_feature_rows(rolling_rows, target_name=target_name, window_id=window_id)
                selected = _select_policy_row(feature_rows, feature_mode=feature_mode, policy_name=policy_name)
                history = _first(feature_rows, feature_mode="history_only")
                selected_mode = selected.get("feature_mode", "") if selected else ""
                selected_modes.append(f"{window_id}:{selected_mode}")
                selected_mse = _to_float(selected.get("teacher_real_mse_gain") if selected else "")
                history_mse = _to_float(history.get("teacher_real_mse_gain") if history else "")
                selected_mae = _to_float(selected.get("teacher_real_mae_gain") if selected else "")
                history_mae = _to_float(history.get("teacher_real_mae_gain") if history else "")
                partial_mse = selected_mse - history_mse
                partial_mae = selected_mae - history_mae
                cf_margin = _to_float(
                    selected.get("teacher_real_vs_best_counterfactual_mse_gain_margin") if selected else ""
                )
                selected_mse_values.append(selected_mse)
                selected_mae_values.append(selected_mae)
                partial_mse_values.append(partial_mse)
                partial_mae_values.append(partial_mae)
                cf_margin_values.append(cf_margin)
                if partial_mse > 0.0:
                    positive_window_count += 1

            min_partial_mse = min(partial_mse_values, default=float("nan"))
            min_partial_mae = min(partial_mae_values, default=float("nan"))
            min_cf_margin = min(cf_margin_values, default=float("nan"))
            gate_like_pass = (
                positive_window_count >= required_windows
                and min_partial_mse >= -0.05
                and min_partial_mae >= -0.05
                and min_cf_margin >= 0.0
            )
            rows.append(
                {
                    "strict_rank": candidate.get("strict_rank", ""),
                    "run_id": candidate.get("run_id", ""),
                    "selection_label": candidate.get("selection_label", ""),
                    "target_name": target_name,
                    "policy_name": policy_name,
                    "window_count": len(window_ids),
                    "required_positive_partial_window_count": required_windows,
                    "positive_partial_window_count": positive_window_count,
                    "min_partial_mse_gain_vs_history": min_partial_mse,
                    "min_partial_mae_gain_vs_history": min_partial_mae,
                    "min_counterfactual_margin": min_cf_margin,
                    "mean_selected_mse_gain": sum(selected_mse_values) / len(selected_mse_values)
                    if selected_mse_values
                    else float("nan"),
                    "mean_selected_mae_gain": sum(selected_mae_values) / len(selected_mae_values)
                    if selected_mae_values
                    else float("nan"),
                    "gate_like_pass": gate_like_pass,
                    "selected_feature_modes_by_window": ";".join(selected_modes),
                    "diagnostic_scope": "oracle_validation_upper_bound_not_online",
                }
            )
    rows.sort(key=lambda row: (_to_int(row["strict_rank"]), str(row["policy_name"])))
    return rows


def build_mode_gate_label_rows(
    candidates: list[dict[str, Any]],
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv")
        window_ids = sorted(
            {
                str(row.get("validation_window_id", ""))
                for row in rolling_rows
                if row.get("target_name") == target_name and str(row.get("validation_window_id", "")) != ""
            },
            key=lambda value: _to_int(value),
        )
        for window_id in window_ids:
            feature_rows = _available_feature_rows(rolling_rows, target_name=target_name, window_id=window_id)
            fh = _first(feature_rows, feature_mode=feature_mode)
            history = _first(feature_rows, feature_mode="history_only")
            best = _select_policy_row(feature_rows, feature_mode=feature_mode, policy_name="best_available_feature_oracle")
            cf_safe = _select_policy_row(
                feature_rows,
                feature_mode=feature_mode,
                policy_name="cf_safe_best_feature_oracle",
            )
            protected = _select_policy_row(
                feature_rows,
                feature_mode=feature_mode,
                policy_name="history_protected_oracle",
            )
            best_mode = best.get("feature_mode", "")
            cf_safe_mode = cf_safe.get("feature_mode", "")
            rows.append(
                {
                    "strict_rank": candidate.get("strict_rank", ""),
                    "run_id": candidate.get("run_id", ""),
                    "selection_label": candidate.get("selection_label", ""),
                    "target_name": target_name,
                    "validation_window_id": window_id,
                    "fh_mse_gain": _to_float(fh.get("teacher_real_mse_gain")),
                    "history_mse_gain": _to_float(history.get("teacher_real_mse_gain")),
                    "best_available_feature_label": best_mode,
                    "best_available_feature_mse_gain": _to_float(best.get("teacher_real_mse_gain")),
                    "best_available_feature_cf_margin": _to_float(
                        best.get("teacher_real_vs_best_counterfactual_mse_gain_margin")
                    ),
                    "best_available_feature_cf_safe": _to_float(
                        best.get("teacher_real_vs_best_counterfactual_mse_gain_margin")
                    )
                    >= 0.0,
                    "cf_safe_feature_label": cf_safe_mode,
                    "cf_safe_feature_mse_gain": _to_float(cf_safe.get("teacher_real_mse_gain")),
                    "cf_safe_feature_cf_margin": _to_float(
                        cf_safe.get("teacher_real_vs_best_counterfactual_mse_gain_margin")
                    ),
                    "history_protected_label": protected.get("feature_mode", ""),
                    "fh_vs_history_label": protected.get("feature_mode", ""),
                    "needs_non_fh_mode": best_mode != feature_mode,
                    "cf_safe_requires_non_fh_mode": cf_safe_mode != feature_mode,
                    "label_source": "validation_window_outcome_oracle",
                    "online_label_available": False,
                    "online_gate_feature_status": "not_defined_in_current_v29_artifact",
                }
            )
    rows.sort(key=lambda row: (_to_int(row["strict_rank"]), _to_int(row["validation_window_id"])))
    return rows


def _ordered_feature_modes(feature_rows: list[dict[str, str]]) -> list[str]:
    modes: list[str] = []
    for row in feature_rows:
        mode = str(row.get("feature_mode", ""))
        if mode and mode not in modes:
            modes.append(mode)
    return modes


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _safe_dynamic_feature_rows_by_window(run_dir: Path, *, target_name: str) -> dict[str, dict[str, Any]]:
    path = run_dir / "stage3_v29_online_gate_dynamic_features.csv"
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        if row.get("target_name") != target_name:
            continue
        window_id = str(row.get("validation_window_id", ""))
        if not window_id:
            continue
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if not key.startswith("feature_dynamic_"):
                continue
            if any(token in key for token in BANNED_ONLINE_FEATURE_NAME_TOKENS):
                continue
            safe_row[key] = _coerce_csv_scalar(value)
        result[window_id] = safe_row
    return result


def build_online_gate_feature_rows(
    candidates: list[dict[str, Any]],
    *,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
) -> list[dict[str, Any]]:
    """Build a leakage-guarded design matrix for a future online mode gate.

    Columns prefixed with ``feature_`` are limited to pre-outcome run metadata,
    train-fit signal/history contracts, and validation-window schedule metadata.
    Oracle labels are emitted separately with an ``oracle_label_`` prefix.
    """

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(str(candidate["run_dir"]))
        rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv")
        signal = _read_first_optional_csv_row(run_dir / "stage3_v29_signal_contract_diagnostics.csv")
        history = _read_first_optional_csv_row(run_dir / "stage3_v29_history_contract_diagnostics.csv")
        dynamic_by_window = _safe_dynamic_feature_rows_by_window(run_dir, target_name=target_name)
        resolved_config = _read_json(run_dir / "resolved_config.json")
        data_cfg = _nested_dict(resolved_config, "data")
        invariant_cfg = _nested_dict(resolved_config, "invariant")
        window_ids = sorted(
            {
                str(row.get("validation_window_id", ""))
                for row in rolling_rows
                if row.get("target_name") == target_name and str(row.get("validation_window_id", "")) != ""
            },
            key=lambda value: _to_int(value),
        )
        for window_id in window_ids:
            feature_rows = _available_feature_rows(rolling_rows, target_name=target_name, window_id=window_id)
            if not feature_rows:
                continue
            reference = _first(feature_rows, feature_mode=feature_mode) or feature_rows[0]
            start_sample = _to_int(reference.get("validation_window_start_sample"))
            end_sample = _to_int(reference.get("validation_window_end_sample"))
            modes = _ordered_feature_modes(feature_rows)
            best = _select_policy_row(feature_rows, feature_mode=feature_mode, policy_name="best_available_feature_oracle")
            cf_safe = _select_policy_row(
                feature_rows,
                feature_mode=feature_mode,
                policy_name="cf_safe_best_feature_oracle",
            )
            protected = _select_policy_row(
                feature_rows,
                feature_mode=feature_mode,
                policy_name="history_protected_oracle",
            )
            row: dict[str, Any] = {
                "strict_rank": candidate.get("strict_rank", ""),
                "run_id": candidate.get("run_id", ""),
                "selection_label": candidate.get("selection_label", ""),
                "target_name": target_name,
                "validation_window_id": window_id,
                "feature_dataset": candidate.get("dataset", reference.get("dataset", "")),
                "feature_seed": candidate.get("seed", reference.get("seed", "")),
                "feature_window_start_sample": start_sample,
                "feature_window_end_sample": end_sample,
                "feature_window_sample_count": max(end_sample - start_sample, 0),
                "feature_window_position_index": _to_int(window_id),
                "feature_candidate_mode": feature_mode,
                "feature_available_modes": ";".join(modes),
                "feature_available_mode_count": len(modes),
                "feature_has_history_mode": "history_only" in modes,
                "feature_has_level_offset_mode": any("level_offset" in mode for mode in modes),
                "feature_data_lookback": _to_int(data_cfg.get("lookback")),
                "feature_data_horizon": _to_int(data_cfg.get("horizon")),
                "feature_signal_contract_version": signal.get("signal_contract_version", ""),
                "feature_signal_mode": signal.get("signal_mode", ""),
                "feature_signal_normalize": signal.get("signal_normalize", ""),
                "feature_signal_fit_split": signal.get("signal_fit_split", ""),
                "feature_signal_fit_sample_count": _to_int(signal.get("signal_fit_sample_count")),
                "feature_signal_scale_abs_mean": _to_float(signal.get("signal_scale_abs_mean")),
                "feature_signal_scale_min": _to_float(signal.get("signal_scale_min")),
                "feature_signal_scale_max": _to_float(signal.get("signal_scale_max")),
                "feature_history_protocol": history.get("history_feature_protocol", ""),
                "feature_history_lags": history.get("history_lags", ""),
                "feature_history_feedback_delay": _to_int(history.get("history_feedback_delay")),
                "feature_history_include_availability_mask": history.get("history_include_availability_mask", ""),
                "feature_history_standardize": history.get("history_standardize", ""),
                "feature_history_fit_split": history.get("history_fit_split", ""),
                "feature_history_fit_sample_count": _to_int(history.get("history_fit_sample_count")),
                "feature_history_value_dim": _to_int(history.get("history_value_dim")),
                "feature_history_feature_dim": _to_int(history.get("history_feature_dim")),
                "feature_history_scale_abs_mean": _to_float(history.get("history_scale_abs_mean")),
                "feature_history_scale_min": _to_float(history.get("history_scale_min")),
                "feature_history_scale_max": _to_float(history.get("history_scale_max")),
                "feature_config_release_first_increment": _to_float(invariant_cfg.get("lambda_release_first_increment")),
                "feature_online_scope": "pre_outcome_static_contract_schedule_and_optional_forward_metadata",
                "feature_dynamic_signal_status": "missing_current_serving_window_statistics",
                "feature_leakage_guard": "feature_prefix_excludes_validation_outcome_columns",
                "oracle_label_best_available_feature": best.get("feature_mode", ""),
                "oracle_label_cf_safe_feature": cf_safe.get("feature_mode", ""),
                "oracle_label_history_protected_feature": protected.get("feature_mode", ""),
                "oracle_label_needs_non_candidate_mode": best.get("feature_mode", "") != feature_mode,
                "oracle_label_source": "validation_window_outcome_oracle",
            }
            dynamic_features = dynamic_by_window.get(window_id, {})
            if dynamic_features:
                row.update(dynamic_features)
                row["feature_dynamic_signal_status"] = str(
                    dynamic_features.get("feature_dynamic_status", "available_from_model_forward")
                )
            rows.append(row)
    rows.sort(key=lambda row: (_to_int(row["strict_rank"]), _to_int(row["validation_window_id"])))
    return rows


def write_csv_no_overwrite(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning selection output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_no_overwrite(path: Path, text: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning selection output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_markdown_report(
    candidates: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    target_name: str,
    feature_mode: str,
) -> str:
    lines = [
        "# Stage III V29 Gate-Aware Tuning Selection",
        "",
        f"Target: `{target_name}`.",
        f"Feature mode: `{feature_mode}`.",
        f"Ranked candidate runs: `{len(candidates)}`.",
        f"Skipped run directories: `{len(skipped)}`.",
        "",
        "Ranking priority: teacher gate pass, generator unlock, non-harmful raw window `0`, "
        "counterfactual margin, positive partial-window count, min partial MSE, min partial MAE, "
        "aggregate partial MSE, aggregate partial MAE, aggregate raw MSE.",
        "",
    ]
    if candidates:
        top = candidates[0]
        lines.extend(
            [
                "## Current Top Candidate",
                "",
                f"- Rank: `{top['strict_rank']}`.",
                f"- Run: `{top['run_id']}`.",
                f"- Label: `{top['selection_label']}`.",
                f"- Gate pass: `{top['target_gate_pass']}`.",
                f"- Generator unlocked: `{top['generator_unlocked']}`.",
                f"- Window `0` raw MSE gain: `{_format_float(_to_float(top['window0_fh_mse_gain']))}`.",
                f"- Counterfactual margin: `{_format_float(_to_float(top['counterfactual_margin']))}`.",
                f"- Positive partial windows: `{top['positive_partial_window_count']}/{top['required_positive_partial_window_count']}`.",
                f"- Validation partial MSE gain over history: `{_format_float(_to_float(top['validation_partial_mse_gain_vs_history']))}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranked Runs",
            "",
            "| Rank | Run | Label | Gate | Unlock | W0 MSE | CF Margin | Pos Windows | Val Partial MSE |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidates:
        lines.append(
            "| {rank} | `{run}` | `{label}` | `{gate}` | `{unlock}` | `{w0}` | `{cf}` | `{pos}/{req}` | `{pmse}` |".format(
                rank=row["strict_rank"],
                run=row["run_id"],
                label=row["selection_label"],
                gate=row["target_gate_pass"],
                unlock=row["generator_unlocked"],
                w0=_format_float(_to_float(row["window0_fh_mse_gain"])),
                cf=_format_float(_to_float(row["counterfactual_margin"])),
                pos=row["positive_partial_window_count"],
                req=row["required_positive_partial_window_count"],
                pmse=_format_float(_to_float(row["validation_partial_mse_gain_vs_history"])),
            )
        )
    if skipped:
        lines.extend(
            [
                "",
                "## Skipped Run Directories",
                "",
                "| Run Dir | Reason | Missing Files |",
                "| --- | --- | --- |",
            ]
        )
        for row in skipped:
            lines.append(
                "| `{}` | `{}` | `{}` |".format(
                    row.get("run_dir", ""),
                    row.get("skip_reason", ""),
                    row.get("missing_files", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "This selector is a tuning-control artifact. It does not convert a gate-negative run "
            "into a manuscript performance result. A run remains non-promotable unless a primary "
            "target passes the strict teacher gate and the generator path is unlocked.",
            "",
        ]
    )
    return "\n".join(lines)


def build_window_markdown_report(window_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Window-Level Tuning Breakdown",
        "",
        f"Window rows: `{len(window_rows)}`.",
        "",
        "Failure reasons are diagnostic flags only. The official gate remains "
        "`teacher_only_v29_target_gate.csv`.",
        "",
        "| Rank | Run | Window | FH MSE | History MSE | Partial MSE | CF Margin | Failures |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in window_rows:
        lines.append(
            "| {rank} | `{run}` | {window} | `{fh}` | `{history}` | `{partial}` | `{cf}` | `{failures}` |".format(
                rank=row.get("strict_rank", ""),
                run=row.get("run_id", ""),
                window=row.get("validation_window_id", ""),
                fh=_format_float(_to_float(row.get("fh_mse_gain"))),
                history=_format_float(_to_float(row.get("history_mse_gain"))),
                partial=_format_float(_to_float(row.get("partial_mse_gain_vs_history"))),
                cf=_format_float(_to_float(row.get("counterfactual_margin"))),
                failures=row.get("failure_reasons", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "Use this table to decide whether a candidate improves the actual failing windows, "
            "not just aggregate validation gain. A candidate that raises aggregate partial MSE "
            "while moving the worst failure to another window should remain diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_feature_mode_policy_markdown_report(policy_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Conditional Feature-Mode Policy Diagnostic",
        "",
        f"Policy rows: `{len(policy_rows)}`.",
        "",
        "These rows are validation-oracle diagnostics. They are not online-available policies "
        "and must not be used as manuscript performance evidence.",
        "",
        "| Rank | Run | Policy | Gate-Like Pass | Pos Windows | Min Partial MSE | Min CF Margin | Selected Modes |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in policy_rows:
        lines.append(
            "| {rank} | `{run}` | `{policy}` | `{passed}` | `{pos}/{req}` | `{pmse}` | `{cf}` | `{modes}` |".format(
                rank=row.get("strict_rank", ""),
                run=row.get("run_id", ""),
                policy=row.get("policy_name", ""),
                passed=row.get("gate_like_pass", ""),
                pos=row.get("positive_partial_window_count", ""),
                req=row.get("required_positive_partial_window_count", ""),
                pmse=_format_float(_to_float(row.get("min_partial_mse_gain_vs_history"))),
                cf=_format_float(_to_float(row.get("min_counterfactual_margin"))),
                modes=row.get("selected_feature_modes_by_window", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "`best_available_feature_oracle` and `history_protected_oracle` use validation-window "
            "outcomes to select modes. They only answer whether feature-mode selection is worth "
            "engineering; they do not define a deployable online policy.",
            "",
        ]
    )
    return "\n".join(lines)


def build_mode_gate_label_markdown_report(label_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Mode-Gate Oracle Labels",
        "",
        f"Label rows: `{len(label_rows)}`.",
        "",
        "These labels are derived from validation-window outcomes. They are useful for "
        "designing an online-available gate, but they are not themselves online labels.",
        "",
        "| Rank | Run | Window | Best Label | Best CF Safe | CF-Safe Label | Needs Non-FH | Online Label |",
        "| ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in label_rows:
        lines.append(
            "| {rank} | `{run}` | {window} | `{best}` | `{best_safe}` | `{cf_safe}` | `{non_fh}` | `{online}` |".format(
                rank=row.get("strict_rank", ""),
                run=row.get("run_id", ""),
                window=row.get("validation_window_id", ""),
                best=row.get("best_available_feature_label", ""),
                best_safe=row.get("best_available_feature_cf_safe", ""),
                cf_safe=row.get("cf_safe_feature_label", ""),
                non_fh=row.get("needs_non_fh_mode", ""),
                online=row.get("online_label_available", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "A deployable V29 mode gate still needs pre-outcome features that are available "
            "at serving time. This artifact only defines what the oracle would have selected "
            "after observing validation-window outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_online_gate_feature_markdown_report(feature_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Online Mode-Gate Feature Matrix",
        "",
        f"Feature rows: `{len(feature_rows)}`.",
        "",
        "This matrix separates online-candidate features from oracle labels. Columns with "
        "`feature_` prefix intentionally exclude validation-window gains, counterfactual "
        "margins, best-mode outcomes, and partial-gain results.",
        "",
        "| Rank | Run | Window | Sample Span | Modes | Signal Fit | History Fit | Dynamic Status | Oracle Label |",
        "| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in feature_rows:
        lines.append(
            "| {rank} | `{run}` | {window} | `{start}-{end}` | `{modes}` | `{signal}` | `{history}` | `{dynamic}` | `{label}` |".format(
                rank=row.get("strict_rank", ""),
                run=row.get("run_id", ""),
                window=row.get("validation_window_id", ""),
                start=row.get("feature_window_start_sample", ""),
                end=row.get("feature_window_end_sample", ""),
                modes=row.get("feature_available_modes", ""),
                signal=row.get("feature_signal_fit_split", ""),
                history=row.get("feature_history_fit_split", ""),
                dynamic=row.get("feature_dynamic_signal_status", ""),
                label=row.get("oracle_label_best_available_feature", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            "The `feature_` namespace is reserved for pre-outcome metadata only. Oracle "
            "targets stay under `oracle_label_` and must not be fed into an online selector.",
            "",
            "## Remaining Online Gap",
            "",
            "Historical V29 runs that do not contain "
            "`stage3_v29_online_gate_dynamic_features.csv` still show "
            "`missing_current_serving_window_statistics`. The V29 probe wrapper now "
            "writes that file for new bounded runs from model-forward state before "
            "validation outcomes are known, but this artifact remains a feature "
            "specification and leakage guard, not a trained online gate.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_selection(
    *,
    run_dirs: list[Path],
    roots: list[Path],
    out_dir: Path,
    prefix: str,
    target_name: str = "raw_residual_vector",
    feature_mode: str = "fluctuation_plus_history",
    require_complete_selector_metrics: bool = False,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_run_dirs = list(run_dirs) + _discover_run_dirs(roots)
    candidates, skipped = select_runs(all_run_dirs, target_name=target_name, feature_mode=feature_mode)
    if require_complete_selector_metrics:
        candidates, skipped = _require_complete_selector_metrics(candidates, skipped)
    window_rows = build_window_breakdown(candidates, target_name=target_name, feature_mode=feature_mode)
    policy_rows = build_feature_mode_policy_diagnostics(
        candidates,
        target_name=target_name,
        feature_mode=feature_mode,
    )
    label_rows = build_mode_gate_label_rows(candidates, target_name=target_name, feature_mode=feature_mode)
    online_feature_rows = build_online_gate_feature_rows(candidates, target_name=target_name, feature_mode=feature_mode)
    write_csv_no_overwrite(out_dir / f"{prefix}.csv", candidates, overwrite=overwrite)
    write_csv_no_overwrite(out_dir / f"{prefix}_windows.csv", window_rows, overwrite=overwrite)
    write_csv_no_overwrite(out_dir / f"{prefix}_feature_mode_policy.csv", policy_rows, overwrite=overwrite)
    write_csv_no_overwrite(out_dir / f"{prefix}_mode_gate_labels.csv", label_rows, overwrite=overwrite)
    write_csv_no_overwrite(out_dir / f"{prefix}_online_gate_features.csv", online_feature_rows, overwrite=overwrite)
    write_csv_no_overwrite(out_dir / f"{prefix}_skipped.csv", skipped, overwrite=overwrite)
    write_text_no_overwrite(
        out_dir / f"{prefix}.md",
        build_markdown_report(candidates, skipped, target_name=target_name, feature_mode=feature_mode),
        overwrite=overwrite,
    )
    write_text_no_overwrite(
        out_dir / f"{prefix}_windows.md",
        build_window_markdown_report(window_rows),
        overwrite=overwrite,
    )
    write_text_no_overwrite(
        out_dir / f"{prefix}_feature_mode_policy.md",
        build_feature_mode_policy_markdown_report(policy_rows),
        overwrite=overwrite,
    )
    write_text_no_overwrite(
        out_dir / f"{prefix}_mode_gate_labels.md",
        build_mode_gate_label_markdown_report(label_rows),
        overwrite=overwrite,
    )
    write_text_no_overwrite(
        out_dir / f"{prefix}_online_gate_features.md",
        build_online_gate_feature_markdown_report(online_feature_rows),
        overwrite=overwrite,
    )
    return candidates, skipped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank completed Stage III V29 tuning runs by strict gate-aware priorities.")
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="v29_tuning_selection")
    parser.add_argument("--target-name", default="raw_residual_vector")
    parser.add_argument("--feature-mode", default="fluctuation_plus_history")
    parser.add_argument("--require-complete-selector-metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze_selection(
        run_dirs=args.run_dir,
        roots=args.root,
        out_dir=args.out_dir,
        prefix=args.prefix,
        target_name=args.target_name,
        feature_mode=args.feature_mode,
        require_complete_selector_metrics=args.require_complete_selector_metrics,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
