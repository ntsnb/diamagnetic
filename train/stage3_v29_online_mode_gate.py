from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.analyze_stage3_v29_tuning_selection import BANNED_ONLINE_FEATURE_NAME_TOKENS


PROTOCOL_VERSION = "v29_online_mode_gate_v1"
DEFAULT_LABEL_COLUMN = "oracle_label_cf_safe_feature"


@dataclass
class OnlineModeGateArtifacts:
    summary: dict[str, Any]
    model: dict[str, Any]
    prediction_rows: list[dict[str, Any]]
    markdown: str


@dataclass
class OnlineModeGateOutcomeArtifacts:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    markdown: str


@dataclass
class OnlineModeGatePolicy:
    summary: dict[str, Any]
    model: dict[str, Any]
    source_prefix: str
    source_dir: str

    @property
    def min_confidence(self) -> float:
        return float(self.summary.get("min_confidence", 0.15))

    @property
    def fallback_mode(self) -> str:
        return str(self.summary.get("fallback_mode", "history_only"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 online mode-gate output: {path}")
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


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 online mode-gate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _window_index(row: dict[str, Any]) -> int:
    value = _coerce_number(row.get("feature_window_position_index"))
    if value is None:
        value = _coerce_number(row.get("validation_window_id"))
    return int(value if value is not None else -1)


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get("run_id", "")), _window_index(row)))


def select_online_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return feature columns that are available online and name-safe.

    The selector upstream reserves ``feature_`` for pre-outcome metadata and
    forward-state statistics. This function adds a second local guard by
    rejecting any feature column whose name contains outcome-like tokens.
    """

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key in columns:
                continue
            if not key.startswith("feature_"):
                continue
            if any(token in key for token in BANNED_ONLINE_FEATURE_NAME_TOKENS):
                continue
            columns.append(key)
    return columns


def split_rows_for_heldout_windows(
    rows: list[dict[str, Any]],
    *,
    heldout_tail_windows: int = 1,
    heldout_start_window: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered = _sorted_rows(rows)
    windows = sorted({_window_index(row) for row in ordered if _window_index(row) >= 0})
    if heldout_start_window is None:
        if not windows:
            split = {
                "strategy": "global_window_order_tail_holdout",
                "heldout_tail_windows": heldout_tail_windows,
                "heldout_start_window": None,
                "available_windows": [],
                "status": "insufficient_windows",
            }
            return [], [], split
        tail = max(int(heldout_tail_windows), 1)
        heldout_start_window = windows[max(len(windows) - tail, 0)]
    train_rows = [row for row in ordered if _window_index(row) < int(heldout_start_window)]
    eval_rows = [row for row in ordered if _window_index(row) >= int(heldout_start_window)]
    split = {
        "strategy": "global_window_order_tail_holdout",
        "heldout_tail_windows": int(heldout_tail_windows),
        "heldout_start_window": int(heldout_start_window),
        "available_windows": windows,
        "status": "ok" if train_rows and eval_rows else "insufficient_train_or_eval_rows",
    }
    return train_rows, eval_rows, split


def _infer_feature_schema(
    train_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    numeric_stats: dict[str, dict[str, float]] = {}
    categorical_vocab: dict[str, list[str]] = {}
    for column in feature_columns:
        numeric_values = [_coerce_number(row.get(column)) for row in train_rows]
        nonmissing = [value for value in numeric_values if value is not None]
        if nonmissing and len(nonmissing) == sum(value is not None for value in numeric_values):
            mean = sum(nonmissing) / len(nonmissing)
            variance = sum((value - mean) ** 2 for value in nonmissing) / max(len(nonmissing), 1)
            std = math.sqrt(variance) or 1.0
            numeric_stats[column] = {"mean": mean, "std": std}
        else:
            values = sorted({str(row.get(column, "")) for row in train_rows})
            categorical_vocab[column] = values
    return numeric_stats, categorical_vocab


def _encode_row(
    row: dict[str, Any],
    *,
    feature_columns: list[str],
    numeric_stats: dict[str, dict[str, float]],
    categorical_vocab: dict[str, list[str]],
) -> list[float]:
    vector: list[float] = []
    for column in feature_columns:
        if column in numeric_stats:
            stats = numeric_stats[column]
            value = _coerce_number(row.get(column))
            raw = stats["mean"] if value is None else value
            vector.append((raw - stats["mean"]) / stats["std"])
            continue
        vocab = categorical_vocab.get(column, [])
        value = str(row.get(column, ""))
        vector.extend(1.0 if value == item else 0.0 for item in vocab)
    return vector


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _fit_nearest_centroid(
    train_rows: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    label_column: str,
) -> dict[str, Any]:
    numeric_stats, categorical_vocab = _infer_feature_schema(train_rows, feature_columns)
    vectors_by_label: dict[str, list[list[float]]] = {}
    for row in train_rows:
        label = str(row.get(label_column, ""))
        vector = _encode_row(
            row,
            feature_columns=feature_columns,
            numeric_stats=numeric_stats,
            categorical_vocab=categorical_vocab,
        )
        vectors_by_label.setdefault(label, []).append(vector)

    centroids: dict[str, list[float]] = {}
    for label, vectors in vectors_by_label.items():
        if not vectors:
            continue
        width = len(vectors[0])
        centroids[label] = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]

    class_counts = {label: len(vectors) for label, vectors in vectors_by_label.items()}
    majority_label = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if class_counts else ""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_type": "standardized_nearest_centroid_mode_gate",
        "label_column": label_column,
        "feature_columns": feature_columns,
        "numeric_stats": numeric_stats,
        "categorical_vocab": categorical_vocab,
        "encoded_feature_count": len(next(iter(centroids.values()), [])),
        "centroids": centroids,
        "class_counts": class_counts,
        "majority_label": majority_label,
        "leakage_guard": "features_only_from_feature_prefix_with_banned_outcome_tokens_removed",
    }


def _predict_one(
    row: dict[str, Any],
    *,
    model: dict[str, Any],
    min_confidence: float,
    fallback_mode: str,
) -> tuple[str, float, bool]:
    centroids: dict[str, list[float]] = model.get("centroids", {})
    if not centroids:
        return fallback_mode, 0.0, True
    vector = _encode_row(
        row,
        feature_columns=list(model["feature_columns"]),
        numeric_stats=dict(model["numeric_stats"]),
        categorical_vocab=dict(model["categorical_vocab"]),
    )
    distances = sorted(
        ((label, _squared_distance(vector, centroid)) for label, centroid in centroids.items()),
        key=lambda item: (item[1], item[0]),
    )
    best_label, best_distance = distances[0]
    if len(distances) == 1:
        confidence = 1.0
    else:
        second_distance = distances[1][1]
        confidence = max(0.0, min(1.0, (second_distance - best_distance) / (second_distance + best_distance + 1e-12)))
    if confidence < min_confidence:
        return fallback_mode, confidence, True
    return best_label, confidence, False


def apply_online_mode_gate_policy(
    row: dict[str, Any],
    *,
    model: dict[str, Any] | None,
    min_confidence: float = 0.15,
    fallback_mode: str = "history_only",
    allowed_modes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    oracle_columns = sorted(key for key in row if key.startswith("oracle_label_"))
    if not model or not model.get("centroids"):
        return {
            "policy_status": "fallback",
            "reject_reason": "missing_or_empty_mode_gate_model",
            "selected_mode": fallback_mode,
            "raw_predicted_mode": "",
            "online_gate_confidence": 0.0,
            "used_fallback": True,
            "fallback_mode": fallback_mode,
            "oracle_label_columns_seen": oracle_columns,
        }

    prediction, confidence, used_fallback = _predict_one(
        row,
        model=model,
        min_confidence=min_confidence,
        fallback_mode=fallback_mode,
    )
    reject_reason = ""
    selected_mode = prediction
    if allowed_modes is not None and selected_mode not in set(str(mode) for mode in allowed_modes):
        reject_reason = "predicted_mode_not_allowed"
        selected_mode = fallback_mode
        used_fallback = True
    elif used_fallback:
        reject_reason = "confidence_below_threshold"

    return {
        "policy_status": "applied",
        "reject_reason": reject_reason,
        "selected_mode": selected_mode,
        "raw_predicted_mode": prediction,
        "online_gate_confidence": confidence,
        "used_fallback": used_fallback,
        "fallback_mode": fallback_mode,
        "min_confidence": min_confidence,
        "oracle_label_columns_seen": oracle_columns,
        "feature_column_count": len(model.get("feature_columns", [])),
        "leakage_guard": "oracle_label_columns_not_used_as_inputs",
    }


def load_online_mode_gate_policy(*, out_dir: Path, prefix: str) -> OnlineModeGatePolicy:
    summary_path = out_dir / f"{prefix}_summary.json"
    model_path = out_dir / f"{prefix}_model.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    return OnlineModeGatePolicy(
        summary=summary,
        model=model,
        source_prefix=prefix,
        source_dir=str(out_dir),
    )


def _artifact_manifest_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _admission_checks(
    outcome_summary: dict[str, Any],
    *,
    min_evaluated_rows: int,
    max_selected_harmful_windows: int,
    min_selected_minus_candidate_mean_mse_gain: float,
    min_selected_minus_history_mean_mse_gain: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(outcome_summary.get("status", "")) != "evaluated":
        reasons.append("outcome_replay_not_evaluated")
    if int(outcome_summary.get("evaluated_rows", 0) or 0) < int(min_evaluated_rows):
        reasons.append("insufficient_evaluated_rows")
    if int(outcome_summary.get("missing_outcome_rows", 0) or 0) != 0:
        reasons.append("missing_outcome_rows")
    if int(outcome_summary.get("selected_harmful_window_count", 0) or 0) > int(max_selected_harmful_windows):
        reasons.append("selected_harmful_windows_exceed_limit")
    selected_minus_candidate = _coerce_number(outcome_summary.get("selected_minus_candidate_mean_mse_gain"))
    if selected_minus_candidate is None or selected_minus_candidate < min_selected_minus_candidate_mean_mse_gain:
        reasons.append("selected_mean_mse_gain_not_above_candidate")
    selected_minus_history = _coerce_number(outcome_summary.get("selected_minus_history_mean_mse_gain"))
    if selected_minus_history is None or selected_minus_history < min_selected_minus_history_mean_mse_gain:
        reasons.append("selected_mean_mse_gain_not_above_history")
    return len(reasons) == 0, reasons


def promote_online_mode_gate_serving_package(
    *,
    policy_dir: Path,
    policy_prefix: str,
    outcome_prefix: str,
    out_dir: Path,
    package_id: str,
    min_evaluated_rows: int = 1,
    max_selected_harmful_windows: int = 0,
    min_selected_minus_candidate_mean_mse_gain: float = 0.0,
    min_selected_minus_history_mean_mse_gain: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    policy_summary_path = policy_dir / f"{policy_prefix}_summary.json"
    policy_model_path = policy_dir / f"{policy_prefix}_model.json"
    policy_predictions_path = policy_dir / f"{policy_prefix}_predictions.csv"
    policy_report_path = policy_dir / f"{policy_prefix}.md"
    outcome_summary_path = policy_dir / f"{outcome_prefix}_summary.json"
    outcome_rows_path = policy_dir / f"{outcome_prefix}_rows.csv"
    outcome_report_path = policy_dir / f"{outcome_prefix}.md"

    outcome_summary = json.loads(outcome_summary_path.read_text(encoding="utf-8"))
    admission_pass, admission_reasons = _admission_checks(
        outcome_summary,
        min_evaluated_rows=min_evaluated_rows,
        max_selected_harmful_windows=max_selected_harmful_windows,
        min_selected_minus_candidate_mean_mse_gain=min_selected_minus_candidate_mean_mse_gain,
        min_selected_minus_history_mean_mse_gain=min_selected_minus_history_mean_mse_gain,
    )
    manifest = {
        "protocol_version": "v29_served_online_mode_gate_manifest_v1",
        "package_id": package_id,
        "policy_prefix": policy_prefix,
        "outcome_prefix": outcome_prefix,
        "policy_dir": str(policy_dir),
        "servable_scope": "mode_selection_only_no_generator_unlock",
        "admission_pass": admission_pass,
        "admission_reasons": admission_reasons,
        "admission_criteria": {
            "min_evaluated_rows": int(min_evaluated_rows),
            "max_selected_harmful_windows": int(max_selected_harmful_windows),
            "min_selected_minus_candidate_mean_mse_gain": float(min_selected_minus_candidate_mean_mse_gain),
            "min_selected_minus_history_mean_mse_gain": float(min_selected_minus_history_mean_mse_gain),
        },
        "outcome_summary": {
            "evaluated_rows": outcome_summary.get("evaluated_rows"),
            "selected_mean_mse_gain": outcome_summary.get("selected_mean_mse_gain"),
            "candidate_mean_mse_gain": outcome_summary.get("candidate_mean_mse_gain"),
            "history_mean_mse_gain": outcome_summary.get("history_mean_mse_gain"),
            "selected_harmful_window_count": outcome_summary.get("selected_harmful_window_count"),
            "candidate_harmful_window_count": outcome_summary.get("candidate_harmful_window_count"),
            "history_harmful_window_count": outcome_summary.get("history_harmful_window_count"),
        },
        "artifacts": {
            "policy_summary": _artifact_manifest_entry(policy_summary_path),
            "policy_model": _artifact_manifest_entry(policy_model_path),
            "policy_predictions": _artifact_manifest_entry(policy_predictions_path),
            "policy_report": _artifact_manifest_entry(policy_report_path),
            "outcome_summary": _artifact_manifest_entry(outcome_summary_path),
            "outcome_rows": _artifact_manifest_entry(outcome_rows_path),
            "outcome_report": _artifact_manifest_entry(outcome_report_path),
        },
        "leakage_guard": "served_policy_manifest_references_feature_only_gate_and_outcome_replay_artifacts",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{package_id}_manifest.json"
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def load_served_online_mode_gate_policy(manifest_path: Path, *, fail_closed: bool = True) -> OnlineModeGatePolicy | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("protocol_version", "")) != "v29_served_online_mode_gate_manifest_v1":
            raise ValueError("unsupported V29 served online mode-gate manifest version")
        if not bool(manifest.get("admission_pass", False)):
            raise ValueError("V29 served online mode-gate manifest did not pass admission")
        for name, entry in dict(manifest.get("artifacts", {})).items():
            path = Path(str(entry.get("path", "")))
            expected = str(entry.get("sha256", ""))
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(f"V29 served online mode-gate artifact hash mismatch: {name}")
        return load_online_mode_gate_policy(
            out_dir=Path(str(manifest["policy_dir"])),
            prefix=str(manifest["policy_prefix"]),
        )
    except Exception:
        if fail_closed:
            return None
        raise


def select_online_mode_for_row(
    row: dict[str, Any],
    *,
    policy: OnlineModeGatePolicy | None,
    allowed_modes: list[str] | tuple[str, ...] | None = None,
    default_mode: str = "history_only",
) -> dict[str, Any]:
    if policy is None:
        return {
            "policy_status": "fallback",
            "reject_reason": "missing_mode_gate_policy",
            "selected_mode": default_mode,
            "raw_predicted_mode": "",
            "online_gate_confidence": 0.0,
            "used_fallback": True,
            "fallback_mode": default_mode,
            "source_prefix": "",
            "source_dir": "",
            "oracle_label_columns_seen": sorted(key for key in row if key.startswith("oracle_label_")),
        }
    if str(policy.summary.get("status", "")) != "evaluated":
        decision = apply_online_mode_gate_policy(
            row,
            model=None,
            min_confidence=policy.min_confidence,
            fallback_mode=policy.fallback_mode,
            allowed_modes=allowed_modes,
        )
        decision["reject_reason"] = "mode_gate_policy_not_evaluated"
    else:
        decision = apply_online_mode_gate_policy(
            row,
            model=policy.model,
            min_confidence=policy.min_confidence,
            fallback_mode=policy.fallback_mode,
            allowed_modes=allowed_modes,
        )
    decision["source_prefix"] = policy.source_prefix
    decision["source_dir"] = policy.source_dir
    return decision


def _prediction_row(
    row: dict[str, Any],
    *,
    split: str,
    label_column: str,
    model: dict[str, Any],
    min_confidence: float,
    fallback_mode: str,
) -> dict[str, Any]:
    prediction, confidence, used_fallback = _predict_one(
        row,
        model=model,
        min_confidence=min_confidence,
        fallback_mode=fallback_mode,
    )
    true_label = str(row.get(label_column, ""))
    return {
        "split": split,
        "run_id": row.get("run_id", ""),
        "target_name": row.get("target_name", ""),
        "validation_window_id": row.get("validation_window_id", ""),
        "feature_window_position_index": row.get("feature_window_position_index", ""),
        "feature_candidate_mode": row.get("feature_candidate_mode", ""),
        "true_label": true_label,
        "predicted_label": prediction,
        "correct": prediction == true_label,
        "online_gate_confidence": confidence,
        "used_fallback": used_fallback,
        "label_column": label_column,
    }


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("correct") is True) / len(rows)


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _harmful_count(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None and value < 0.0)


def _outcome_lookup_key(row: dict[str, Any], mode: str | None = None) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_id", "")),
        str(row.get("target_name", "")),
        str(row.get("validation_window_id", "")),
        str(mode if mode is not None else row.get("feature_mode", "")),
    )


def _build_outcome_lookup(rolling_outcome_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rolling_outcome_rows:
        lookup[_outcome_lookup_key(row)] = row
    return lookup


def _outcome_metrics(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    feature_row: dict[str, Any],
    mode: str,
) -> tuple[float | None, float | None, float | None]:
    outcome = lookup.get(_outcome_lookup_key(feature_row, mode))
    if outcome is None:
        return None, None, None
    return (
        _coerce_number(outcome.get("teacher_real_mse_gain")),
        _coerce_number(outcome.get("teacher_real_mae_gain")),
        _coerce_number(outcome.get("teacher_real_vs_best_counterfactual_mse_gain_margin")),
    )


def evaluate_online_mode_gate_serving_outcomes(
    feature_rows: list[dict[str, Any]],
    *,
    rolling_outcome_rows: list[dict[str, Any]],
    policy: OnlineModeGatePolicy | None,
    candidate_mode: str = "fluctuation_plus_history",
    history_mode: str = "history_only",
    default_mode: str = "history_only",
) -> OnlineModeGateOutcomeArtifacts:
    lookup = _build_outcome_lookup(rolling_outcome_rows)
    rows: list[dict[str, Any]] = []
    for feature_row in _sorted_rows(feature_rows):
        serving_row = {key: value for key, value in feature_row.items() if not key.startswith("oracle_label_")}
        allowed_modes = [mode for mode in str(feature_row.get("feature_available_modes", "")).split(";") if mode]
        decision = select_online_mode_for_row(
            serving_row,
            policy=policy,
            allowed_modes=allowed_modes or None,
            default_mode=default_mode,
        )
        selected_mode = str(decision.get("selected_mode", default_mode))
        selected_mse, selected_mae, selected_cf_margin = _outcome_metrics(lookup, feature_row, selected_mode)
        candidate_mse, candidate_mae, candidate_cf_margin = _outcome_metrics(lookup, feature_row, candidate_mode)
        history_mse, history_mae, history_cf_margin = _outcome_metrics(lookup, feature_row, history_mode)
        oracle_mode = str(feature_row.get("oracle_label_cf_safe_feature", ""))
        oracle_mse, oracle_mae, oracle_cf_margin = _outcome_metrics(lookup, feature_row, oracle_mode) if oracle_mode else (None, None, None)
        rows.append(
            {
                "run_id": feature_row.get("run_id", ""),
                "target_name": feature_row.get("target_name", ""),
                "validation_window_id": feature_row.get("validation_window_id", ""),
                "feature_candidate_mode": feature_row.get("feature_candidate_mode", candidate_mode),
                "selected_mode": selected_mode,
                "raw_predicted_mode": decision.get("raw_predicted_mode", ""),
                "used_fallback": decision.get("used_fallback", ""),
                "reject_reason": decision.get("reject_reason", ""),
                "online_gate_confidence": decision.get("online_gate_confidence", ""),
                "oracle_label_columns_seen": ";".join(decision.get("oracle_label_columns_seen", [])),
                "selected_mse_gain": selected_mse,
                "selected_mae_gain": selected_mae,
                "selected_counterfactual_margin": selected_cf_margin,
                "candidate_mode": candidate_mode,
                "candidate_mse_gain": candidate_mse,
                "candidate_mae_gain": candidate_mae,
                "candidate_counterfactual_margin": candidate_cf_margin,
                "history_mode": history_mode,
                "history_mse_gain": history_mse,
                "history_mae_gain": history_mae,
                "history_counterfactual_margin": history_cf_margin,
                "oracle_cf_safe_mode": oracle_mode,
                "oracle_cf_safe_mse_gain": oracle_mse,
                "oracle_cf_safe_mae_gain": oracle_mae,
                "oracle_cf_safe_counterfactual_margin": oracle_cf_margin,
                "selected_minus_candidate_mse_gain": None if selected_mse is None or candidate_mse is None else selected_mse - candidate_mse,
                "selected_minus_history_mse_gain": None if selected_mse is None or history_mse is None else selected_mse - history_mse,
            }
        )

    selected_values = [_coerce_number(row.get("selected_mse_gain")) for row in rows]
    candidate_values = [_coerce_number(row.get("candidate_mse_gain")) for row in rows]
    history_values = [_coerce_number(row.get("history_mse_gain")) for row in rows]
    oracle_values = [_coerce_number(row.get("oracle_cf_safe_mse_gain")) for row in rows]
    missing_outcomes = sum(
        1
        for row in rows
        if row.get("selected_mse_gain") is None
        or row.get("candidate_mse_gain") is None
        or row.get("history_mse_gain") is None
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "evaluated" if rows and missing_outcomes == 0 else "incomplete_outcome_rows",
        "row_count": len(feature_rows),
        "evaluated_rows": len(rows) - missing_outcomes,
        "missing_outcome_rows": missing_outcomes,
        "candidate_mode": candidate_mode,
        "history_mode": history_mode,
        "selected_mean_mse_gain": _mean(selected_values),
        "candidate_mean_mse_gain": _mean(candidate_values),
        "history_mean_mse_gain": _mean(history_values),
        "oracle_cf_safe_mean_mse_gain": _mean(oracle_values),
        "selected_minus_candidate_mean_mse_gain": None
        if _mean(selected_values) is None or _mean(candidate_values) is None
        else _mean(selected_values) - _mean(candidate_values),
        "selected_minus_history_mean_mse_gain": None
        if _mean(selected_values) is None or _mean(history_values) is None
        else _mean(selected_values) - _mean(history_values),
        "selected_harmful_window_count": _harmful_count(selected_values),
        "candidate_harmful_window_count": _harmful_count(candidate_values),
        "history_harmful_window_count": _harmful_count(history_values),
        "fallback_predictions": sum(1 for row in rows if str(row.get("used_fallback")) == "True" or row.get("used_fallback") is True),
        "leakage_guard": "serving_rows_strip_oracle_label_columns_before_policy_prediction",
    }
    markdown = _build_outcome_markdown_report(summary=summary, rows=rows)
    return OnlineModeGateOutcomeArtifacts(summary=summary, rows=rows, markdown=markdown)


def build_online_mode_gate_artifacts(
    rows: list[dict[str, Any]],
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    heldout_tail_windows: int = 1,
    heldout_start_window: int | None = None,
    min_confidence: float = 0.15,
    fallback_mode: str = "history_only",
) -> OnlineModeGateArtifacts:
    if not label_column.startswith("oracle_label_"):
        raise ValueError("V29 online mode gate label_column must use the oracle_label_ namespace")

    labeled_rows = [row for row in rows if str(row.get(label_column, "")).strip()]
    feature_columns = select_online_feature_columns(labeled_rows)
    train_rows, eval_rows, split = split_rows_for_heldout_windows(
        labeled_rows,
        heldout_tail_windows=heldout_tail_windows,
        heldout_start_window=heldout_start_window,
    )

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "not_evaluated",
        "label_column": label_column,
        "row_count": len(rows),
        "labeled_row_count": len(labeled_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "feature_column_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split": split,
        "min_confidence": min_confidence,
        "fallback_mode": fallback_mode,
        "leakage_guard": "only_feature_prefix_columns_used_and_oracle_columns_excluded_from_inputs",
    }

    if not feature_columns or not train_rows or not eval_rows:
        summary["status"] = "insufficient_data"
        summary["insufficient_reason"] = "requires_nonempty_feature_columns_train_rows_and_heldout_rows"
        markdown = _build_markdown_report(summary=summary, prediction_rows=[])
        return OnlineModeGateArtifacts(summary=summary, model={}, prediction_rows=[], markdown=markdown)

    model = _fit_nearest_centroid(train_rows, feature_columns=feature_columns, label_column=label_column)
    prediction_rows = [
        _prediction_row(
            row,
            split="train",
            label_column=label_column,
            model=model,
            min_confidence=min_confidence,
            fallback_mode=fallback_mode,
        )
        for row in train_rows
    ] + [
        _prediction_row(
            row,
            split="heldout",
            label_column=label_column,
            model=model,
            min_confidence=min_confidence,
            fallback_mode=fallback_mode,
        )
        for row in eval_rows
    ]
    heldout_predictions = [row for row in prediction_rows if row["split"] == "heldout"]
    train_predictions = [row for row in prediction_rows if row["split"] == "train"]
    summary.update(
        {
            "status": "evaluated",
            "train_accuracy": _accuracy(train_predictions),
            "heldout_accuracy": _accuracy(heldout_predictions),
            "heldout_correct": sum(1 for row in heldout_predictions if row["correct"] is True),
            "heldout_total": len(heldout_predictions),
            "fallback_predictions": sum(1 for row in prediction_rows if row["used_fallback"] is True),
            "class_counts": model.get("class_counts", {}),
            "model_type": model.get("model_type"),
        }
    )
    markdown = _build_markdown_report(summary=summary, prediction_rows=prediction_rows)
    return OnlineModeGateArtifacts(summary=summary, model=model, prediction_rows=prediction_rows, markdown=markdown)


def _build_markdown_report(*, summary: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Online Mode Gate",
        "",
        f"Status: `{summary.get('status')}`.",
        f"Label column: `{summary.get('label_column')}`.",
        f"Train rows: `{summary.get('train_rows')}`.",
        f"Held-out rows: `{summary.get('eval_rows')}`.",
        f"Feature columns: `{summary.get('feature_column_count')}`.",
        "",
    ]
    if summary.get("status") == "evaluated":
        lines.extend(
            [
                f"Train accuracy: `{summary.get('train_accuracy')}`.",
                f"Held-out accuracy: `{summary.get('heldout_accuracy')}`.",
                f"Fallback predictions: `{summary.get('fallback_predictions')}`.",
                "",
                "| Split | Run | Window | True | Predicted | Confidence | Fallback |",
                "| --- | --- | ---: | --- | --- | ---: | --- |",
            ]
        )
        for row in prediction_rows:
            lines.append(
                "| {split} | `{run}` | {window} | `{true}` | `{pred}` | {conf:.6g} | `{fallback}` |".format(
                    split=row.get("split", ""),
                    run=row.get("run_id", ""),
                    window=row.get("validation_window_id", ""),
                    true=row.get("true_label", ""),
                    pred=row.get("predicted_label", ""),
                    conf=float(row.get("online_gate_confidence", 0.0)),
                    fallback=row.get("used_fallback", ""),
                )
            )
    else:
        lines.append(f"Insufficient reason: `{summary.get('insufficient_reason', '')}`.")
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            "Inputs are restricted to `feature_` columns after removing names with outcome-like tokens. "
            "`oracle_label_` columns are used only as supervision/evaluation labels, never as model inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_online_mode_gate_artifacts(
    *,
    out_dir: Path,
    prefix: str,
    artifacts: OnlineModeGateArtifacts,
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / f"{prefix}_summary.json", artifacts.summary, overwrite=overwrite)
    _write_json(out_dir / f"{prefix}_model.json", artifacts.model, overwrite=overwrite)
    _write_csv(out_dir / f"{prefix}_predictions.csv", artifacts.prediction_rows, overwrite=overwrite)
    md_path = out_dir / f"{prefix}.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 online mode-gate output: {md_path}")
    md_path.write_text(artifacts.markdown, encoding="utf-8")


def _build_outcome_markdown_report(*, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Online Mode Gate Outcome Replay",
        "",
        f"Status: `{summary.get('status')}`.",
        f"Rows: `{summary.get('row_count')}`.",
        f"Evaluated rows: `{summary.get('evaluated_rows')}`.",
        f"Missing outcome rows: `{summary.get('missing_outcome_rows')}`.",
        f"Candidate mode: `{summary.get('candidate_mode')}`.",
        f"History mode: `{summary.get('history_mode')}`.",
        "",
        f"Selected mean MSE gain: `{summary.get('selected_mean_mse_gain')}`.",
        f"Candidate mean MSE gain: `{summary.get('candidate_mean_mse_gain')}`.",
        f"History mean MSE gain: `{summary.get('history_mean_mse_gain')}`.",
        f"Oracle cf-safe mean MSE gain: `{summary.get('oracle_cf_safe_mean_mse_gain')}`.",
        "",
        f"Selected harmful windows: `{summary.get('selected_harmful_window_count')}`.",
        f"Candidate harmful windows: `{summary.get('candidate_harmful_window_count')}`.",
        f"History harmful windows: `{summary.get('history_harmful_window_count')}`.",
        f"Fallback predictions: `{summary.get('fallback_predictions')}`.",
        "",
        "| Run | Window | Selected | Sel MSE | Candidate MSE | History MSE | Reject |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{run}` | {window} | `{mode}` | {selected} | {candidate} | {history} | `{reject}` |".format(
                run=row.get("run_id", ""),
                window=row.get("validation_window_id", ""),
                mode=row.get("selected_mode", ""),
                selected=row.get("selected_mse_gain", ""),
                candidate=row.get("candidate_mse_gain", ""),
                history=row.get("history_mse_gain", ""),
                reject=row.get("reject_reason", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            "Outcome replay strips `oracle_label_` columns before calling the serving policy. "
            "Rolling outcome columns are used only after mode selection to score the selected mode.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_online_mode_gate_outcome_artifacts(
    *,
    out_dir: Path,
    prefix: str,
    artifacts: OnlineModeGateOutcomeArtifacts,
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / f"{prefix}_summary.json", artifacts.summary, overwrite=overwrite)
    _write_csv(out_dir / f"{prefix}_rows.csv", artifacts.rows, overwrite=overwrite)
    md_path = out_dir / f"{prefix}.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 online mode-gate output: {md_path}")
    md_path.write_text(artifacts.markdown, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a leakage-guarded V29 online mode gate.")
    parser.add_argument("--evaluate-outcomes", action="store_true")
    parser.add_argument("--features-csv", type=str, required=True)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--policy-dir", type=str, default=None)
    parser.add_argument("--policy-prefix", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="stage3_v29_online_mode_gate")
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--heldout-tail-windows", type=int, default=1)
    parser.add_argument("--heldout-start-window", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--fallback-mode", type=str, default="history_only")
    parser.add_argument("--candidate-mode", type=str, default="fluctuation_plus_history")
    parser.add_argument("--history-mode", type=str, default="history_only")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    features_path = Path(args.features_csv)
    rows = _read_csv(features_path)
    if bool(args.evaluate_outcomes):
        if not args.policy_dir or not args.policy_prefix:
            raise ValueError("--evaluate-outcomes requires --policy-dir and --policy-prefix")
        if not args.run_dir:
            raise ValueError("--evaluate-outcomes requires at least one --run-dir")
        policy = load_online_mode_gate_policy(out_dir=Path(args.policy_dir), prefix=str(args.policy_prefix))
        rolling_rows: list[dict[str, str]] = []
        for run_dir in args.run_dir:
            rolling_rows.extend(_read_csv(Path(run_dir) / "teacher_only_rolling_validation_window_summary_v29.csv"))
        outcome_artifacts = evaluate_online_mode_gate_serving_outcomes(
            rows,
            rolling_outcome_rows=rolling_rows,
            policy=policy,
            candidate_mode=str(args.candidate_mode),
            history_mode=str(args.history_mode),
            default_mode=str(args.fallback_mode),
        )
        outcome_artifacts.summary["features_csv"] = str(features_path)
        outcome_artifacts.summary["features_csv_sha256"] = _sha256_file(features_path)
        outcome_artifacts.summary["policy_dir"] = str(args.policy_dir)
        outcome_artifacts.summary["policy_prefix"] = str(args.policy_prefix)
        outcome_artifacts.summary["run_dirs"] = list(args.run_dir)
        write_online_mode_gate_outcome_artifacts(
            out_dir=Path(args.out_dir),
            prefix=str(args.prefix),
            artifacts=outcome_artifacts,
            overwrite=bool(args.overwrite),
        )
        return

    artifacts = build_online_mode_gate_artifacts(
        rows,
        label_column=args.label_column,
        heldout_tail_windows=args.heldout_tail_windows,
        heldout_start_window=args.heldout_start_window,
        min_confidence=args.min_confidence,
        fallback_mode=args.fallback_mode,
    )
    artifacts.summary["features_csv"] = str(features_path)
    artifacts.summary["features_csv_sha256"] = _sha256_file(features_path)
    write_online_mode_gate_artifacts(
        out_dir=Path(args.out_dir),
        prefix=str(args.prefix),
        artifacts=artifacts,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
