from __future__ import annotations

from typing import Any, Iterable


COUNTERFACTUAL_MODES = {"zero", "shuffled", "sign_flipped", "random_matched_norm", "amplified_2x", "amplified_5x"}


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _gain(row: dict[str, Any], base_key: str, stage3_key: str) -> float:
    base = _as_float(row, base_key)
    stage3 = _as_float(row, stage3_key)
    return (base - stage3) / base if base != 0.0 else 0.0


def evaluate_stage3_sensitivity_gate(
    rows: Iterable[dict[str, Any]],
    *,
    min_real_mse_gain: float = 0.0,
    min_real_mae_gain: float | None = None,
    counterfactual_margin: float = 0.0,
    max_real_reject_rate: float | None = None,
    require_nonzero_delta: bool = True,
    require_prediction_shift: bool = True,
) -> dict[str, Any]:
    by_mode = {str(row["mode"]): row for row in rows}
    if "real" not in by_mode:
        raise ValueError("Stage III sensitivity gate requires a real mode row")

    real = by_mode["real"]
    real_mse_gain = _gain(real, "online_base_mse", "online_stage3_mse")
    real_mae_gain = _gain(real, "online_base_mae", "online_stage3_mae")
    real_delta_norm = _as_float(real, "online_stage3_delta_norm_pre_clamp")
    real_pred_shift = _as_float(real, "online_stage3_pred_shift")
    real_shift_reject_count = float(real.get("online_stage3_reject_shift_count", 0.0))
    real_risk_reject_count = float(real.get("online_stage3_reject_risk_count", 0.0))
    real_missing_reject_count = float(real.get("online_stage3_reject_missing_drift_count", 0.0))
    real_reject_count = real_shift_reject_count + real_risk_reject_count + real_missing_reject_count
    real_batch_count = float(real.get("online_batch_count", 0.0))
    real_reject_rate = real_reject_count / real_batch_count if real_batch_count > 0.0 else 0.0
    real_shift_reject_rate = real_shift_reject_count / real_batch_count if real_batch_count > 0.0 else 0.0
    real_risk_reject_rate = real_risk_reject_count / real_batch_count if real_batch_count > 0.0 else 0.0
    real_missing_reject_rate = real_missing_reject_count / real_batch_count if real_batch_count > 0.0 else 0.0

    counterfactual_rows = {
        mode: row for mode, row in by_mode.items() if mode in COUNTERFACTUAL_MODES
    }
    best_mode = ""
    best_gain = float("-inf")
    for mode, row in counterfactual_rows.items():
        gain = _gain(row, "online_base_mse", "online_stage3_mse")
        if gain > best_gain:
            best_gain = gain
            best_mode = mode
    if not counterfactual_rows:
        best_gain = float("nan")

    failure_reasons: list[str] = []
    if real_mse_gain < min_real_mse_gain:
        failure_reasons.append("real_mse_gain")
    if min_real_mae_gain is not None and real_mae_gain < min_real_mae_gain:
        failure_reasons.append("real_mae_gain")
    if counterfactual_rows and real_mse_gain <= best_gain + counterfactual_margin:
        failure_reasons.append("counterfactual")
    if max_real_reject_rate is not None and real_reject_rate > max_real_reject_rate:
        failure_reasons.append("reject_rate")
    if require_nonzero_delta and real_delta_norm <= 0.0:
        failure_reasons.append("delta_norm")
    if require_prediction_shift and real_pred_shift <= 0.0:
        failure_reasons.append("prediction_shift")

    return {
        "claim_gate_pass": not failure_reasons,
        "failure_reasons": ";".join(failure_reasons),
        "real_mse_gain": real_mse_gain,
        "real_mae_gain": real_mae_gain,
        "best_counterfactual_mode": best_mode,
        "best_counterfactual_mse_gain": best_gain,
        "real_vs_best_counterfactual_mse_gain_margin": (
            real_mse_gain - best_gain if counterfactual_rows else float("nan")
        ),
        "real_delta_norm": real_delta_norm,
        "real_prediction_shift": real_pred_shift,
        "real_reject_rate": real_reject_rate,
        "real_shift_reject_rate": real_shift_reject_rate,
        "real_risk_reject_rate": real_risk_reject_rate,
        "real_missing_drift_reject_rate": real_missing_reject_rate,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def evaluate_rolling_stage3_sensitivity_gates(
    window_gate_rows: Iterable[dict[str, Any]],
    *,
    min_window_pass_rate: float = 1.0,
) -> dict[str, Any]:
    rows = list(window_gate_rows)
    if not rows:
        raise ValueError("rolling Stage III gate requires at least one validation window")

    pass_rows = [row for row in rows if _as_bool(row.get("claim_gate_pass", False))]
    failed_rows = [row for row in rows if not _as_bool(row.get("claim_gate_pass", False))]
    pass_rate = len(pass_rows) / len(rows)
    worst_mse_gain = min(float(row.get("real_mse_gain", 0.0)) for row in rows)
    worst_mae_gain = min(float(row.get("real_mae_gain", 0.0)) for row in rows)
    best_counterfactual_gain = max(float(row.get("best_counterfactual_mse_gain", float("-inf"))) for row in rows)
    worst_counterfactual_margin = min(
        float(row.get("real_vs_best_counterfactual_mse_gain_margin", 0.0)) for row in rows
    )
    max_real_reject_rate = max(float(row.get("real_reject_rate", 0.0)) for row in rows)

    failure_reasons: list[str] = []
    if pass_rate < min_window_pass_rate:
        failure_reasons.append("window_pass_rate")
    for row in failed_rows:
        for reason in str(row.get("failure_reasons", "")).split(";"):
            if reason and reason not in failure_reasons:
                failure_reasons.append(reason)

    failed_ids = [
        str(row.get("validation_window_id", row.get("split", "")))
        for row in failed_rows
    ]
    return {
        "rolling_claim_gate_pass": not failure_reasons,
        "rolling_failure_reasons": ";".join(failure_reasons),
        "validation_window_count": len(rows),
        "validation_window_pass_count": len(pass_rows),
        "validation_window_pass_rate": pass_rate,
        "failed_validation_windows": ";".join(failed_ids),
        "worst_window_real_mse_gain": worst_mse_gain,
        "worst_window_real_mae_gain": worst_mae_gain,
        "best_window_counterfactual_mse_gain": best_counterfactual_gain,
        "worst_window_real_vs_best_counterfactual_mse_gain_margin": worst_counterfactual_margin,
        "max_window_real_reject_rate": max_real_reject_rate,
    }
