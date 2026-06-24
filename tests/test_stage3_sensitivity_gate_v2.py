from __future__ import annotations

import pytest

from train.stage3_sensitivity_gate_v2 import (
    evaluate_rolling_stage3_sensitivity_gates,
    evaluate_stage3_sensitivity_gate,
)


def test_sensitivity_gate_passes_only_when_real_beats_base_and_counterfactuals():
    rows = [
        {
            "mode": "real",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.80,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.82,
            "online_stage3_delta_norm_pre_clamp": 0.4,
            "online_stage3_pred_shift": 0.1,
        },
        {
            "mode": "zero",
            "online_base_mse": 1.0,
            "online_stage3_mse": 1.0,
            "online_base_mae": 1.0,
            "online_stage3_mae": 1.0,
            "online_stage3_delta_norm_pre_clamp": 0.0,
            "online_stage3_pred_shift": 0.0,
        },
        {
            "mode": "shuffled",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.95,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.96,
            "online_stage3_delta_norm_pre_clamp": 0.2,
            "online_stage3_pred_shift": 0.05,
        },
        {
            "mode": "sign_flipped",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.97,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.98,
            "online_stage3_delta_norm_pre_clamp": 0.4,
            "online_stage3_pred_shift": 0.09,
        },
    ]

    result = evaluate_stage3_sensitivity_gate(rows, min_real_mse_gain=0.01, counterfactual_margin=0.01)

    assert result["claim_gate_pass"] is True
    assert result["real_mse_gain"] > 0.0
    assert result["best_counterfactual_mode"] == "shuffled"


def test_sensitivity_gate_fails_when_sign_flipped_beats_real():
    rows = [
        {
            "mode": "real",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.99,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.99,
            "online_stage3_delta_norm_pre_clamp": 0.4,
            "online_stage3_pred_shift": 0.1,
        },
        {
            "mode": "sign_flipped",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.90,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.91,
            "online_stage3_delta_norm_pre_clamp": 0.4,
            "online_stage3_pred_shift": 0.1,
        },
    ]

    result = evaluate_stage3_sensitivity_gate(rows, min_real_mse_gain=0.01, counterfactual_margin=0.01)

    assert result["claim_gate_pass"] is False
    assert result["best_counterfactual_mode"] == "sign_flipped"
    assert "counterfactual" in result["failure_reasons"]


def test_sensitivity_gate_can_require_mae_gain_and_reject_rate_budget():
    rows = [
        {
            "mode": "real",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.8,
            "online_base_mae": 1.0,
            "online_stage3_mae": 1.01,
            "online_stage3_delta_norm_pre_clamp": 0.3,
            "online_stage3_pred_shift": 0.1,
            "online_stage3_reject_shift_count": 4.0,
            "online_batch_count": 10.0,
        },
        {
            "mode": "zero",
            "online_base_mse": 1.0,
            "online_stage3_mse": 1.0,
            "online_base_mae": 1.0,
            "online_stage3_mae": 1.0,
            "online_stage3_delta_norm_pre_clamp": 0.0,
            "online_stage3_pred_shift": 0.0,
            "online_stage3_reject_shift_count": 0.0,
            "online_batch_count": 10.0,
        },
    ]

    result = evaluate_stage3_sensitivity_gate(
        rows,
        min_real_mse_gain=0.01,
        min_real_mae_gain=0.0,
        max_real_reject_rate=0.2,
    )

    assert result["claim_gate_pass"] is False
    assert "real_mae_gain" in result["failure_reasons"]
    assert "reject_rate" in result["failure_reasons"]


def test_sensitivity_gate_counts_risk_rejections_in_total_reject_rate():
    rows = [
        {
            "mode": "real",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.8,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.8,
            "online_stage3_delta_norm_pre_clamp": 0.3,
            "online_stage3_pred_shift": 0.1,
            "online_stage3_reject_shift_count": 0.0,
            "online_stage3_reject_risk_count": 4.0,
            "online_stage3_reject_missing_drift_count": 1.0,
            "online_batch_count": 10.0,
        },
        {
            "mode": "zero",
            "online_base_mse": 1.0,
            "online_stage3_mse": 1.0,
            "online_base_mae": 1.0,
            "online_stage3_mae": 1.0,
            "online_stage3_delta_norm_pre_clamp": 0.0,
            "online_stage3_pred_shift": 0.0,
            "online_batch_count": 10.0,
        },
    ]

    result = evaluate_stage3_sensitivity_gate(
        rows,
        min_real_mse_gain=0.01,
        min_real_mae_gain=0.01,
        max_real_reject_rate=0.2,
    )

    assert result["claim_gate_pass"] is False
    assert result["real_reject_rate"] == 0.5
    assert result["real_shift_reject_rate"] == 0.0
    assert result["real_risk_reject_rate"] == 0.4
    assert result["real_missing_drift_reject_rate"] == 0.1
    assert "reject_rate" in result["failure_reasons"]


def test_rolling_sensitivity_gate_requires_stable_validation_windows():
    window_gates = [
        {
            "validation_window_id": 0,
            "claim_gate_pass": True,
            "failure_reasons": "",
            "real_mse_gain": 0.02,
            "real_mae_gain": 0.01,
            "best_counterfactual_mse_gain": 0.0,
        },
        {
            "validation_window_id": 1,
            "claim_gate_pass": False,
            "failure_reasons": "counterfactual",
            "real_mse_gain": -0.01,
            "real_mae_gain": -0.02,
            "best_counterfactual_mse_gain": 0.03,
        },
    ]

    result = evaluate_rolling_stage3_sensitivity_gates(window_gates, min_window_pass_rate=1.0)

    assert result["rolling_claim_gate_pass"] is False
    assert result["validation_window_count"] == 2
    assert result["validation_window_pass_count"] == 1
    assert result["validation_window_pass_rate"] == 0.5
    assert result["failed_validation_windows"] == "1"
    assert result["worst_window_real_mse_gain"] == -0.01
    assert "counterfactual" in result["rolling_failure_reasons"]


def test_rolling_gate_reports_worst_margin_and_max_reject_rate():
    result = evaluate_rolling_stage3_sensitivity_gates(
        [
            {
                "claim_gate_pass": False,
                "failure_reasons": "counterfactual",
                "validation_window_id": 0,
                "real_mse_gain": 0.04,
                "real_mae_gain": 0.03,
                "best_counterfactual_mse_gain": 0.041,
                "real_vs_best_counterfactual_mse_gain_margin": -0.001,
                "real_reject_rate": 0.05,
            },
            {
                "claim_gate_pass": True,
                "failure_reasons": "",
                "validation_window_id": 1,
                "real_mse_gain": 0.03,
                "real_mae_gain": 0.02,
                "best_counterfactual_mse_gain": 0.027,
                "real_vs_best_counterfactual_mse_gain_margin": 0.003,
                "real_reject_rate": 0.15,
            },
        ],
        min_window_pass_rate=0.0,
    )

    assert result["worst_window_real_vs_best_counterfactual_mse_gain_margin"] == pytest.approx(-0.001)
    assert result["max_window_real_reject_rate"] == pytest.approx(0.15)
