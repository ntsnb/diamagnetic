from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from train.analyze_stage3_step_metrics import (
    build_common_accept_mask_summary,
    build_paired_mode_differences,
    build_validation_window_failure_breakdown,
    write_analysis_outputs,
)


def _toy_step_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "test",
                "mode": "real",
                "step_index": 0,
                "used": 1.0,
                "final_mse": 0.8,
                "would_apply_blended_mse": 0.8,
                "pred_shift": 0.1,
                "risk_score": 0.6,
                "delta_norm_pre_clamp": 1.0,
                "drift_norm": 2.0,
            },
            {
                "split": "test",
                "mode": "amplified_2x",
                "step_index": 0,
                "used": 1.0,
                "final_mse": 0.7,
                "would_apply_blended_mse": 0.7,
                "pred_shift": 0.11,
                "risk_score": 0.8,
                "delta_norm_pre_clamp": 1.01,
                "drift_norm": 4.0,
            },
            {
                "split": "test",
                "mode": "real",
                "step_index": 1,
                "used": 1.0,
                "final_mse": 0.5,
                "would_apply_blended_mse": 0.5,
                "pred_shift": 0.2,
                "risk_score": 0.7,
                "delta_norm_pre_clamp": 1.2,
                "drift_norm": 3.0,
            },
            {
                "split": "test",
                "mode": "amplified_2x",
                "step_index": 1,
                "used": 0.0,
                "final_mse": 0.9,
                "would_apply_blended_mse": 0.4,
                "pred_shift": 0.21,
                "risk_score": 0.9,
                "delta_norm_pre_clamp": 1.2,
                "drift_norm": 6.0,
            },
        ]
    )


def test_paired_mode_differences_preserve_stepwise_real_counterfactual_margins():
    paired = build_paired_mode_differences(_toy_step_rows(), reference_mode="real")

    assert list(paired["mode"]) == ["amplified_2x", "amplified_2x"]
    assert paired.loc[0, "final_mse_minus_reference"] == pytest.approx(-0.1)
    assert paired.loc[1, "would_apply_blended_mse_minus_reference"] == pytest.approx(-0.1)
    assert paired.loc[1, "mode_used_reference_rejected"] == 0.0
    assert paired.loc[1, "reference_used_mode_rejected"] == 1.0
    assert paired.loc[0, "risk_score_minus_reference"] == pytest.approx(0.2)


def test_common_accept_mask_summary_reports_gate_independent_would_blend_margin():
    summary = build_common_accept_mask_summary(_toy_step_rows(), reference_mode="real")

    row = summary.iloc[0]
    assert row["mode"] == "amplified_2x"
    assert row["paired_step_count"] == 2
    assert row["both_used_step_count"] == 1
    assert row["reference_used_step_count"] == 2
    assert row["mode_used_step_count"] == 1
    assert row["all_final_mse_minus_reference_mean"] == pytest.approx(0.15)
    assert row["all_would_apply_blended_mse_minus_reference_mean"] == pytest.approx(-0.1)
    assert row["reference_used_would_apply_blended_mse_minus_reference_mean"] == pytest.approx(-0.1)
    assert row["mode_used_reference_rejected_rate"] == pytest.approx(0.0)
    assert row["reference_used_mode_rejected_rate"] == pytest.approx(0.5)


def test_common_accept_mask_summary_keeps_validation_windows_separate_when_present():
    rows = _toy_step_rows()
    duplicate = rows.copy()
    rows["split"] = "validation_window_0"
    duplicate["split"] = "validation_window_1"
    duplicate["final_mse"] = duplicate["final_mse"] + 1.0
    summary = build_common_accept_mask_summary(pd.concat([rows, duplicate], ignore_index=True), reference_mode="real")

    assert list(summary["split"]) == ["validation_window_0", "validation_window_1"]
    assert list(summary["mode"]) == ["amplified_2x", "amplified_2x"]
    assert summary.loc[0, "all_final_mse_minus_reference_mean"] == pytest.approx(0.15)
    assert summary.loc[1, "all_final_mse_minus_reference_mean"] == pytest.approx(0.15)


def test_validation_window_breakdown_reports_best_counterfactual_and_failure_reason():
    sensitivity_rows = pd.DataFrame(
        [
            {
                "validation_window_id": 0,
                "split": "validation_window_0",
                "mode": "real",
                "online_base_mse": 1.0,
                "online_stage3_mse": 0.8,
                "online_base_mae": 1.0,
                "online_stage3_mae": 0.9,
                "online_stage3_use_rate": 0.9,
            },
            {
                "validation_window_id": 0,
                "split": "validation_window_0",
                "mode": "amplified_2x",
                "online_base_mse": 1.0,
                "online_stage3_mse": 0.7,
                "online_base_mae": 1.0,
                "online_stage3_mae": 0.8,
                "online_stage3_use_rate": 0.95,
            },
            {
                "validation_window_id": 0,
                "split": "validation_window_0",
                "mode": "zero",
                "online_base_mse": 1.0,
                "online_stage3_mse": 1.0,
                "online_base_mae": 1.0,
                "online_stage3_mae": 1.0,
                "online_stage3_use_rate": 0.0,
            },
        ]
    )
    gate_rows = pd.DataFrame(
        [
            {
                "validation_window_id": 0,
                "claim_gate_pass": False,
                "failure_reasons": "counterfactual",
                "real_reject_rate": 0.1,
            }
        ]
    )

    breakdown = build_validation_window_failure_breakdown(sensitivity_rows, gate_rows)

    assert breakdown.loc[0, "validation_window_id"] == 0
    assert breakdown.loc[0, "best_counterfactual_mode"] == "amplified_2x"
    assert breakdown.loc[0, "real_mse_gain"] == pytest.approx(0.2)
    assert breakdown.loc[0, "best_counterfactual_mse_gain"] == pytest.approx(0.3)
    assert breakdown.loc[0, "real_vs_best_counterfactual_mse_gain_margin"] == pytest.approx(-0.1)
    assert breakdown.loc[0, "failure_reasons"] == "counterfactual"


def test_write_analysis_outputs_refuses_to_overwrite_existing_files(tmp_path: Path):
    existing = tmp_path / "common_accept_mask_summary.csv"
    existing.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        write_analysis_outputs(
            output_dir=tmp_path,
            common_accept_mask_summary=pd.DataFrame([{"mode": "x"}]),
            paired_mode_differences=pd.DataFrame([{"mode": "x"}]),
            validation_window_failure_breakdown=pd.DataFrame([{"window": 0}]),
            overwrite=False,
        )

    assert existing.read_text(encoding="utf-8") == "sentinel"


def test_direct_script_invocation_can_import_project_package(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _toy_step_rows().to_csv(run_dir / "step_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "validation_window_id": 0,
                "split": "validation_window_0",
                "mode": "real",
                "online_base_mse": 1.0,
                "online_stage3_mse": 0.9,
                "online_base_mae": 1.0,
                "online_stage3_mae": 0.95,
                "online_stage3_use_rate": 1.0,
            },
            {
                "validation_window_id": 0,
                "split": "validation_window_0",
                "mode": "zero",
                "online_base_mse": 1.0,
                "online_stage3_mse": 1.0,
                "online_base_mae": 1.0,
                "online_stage3_mae": 1.0,
                "online_stage3_use_rate": 0.0,
            },
        ]
    ).to_csv(run_dir / "rolling_validation_sensitivity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "validation_window_id": 0,
                "claim_gate_pass": True,
                "failure_reasons": "",
                "real_reject_rate": 0.0,
            }
        ]
    ).to_csv(run_dir / "rolling_validation_claim_gate.csv", index=False)

    result = subprocess.run(
        [
            sys.executable,
            "train/analyze_stage3_step_metrics.py",
            "--run-dir",
            str(run_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (run_dir / "common_accept_mask_summary.csv").exists()
