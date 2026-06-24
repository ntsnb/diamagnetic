from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

from train.analyze_stage3_v13_failure_diagnostics import (
    build_alpha_candidate_stability,
    build_split_generalization_diagnostics,
    build_window_failure_diagnostics,
    write_csv_no_overwrite,
)


def _summary_row(
    *,
    feature_mode: str,
    mode: str,
    window: int,
    mse_gain: float,
    mae_gain: float,
    sign: float,
) -> dict[str, str]:
    return {
        "feature_mode": feature_mode,
        "mode": mode,
        "validation_window_id": str(window),
        "split": f"validation_window_{window}",
        "teacher_mse_gain": str(mse_gain),
        "teacher_mae_gain": str(mae_gain),
        "teacher_sign_agreement_rate": str(sign),
    }


def test_window_failure_diagnostics_separate_attribution_and_real_failures() -> None:
    rows = [
        _summary_row(
            feature_mode="fluctuation_plus_history",
            mode="real",
            window=1,
            mse_gain=0.522514,
            mae_gain=0.324470,
            sign=0.763889,
        ),
        _summary_row(
            feature_mode="fluctuation_plus_history",
            mode="zero",
            window=1,
            mse_gain=0.520657,
            mae_gain=0.311043,
            sign=0.854167,
        ),
        _summary_row(
            feature_mode="fluctuation_plus_history",
            mode="real",
            window=3,
            mse_gain=0.011052,
            mae_gain=-0.008035,
            sign=0.506944,
        ),
        _summary_row(
            feature_mode="fluctuation_plus_history",
            mode="zero",
            window=3,
            mse_gain=-0.028928,
            mae_gain=-0.002034,
            sign=0.538194,
        ),
    ]
    partial_rows = [
        {
            "validation_window_id": "1",
            "partial_mse_gain_vs_history": "0.301187",
            "partial_mae_gain_vs_history": "0.222338",
        },
        {
            "validation_window_id": "3",
            "partial_mse_gain_vs_history": "-0.034105",
            "partial_mae_gain_vs_history": "-0.022592",
        },
    ]

    diagnostics = build_window_failure_diagnostics(
        rows,
        partial_rows=partial_rows,
        feature_mode="fluctuation_plus_history",
        min_real_mse_gain=0.005,
        min_real_mae_gain=0.002,
        counterfactual_margin=0.002,
        min_sign_agreement=0.55,
        min_partial_mse_gain_vs_history=0.001,
        min_partial_mae_gain_vs_history=0.0005,
    )

    by_window = {row["validation_window_id"]: row for row in diagnostics}
    assert by_window["1"]["failure_family"] == "attribution"
    assert by_window["1"]["failure_reasons"] == "counterfactual"
    assert float(by_window["1"]["real_vs_best_counterfactual_mse_gain_margin"]) == pytest.approx(0.001857)
    assert by_window["3"]["failure_family"] == "real_behavior;incremental"
    assert "real_mae_gain" in by_window["3"]["failure_reasons"]
    assert "partial_mse_gain_vs_history" in by_window["3"]["failure_reasons"]


def test_split_generalization_diagnostics_reports_validation_to_test_reversal() -> None:
    validation_rows = [
        {"feature_mode": "fluctuation_plus_history", "mode": "real", "teacher_mse_gain": "0.224375", "teacher_mae_gain": "0.115998"},
        {"feature_mode": "fluctuation_plus_history", "mode": "zero", "teacher_mse_gain": "0.187673", "teacher_mae_gain": "0.100000"},
    ]
    test_rows = [
        {"feature_mode": "fluctuation_plus_history", "mode": "real", "teacher_mse_gain": "0.003228", "teacher_mae_gain": "0.009038"},
        {"feature_mode": "fluctuation_plus_history", "mode": "zero", "teacher_mse_gain": "0.008873", "teacher_mae_gain": "0.010000"},
    ]
    validation_partial = [{"partial_mse_gain_vs_history": "0.151253", "partial_mae_gain_vs_history": "0.097218"}]
    test_partial = [{"partial_mse_gain_vs_history": "-0.039385", "partial_mae_gain_vs_history": "-0.009873"}]

    diagnostics = build_split_generalization_diagnostics(
        validation_rows,
        test_rows,
        validation_partial_rows=validation_partial,
        test_partial_rows=test_partial,
        feature_mode="fluctuation_plus_history",
    )

    row = diagnostics[0]
    assert row["feature_mode"] == "fluctuation_plus_history"
    assert float(row["validation_to_test_real_mse_gain_change"]) == pytest.approx(-0.221147)
    assert row["test_best_counterfactual_mode"] == "zero"
    assert row["incremental_value_reversal"] == "True"


def test_alpha_candidate_stability_summarizes_per_output_blocks() -> None:
    alpha_rows = [
        {"ridge": "0.01", "output_idx": "0", "alpha": "0.5", "fold_idx": "0", "mse_gain": "0.1", "mae_gain": "0.03", "sign_agreement": "0.6", "block_pass": "True"},
        {"ridge": "0.01", "output_idx": "0", "alpha": "0.5", "fold_idx": "1", "mse_gain": "-0.2", "mae_gain": "-0.05", "sign_agreement": "0.4", "block_pass": "False"},
        {"ridge": "0.01", "output_idx": "0", "alpha": "0.0", "fold_idx": "0", "mse_gain": "0.0", "mae_gain": "0.0", "sign_agreement": "0.0", "block_pass": "True"},
        {"ridge": "0.01", "output_idx": "0", "alpha": "0.0", "fold_idx": "1", "mse_gain": "0.0", "mae_gain": "0.0", "sign_agreement": "0.0", "block_pass": "True"},
    ]

    diagnostics = build_alpha_candidate_stability(alpha_rows)

    by_alpha = {row["alpha"]: row for row in diagnostics}
    assert by_alpha["0.5"]["fold_count"] == "2"
    assert by_alpha["0.5"]["block_pass_rate"] == "0.500000"
    assert by_alpha["0.5"]["stability_warning"] == "negative_block_gain"
    assert by_alpha["0.0"]["stability_warning"] == ""


def test_write_csv_no_overwrite_refuses_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "diagnostic.csv"
    target.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        write_csv_no_overwrite(target, [{"a": "1"}], overwrite=False)

    assert target.read_text(encoding="utf-8") == "sentinel"


def test_direct_v13_failure_diagnostic_script_invocation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()

    def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    rolling_rows = [
        _summary_row(feature_mode="fluctuation_plus_history", mode="real", window=0, mse_gain=0.1, mae_gain=0.02, sign=0.7),
        _summary_row(feature_mode="fluctuation_plus_history", mode="zero", window=0, mse_gain=0.0, mae_gain=0.0, sign=0.0),
    ]
    split_rows = [
        {"feature_mode": "fluctuation_plus_history", "mode": "real", "teacher_mse_gain": "0.1", "teacher_mae_gain": "0.02"},
        {"feature_mode": "fluctuation_plus_history", "mode": "zero", "teacher_mse_gain": "0.0", "teacher_mae_gain": "0.0"},
    ]
    partial_rows = [{"validation_window_id": "0", "partial_mse_gain_vs_history": "0.1", "partial_mae_gain_vs_history": "0.02"}]
    alpha_rows = [
        {"ridge": "0.01", "output_idx": "0", "alpha": "0.0", "fold_idx": "0", "mse_gain": "0.0", "mae_gain": "0.0", "sign_agreement": "0.0", "block_pass": "True"}
    ]
    write_csv(run_dir / "teacher_only_rolling_validation_summary.csv", rolling_rows)
    write_csv(run_dir / "teacher_only_validation_summary.csv", split_rows)
    write_csv(run_dir / "teacher_only_test_summary.csv", split_rows)
    write_csv(run_dir / "teacher_only_rolling_partial_gain_vs_history.csv", partial_rows)
    write_csv(run_dir / "teacher_only_validation_partial_gain_vs_history.csv", [{"partial_mse_gain_vs_history": "0.1", "partial_mae_gain_vs_history": "0.02"}])
    write_csv(run_dir / "teacher_only_test_partial_gain_vs_history.csv", [{"partial_mse_gain_vs_history": "0.1", "partial_mae_gain_vs_history": "0.02"}])
    write_csv(run_dir / "v13_alpha_blocks_fh.csv", alpha_rows)

    result = subprocess.run(
        [
            sys.executable,
            "train/analyze_stage3_v13_failure_diagnostics.py",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "v13_window_failure_diagnostics.csv").exists()
    assert (out_dir / "v13_split_generalization_diagnostics.csv").exists()
    assert (out_dir / "v13_alpha_candidate_stability_fh.csv").exists()
