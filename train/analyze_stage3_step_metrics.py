from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.stage3_sensitivity_gate_v2 import COUNTERFACTUAL_MODES


PAIR_CONTEXT_COLUMNS = ("run_id", "dataset", "seed", "split", "validation_window_id", "step_index")
SUMMARY_CONTEXT_COLUMNS = ("run_id", "dataset", "seed", "split", "validation_window_id", "mode")
STEP_NUMERIC_COLUMNS = (
    "used",
    "base_mse",
    "base_mae",
    "candidate_mse",
    "candidate_mae",
    "would_apply_blended_mse",
    "would_apply_blended_mae",
    "final_mse",
    "final_mae",
    "pred_shift",
    "risk_score",
    "drift_norm",
    "delta_norm_pre_clamp",
    "delta_norm_post_clamp",
    "clamp_fraction",
)


def _as_numeric(df: pd.DataFrame, columns: tuple[str, ...] = STEP_NUMERIC_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _pair_context_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in PAIR_CONTEXT_COLUMNS if column in df.columns]
    if "step_index" not in columns:
        raise ValueError("step metrics must include step_index")
    return columns


def _mean_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _sum_bool(series: pd.Series) -> int:
    return int(series.astype(bool).sum())


def _paired_with_reference(step_metrics: pd.DataFrame, reference_mode: str) -> pd.DataFrame:
    df = _as_numeric(step_metrics)
    if "mode" not in df.columns:
        raise ValueError("step metrics must include mode")
    if reference_mode not in set(df["mode"].astype(str)):
        raise ValueError(f"reference mode not found in step metrics: {reference_mode}")

    context_columns = _pair_context_columns(df)
    reference = df[df["mode"].astype(str) == reference_mode].copy()
    others = df[df["mode"].astype(str) != reference_mode].copy()
    return others.merge(
        reference,
        on=context_columns,
        how="inner",
        suffixes=("", "_reference"),
    )


def build_paired_mode_differences(
    step_metrics: pd.DataFrame,
    *,
    reference_mode: str = "real",
) -> pd.DataFrame:
    """Build one row per step and non-reference mode against the reference mode."""

    paired = _paired_with_reference(step_metrics, reference_mode)
    rows: list[dict[str, Any]] = []
    context_columns = _pair_context_columns(step_metrics)
    difference_columns = (
        "candidate_mse",
        "would_apply_blended_mse",
        "final_mse",
        "candidate_mae",
        "would_apply_blended_mae",
        "final_mae",
        "pred_shift",
        "risk_score",
        "drift_norm",
        "delta_norm_pre_clamp",
        "delta_norm_post_clamp",
        "clamp_fraction",
    )
    for _, row in paired.iterrows():
        out = {column: row[column] for column in context_columns if column in row}
        out["reference_mode"] = reference_mode
        out["mode"] = row["mode"]
        out["used"] = float(row.get("used", 0.0))
        out["reference_used"] = float(row.get("used_reference", 0.0))
        out["mode_used_reference_rejected"] = float(out["used"] > 0.5 and out["reference_used"] <= 0.5)
        out["reference_used_mode_rejected"] = float(out["reference_used"] > 0.5 and out["used"] <= 0.5)
        for column in difference_columns:
            if column in paired.columns and f"{column}_reference" in paired.columns:
                out[column] = row[column]
                out[f"{column}_reference"] = row[f"{column}_reference"]
                out[f"{column}_minus_reference"] = row[column] - row[f"{column}_reference"]
        if "final_mse_minus_reference" in out:
            out["mode_final_mse_better_than_reference"] = float(out["final_mse_minus_reference"] < 0.0)
        if "would_apply_blended_mse_minus_reference" in out:
            out["mode_would_apply_blended_mse_better_than_reference"] = float(
                out["would_apply_blended_mse_minus_reference"] < 0.0
            )
        rows.append(out)
    return pd.DataFrame(rows)


def build_common_accept_mask_summary(
    step_metrics: pd.DataFrame,
    *,
    reference_mode: str = "real",
) -> pd.DataFrame:
    """Summarize per-mode margins under all, common-accepted, and matched accept masks."""

    paired = build_paired_mode_differences(step_metrics, reference_mode=reference_mode)
    if paired.empty:
        return pd.DataFrame()
    group_columns = [column for column in SUMMARY_CONTEXT_COLUMNS if column in paired.columns]
    if "mode" not in group_columns:
        group_columns.append("mode")
    rows: list[dict[str, Any]] = []
    for group_key, group in paired.groupby(group_columns, sort=False, dropna=False):
        if len(group_columns) == 1:
            group_key = (group_key,)
        context = dict(zip(group_columns, group_key, strict=True))
        mode_used = group["used"] > 0.5
        ref_used = group["reference_used"] > 0.5
        both_used = mode_used & ref_used
        mode_only = mode_used & ~ref_used
        ref_only = ref_used & ~mode_used
        row: dict[str, Any] = {
            "reference_mode": reference_mode,
            **context,
            "paired_step_count": int(len(group)),
            "both_used_step_count": _sum_bool(both_used),
            "reference_used_step_count": _sum_bool(ref_used),
            "mode_used_step_count": _sum_bool(mode_used),
            "mode_used_reference_rejected_rate": _mean_or_nan(mode_only.astype(float)),
            "reference_used_mode_rejected_rate": _mean_or_nan(ref_only.astype(float)),
        }
        for column in (
            "final_mse_minus_reference",
            "final_mae_minus_reference",
            "would_apply_blended_mse_minus_reference",
            "would_apply_blended_mae_minus_reference",
            "pred_shift_minus_reference",
            "risk_score_minus_reference",
            "delta_norm_pre_clamp_minus_reference",
            "drift_norm_minus_reference",
        ):
            if column not in group.columns:
                continue
            row[f"all_{column}_mean"] = _mean_or_nan(group[column].dropna())
            row[f"both_used_{column}_mean"] = _mean_or_nan(group.loc[both_used, column].dropna())
            row[f"reference_used_{column}_mean"] = _mean_or_nan(group.loc[ref_used, column].dropna())
            row[f"mode_used_{column}_mean"] = _mean_or_nan(group.loc[mode_used, column].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def _gain(row: pd.Series, base_column: str, stage3_column: str) -> float:
    base = float(row[base_column])
    value = float(row[stage3_column])
    return (base - value) / base if base != 0.0 else 0.0


def build_validation_window_failure_breakdown(
    sensitivity_rows: pd.DataFrame,
    gate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per validation window with real-vs-counterfactual margins."""

    if "mode" not in sensitivity_rows.columns:
        raise ValueError("rolling sensitivity rows must include mode")
    if "validation_window_id" not in sensitivity_rows.columns:
        raise ValueError("rolling sensitivity rows must include validation_window_id")
    sensitivity_rows = sensitivity_rows.copy()
    gate_rows = gate_rows.copy()
    sensitivity_rows["validation_window_id"] = pd.to_numeric(
        sensitivity_rows["validation_window_id"], errors="coerce"
    ).astype("Int64")
    if not gate_rows.empty and "validation_window_id" in gate_rows.columns:
        gate_rows["validation_window_id"] = pd.to_numeric(gate_rows["validation_window_id"], errors="coerce").astype(
            "Int64"
        )

    rows: list[dict[str, Any]] = []
    for window_id, window in sensitivity_rows.groupby("validation_window_id", sort=True):
        by_mode = {str(row["mode"]): row for _, row in window.iterrows()}
        if "real" not in by_mode:
            continue
        real = by_mode["real"]
        real_mse_gain = _gain(real, "online_base_mse", "online_stage3_mse")
        real_mae_gain = _gain(real, "online_base_mae", "online_stage3_mae")
        best_mode = ""
        best_gain = float("-inf")
        best_row: pd.Series | None = None
        for mode, row in by_mode.items():
            if mode not in COUNTERFACTUAL_MODES:
                continue
            gain = _gain(row, "online_base_mse", "online_stage3_mse")
            if gain > best_gain:
                best_mode = mode
                best_gain = gain
                best_row = row
        gate_match = gate_rows[gate_rows.get("validation_window_id", pd.Series(dtype="Int64")) == window_id]
        gate = gate_match.iloc[0] if not gate_match.empty else pd.Series(dtype=object)
        rows.append(
            {
                "validation_window_id": int(window_id),
                "split": real.get("split", f"validation_window_{int(window_id)}"),
                "claim_gate_pass": gate.get("claim_gate_pass", ""),
                "failure_reasons": gate.get("failure_reasons", ""),
                "real_mse_gain": real_mse_gain,
                "real_mae_gain": real_mae_gain,
                "real_online_stage3_mse": float(real["online_stage3_mse"]),
                "real_online_stage3_mae": float(real["online_stage3_mae"]),
                "real_use_rate": float(real.get("online_stage3_use_rate", float("nan"))),
                "real_reject_rate": float(gate.get("real_reject_rate", float("nan"))),
                "best_counterfactual_mode": best_mode,
                "best_counterfactual_mse_gain": best_gain,
                "real_vs_best_counterfactual_mse_gain_margin": real_mse_gain - best_gain,
                "best_counterfactual_stage3_mse": (
                    float(best_row["online_stage3_mse"]) if best_row is not None else float("nan")
                ),
                "best_counterfactual_use_rate": (
                    float(best_row.get("online_stage3_use_rate", float("nan")))
                    if best_row is not None
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _preflight_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "; ".join(str(path) for path in existing)
        raise FileExistsError(f"Refuse to overwrite existing Stage III step-metrics analysis files: {joined}")


def write_analysis_outputs(
    *,
    output_dir: Path,
    common_accept_mask_summary: pd.DataFrame,
    paired_mode_differences: pd.DataFrame,
    validation_window_failure_breakdown: pd.DataFrame,
    overwrite: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "common_accept_mask_summary": output_dir / "common_accept_mask_summary.csv",
        "paired_mode_differences": output_dir / "paired_mode_differences.csv",
        "validation_window_failure_breakdown": output_dir / "validation_window_failure_breakdown.csv",
    }
    _preflight_no_overwrite(list(outputs.values()), overwrite=overwrite)
    common_accept_mask_summary.to_csv(outputs["common_accept_mask_summary"], index=False)
    paired_mode_differences.to_csv(outputs["paired_mode_differences"], index=False)
    validation_window_failure_breakdown.to_csv(outputs["validation_window_failure_breakdown"], index=False)
    return {name: str(path) for name, path in outputs.items()}


def analyze_run_dir(run_dir: Path, *, overwrite: bool = False) -> dict[str, str]:
    step_path = run_dir / "step_metrics.csv"
    rolling_summary_path = run_dir / "rolling_validation_sensitivity_summary.csv"
    rolling_gate_path = run_dir / "rolling_validation_claim_gate.csv"
    if not step_path.exists():
        raise FileNotFoundError(f"missing step metrics: {step_path}")
    if not rolling_summary_path.exists():
        raise FileNotFoundError(f"missing rolling validation sensitivity summary: {rolling_summary_path}")
    if not rolling_gate_path.exists():
        raise FileNotFoundError(f"missing rolling validation claim gate: {rolling_gate_path}")

    step_metrics = pd.read_csv(step_path)
    rolling_summary = pd.read_csv(rolling_summary_path)
    rolling_gate = pd.read_csv(rolling_gate_path)
    outputs = write_analysis_outputs(
        output_dir=run_dir,
        common_accept_mask_summary=build_common_accept_mask_summary(step_metrics),
        paired_mode_differences=build_paired_mode_differences(step_metrics),
        validation_window_failure_breakdown=build_validation_window_failure_breakdown(rolling_summary, rolling_gate),
        overwrite=overwrite,
    )

    rolling_step_path = run_dir / "rolling_validation_step_metrics.csv"
    if rolling_step_path.exists():
        rolling_step_metrics = pd.read_csv(rolling_step_path)
        rolling_outputs = {
            "rolling_common_accept_mask_summary": run_dir / "rolling_common_accept_mask_summary.csv",
            "rolling_paired_mode_differences": run_dir / "rolling_paired_mode_differences.csv",
        }
        _preflight_no_overwrite(list(rolling_outputs.values()), overwrite=overwrite)
        build_common_accept_mask_summary(rolling_step_metrics).to_csv(
            rolling_outputs["rolling_common_accept_mask_summary"],
            index=False,
        )
        build_paired_mode_differences(rolling_step_metrics).to_csv(
            rolling_outputs["rolling_paired_mode_differences"],
            index=False,
        )
        outputs.update({name: str(path) for name, path in rolling_outputs.items()})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage III step_metrics.csv diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Stage III run directory containing step_metrics.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing derived analysis CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze_run_dir(Path(args.run_dir), overwrite=bool(args.overwrite))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
