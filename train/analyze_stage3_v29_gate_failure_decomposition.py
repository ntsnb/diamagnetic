from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "stage3_v29_gate_failure_decomposition_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _split_semicolon(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _write_text_no_overwrite(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _write_json_no_overwrite(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    _write_text_no_overwrite(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def _write_csv_no_overwrite(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(str(item) for item in value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _preflight_blocker_counts(preflight_report: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in preflight_report.get("candidate_reports", []):
        counts.update(str(blocker) for blocker in candidate.get("blockers", []))
    return counts


def _window_failure_counts(window_rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in window_rows:
        counts.update(_split_semicolon(row.get("failure_reasons")))
    return counts


def _window_summary_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[str(row.get("validation_window_id", ""))].append(row)

    summaries: list[dict[str, Any]] = []
    for window_id in sorted(grouped, key=lambda value: _to_int(value, default=999999)):
        rows = grouped[window_id]
        failure_counts: Counter[str] = Counter()
        counterfactual_negative = 0
        partial_mse_floor = 0
        partial_mae_floor = 0
        real_best = 0
        for row in rows:
            reasons = _split_semicolon(row.get("failure_reasons"))
            failure_counts.update(reasons)
            if _to_float(row.get("counterfactual_margin")) < 0.0:
                counterfactual_negative += 1
            if "partial_mse_below_gate_floor" in reasons:
                partial_mse_floor += 1
            if "partial_mae_below_gate_floor" in reasons:
                partial_mae_floor += 1
            if _truthy(row.get("real_is_best_mode")):
                real_best += 1

        summaries.append(
            {
                "validation_window_id": window_id,
                "row_count": len(rows),
                "partial_mse_floor_failures": partial_mse_floor,
                "partial_mae_floor_failures": partial_mae_floor,
                "counterfactual_negative_rows": counterfactual_negative,
                "real_best_rows": real_best,
                "top_failure_reasons": ";".join(reason for reason, _ in failure_counts.most_common(4)),
            }
        )
    return summaries


def _candidate_diagnosis(row: dict[str, Any], *, min_partial_floor: float) -> dict[str, Any]:
    min_partial_mse = _to_float(row.get("min_partial_mse_gain_vs_history"))
    min_partial_mae = _to_float(row.get("min_partial_mae_gain_vs_history"))
    validation_partial_mse = _to_float(row.get("validation_partial_mse_gain_vs_history"))
    counterfactual_margin = _to_float(row.get("counterfactual_margin"))
    window0_fh = _to_float(row.get("window0_fh_mse_gain"))
    target_gate_pass = _truthy(row.get("target_gate_pass"))
    generator_unlocked = _truthy(row.get("generator_unlocked"))

    tags: list[str] = []
    if window0_fh >= 0.0 and (min_partial_mse < min_partial_floor or min_partial_mae < min_partial_floor):
        tags.append("window0_nonharm_but_partial_floor_fails")
    if validation_partial_mse > 0.0 and min_partial_mse < min_partial_floor:
        tags.append("aggregate_partial_positive_but_min_window_fails")
    if counterfactual_margin < 0.0:
        tags.append("counterfactual_margin_negative")
    if not generator_unlocked:
        tags.append("generator_locked")

    return {
        "strict_rank": _to_int(row.get("strict_rank")),
        "run_id": row.get("run_id", ""),
        "selection_label": row.get("selection_label", ""),
        "target_gate_pass": target_gate_pass,
        "generator_unlocked": generator_unlocked,
        "positive_partial_window_count": _to_int(row.get("positive_partial_window_count")),
        "required_positive_partial_window_count": _to_int(row.get("required_positive_partial_window_count")),
        "min_partial_mse_gain_vs_history": min_partial_mse,
        "min_partial_mae_gain_vs_history": min_partial_mae,
        "counterfactual_margin": counterfactual_margin,
        "window0_fh_mse_gain": window0_fh,
        "validation_partial_mse_gain_vs_history": validation_partial_mse,
        "conflict_tags": tags,
    }


def analyze_gate_failure_decomposition(
    *,
    selector_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    preflight_report: dict[str, Any],
    min_partial_floor: float = -0.05,
) -> dict[str, Any]:
    """Summarize why a V29 candidate set is still blocked before generator unlock.

    This function is intentionally post-hoc and read-only. It does not relax the
    official teacher gate; it only decomposes repeated failure modes so the next
    experiment is not another blind scalar sweep.
    """

    blocker_counts = _preflight_blocker_counts(preflight_report)
    window_counts = _window_failure_counts(window_rows)
    diagnoses = [_candidate_diagnosis(row, min_partial_floor=min_partial_floor) for row in selector_rows]
    gate_pass_count = sum(1 for row in selector_rows if _truthy(row.get("target_gate_pass")))
    generator_unlocked_count = sum(1 for row in selector_rows if _truthy(row.get("generator_unlocked")))

    dominant_axes: list[str] = []
    if gate_pass_count == 0:
        dominant_axes.append("no_primary_teacher_gate_pass")
    if blocker_counts.get("positive_partial_windows_below_required", 0) > 0:
        dominant_axes.append("positive_partial_window_shortfall")
    if (
        blocker_counts.get("min_partial_mse_below_floor", 0) > 0
        or blocker_counts.get("min_partial_mae_below_floor", 0) > 0
        or window_counts.get("partial_mse_below_gate_floor", 0) > 0
        or window_counts.get("partial_mae_below_gate_floor", 0) > 0
    ):
        dominant_axes.append("partial_floor_violation")
    if blocker_counts.get("counterfactual_margin_negative", 0) > 0 or window_counts.get("counterfactual_negative", 0) > 0:
        dominant_axes.append("counterfactual_ordering_failure")
    if any(
        "window0_nonharm_but_partial_floor_fails" in diagnosis["conflict_tags"]
        or "aggregate_partial_positive_but_min_window_fails" in diagnosis["conflict_tags"]
        for diagnosis in diagnoses
    ):
        dominant_axes.append("window_level_conflict")
    if generator_unlocked_count == 0:
        dominant_axes.append("generator_unlock_blocked")

    matrix_decision = (
        "stop_local_scalar_matrix"
        if selector_rows and gate_pass_count == 0 and generator_unlocked_count == 0
        else "continue_preflight_controlled_tuning"
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_count": len(selector_rows),
        "gate_pass_count": gate_pass_count,
        "generator_unlocked_count": generator_unlocked_count,
        "preflight_pass": bool(preflight_report.get("preflight_pass")),
        "preflight_decision": preflight_report.get("decision", ""),
        "matrix_decision": matrix_decision,
        "dominant_failure_axes": dominant_axes,
        "primary_blocker_counts": dict(sorted(blocker_counts.items())),
        "window_failure_counts": dict(sorted(window_counts.items())),
        "candidate_diagnoses": diagnoses,
        "window_summaries": _window_summary_rows(window_rows),
        "next_boundary": (
            "diagnose_window_level_partial_and_counterfactual_conflict_before_any_new_scalar_sweep"
            if matrix_decision == "stop_local_scalar_matrix"
            else "rerun_generator_unlock_preflight_before_generator_training"
        ),
    }


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage III V29 Gate-Failure Decomposition",
        "",
        "This is a read-only diagnostic artifact. It does not relax the V29 teacher gate and does not unlock generator training.",
        "",
        "## Summary",
        "",
        f"- Protocol: `{report['protocol_version']}`.",
        f"- Candidate count: `{report['candidate_count']}`.",
        f"- Teacher gate pass count: `{report['gate_pass_count']}`.",
        f"- Generator unlocked count: `{report['generator_unlocked_count']}`.",
        f"- Preflight decision: `{report['preflight_decision']}`.",
        f"- Matrix decision: `{report['matrix_decision']}`.",
        "",
        "Do not train the generator unless a future preflight returns `preflight_pass=true`.",
        "",
        "## Dominant Failure Axes",
        "",
    ]
    for axis in report["dominant_failure_axes"]:
        lines.append(f"- `{axis}`")

    lines.extend(["", "## Candidate Diagnoses", ""])
    lines.append("| Rank | Run | Gate | Unlock | Min partial MSE | Min partial MAE | CF margin | Tags |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["candidate_diagnoses"]:
        lines.append(
            "| {rank} | `{run}` | `{gate}` | `{unlock}` | `{pmse}` | `{pmae}` | `{cf}` | `{tags}` |".format(
                rank=row["strict_rank"],
                run=row["run_id"],
                gate=row["target_gate_pass"],
                unlock=row["generator_unlocked"],
                pmse=row["min_partial_mse_gain_vs_history"],
                pmae=row["min_partial_mae_gain_vs_history"],
                cf=row["counterfactual_margin"],
                tags=";".join(row["conflict_tags"]),
            )
        )

    lines.extend(["", "## Window-Level Failure Summary", ""])
    lines.append("| Window | Rows | Partial MSE floor | Partial MAE floor | Negative CF | Real best | Top reasons |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report["window_summaries"]:
        lines.append(
            "| {window} | {rows} | {pmse} | {pmae} | {cf} | {real} | `{reasons}` |".format(
                window=row["validation_window_id"],
                rows=row["row_count"],
                pmse=row["partial_mse_floor_failures"],
                pmae=row["partial_mae_floor_failures"],
                cf=row["counterfactual_negative_rows"],
                real=row["real_best_rows"],
                reasons=row["top_failure_reasons"],
            )
        )

    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "The correct next move is to stop this local scalar matrix and explain the window-level conflict before adding another candidate.",
            "A new tuning block should be predeclared only after it targets both partial-floor stability and counterfactual ordering.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gate_failure_decomposition(
    *,
    selector_csv: Path,
    windows_csv: Path,
    preflight_json: Path,
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    report = analyze_gate_failure_decomposition(
        selector_rows=_read_csv(selector_csv),
        window_rows=_read_csv(windows_csv),
        preflight_report=_read_json(preflight_json),
    )
    _write_json_no_overwrite(out_dir / f"{prefix}_gate_failure_decomposition.json", report, overwrite=overwrite)
    _write_csv_no_overwrite(
        out_dir / f"{prefix}_gate_failure_decomposition.csv",
        report["candidate_diagnoses"],
        overwrite=overwrite,
    )
    _write_csv_no_overwrite(
        out_dir / f"{prefix}_gate_failure_decomposition_windows.csv",
        report["window_summaries"],
        overwrite=overwrite,
    )
    _write_text_no_overwrite(
        out_dir / f"{prefix}_gate_failure_decomposition.md",
        _build_markdown_report(report),
        overwrite=overwrite,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose Stage III V29 teacher-gate failures from selector artifacts.")
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v29_gate_failure")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = write_gate_failure_decomposition(
        selector_csv=args.selector_csv,
        windows_csv=args.windows_csv,
        preflight_json=args.preflight_json,
        out_dir=args.out_dir,
        prefix=args.prefix,
        overwrite=args.overwrite,
    )
    print(
        "matrix_decision={decision} candidates={candidates} gate_pass={gate_pass} generator_unlocked={unlock}".format(
            decision=report["matrix_decision"],
            candidates=report["candidate_count"],
            gate_pass=report["gate_pass_count"],
            unlock=report["generator_unlocked_count"],
        )
    )


if __name__ == "__main__":
    main()
