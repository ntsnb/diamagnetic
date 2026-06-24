from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PREFLIGHT_PROTOCOL_VERSION = "stage3_v29_generator_unlock_preflight_v1"
SERVED_MODE_GATE_PROTOCOL_VERSION = "v29_served_online_mode_gate_manifest_v1"
SERVED_MODE_GATE_PREREQUISITE_SCOPE = "mode_selection_only_no_generator_unlock"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _split_stop_reasons(value: object) -> list[str]:
    text = str(value or "").replace(",", ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def _served_manifest_blockers(served_manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(served_manifest.get("protocol_version", "")) != SERVED_MODE_GATE_PROTOCOL_VERSION:
        blockers.append("served_manifest_protocol_unsupported")
    if not bool(served_manifest.get("admission_pass", False)):
        blockers.append("served_mode_gate_admission_not_passed")
    if str(served_manifest.get("servable_scope", "")) != SERVED_MODE_GATE_PREREQUISITE_SCOPE:
        blockers.append("served_scope_not_mode_gate_prerequisite")
    for reason in served_manifest.get("admission_reasons", []) or []:
        blockers.append(f"served_admission_reason:{reason}")
    return blockers


def _candidate_blockers(
    row: dict[str, Any],
    *,
    min_partial_gain_floor: float,
    min_counterfactual_margin: float,
    min_window0_fh_mse_gain: float,
    min_positive_partial_windows: int | None,
) -> list[str]:
    blockers: list[str] = []
    if not _truthy(row.get("selector_metric_complete", True)):
        blockers.append("selector_metric_incomplete")
    if not _truthy(row.get("target_gate_pass")):
        blockers.append("target_gate_not_passed")
    if not _truthy(row.get("any_primary_gate_pass")):
        blockers.append("no_primary_gate_passed")
    if _truthy(row.get("teacher_abort_written")):
        blockers.append("teacher_abort_written")
    if not _truthy(row.get("generator_unlocked")):
        blockers.append("generator_not_unlocked")
    for reason in _split_stop_reasons(row.get("stop_reasons")):
        blockers.append(f"stop_reason:{reason}")

    positive_windows = _to_int(row.get("positive_partial_window_count"))
    required_windows = _to_int(row.get("required_positive_partial_window_count"))
    if min_positive_partial_windows is not None:
        required_windows = max(required_windows, int(min_positive_partial_windows))
    if positive_windows < required_windows:
        blockers.append("positive_partial_windows_below_required")

    min_partial_mse = _to_float(row.get("min_partial_mse_gain_vs_history"))
    if not math.isfinite(min_partial_mse):
        blockers.append("min_partial_mse_missing_or_nonfinite")
    elif min_partial_mse < min_partial_gain_floor:
        blockers.append("min_partial_mse_below_floor")

    min_partial_mae = _to_float(row.get("min_partial_mae_gain_vs_history"))
    if not math.isfinite(min_partial_mae):
        blockers.append("min_partial_mae_missing_or_nonfinite")
    elif min_partial_mae < min_partial_gain_floor:
        blockers.append("min_partial_mae_below_floor")

    counterfactual_margin = _to_float(row.get("counterfactual_margin"))
    if not math.isfinite(counterfactual_margin):
        blockers.append("counterfactual_margin_missing_or_nonfinite")
    elif counterfactual_margin < min_counterfactual_margin:
        blockers.append(
            "counterfactual_margin_negative"
            if min_counterfactual_margin == 0.0
            else "counterfactual_margin_below_floor"
        )

    window0_fh_mse = _to_float(row.get("window0_fh_mse_gain"))
    if not math.isfinite(window0_fh_mse):
        blockers.append("window0_fh_mse_missing_or_nonfinite")
    elif window0_fh_mse < min_window0_fh_mse_gain:
        blockers.append("window0_fh_mse_below_floor")
    return blockers


def _candidate_report(
    row: dict[str, Any],
    *,
    min_partial_gain_floor: float,
    min_counterfactual_margin: float,
    min_window0_fh_mse_gain: float,
    min_positive_partial_windows: int | None,
) -> dict[str, Any]:
    blockers = _candidate_blockers(
        row,
        min_partial_gain_floor=min_partial_gain_floor,
        min_counterfactual_margin=min_counterfactual_margin,
        min_window0_fh_mse_gain=min_window0_fh_mse_gain,
        min_positive_partial_windows=min_positive_partial_windows,
    )
    return {
        "strict_rank": _to_int(row.get("strict_rank")),
        "run_id": str(row.get("run_id", "")),
        "selection_label": str(row.get("selection_label", "")),
        "preflight_pass": len(blockers) == 0,
        "blockers": blockers,
        "metrics": {
            "target_gate_pass": _truthy(row.get("target_gate_pass")),
            "any_primary_gate_pass": _truthy(row.get("any_primary_gate_pass")),
            "teacher_abort_written": _truthy(row.get("teacher_abort_written")),
            "generator_unlocked": _truthy(row.get("generator_unlocked")),
            "positive_partial_window_count": _to_int(row.get("positive_partial_window_count")),
            "required_positive_partial_window_count": _to_int(row.get("required_positive_partial_window_count")),
            "min_partial_mse_gain_vs_history": _to_float(row.get("min_partial_mse_gain_vs_history")),
            "min_partial_mae_gain_vs_history": _to_float(row.get("min_partial_mae_gain_vs_history")),
            "counterfactual_margin": _to_float(row.get("counterfactual_margin")),
            "window0_fh_mse_gain": _to_float(row.get("window0_fh_mse_gain")),
            "stop_reasons": _split_stop_reasons(row.get("stop_reasons")),
        },
    }


def build_generator_unlock_preflight(
    selector_rows: list[dict[str, Any]],
    served_manifest: dict[str, Any],
    *,
    top_k: int = 1,
    min_partial_gain_floor: float = -0.05,
    min_counterfactual_margin: float = 0.0,
    min_window0_fh_mse_gain: float = 0.0,
    min_positive_partial_windows: int | None = None,
) -> dict[str, Any]:
    """Audit whether a V29 selector result is ready for generator-unlock experiments.

    This is intentionally a read-only admission report. A mode-gate serving
    package can satisfy the online mode-selection prerequisite, but it does not
    itself authorize generator training unless the selected teacher run also
    passes the strict V29 target gate and generator-unlock conditions.
    """

    selected_rows = list(selector_rows)[: max(int(top_k), 0)]
    mode_gate_blockers = _served_manifest_blockers(served_manifest)
    candidate_reports = [
        _candidate_report(
            row,
            min_partial_gain_floor=min_partial_gain_floor,
            min_counterfactual_margin=min_counterfactual_margin,
            min_window0_fh_mse_gain=min_window0_fh_mse_gain,
            min_positive_partial_windows=min_positive_partial_windows,
        )
        for row in selected_rows
    ]
    global_blockers = list(mode_gate_blockers)
    if not selected_rows:
        global_blockers.append("empty_selector")

    any_candidate_pass = any(bool(report["preflight_pass"]) for report in candidate_reports)
    preflight_pass = not global_blockers and any_candidate_pass
    return {
        "protocol_version": PREFLIGHT_PROTOCOL_VERSION,
        "preflight_pass": preflight_pass,
        "decision": "allow_generator_unlock_experiment" if preflight_pass else "do_not_train_generator",
        "recommendation": (
            "Proceed only to a bounded generator-unlock experiment with the audited candidate."
            if preflight_pass
            else "Do not train the generator; improve the teacher gate before generator unlock."
        ),
        "selector_scope": {
            "top_k": int(top_k),
            "candidate_count": len(candidate_reports),
        },
        "candidate_count": len(candidate_reports),
        "mode_gate_prerequisite_pass": len(mode_gate_blockers) == 0,
        "mode_gate_blockers": mode_gate_blockers,
        "global_blockers": global_blockers,
        "served_manifest": {
            "protocol_version": served_manifest.get("protocol_version", ""),
            "package_id": served_manifest.get("package_id", ""),
            "servable_scope": served_manifest.get("servable_scope", ""),
            "admission_pass": bool(served_manifest.get("admission_pass", False)),
            "admission_reasons": served_manifest.get("admission_reasons", []),
        },
        "thresholds": {
            "min_partial_gain_floor": float(min_partial_gain_floor),
            "min_counterfactual_margin": float(min_counterfactual_margin),
            "min_window0_fh_mse_gain": float(min_window0_fh_mse_gain),
            "min_positive_partial_windows": min_positive_partial_windows,
        },
        "candidate_reports": candidate_reports,
        "interpretation_guard": (
            "A served mode gate with scope mode_selection_only_no_generator_unlock is only a "
            "mode-selection prerequisite. Generator training remains blocked unless the teacher "
            "target gate has passed and no teacher abort/generator stop reason is present."
        ),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage III V29 Generator-Unlock Preflight",
        "",
        f"Protocol: `{report.get('protocol_version')}`.",
        f"Decision: `{report.get('decision')}`.",
        f"Preflight pass: `{report.get('preflight_pass')}`.",
        f"Mode-gate prerequisite pass: `{report.get('mode_gate_prerequisite_pass')}`.",
        f"Candidate count: `{report.get('candidate_count')}`.",
        "",
        "## Recommendation",
        "",
        str(report.get("recommendation", "")),
        "",
        "## Served Mode-Gate Prerequisite",
        "",
        f"- Package: `{report.get('served_manifest', {}).get('package_id', '')}`.",
        f"- Scope: `{report.get('served_manifest', {}).get('servable_scope', '')}`.",
        f"- Admission pass: `{report.get('served_manifest', {}).get('admission_pass', '')}`.",
        f"- Blockers: `{';'.join(report.get('mode_gate_blockers', []))}`.",
        "",
        "## Candidate Gate Audit",
        "",
        "| Rank | Run | Label | Pass | Blockers |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for candidate in report.get("candidate_reports", []):
        lines.append(
            "| {rank} | `{run}` | `{label}` | `{passed}` | `{blockers}` |".format(
                rank=candidate.get("strict_rank", ""),
                run=candidate.get("run_id", ""),
                label=candidate.get("selection_label", ""),
                passed=candidate.get("preflight_pass", ""),
                blockers=";".join(candidate.get("blockers", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            str(report.get("interpretation_guard", "")),
            "",
        ]
    )
    return "\n".join(lines)


def write_generator_unlock_preflight_artifacts(
    *,
    out_dir: Path,
    prefix: str,
    report: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    markdown_path = out_dir / f"{prefix}.md"
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 generator-unlock preflight output: {json_path}")
    if markdown_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 generator-unlock preflight output: {markdown_path}")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage III V29 readiness for generator-unlock experiments.")
    parser.add_argument("--selector-csv", required=True, type=Path)
    parser.add_argument("--served-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="stage3_v29_generator_unlock_preflight")
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--min-partial-gain-floor", default=-0.05, type=float)
    parser.add_argument("--min-counterfactual-margin", default=0.0, type=float)
    parser.add_argument("--min-window0-fh-mse-gain", default=0.0, type=float)
    parser.add_argument("--min-positive-partial-windows", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_generator_unlock_preflight(
        _read_csv(args.selector_csv),
        _read_json(args.served_manifest),
        top_k=args.top_k,
        min_partial_gain_floor=args.min_partial_gain_floor,
        min_counterfactual_margin=args.min_counterfactual_margin,
        min_window0_fh_mse_gain=args.min_window0_fh_mse_gain,
        min_positive_partial_windows=args.min_positive_partial_windows,
    )
    write_generator_unlock_preflight_artifacts(
        out_dir=args.out_dir,
        prefix=args.prefix,
        report=report,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
