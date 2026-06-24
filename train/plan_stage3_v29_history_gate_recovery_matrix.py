from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "stage3_v29_history_gate_recovery_matrix_v1"
SOURCE_DECOMPOSITION_PROTOCOL = "stage3_v29_gate_failure_decomposition_v1"
DEFAULT_CONFIG_DIR = Path("configs/stage3_v29_history_gate_recovery_matrix")
DEFAULT_BUDGET_ARGS = (
    "--epochs 1",
    "--max-train-batches 60",
    "--max-generator-batches 60",
    "--max-eval-batches 60",
    "--validation-windows 5",
)

OLD_AXES = {
    "history_guarded_release_floor",
    "counterfactual_margin_micro_repair",
    "late_window_partial_floor_guard",
    "delayed_change_tempered_history",
    "release_null_coupled_guard",
    "window0_partial_floor_rebalance",
    "counterfactual_ordering_without_floor_pressure",
    "dual_axis_floor_and_cf_coupled",
    "mode_abstention_teacher_selection_probe",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rounded(value: float) -> float:
    return float(f"{value:.12g}")


def _num(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        value = mapping.get(key, default)
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _base_invariant(base_config: dict[str, Any]) -> dict[str, Any]:
    invariant = base_config.get("invariant", {})
    return invariant if isinstance(invariant, dict) else {}


def _window_summary(report: dict[str, Any], window_id: str) -> dict[str, Any]:
    rows = report.get("window_summaries", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("validation_window_id")) == str(window_id):
            return row
    return {}


def _top_gate_near_run_id(selector_rows: list[dict[str, str]]) -> str:
    raw_rows = [
        row
        for row in selector_rows
        if row.get("target_name") == "raw_residual_vector"
        and row.get("feature_mode") == "fluctuation_plus_history"
    ]
    if not raw_rows:
        return ""
    raw_rows.sort(key=lambda row: _as_int(row.get("strict_rank"), 9999))
    return str(raw_rows[0].get("run_id", ""))


def _top_window_rows(window_rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in window_rows
        if row.get("run_id") == run_id
        and row.get("target_name") == "raw_residual_vector"
        and row.get("feature_mode") == "fluctuation_plus_history"
    ]
    return sorted(rows, key=lambda row: _as_int(row.get("validation_window_id"), 9999))


def _history_dominant_windows(window_rows: list[dict[str, str]], run_id: str) -> list[str]:
    dominant: list[str] = []
    for row in _top_window_rows(window_rows, run_id):
        partial = _num(row, "partial_mse_gain_vs_history", 0.0)
        history_gain = _num(row, "history_mse_gain", 0.0)
        if partial < 0.0 and history_gain > 0.0:
            dominant.append(str(row.get("validation_window_id", "")))
    return dominant


def _counterfactual_tie_windows(window_rows: list[dict[str, str]], run_id: str) -> list[str]:
    tied: list[str] = []
    for row in _top_window_rows(window_rows, run_id):
        margin = _num(row, "counterfactual_margin", 0.0)
        if margin < 0.0 and abs(margin) < 0.02:
            tied.append(str(row.get("validation_window_id", "")))
    return tied


def _preconditions_hold(decomposition_report: dict[str, Any]) -> bool:
    if decomposition_report.get("protocol_version") != SOURCE_DECOMPOSITION_PROTOCOL:
        return False
    if decomposition_report.get("matrix_decision") != "stop_local_scalar_matrix":
        return False
    if _as_int(decomposition_report.get("gate_pass_count"), 0) != 0:
        return False
    if _as_int(decomposition_report.get("generator_unlocked_count"), 0) != 0:
        return False
    blocker_counts = decomposition_report.get("primary_blocker_counts", {})
    if not isinstance(blocker_counts, dict):
        return False
    required_blockers = (
        "counterfactual_margin_negative",
        "min_partial_mse_below_floor",
        "min_partial_mae_below_floor",
    )
    return all(_as_int(blocker_counts.get(key), 0) > 0 for key in required_blockers)


def _candidate_template(
    *,
    axis: str,
    run_slug: str,
    priority: int,
    addresses_failure_axes: list[str],
    invariant_overrides: dict[str, float],
    mechanism_shift: str,
    rationale: str,
    window_focus: str,
    diagnostic_metadata: dict[str, Any],
) -> dict[str, Any]:
    if axis in OLD_AXES:
        raise ValueError(f"History-gate recovery axis repeats an exhausted axis: {axis}")
    return {
        "axis": axis,
        "run_slug": run_slug,
        "priority": priority,
        "addresses_failure_axes": addresses_failure_axes,
        "invariant_overrides": invariant_overrides,
        "mechanism_shift": mechanism_shift,
        "rationale": rationale,
        "window_focus": window_focus,
        "diagnostic_metadata": diagnostic_metadata,
    }


def _candidate_templates(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    selector_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    invariant = _base_invariant(base_config)
    top_run_id = _top_gate_near_run_id(selector_rows)
    history_dominant = _history_dominant_windows(window_rows, top_run_id)
    cf_ties = _counterfactual_tie_windows(window_rows, top_run_id)
    window0 = _window_summary(decomposition_report, "0")
    window2 = _window_summary(decomposition_report, "2")
    row_count = _as_int(window0.get("row_count"), 0)
    w0_floor_failures = _as_int(window0.get("partial_mse_floor_failures"), 0)
    w2_cf_failures = _as_int(window2.get("counterfactual_negative_rows"), 0)

    shared_meta = {
        "top_gate_near_run_id": top_run_id,
        "history_dominant_windows": history_dominant,
        "counterfactual_tie_windows": cf_ties,
        "window0_partial_mse_floor_failures": w0_floor_failures,
        "window0_row_count": row_count,
        "window2_counterfactual_negative_rows": w2_cf_failures,
    }

    return [
        _candidate_template(
            axis="window0_safe_release_ramp",
            run_slug="w0ramp",
            priority=10,
            addresses_failure_axes=[
                "history_dominance_partial_floor_conflict",
                "partial_floor_violation",
                "counterfactual_tie_near_gate",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_release_first_increment": _rounded(_num(invariant, "lambda_release_first_increment", 0.4) + 0.01),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.025),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.12) + 0.02),
                "lambda_history_budget_cap": _rounded(_num(invariant, "lambda_history_budget_cap", 0.035) + 0.01),
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.0008),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.001) + 0.00008),
            },
            mechanism_shift="small_release_ramp_with_window0_anchor_protection",
            rationale=(
                "The top one-epoch rows are window-0 non-harmful but fail partial floors. "
                "This candidate tests a small release ramp while increasing anchor/router protection."
            ),
            window_focus="window0_nonharm_plus_partial_floor",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="late_history_dominance_bridge",
            run_slug="latehist",
            priority=20,
            addresses_failure_axes=[
                "history_dominance_partial_floor_conflict",
                "positive_partial_window_shortfall",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_support_conditioned_advantage": _rounded(_num(invariant, "lambda_support_conditioned_advantage", 0.08) + 0.035),
                "lambda_residual_release_advantage": _rounded(_num(invariant, "lambda_residual_release_advantage", 0.08) + 0.025),
                "lambda_residual_release_alignment": _rounded(_num(invariant, "lambda_residual_release_alignment", 0.03) + 0.02),
                "lambda_release_state_advantage": _rounded(_num(invariant, "lambda_release_state_advantage", 0.12) + 0.015),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.015),
                "lambda_release_first_increment": _rounded(_num(invariant, "lambda_release_first_increment", 0.4) + 0.005),
            },
            mechanism_shift="late_window_support_bridge_without_global_release_jump",
            rationale=(
                "History-dominant windows in the top row are not solved by global counterfactual pressure. "
                "This tests support-conditioned residual release to improve late partial windows."
            ),
            window_focus="history_dominant_windows",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="counterfactual_tie_breaker_floor_safe",
            run_slug="cftie",
            priority=30,
            addresses_failure_axes=[
                "counterfactual_tie_near_gate",
                "counterfactual_ordering_failure",
                "partial_floor_violation",
            ],
            invariant_overrides={
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.0012),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.001) + 0.00012),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.12) + 0.025),
                "lambda_local_env_consistency": _rounded(_num(invariant, "lambda_local_env_consistency", 0.04) + 0.015),
                "lambda_release_first_increment": _rounded(max(0.0, _num(invariant, "lambda_release_first_increment", 0.4) - 0.005)),
            },
            mechanism_shift="counterfactual_tie_breaker_without_release_amplitude_growth",
            rationale=(
                "The top row has very small negative counterfactual margins in several windows. "
                "This tests ordering repair while avoiding a larger release amplitude."
            ),
            window_focus="near_zero_counterfactual_margins",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="history_cap_release_balance",
            run_slug="histcap",
            priority=40,
            addresses_failure_axes=[
                "history_dominance_partial_floor_conflict",
                "positive_partial_window_shortfall",
                "counterfactual_ordering_failure",
            ],
            invariant_overrides={
                "lambda_history_after_release": _rounded(max(0.0, _num(invariant, "lambda_history_after_release", 0.08) - 0.015)),
                "lambda_history_budget_cap": _rounded(max(0.0, _num(invariant, "lambda_history_budget_cap", 0.035) - 0.008)),
                "lambda_release_priority_advantage": _rounded(_num(invariant, "lambda_release_priority_advantage", 0.18) + 0.025),
                "lambda_support_alignment": _rounded(_num(invariant, "lambda_support_alignment", 0.14) + 0.015),
                "lambda_suppression_alignment": _rounded(_num(invariant, "lambda_suppression_alignment", 0.1) + 0.01),
                "lambda_suppression_gate_sparsity": _rounded(_num(invariant, "lambda_suppression_gate_sparsity", 0.025) + 0.005),
            },
            mechanism_shift="reduce_history_dominance_while_preserving_abstention_safety",
            rationale=(
                "Several gate-near windows fail because history-only is stronger than fluctuation-plus-history. "
                "This tests whether a modest history cap lets the release branch add value without forcing generator unlock."
            ),
            window_focus="history_cap_balance",
            diagnostic_metadata=shared_meta,
        ),
    ]


def _suggested_command(*, config_name: str, run_id: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    budget = " ".join(DEFAULT_BUDGET_ARGS)
    return (
        "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
        f"--config {config_dir / config_name} "
        f"--run-id {run_id} {budget}"
    )


def build_history_gate_recovery_matrix(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    selector_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    run_id_prefix: str = "v29hgr_conf015",
) -> list[dict[str, Any]]:
    if not _preconditions_hold(decomposition_report):
        return []

    rows: list[dict[str, Any]] = []
    for index, template in enumerate(
        _candidate_templates(
            base_config=base_config,
            decomposition_report=decomposition_report,
            selector_rows=selector_rows,
            window_rows=window_rows,
        ),
        start=1,
    ):
        run_id = f"{run_id_prefix}_{index:02d}_{template['run_slug']}"
        config_name = f"stage3_v29_{run_id}.yaml"
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "candidate_id": f"{index:02d}_{template['axis']}",
                "priority": template["priority"],
                "axis": template["axis"],
                "run_slug": template["run_slug"],
                "run_id": run_id,
                "config_name": config_name,
                "source_decomposition_protocol": SOURCE_DECOMPOSITION_PROTOCOL,
                "source_matrix_decision": str(decomposition_report.get("matrix_decision", "")),
                "source_candidate_count": _as_int(decomposition_report.get("candidate_count"), 0),
                "addresses_failure_axes": template["addresses_failure_axes"],
                "window_focus": template["window_focus"],
                "architecture_policy": "frozen_v29_no_model_topology_change",
                "generator_policy": "preflight_required_do_not_unlock_from_matrix",
                "repair_matrix_status_required": "exhausted_meaningful_rows",
                "known_failed_axis_guard": "not_old_scalar_or_repair_axis",
                "mechanism_shift": template["mechanism_shift"],
                "model_overrides": {},
                "train_overrides": {},
                "invariant_overrides": template["invariant_overrides"],
                "diagnostic_metadata": template["diagnostic_metadata"],
                "command_budget": " ".join(DEFAULT_BUDGET_ARGS),
                "suggested_probe_command": _suggested_command(config_name=config_name, run_id=run_id),
                "rationale": template["rationale"],
                "promotion_gate": (
                    "rerun complete selector, gate-failure decomposition, and generator-unlock preflight; "
                    "do not train generator unless preflight_pass becomes true"
                ),
            }
        )
    return sorted(rows, key=lambda row: (int(row["priority"]), str(row["run_id"])))


def _merge_config(base_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    invariant = cfg.setdefault("invariant", {})
    if not isinstance(invariant, dict):
        invariant = {}
        cfg["invariant"] = invariant
    invariant.update(dict(candidate["invariant_overrides"]))

    stage3 = cfg.setdefault("stage3_v2", {})
    if not isinstance(stage3, dict):
        stage3 = {}
        cfg["stage3_v2"] = stage3
    stage3["run_id"] = str(candidate["run_id"])
    stage3["history_gate_recovery_matrix_protocol"] = PROTOCOL_VERSION
    stage3["history_gate_recovery_candidate_id"] = str(candidate["candidate_id"])

    cfg["history_gate_recovery_matrix"] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": candidate["candidate_id"],
        "axis": candidate["axis"],
        "source_decomposition_protocol": candidate["source_decomposition_protocol"],
        "source_matrix_decision": candidate["source_matrix_decision"],
        "source_candidate_count": candidate["source_candidate_count"],
        "addresses_failure_axes": candidate["addresses_failure_axes"],
        "window_focus": candidate["window_focus"],
        "architecture_policy": candidate["architecture_policy"],
        "generator_policy": candidate["generator_policy"],
        "repair_matrix_status_required": candidate["repair_matrix_status_required"],
        "known_failed_axis_guard": candidate["known_failed_axis_guard"],
        "mechanism_shift": candidate["mechanism_shift"],
        "diagnostic_metadata": candidate["diagnostic_metadata"],
        "promotion_gate": candidate["promotion_gate"],
    }
    return cfg


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 history-gate recovery artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_scalar(value) for key, value in row.items()})


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 History-Gate Recovery Matrix",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Rows: `{len(rows)}`.",
        "",
        "This matrix does not train the generator and does not change V29 model topology. "
        "It is a post-repair tuning plan for the history-dominance versus counterfactual-ordering conflict.",
        "",
        "| Priority | Candidate | Axis | Mechanism shift | Failure axes |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | `{candidate}` | `{axis}` | `{mechanism}` | `{axes}` |".format(
                priority=row["priority"],
                candidate=row["candidate_id"],
                axis=row["axis"],
                mechanism=row["mechanism_shift"],
                axes=";".join(row["addresses_failure_axes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These candidates are next-run tuning controls only. They are not generator-unlock evidence, "
            "not online-validation evidence, and not manuscript performance evidence. After each run, rerun "
            "the complete selector, gate-failure decomposition, and generator-unlock preflight.",
            "",
        ]
    )
    return "\n".join(lines)


def write_history_gate_recovery_matrix_artifacts(
    *,
    rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    config_dir: Path,
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 history-gate recovery artifact: {md_path}")
    _write_csv(csv_path, rows, overwrite=overwrite)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[str] = []
    for row in rows:
        config_path = config_dir / str(row["config_name"])
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Refuse to overwrite existing V29 history-gate recovery config: {config_path}")
        payload = _merge_config(base_config, row)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        config_paths.append(str(config_path))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "configs": config_paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage III V29 history-gate recovery matrix configs.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--decomposition-json", type=Path, required=True)
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v29_history_gate_recovery_matrix")
    parser.add_argument("--run-id-prefix", default="v29hgr_conf015")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_config = _read_yaml(args.base_config)
    decomposition_report = _read_json(args.decomposition_json)
    selector_rows = _read_csv(args.selector_csv)
    window_rows = _read_csv(args.windows_csv)
    rows = build_history_gate_recovery_matrix(
        base_config=base_config,
        decomposition_report=decomposition_report,
        selector_rows=selector_rows,
        window_rows=window_rows,
        run_id_prefix=str(args.run_id_prefix),
    )
    outputs = write_history_gate_recovery_matrix_artifacts(
        rows=rows,
        base_config=base_config,
        config_dir=args.config_dir,
        out_dir=args.out_dir,
        prefix=str(args.prefix),
        overwrite=bool(args.overwrite),
    )
    print(f"rows={len(rows)} csv={outputs['csv']} markdown={outputs['markdown']} configs={len(outputs['configs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
