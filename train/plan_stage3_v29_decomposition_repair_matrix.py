from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "stage3_v29_decomposition_repair_matrix_v1"
SOURCE_DECOMPOSITION_PROTOCOL = "stage3_v29_gate_failure_decomposition_v1"
DEFAULT_CONFIG_DIR = Path("configs/stage3_v29_decomposition_repair_matrix")
DEFAULT_BUDGET_ARGS = (
    "--epochs 1",
    "--max-train-batches 60",
    "--max-generator-batches 60",
    "--max-eval-batches 60",
    "--validation-windows 5",
)
OLD_SCALAR_AXES = {
    "history_guarded_release_floor",
    "counterfactual_margin_micro_repair",
    "late_window_partial_floor_guard",
    "delayed_change_tempered_history",
    "release_null_coupled_guard",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _rounded(value: float) -> float:
    return float(f"{value:.12g}")


def _num(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _dominant_axes(report: dict[str, Any]) -> set[str]:
    axes = report.get("dominant_failure_axes", [])
    return {str(axis) for axis in axes} if isinstance(axes, list) else set()


def _window_summary(report: dict[str, Any], window_id: str) -> dict[str, Any]:
    rows = report.get("window_summaries", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("validation_window_id")) == str(window_id):
            return row
    return {}


def _base_invariant(base_config: dict[str, Any]) -> dict[str, Any]:
    invariant = base_config.get("invariant", {})
    return invariant if isinstance(invariant, dict) else {}


def _candidate_template(
    *,
    axis: str,
    run_slug: str,
    priority: int,
    addresses_failure_axes: list[str],
    invariant_overrides: dict[str, float],
    teacher_selection_overrides: dict[str, float | str | bool],
    rationale: str,
    window_focus: str,
) -> dict[str, Any]:
    if axis in OLD_SCALAR_AXES:
        raise ValueError(f"Repair candidate repeats an exhausted scalar axis: {axis}")
    return {
        "axis": axis,
        "run_slug": run_slug,
        "priority": priority,
        "addresses_failure_axes": addresses_failure_axes,
        "invariant_overrides": invariant_overrides,
        "teacher_selection_overrides": teacher_selection_overrides,
        "rationale": rationale,
        "window_focus": window_focus,
    }


def _candidate_templates(base_config: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    invariant = _base_invariant(base_config)
    axes = _dominant_axes(report)
    window0 = _window_summary(report, "0")
    window2 = _window_summary(report, "2")
    window0_floor_failures = _num(window0, "partial_mse_floor_failures", 0.0)
    window2_cf_failures = _num(window2, "counterfactual_negative_rows", 0.0)

    templates = [
        _candidate_template(
            axis="window0_partial_floor_rebalance",
            run_slug="w0floor",
            priority=10,
            addresses_failure_axes=[
                "partial_floor_violation",
                "positive_partial_window_shortfall",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_release_first_increment": _rounded(max(0.0, _num(invariant, "lambda_release_first_increment", 0.4) - 0.035)),
                "lambda_history_after_release": _rounded(_num(invariant, "lambda_history_after_release", 0.08) + 0.055),
                "lambda_history_budget_cap": _rounded(_num(invariant, "lambda_history_budget_cap", 0.035) + 0.025),
                "lambda_invariant_correction_budget": _rounded(max(0.0, _num(invariant, "lambda_invariant_correction_budget", 0.02) - 0.004)),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.025),
            },
            teacher_selection_overrides={
                "window0_partial_floor_guard": True,
                "window0_partial_floor_failure_count": window0_floor_failures,
            },
            rationale=(
                "Window 0 fails partial MSE/MAE floors for the full candidate set, so this candidate "
                "reduces release amplitude and increases history/anchor support without increasing "
                "counterfactual pressure."
            ),
            window_focus="window0_partial_floor",
        ),
        _candidate_template(
            axis="counterfactual_ordering_without_floor_pressure",
            run_slug="cforder",
            priority=20,
            addresses_failure_axes=[
                "counterfactual_ordering_failure",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.0025),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.001) + 0.00025),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.12) + 0.0125),
                "lambda_release_temporal": _rounded(_num(invariant, "lambda_release_temporal", 0.04) + 0.0025),
                "lambda_release_first_increment": _rounded(max(0.0, _num(invariant, "lambda_release_first_increment", 0.4) - 0.005)),
            },
            teacher_selection_overrides={
                "counterfactual_only_window_probe": "window2",
                "window2_counterfactual_negative_count": window2_cf_failures,
            },
            rationale=(
                "Window 2 has no partial-floor failure but still has counterfactual-ordering failure, "
                "so this isolates a very small ordering repair without pushing correction amplitude."
            ),
            window_focus="window2_counterfactual_ordering",
        ),
        _candidate_template(
            axis="dual_axis_floor_and_cf_coupled",
            run_slug="floorcf",
            priority=30,
            addresses_failure_axes=[
                "partial_floor_violation",
                "counterfactual_ordering_failure",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_release_first_increment": _rounded(max(0.0, _num(invariant, "lambda_release_first_increment", 0.4) - 0.02)),
                "lambda_history_after_release": _rounded(_num(invariant, "lambda_history_after_release", 0.08) + 0.035),
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.0015),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.001) + 0.00015),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.015),
                "lambda_residual_release_alignment": _rounded(_num(invariant, "lambda_residual_release_alignment", 0.16) + 0.01),
            },
            teacher_selection_overrides={
                "dual_axis_gate_required": True,
                "require_partial_floor_and_counterfactual_nonnegative": True,
            },
            rationale=(
                "The decomposition shows partial floors and counterfactual ordering fail together in most "
                "windows, so this candidate couples mild floor support with mild ordering pressure."
            ),
            window_focus="joint_floor_and_counterfactual",
        ),
        _candidate_template(
            axis="mode_abstention_teacher_selection_probe",
            run_slug="abstain",
            priority=40,
            addresses_failure_axes=[
                "no_primary_teacher_gate_pass",
                "generator_unlock_blocked",
                "counterfactual_ordering_failure",
            ],
            invariant_overrides={
                "lambda_suppression_alignment": _rounded(_num(invariant, "lambda_suppression_alignment", 0.11) + 0.02),
                "lambda_suppression_gate_sparsity": _rounded(_num(invariant, "lambda_suppression_gate_sparsity", 0.03) + 0.01),
                "lambda_support_alignment": _rounded(max(0.0, _num(invariant, "lambda_support_alignment", 0.14) - 0.01)),
                "lambda_release_state_advantage": _rounded(max(0.0, _num(invariant, "lambda_release_state_advantage", 0.12) - 0.015)),
            },
            teacher_selection_overrides={
                "teacher_selection_probe": "abstain_unless_partial_and_counterfactual_agree",
                "served_mode_scope": "mode_selection_only_no_generator_unlock",
            },
            rationale=(
                "If correction and counterfactual ordering disagree, stricter teacher selection or abstention "
                "may be safer than additional correction. This probes selection controls, not topology."
            ),
            window_focus="teacher_selection_abstention",
        ),
    ]

    if "partial_floor_violation" not in axes and "counterfactual_ordering_failure" not in axes:
        return []
    return templates


def _suggested_command(*, config_name: str, run_id: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    budget = " ".join(DEFAULT_BUDGET_ARGS)
    return (
        "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
        f"--config {config_dir / config_name} "
        f"--run-id {run_id} {budget}"
    )


def build_decomposition_repair_matrix(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    run_id_prefix: str = "v29repair",
) -> list[dict[str, Any]]:
    if decomposition_report.get("protocol_version") != SOURCE_DECOMPOSITION_PROTOCOL:
        return []
    if decomposition_report.get("matrix_decision") != "stop_local_scalar_matrix":
        return []

    rows: list[dict[str, Any]] = []
    for index, template in enumerate(_candidate_templates(base_config, decomposition_report), start=1):
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
                "addresses_failure_axes": template["addresses_failure_axes"],
                "window_focus": template["window_focus"],
                "architecture_policy": "frozen_v29_no_model_topology_change",
                "generator_policy": "preflight_required_do_not_unlock_from_matrix",
                "known_failed_axis_guard": "not_one_of_prior_five_scalar_axes",
                "model_overrides": {},
                "train_overrides": {},
                "teacher_selection_overrides": template["teacher_selection_overrides"],
                "invariant_overrides": template["invariant_overrides"],
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
    stage3["decomposition_repair_matrix_protocol"] = PROTOCOL_VERSION
    stage3["decomposition_repair_candidate_id"] = str(candidate["candidate_id"])

    cfg["decomposition_repair_matrix"] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": candidate["candidate_id"],
        "axis": candidate["axis"],
        "source_decomposition_protocol": candidate["source_decomposition_protocol"],
        "source_matrix_decision": candidate["source_matrix_decision"],
        "addresses_failure_axes": candidate["addresses_failure_axes"],
        "window_focus": candidate["window_focus"],
        "architecture_policy": candidate["architecture_policy"],
        "generator_policy": candidate["generator_policy"],
        "known_failed_axis_guard": candidate["known_failed_axis_guard"],
        "teacher_selection_overrides": candidate["teacher_selection_overrides"],
        "promotion_gate": candidate["promotion_gate"],
    }
    return cfg


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 decomposition repair matrix artifact: {path}")
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
        "# Stage III V29 Decomposition-Driven Repair Matrix",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Rows: `{len(rows)}`.",
        "",
        "This matrix does not train the generator, does not change V29 model topology, "
        "and does not overwrite older V29 versions. It translates the gate-failure "
        "decomposition into predeclared candidates for partial-floor stability and "
        "counterfactual ordering diagnostics.",
        "",
        "| Priority | Candidate | Axis | Failure axes | Window focus |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | `{candidate}` | `{axis}` | `{axes}` | `{focus}` |".format(
                priority=row["priority"],
                candidate=row["candidate_id"],
                axis=row["axis"],
                axes=";".join(row["addresses_failure_axes"]),
                focus=row["window_focus"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These candidates are next-run tuning controls only. After each run, rerun the complete selector, "
            "gate-failure decomposition, and generator-unlock preflight. Do not train generator unless "
            "`preflight_pass=true`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_decomposition_repair_matrix_artifacts(
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
        raise FileExistsError(f"Refuse to overwrite existing V29 decomposition repair matrix artifact: {md_path}")
    _write_csv(csv_path, rows, overwrite=overwrite)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[str] = []
    for row in rows:
        config_path = config_dir / str(row["config_name"])
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Refuse to overwrite existing V29 decomposition repair config: {config_path}")
        payload = _merge_config(base_config, row)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        config_paths.append(str(config_path))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "configs": config_paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build decomposition-driven Stage III V29 repair matrix configs.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--decomposition-json", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v29_decomposition_repair_matrix")
    parser.add_argument("--run-id-prefix", default="v29repair_conf015")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_config = _read_yaml(args.base_config)
    decomposition_report = _read_json(args.decomposition_json)
    rows = build_decomposition_repair_matrix(
        base_config=base_config,
        decomposition_report=decomposition_report,
        run_id_prefix=str(args.run_id_prefix),
    )
    outputs = write_decomposition_repair_matrix_artifacts(
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
