from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "stage3_v29_teacher_gate_tuning_matrix_v1"
DEFAULT_CONFIG_DIR = Path("configs/stage3_v29_teacher_gate_tuning_matrix")
DEFAULT_BUDGET_ARGS = (
    "--epochs 1",
    "--max-train-batches 60",
    "--max-generator-batches 60",
    "--max-eval-batches 60",
    "--validation-windows 5",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _rounded(value: float) -> float:
    return float(f"{value:.12g}")


def _candidate_template(
    *,
    axis: str,
    priority: int,
    addresses: list[str],
    invariant_overrides: dict[str, float],
    rationale: str,
) -> dict[str, Any]:
    return {
        "axis": axis,
        "priority": priority,
        "addresses_blockers": addresses,
        "invariant_overrides": invariant_overrides,
        "rationale": rationale,
    }


def _candidate_templates(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    invariant = base_config.get("invariant", {})
    invariant = invariant if isinstance(invariant, dict) else {}
    return [
        _candidate_template(
            axis="history_guarded_release_floor",
            priority=10,
            addresses=[
                "positive_partial_windows_below_required",
                "min_partial_mse_below_floor",
                "min_partial_mae_below_floor",
            ],
            invariant_overrides={
                "lambda_release_first_increment": _rounded(_num(invariant, "lambda_release_first_increment", 0.4) - 0.02),
                "lambda_history_budget_cap": _rounded(_num(invariant, "lambda_history_budget_cap", 0.035) + 0.015),
                "lambda_history_after_release": _rounded(_num(invariant, "lambda_history_after_release", 0.08) + 0.04),
                "lambda_invariant_correction_budget": _rounded(max(0.0, _num(invariant, "lambda_invariant_correction_budget", 0.02) - 0.002)),
            },
            rationale=(
                "Reduce release-first amplitude while allowing more post-release history support. "
                "This targets partial MSE/MAE floors without changing V29 topology."
            ),
        ),
        _candidate_template(
            axis="counterfactual_margin_micro_repair",
            priority=20,
            addresses=[
                "counterfactual_margin_negative",
                "min_partial_mse_below_floor",
            ],
            invariant_overrides={
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.005),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.001) + 0.0005),
                "lambda_release_temporal": _rounded(_num(invariant, "lambda_release_temporal", 0.04) + 0.005),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.12) + 0.02),
            },
            rationale=(
                "Use a small counterfactual-margin increase only with temporal/router stabilizers, "
                "avoiding the known failed single-axis counterfactual-shuffle sweep."
            ),
        ),
        _candidate_template(
            axis="late_window_partial_floor_guard",
            priority=30,
            addresses=[
                "positive_partial_windows_below_required",
                "min_partial_mse_below_floor",
                "min_partial_mae_below_floor",
            ],
            invariant_overrides={
                "lambda_release_temporal": _rounded(_num(invariant, "lambda_release_temporal", 0.04) + 0.01),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.02),
                "lambda_completion_after_release": _rounded(_num(invariant, "lambda_completion_after_release", 0.08) + 0.02),
                "lambda_release_state_advantage": _rounded(max(0.0, _num(invariant, "lambda_release_state_advantage", 0.12) - 0.02)),
            },
            rationale=(
                "Stabilize release completion and anchor preservation to reduce late-window partial "
                "floor failures while keeping the one-epoch stability anchor."
            ),
        ),
        _candidate_template(
            axis="delayed_change_tempered_history",
            priority=40,
            addresses=[
                "positive_partial_windows_below_required",
                "counterfactual_margin_negative",
            ],
            invariant_overrides={
                "lambda_delayed_change": _rounded(max(0.0, _num(invariant, "lambda_delayed_change", 0.28) - 0.02)),
                "lambda_change_readout": _rounded(max(0.0, _num(invariant, "lambda_change_readout", 0.3) - 0.02)),
                "lambda_history_budget_cap": _rounded(_num(invariant, "lambda_history_budget_cap", 0.035) + 0.01),
                "lambda_support_alignment": _rounded(max(0.0, _num(invariant, "lambda_support_alignment", 0.14) - 0.02)),
            },
            rationale=(
                "Temper delayed-change/readout pressure and add history budget as a coupled check "
                "against over-corrected release states."
            ),
        ),
        _candidate_template(
            axis="release_null_coupled_guard",
            priority=50,
            addresses=[
                "counterfactual_margin_negative",
                "min_partial_mae_below_floor",
            ],
            invariant_overrides={
                "lambda_release_null": _rounded(_num(invariant, "lambda_release_null", 0.05) + 0.002),
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.08) + 0.003),
                "lambda_release_first_increment": _rounded(_num(invariant, "lambda_release_first_increment", 0.4) - 0.01),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.12) + 0.01),
            },
            rationale=(
                "Probe release-null only as a coupled guard with amplitude reduction and anchor "
                "preservation, not as the previously failed single-axis null sweep."
            ),
        ),
    ]


def _top_candidate_report(preflight_report: dict[str, Any]) -> dict[str, Any]:
    reports = preflight_report.get("candidate_reports", [])
    if isinstance(reports, list) and reports:
        report = reports[0]
        return report if isinstance(report, dict) else {}
    return {}


def _suggested_command(*, config_name: str, run_id: str) -> str:
    budget = " ".join(DEFAULT_BUDGET_ARGS)
    return (
        "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
        f"--config {DEFAULT_CONFIG_DIR / config_name} "
        f"--run-id {run_id} {budget}"
    )


def build_teacher_gate_tuning_matrix(
    *,
    base_config: dict[str, Any],
    preflight_report: dict[str, Any],
    run_id_prefix: str = "v29teachergate",
) -> list[dict[str, Any]]:
    report = _top_candidate_report(preflight_report)
    blockers = [str(item) for item in report.get("blockers", [])] if isinstance(report.get("blockers", []), list) else []
    blocker_set = set(blockers)
    if bool(preflight_report.get("preflight_pass", False)):
        return []

    rows: list[dict[str, Any]] = []
    for index, template in enumerate(_candidate_templates(base_config), start=1):
        addresses = [item for item in template["addresses_blockers"] if item in blocker_set]
        if not addresses:
            continue
        run_id = f"{run_id_prefix}_{index:02d}_{template['axis']}"
        config_name = f"stage3_v29_{run_id}.yaml"
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "candidate_id": f"{index:02d}_{template['axis']}",
                "priority": int(template["priority"]),
                "axis": template["axis"],
                "run_id": run_id,
                "config_name": config_name,
                "source_preflight_decision": str(preflight_report.get("decision", "")),
                "source_candidate_run_id": str(report.get("run_id", "")),
                "addresses_blockers": addresses,
                "all_preflight_blockers": blockers,
                "architecture_policy": "frozen_v29_no_model_topology_change",
                "known_failed_axis_guard": "coupled_overrides_only_no_single_axis_repeat",
                "model_overrides": {},
                "train_overrides": {},
                "invariant_overrides": template["invariant_overrides"],
                "command_budget": " ".join(DEFAULT_BUDGET_ARGS),
                "suggested_probe_command": _suggested_command(config_name=config_name, run_id=run_id),
                "rationale": template["rationale"],
                "promotion_gate": (
                    "rerun selector, complete-only mode-gate replay, and generator-unlock preflight; "
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
    train = cfg.setdefault("train", {})
    if not isinstance(train, dict):
        train = {}
        cfg["train"] = train
    train.update(dict(candidate.get("train_overrides", {})))
    stage3 = cfg.setdefault("stage3_v2", {})
    if not isinstance(stage3, dict):
        stage3 = {}
        cfg["stage3_v2"] = stage3
    stage3["run_id"] = str(candidate["run_id"])
    stage3["tuning_matrix_protocol"] = PROTOCOL_VERSION
    stage3["tuning_matrix_candidate_id"] = str(candidate["candidate_id"])
    stage3["tuning_matrix_source_candidate_run_id"] = str(candidate["source_candidate_run_id"])
    cfg["tuning_matrix"] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": candidate["candidate_id"],
        "axis": candidate["axis"],
        "addresses_blockers": candidate["addresses_blockers"],
        "architecture_policy": candidate["architecture_policy"],
        "known_failed_axis_guard": candidate["known_failed_axis_guard"],
        "promotion_gate": candidate["promotion_gate"],
    }
    return cfg


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 teacher-gate matrix output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_scalar(value) for key, value in row.items()})


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Teacher-Gate Tuning Matrix",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Candidate rows: `{len(rows)}`.",
        "",
        "This is a non-training run-control artifact. It converts the current "
        "generator-unlock preflight blockers into frozen-architecture V29 tuning "
        "candidates. It is not performance evidence.",
        "",
        "| Priority | Candidate | Axis | Addresses | Run ID |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | `{candidate}` | `{axis}` | `{addresses}` | `{run}` |".format(
                priority=row["priority"],
                candidate=row["candidate_id"],
                axis=row["axis"],
                addresses=";".join(row["addresses_blockers"]),
                run=row["run_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Execution Gate",
            "",
            "Run these candidates only after selector/preflight artifacts are current. "
            "After each completed candidate, refresh the complete-only selector and "
            "rerun generator-unlock preflight. Do not start generator training unless "
            "`preflight_pass=true`.",
            "",
            "## Architecture Guard",
            "",
            "Every candidate keeps V29 model topology frozen and changes only coupled "
            "training/invariant scalar controls. Single-axis repeats of already failed "
            "counterfactual, release-null, release-first, learning-rate, readout, or "
            "delayed-change sweeps are intentionally avoided.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_teacher_gate_tuning_matrix_artifacts(
    *,
    candidates: list[dict[str, Any]],
    base_config: dict[str, Any],
    out_dir: Path,
    config_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    config_paths = [config_dir / str(candidate["config_name"]) for candidate in candidates]
    targets = [csv_path, md_path, *config_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 teacher-gate matrix output: {existing[0]}")

    _write_csv(csv_path, candidates, overwrite=overwrite)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_build_markdown(candidates), encoding="utf-8")
    config_dir.mkdir(parents=True, exist_ok=True)
    for candidate, config_path in zip(candidates, config_paths):
        cfg = _merge_config(base_config, candidate)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "configs": [str(path) for path in config_paths],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan frozen-architecture V29 teacher-gate tuning candidates.")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--preflight-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), type=Path)
    parser.add_argument("--prefix", default="v29_teacher_gate_tuning_matrix")
    parser.add_argument("--run-id-prefix", default="v29teachergate")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_config = _read_yaml(args.base_config)
    preflight_report = _read_json(args.preflight_json)
    candidates = build_teacher_gate_tuning_matrix(
        base_config=base_config,
        preflight_report=preflight_report,
        run_id_prefix=str(args.run_id_prefix),
    )
    paths = write_teacher_gate_tuning_matrix_artifacts(
        candidates=candidates,
        base_config=base_config,
        out_dir=args.out_dir,
        config_dir=args.config_dir,
        prefix=str(args.prefix),
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['markdown']}")
    for config_path in paths["configs"]:
        print(f"wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
