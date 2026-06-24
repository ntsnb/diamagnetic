from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "stage3_v29_tuning_queue_v1"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return ""


def infer_candidate_axis(config_path: Path, cfg: dict[str, Any]) -> str:
    name = config_path.stem.lower()
    invariant = cfg.get("invariant", {})
    invariant = invariant if isinstance(invariant, dict) else {}
    if "cf_shuffle" in name or "counterfactual_shuffle" in name:
        return "counterfactual_shuffle"
    if "change_readout" in name or "_cr" in name:
        return "change_readout"
    if "delayed_change" in name or "_dc" in name:
        return "delayed_change"
    if "release_null" in name:
        return "release_null"
    if "release_first" in name:
        return "release_first"
    if "_lr_" in name or name.endswith("_lr"):
        return "learning_rate"
    if "stability" in name:
        return "stability"
    if "guard" in name:
        return "guarded_counterfactual"
    if "lambda_counterfactual_shuffle" in invariant:
        return "counterfactual_shuffle"
    if "lambda_change_readout" in invariant:
        return "change_readout"
    if "lambda_delayed_change" in invariant:
        return "delayed_change"
    if "lambda_release_null" in invariant:
        return "release_null"
    if "lambda_release_first_increment" in invariant:
        return "release_first"
    return "other"


def _extract_run_id(config_path: Path, cfg: dict[str, Any]) -> str:
    stage3 = cfg.get("stage3_v2", {})
    if isinstance(stage3, dict) and stage3.get("run_id"):
        return str(stage3["run_id"])
    return config_path.stem


def _candidate_dataset(cfg: dict[str, Any]) -> str:
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return ""
    for key in ("dataset", "name", "target_dataset"):
        if data.get(key):
            return str(data[key]).strip().lower()
    raw_path = str(data.get("path", "")).strip()
    if raw_path:
        return Path(raw_path).stem.lower()
    return ""


def _run_dir_dataset(run_dir: Path, manifest: dict[str, Any]) -> str:
    if manifest.get("dataset"):
        return str(manifest["dataset"]).strip().lower()
    try:
        return run_dir.parent.name.strip().lower()
    except IndexError:
        return ""


def _dataset_compatible(candidate_dataset: str, completed_dataset: str) -> bool:
    return not candidate_dataset or not completed_dataset or candidate_dataset == completed_dataset


def _tuning_signature(cfg: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    train = cfg.get("train", {})
    invariant = cfg.get("invariant", {})
    train = train if isinstance(train, dict) else {}
    invariant = invariant if isinstance(invariant, dict) else {}
    parts: list[tuple[str, str]] = []
    if "lr" in train:
        parts.append(("train.lr", _format_float(train.get("lr"))))
    for key in sorted(invariant):
        value = invariant.get(key)
        if isinstance(value, (int, float, str, bool)):
            parts.append((f"invariant.{key}", str(value)))
    return tuple(parts)


def _read_resolved_config(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    resolved = manifest.get("resolved_config_json", {})
    if not isinstance(resolved, dict):
        return {}
    raw_path = str(resolved.get("resolved_config_path", ""))
    if not raw_path:
        return {}
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(run_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate)
    return {}


def _signature_matches(candidate: tuple[tuple[str, str], ...], completed: tuple[tuple[str, str], ...]) -> bool:
    if not candidate or not completed:
        return False
    return set(candidate).issubset(set(completed)) or set(completed).issubset(set(candidate))


def _signature_match_allowed_for_run_dir(run_dir: Path) -> bool:
    parts = [part.lower() for part in run_dir.parts]
    stage_roots = [part for part in parts if part.startswith("s3v29expert")]
    if not stage_roots:
        return True
    return any(part.startswith("s3v29expert_tuning") for part in stage_roots)


def _discover_completed_runs(log_roots: list[Path]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    runs: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    recorded_dirs: set[Path] = set()
    for root in log_roots:
        if not root.exists():
            continue
        for manifest_path in root.rglob("run_manifest_v29.json"):
            run_dir = manifest_path.parent
            manifest = _read_json(manifest_path)
            run_id = str(manifest.get("run_id") or run_dir.parents[1].name)
            if run_id:
                runs.setdefault(run_id, run_dir)
            resolved_cfg = _read_resolved_config(manifest, run_dir)
            records.append(
                {
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "manifest": manifest,
                    "dataset": _run_dir_dataset(run_dir, manifest),
                    "signature": _tuning_signature(resolved_cfg) if resolved_cfg else (),
                }
            )
            recorded_dirs.add(run_dir.resolve())
        for gate_path in root.rglob("teacher_only_v29_target_gate.csv"):
            run_dir = gate_path.parent
            run_id = run_dir.parents[1].name
            runs.setdefault(run_id, run_dir)
            if run_dir.resolve() not in recorded_dirs:
                records.append(
                    {
                        "run_id": run_id,
                        "run_dir": run_dir,
                        "manifest": {},
                        "dataset": _run_dir_dataset(run_dir, {}),
                        "signature": (),
                    }
                )
                recorded_dirs.add(run_dir.resolve())
    return runs, records


def _selector_rows(selector_csv: Path | None) -> dict[str, dict[str, str]]:
    if selector_csv is None:
        return {}
    rows = _read_csv(selector_csv)
    return {str(row.get("run_id", "")): row for row in rows if row.get("run_id")}


def _gate_summary(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return {}
    rows = _read_csv(run_dir / "teacher_only_v29_target_gate.csv")
    for row in rows:
        if row.get("target_name") == "raw_residual_vector":
            return row
    return rows[0] if rows else {}


def _manifest_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    return _read_json(run_dir / "run_manifest_v29.json")


def _status_and_action(*, completed: bool, in_selector: bool, gate_pass: bool) -> tuple[str, str, int]:
    if completed and gate_pass and not in_selector:
        return "completed_gate_pass_needs_selector_refresh", "refresh_selector_and_promotion_review", 5
    if completed and not in_selector:
        return "completed_needs_selector_refresh", "refresh_selector_before_training", 10
    if completed and in_selector:
        return "covered_by_selector", "inspect_selector_rank_and_failure_windows", 80
    return "queued_not_run", "run_only_after_selector_refresh_queue_is_empty", 40


def _axis_priority_adjustment(axis: str) -> int:
    if axis in {"release_first", "release_null", "change_readout", "delayed_change"}:
        return 0
    if axis in {"counterfactual_shuffle", "guarded_counterfactual"}:
        return 5
    if axis == "learning_rate":
        return 15
    return 10


def build_tuning_queue(
    *,
    config_paths: list[Path],
    log_roots: list[Path],
    selector_csv: Path | None = None,
) -> list[dict[str, Any]]:
    completed_by_run, completed_records = _discover_completed_runs(log_roots)
    selector_by_run = _selector_rows(selector_csv)
    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        cfg = _read_yaml(config_path)
        run_id = _extract_run_id(config_path, cfg)
        candidate_dataset = _candidate_dataset(cfg)
        axis = infer_candidate_axis(config_path, cfg)
        signature = _tuning_signature(cfg)
        direct_run_dir = completed_by_run.get(run_id)
        matched_records = [
            record
            for record in completed_records
            if _dataset_compatible(candidate_dataset, str(record.get("dataset", "")))
            and (
                record["run_id"] == run_id
                or (
                    _signature_match_allowed_for_run_dir(Path(record["run_dir"]))
                    and _signature_matches(signature, record.get("signature", ()))
                )
            )
        ]
        if direct_run_dir is not None and all(record["run_dir"] != direct_run_dir for record in matched_records):
            matched_records.append(
                {
                    "run_id": run_id,
                    "run_dir": direct_run_dir,
                    "manifest": _manifest_summary(direct_run_dir),
                    "signature": signature,
                }
            )
        matched_run_ids = [str(record.get("run_id", "")) for record in matched_records if record.get("run_id")]
        run_dir = matched_records[0]["run_dir"] if matched_records else None
        selector_row = {}
        for matched_run_id in [run_id, *matched_run_ids]:
            selector_row = selector_by_run.get(matched_run_id, {})
            if selector_row:
                break
        gate_row = _gate_summary(run_dir)
        manifest = matched_records[0].get("manifest", {}) if matched_records else _manifest_summary(run_dir)
        completed = run_dir is not None
        in_selector = bool(selector_row)
        gate_pass = _truthy(gate_row.get("primary_gate_pass"))
        status, next_action, base_priority = _status_and_action(
            completed=completed,
            in_selector=in_selector,
            gate_pass=gate_pass,
        )
        priority = base_priority
        if status == "queued_not_run":
            priority += _axis_priority_adjustment(axis)
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "config_path": str(config_path),
                "config_name": config_path.name,
                "run_id": run_id,
                "matched_completed_run_ids": ";".join(sorted(set(matched_run_ids))),
                "matched_completed_run_dirs": ";".join(
                    sorted({str(record["run_dir"]) for record in matched_records if record.get("run_dir")})
                ),
                "candidate_axis": axis,
                "status": status,
                "next_action": next_action,
                "priority_band": priority,
                "completed_run_dir": str(run_dir or ""),
                "selector_rank": selector_row.get("strict_rank", ""),
                "dataset": manifest.get("dataset", ""),
                "seed": manifest.get("seed", cfg.get("seed", "")),
                "primary_gate_pass": str(gate_pass) if completed else "",
                "positive_partial_window_count": gate_row.get("positive_partial_window_count", ""),
                "required_positive_partial_window_count": gate_row.get("required_positive_partial_window_count", ""),
                "counterfactual_margin": _format_float(gate_row.get("counterfactual_margin")),
                "stop_reasons": ";".join(str(reason) for reason in manifest.get("stage3_stop_reasons", [])),
                "suggested_probe_command": (
                    "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
                    f"--config {config_path} --run-id {run_id}"
                ),
                "evidence_note": _evidence_note(status),
            }
        )
    return sorted(rows, key=lambda row: (int(row["priority_band"]), str(row["run_id"])))


def _evidence_note(status: str) -> str:
    if status == "covered_by_selector":
        return "Already in selector; inspect failure windows before further tuning."
    if status == "completed_needs_selector_refresh":
        return "Completed run exists but current selector omitted it; refresh selector before launching more training."
    if status == "completed_gate_pass_needs_selector_refresh":
        return "Gate-passing run exists outside selector; refresh selector and review promotion evidence first."
    return "Configuration exists but no completed V29 manifest/target gate was found."


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning queue artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning queue artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Stage III V29 Tuning Queue",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Candidate rows: `{len(rows)}`.",
        "",
        "Do not treat queued rows as performance evidence. They are run-control records only.",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Highest Priority Rows",
            "",
            "| Priority | Run | Axis | Status | Next action |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows[:20]:
        lines.append(
            "| {priority} | `{run}` | `{axis}` | `{status}` | `{action}` |".format(
                priority=row["priority_band"],
                run=row["run_id"],
                axis=row["candidate_axis"],
                status=row["status"],
                action=row["next_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "Refresh selector coverage before launching queued training if any completed run is outside the selector.",
            "Only launch `queued_not_run` rows after the selector and replay artifacts are synchronized.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tuning_queue_artifacts(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    _write_csv(csv_path, rows, overwrite=overwrite)
    _write_markdown(md_path, rows, overwrite=overwrite)
    return {"csv": csv_path, "md": md_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a non-training Stage III V29 tuning queue.")
    parser.add_argument(
        "--config-glob",
        action="append",
        default=["configs/stage3_v29_expert_local_invariant_release_probe_tune_*.yaml"],
    )
    parser.add_argument("--log-root", action="append", default=["logs"])
    parser.add_argument("--selector-csv", default="artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.csv")
    parser.add_argument("--out-dir", default="artifacts/stage3_v29_tuning_queue")
    parser.add_argument("--prefix", default="current_v29_tuning_queue")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_paths: list[Path] = []
    for pattern in args.config_glob:
        config_paths.extend(Path().glob(pattern))
    rows = build_tuning_queue(
        config_paths=sorted(set(config_paths)),
        log_roots=[Path(root) for root in args.log_root],
        selector_csv=Path(args.selector_csv) if args.selector_csv else None,
    )
    written = write_tuning_queue_artifacts(
        rows,
        out_dir=Path(args.out_dir),
        prefix=str(args.prefix),
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {written['csv']}")
    print(f"wrote {written['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
