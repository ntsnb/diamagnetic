from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
from typing import Any


PROTOCOL_VERSION = "stage3_v29_tuning_execution_plan_v1"
MAX_WINDOWS_PATH_LENGTH = 259
LONGEST_STAGE3_V29_TEACHER_CHECKPOINT_NAME = "teacher_v18_residual_change_vector_leveloffsetplushistory.pt"
DEFAULT_BUDGET = {
    "epochs": 1,
    "max_train_batches": 60,
    "max_generator_batches": 60,
    "max_eval_batches": 60,
    "validation_windows": 5,
}
SUPPORTED_DATASET_OVERRIDES = {"panama", "tetouan", "spain_energy_weather"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_int(value: object, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _quote_path(path: Path) -> str:
    text = str(path)
    if " " in text:
        return shlex.quote(text)
    return text


def _config_path(config_dir: Path, row: dict[str, Any]) -> Path:
    raw = str(row.get("config_path") or row.get("config_name") or "")
    path = Path(raw)
    return path if path.is_absolute() else config_dir / path


def _run_output_dir(out_root: Path, run_id: str, dataset: str, seed: int) -> Path:
    return out_root / run_id / dataset / f"seed_{seed}"


def _checkpoint_output_dir(checkpoint_root: Path, run_id: str, dataset: str, seed: int) -> Path:
    return checkpoint_root / dataset / run_id / f"seed_{seed}"


def _path_length(path: Path) -> int:
    try:
        return len(str(path.resolve()))
    except OSError:
        return len(str(path.absolute()))


def _command(
    *,
    config_path: Path,
    run_id: str,
    dataset: str,
    seed: int,
    out_root: Path,
    checkpoint_root: Path,
    device: str,
    budget: dict[str, int],
) -> str:
    dataset_arg = f"--dataset {dataset} " if dataset in SUPPORTED_DATASET_OVERRIDES else ""
    return (
        "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
        f"--config {_quote_path(config_path)} "
        f"{dataset_arg}"
        f"--run-id {run_id} "
        f"--seed {seed} "
        f"--device {device} "
        f"--epochs {budget['epochs']} "
        f"--max-train-batches {budget['max_train_batches']} "
        f"--max-generator-batches {budget['max_generator_batches']} "
        f"--max-eval-batches {budget['max_eval_batches']} "
        f"--validation-windows {budget['validation_windows']} "
        f"--out-root {_quote_path(out_root)} "
        f"--checkpoint-root {_quote_path(checkpoint_root)}"
    )


def build_tuning_execution_plan(
    *,
    matrix_rows: list[dict[str, Any]],
    config_dir: Path,
    out_root: Path,
    checkpoint_root: Path,
    dataset: str = "weather",
    seed: int = 42,
    device: str = "auto",
    budget: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    resolved_budget = dict(DEFAULT_BUDGET)
    if budget:
        resolved_budget.update({key: int(value) for key, value in budget.items()})
    rows: list[dict[str, Any]] = []
    sorted_rows = sorted(matrix_rows, key=lambda row: (_to_int(row.get("priority"), 999), str(row.get("run_id", ""))))
    for order, matrix_row in enumerate(sorted_rows, start=1):
        run_id = str(matrix_row.get("run_id", "")).strip()
        config_path = _config_path(config_dir, matrix_row)
        run_dir = _run_output_dir(out_root, run_id, dataset, seed)
        ckpt_dir = _checkpoint_output_dir(checkpoint_root, run_id, dataset, seed)
        max_checkpoint_file_path = ckpt_dir / LONGEST_STAGE3_V29_TEACHER_CHECKPOINT_NAME
        max_checkpoint_file_path_length = _path_length(max_checkpoint_file_path)
        block_reasons: list[str] = []
        if not run_id:
            block_reasons.append("missing_run_id")
        if not config_path.exists():
            block_reasons.append("config_path_missing")
        if run_dir.exists():
            block_reasons.append("run_output_dir_exists")
        if ckpt_dir.exists():
            block_reasons.append("checkpoint_output_dir_exists")
        if max_checkpoint_file_path_length > MAX_WINDOWS_PATH_LENGTH:
            block_reasons.append("checkpoint_file_path_too_long")
        status = "ready_to_run" if not block_reasons else "blocked_existing_output"
        if "config_path_missing" in block_reasons or "missing_run_id" in block_reasons:
            status = "blocked_invalid_candidate"
        elif block_reasons == ["checkpoint_file_path_too_long"]:
            status = "blocked_runtime_path_length"
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "execution_order": order,
                "candidate_id": matrix_row.get("candidate_id", ""),
                "axis": matrix_row.get("axis", ""),
                "priority": matrix_row.get("priority", ""),
                "run_id": run_id,
                "dataset": dataset,
                "seed": seed,
                "config_path": str(config_path),
                "run_output_dir": str(run_dir),
                "checkpoint_output_dir": str(ckpt_dir),
                "execution_status": status,
                "block_reasons": ";".join(block_reasons),
                "max_windows_path_length": MAX_WINDOWS_PATH_LENGTH,
                "max_checkpoint_file_path_length": max_checkpoint_file_path_length,
                "max_checkpoint_file_path": str(max_checkpoint_file_path),
                "overwrite_policy": "no_overwrite",
                "post_run_required_actions": "refresh_complete_selector;rerun_mode_gate_replay;rerun_generator_unlock_preflight",
                "command": _command(
                    config_path=config_path,
                    run_id=run_id,
                    dataset=dataset,
                    seed=seed,
                    out_root=out_root,
                    checkpoint_root=checkpoint_root,
                    device=device,
                    budget=resolved_budget,
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning execution output: {path}")
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


def _markdown(rows: list[dict[str, Any]]) -> str:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("execution_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Stage III V29 Tuning Execution Plan",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Rows: `{len(rows)}`.",
        "",
        "This plan is non-executing. It lists bounded probe commands and checks "
        "whether the intended output paths already exist. Commands intentionally "
        "omit `--overwrite`.",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "| Order | Run | Axis | Status | Blockers |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {order} | `{run}` | `{axis}` | `{status}` | `{blockers}` |".format(
                order=row.get("execution_order", ""),
                run=row.get("run_id", ""),
                axis=row.get("axis", ""),
                status=row.get("execution_status", ""),
                blockers=row.get("block_reasons", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Post-Run Gate",
            "",
            "After each completed probe, refresh the complete-only selector, rerun "
            "mode-gate outcome replay, and rerun generator-unlock preflight. Do not "
            "start generator training unless preflight passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tuning_execution_plan_artifacts(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, str]:
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 tuning execution output: {md_path}")
    _write_csv(csv_path, rows, overwrite=overwrite)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a safe non-overwriting execution plan for V29 tuning candidates.")
    parser.add_argument("--matrix-csv", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="v29_tuning_execution_plan")
    parser.add_argument("--dataset", default="weather")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", default=DEFAULT_BUDGET["epochs"], type=int)
    parser.add_argument("--max-train-batches", default=DEFAULT_BUDGET["max_train_batches"], type=int)
    parser.add_argument("--max-generator-batches", default=DEFAULT_BUDGET["max_generator_batches"], type=int)
    parser.add_argument("--max-eval-batches", default=DEFAULT_BUDGET["max_eval_batches"], type=int)
    parser.add_argument("--validation-windows", default=DEFAULT_BUDGET["validation_windows"], type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = build_tuning_execution_plan(
        matrix_rows=_read_csv(args.matrix_csv),
        config_dir=args.config_dir,
        out_root=args.out_root,
        checkpoint_root=args.checkpoint_root,
        dataset=str(args.dataset),
        seed=int(args.seed),
        device=str(args.device),
        budget={
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_generator_batches": args.max_generator_batches,
            "max_eval_batches": args.max_eval_batches,
            "validation_windows": args.validation_windows,
        },
    )
    written = write_tuning_execution_plan_artifacts(
        rows=rows,
        out_dir=args.out_dir,
        prefix=str(args.prefix),
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {written['csv']}")
    print(f"wrote {written['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
