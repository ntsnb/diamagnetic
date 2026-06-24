from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("panama", "tetouan", "spain_energy_weather")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MODES = (
    "real",
    "zero",
    "shuffled",
    "sign_flipped",
    "amplified_2x",
    "amplified_5x",
    "random_matched_norm",
    "conditional_shuffled",
    "conditional_random_matched_norm",
)
DEFAULT_FEATURE_MODES = (
    "history_only",
    "fluctuation_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
)
DEFAULT_TARGETS = ("raw_residual_vector", "residual_change_vector", "residual_sign")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_run(
    *,
    out_dir: Path,
    ckpt_dir: Path,
    command: list[str],
    returncode: int | None,
    elapsed_seconds: float | None,
    status: str,
    log_path: Path,
) -> list[dict[str, Any]]:
    gate_rows = _read_csv_rows(out_dir / "teacher_only_v15_target_gate.csv")
    abort_payload = _read_json(out_dir / "teacher_only_abort.json")
    complete_payload = _read_json(out_dir / "teacher_only_complete.json")
    rows: list[dict[str, Any]] = []
    if not gate_rows:
        gate_rows = [{"target_name": "", "target_role": "", "primary_gate_pass": ""}]
    for gate in gate_rows:
        rows.append(
            {
                "status": status,
                "returncode": "" if returncode is None else returncode,
                "elapsed_seconds": "" if elapsed_seconds is None else f"{elapsed_seconds:.3f}",
                "target_name": gate.get("target_name", ""),
                "target_role": gate.get("target_role", ""),
                "primary_gate_pass": gate.get("primary_gate_pass", ""),
                "positive_partial_window_count": gate.get("positive_partial_window_count", ""),
                "window_count": gate.get("window_count", ""),
                "min_partial_mse_gain_vs_history": gate.get("min_partial_mse_gain_vs_history", ""),
                "min_partial_mae_gain_vs_history": gate.get("min_partial_mae_gain_vs_history", ""),
                "counterfactual_margin": gate.get("counterfactual_margin", ""),
                "counterfactual_pass": gate.get("counterfactual_pass", ""),
                "out_dir": str(out_dir),
                "ckpt_dir": str(ckpt_dir),
                "log_path": str(log_path),
                "command": " ".join(command),
                "teacher_abort_written": bool(abort_payload),
                "teacher_complete_written": bool(complete_payload),
                "directional_stop_reasons": ";".join(str(reason) for reason in abort_payload.get("directional_stop_reasons", [])),
                "generator_artifact_exists": bool(list(ckpt_dir.glob("generator*.pt"))),
            }
        )
    return rows


def _build_command(args: argparse.Namespace, *, dataset: str, seed: int, run_id: str) -> list[str]:
    command = [
        args.python_executable,
        "train\\run_stage3_v15_residual_target_predictability_probe.py",
        "--config",
        args.config,
        "--dataset",
        dataset,
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--validation-windows",
        str(args.validation_windows),
        "--drift-source",
        args.drift_source,
        "--drift-polarity",
        str(args.drift_polarity),
        "--signal-mode",
        args.signal_mode,
        "--epochs",
        str(args.epochs),
        "--invariant-epochs",
        str(args.invariant_epochs),
        "--max-generator-batches",
        str(args.max_generator_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--out-root",
        args.out_root,
        "--checkpoint-root",
        args.checkpoint_root,
        "--modes",
        *args.modes,
        "--feature-modes",
        *args.feature_modes,
        "--targets",
        *args.targets,
    ]
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    command.append("--normalize-vector" if args.normalize_vector else "--no-normalize-vector")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-overwriting V15 residual-target predictability matrix.")
    parser.add_argument("--config", default="configs/stage3_v15_residual_target_predictability_probe.yaml")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--invariant-epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=480)
    parser.add_argument("--max-eval-batches", type=int, default=480)
    parser.add_argument("--validation-windows", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--drift-source", choices=["delta_phi_v"], default="delta_phi_v")
    parser.add_argument("--drift-polarity", type=float, default=1.0)
    parser.add_argument("--signal-mode", choices=["fluctuation_delta"], default="fluctuation_delta")
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--feature-modes", nargs="*", default=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--out-root", default="logs/s3v15target_matrix")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v15target_matrix")
    parser.add_argument("--run-prefix", default="v15targetm")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--stop-on-direction-error", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_id = (
        f"{args.run_prefix}_{timestamp}_"
        f"e{args.epochs}_ie{args.invariant_epochs}_b{args.max_generator_batches}_ew{args.max_eval_batches}"
    )
    manifest_dir = root / args.out_root / "_manifests" / matrix_id
    manifest_dir.mkdir(parents=True, exist_ok=False)

    planned_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    stop_matrix = False
    for dataset in args.datasets:
        for seed in args.seeds:
            run_id = f"{matrix_id}_{dataset}_s{seed}"
            out_dir = root / args.out_root / run_id / dataset / f"seed_{seed}"
            ckpt_dir = root / args.checkpoint_root / dataset / run_id / f"seed_{seed}"
            command = _build_command(args, dataset=dataset, seed=seed, run_id=run_id)
            log_path = manifest_dir / f"{dataset}_seed_{seed}.log"
            planned_runs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "run_id": run_id,
                    "out_dir": str(out_dir),
                    "ckpt_dir": str(ckpt_dir),
                    "log_path": str(log_path),
                    "command": command,
                }
            )
            if args.dry_run:
                summaries.extend(
                    _summarize_run(
                        out_dir=out_dir,
                        ckpt_dir=ckpt_dir,
                        command=command,
                        returncode=None,
                        elapsed_seconds=None,
                        status="planned",
                        log_path=log_path,
                    )
                )
                continue

            start = time.time()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"run_id: {run_id}\n")
                log_file.write(f"dataset: {dataset}\n")
                log_file.write(f"seed: {seed}\n")
                log_file.write("command: " + " ".join(command) + "\n\n")
                log_file.flush()
                proc = subprocess.run(command, cwd=root, stdout=log_file, stderr=subprocess.STDOUT, check=False)
            elapsed = time.time() - start
            status = "completed" if proc.returncode == 0 else "failed"
            summaries.extend(
                _summarize_run(
                    out_dir=out_dir,
                    ckpt_dir=ckpt_dir,
                    command=command,
                    returncode=proc.returncode,
                    elapsed_seconds=elapsed,
                    status=status,
                    log_path=log_path,
                )
            )
            _write_csv(manifest_dir / "matrix_summary.csv", summaries)
            abort_payload = _read_json(out_dir / "teacher_only_abort.json")
            if args.stop_on_direction_error and abort_payload.get("directional_stop_reasons"):
                stop_matrix = True
                break
            if proc.returncode != 0 and args.stop_on_error:
                stop_matrix = True
                break
        if stop_matrix:
            break

    manifest = {
        "matrix_id": matrix_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "feature_modes": args.feature_modes,
        "targets": args.targets,
        "modes": args.modes,
        "out_root": args.out_root,
        "checkpoint_root": args.checkpoint_root,
        "planned_runs": planned_runs,
        "stopped_on_direction_error": stop_matrix,
        "summary_csv": str(manifest_dir / "matrix_summary.csv"),
    }
    _write_json(manifest_dir / "matrix_manifest.json", manifest)
    _write_json(manifest_dir / "planned_runs.json", planned_runs)
    _write_csv(manifest_dir / "matrix_summary.csv", summaries)
    print(manifest_dir)


if __name__ == "__main__":
    main()
