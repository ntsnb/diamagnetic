from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
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
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_single_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


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
        for key in row.keys():
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
) -> dict[str, Any]:
    rolling = _read_single_csv_row(out_dir / "teacher_only_rolling_validation_aggregate.csv")
    test = _read_single_csv_row(out_dir / "teacher_only_test_diagnostic_summary.csv")
    teacher = _read_single_csv_row(out_dir / "residual_teacher_v12c_diagnostics.csv")
    if not teacher:
        teacher = _read_single_csv_row(out_dir / "residual_teacher_v12_diagnostics.csv")
    if not teacher:
        teacher = _read_single_csv_row(out_dir / "residual_teacher_v11_diagnostics.csv")
    alignment_rows_path = out_dir / "alignment_diagnostics.csv"
    alignment = _read_single_csv_row(alignment_rows_path)
    abort_payload = _read_json(out_dir / "teacher_only_abort.json")
    complete_payload = _read_json(out_dir / "teacher_only_complete.json")

    row: dict[str, Any] = {
        "status": status,
        "returncode": "" if returncode is None else returncode,
        "elapsed_seconds": "" if elapsed_seconds is None else f"{elapsed_seconds:.3f}",
        "out_dir": str(out_dir),
        "ckpt_dir": str(ckpt_dir),
        "log_path": str(log_path),
        "command": " ".join(command),
        "teacher_abort_written": bool(abort_payload),
        "teacher_complete_written": bool(complete_payload),
        "teacher_gate_pass": rolling.get("teacher_diagnostic_gate_pass", ""),
        "teacher_gate_failure_reasons": rolling.get("teacher_diagnostic_failure_reasons", ""),
        "teacher_window_pass_count": rolling.get("teacher_window_pass_count", ""),
        "teacher_validation_window_count": rolling.get("teacher_window_count", ""),
        "teacher_worst_real_mse_gain": rolling.get("teacher_worst_real_mse_gain", ""),
        "teacher_worst_real_mae_gain": rolling.get("teacher_worst_real_mae_gain", ""),
        "teacher_worst_real_vs_best_cf_margin": rolling.get(
            "teacher_worst_real_vs_best_counterfactual_mse_gain_margin", ""
        ),
        "teacher_worst_real_sign_agreement": rolling.get("teacher_worst_sign_agreement_rate", ""),
        "teacher_test_mse_gain": test.get("teacher_real_mse_gain", ""),
        "teacher_test_mae_gain": test.get("teacher_real_mae_gain", ""),
        "teacher_test_sign_agreement": test.get("teacher_real_sign_agreement_rate", ""),
        "teacher_train_r2": teacher.get("teacher_residual_r2", ""),
        "teacher_oof_mse_gain": teacher.get("teacher_oof_mse_gain", ""),
        "teacher_oof_mae_gain": teacher.get("teacher_oof_mae_gain", ""),
        "teacher_oof_sign_agreement": teacher.get("teacher_oof_sign_agreement", ""),
        "teacher_output_pass_count": teacher.get("teacher_output_pass_count", ""),
        "teacher_output_count": teacher.get("teacher_output_count", ""),
        "teacher_oof_fold_count": teacher.get("oof_fold_count", ""),
        "teacher_shrinkage_alpha": teacher.get("teacher_shrinkage_alpha", ""),
        "teacher_alpha_source": teacher.get("teacher_alpha_source", ""),
        "alignment_method": alignment.get("alignment_method", ""),
        "alignment_gate_pass": alignment.get("alignment_gate_pass", ""),
        "alignment_active_rank": alignment.get("active_rank", ""),
        "alignment_selected_leakage_rank": alignment.get("selected_leakage_rank", ""),
        "alignment_teacher_residual_r2": alignment.get(
            "teacher_residual_r2",
            alignment.get("selected_residual_predictability_r2", ""),
        ),
        "alignment_oof_transfer_mse_gain": alignment.get("oof_transfer_mse_gain", ""),
        "alignment_active_min_block_transfer_r2": alignment.get("active_min_block_transfer_r2", ""),
        "alignment_active_min_block_sign_agreement": alignment.get("active_min_block_sign_agreement", ""),
        "alignment_retained_variance_fraction": alignment.get("selected_retained_variance_fraction", ""),
        "alignment_invariant_predictability_r2": alignment.get("selected_invariant_predictability_r2", ""),
        "generator_served_artifact_exists": (ckpt_dir / "generator_served_policy.pt").exists(),
    }
    return row


def _build_command(args: argparse.Namespace, *, dataset: str, seed: int, run_id: str) -> list[str]:
    command = [
        args.python_executable,
        "train\\run_stage3_v12c_sensitivity_audit.py",
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
        "--generator-epochs",
        str(args.generator_epochs),
        "--max-generator-batches",
        str(args.max_generator_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--out-root",
        args.out_root,
        "--checkpoint-root",
        args.checkpoint_root,
        "--teacher-only",
        "--modes",
        *args.modes,
    ]
    if args.normalize_vector:
        command.append("--normalize-vector")
    else:
        command.append("--no-normalize-vector")
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-overwriting V12c teacher-only audit matrix.")
    parser.add_argument("--config", default="configs/stage3_v12c_multi_policy_teacher_probe.yaml")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--invariant-epochs", type=int, default=5)
    parser.add_argument("--generator-epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=480)
    parser.add_argument("--max-eval-batches", type=int, default=480)
    parser.add_argument("--validation-windows", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--drift-source", choices=["phi_gap", "delta_phi_v"], default="delta_phi_v")
    parser.add_argument("--drift-polarity", type=float, default=1.0)
    parser.add_argument("--signal-mode", choices=["level_scalar", "delta_scalar", "level_vector", "delta_vector"], default="level_vector")
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--out-root", default="logs/s3v12cm")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v12cm")
    parser.add_argument("--run-prefix", default="v12ctm")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
                summaries.append(
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

            start = datetime.now()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"run_id: {run_id}\n")
                log_file.write(f"dataset: {dataset}\n")
                log_file.write(f"seed: {seed}\n")
                log_file.write("command: " + " ".join(command) + "\n\n")
                log_file.flush()
                proc = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = (datetime.now() - start).total_seconds()
            status = "completed" if proc.returncode == 0 else "failed"
            summary = _summarize_run(
                out_dir=out_dir,
                ckpt_dir=ckpt_dir,
                command=command,
                returncode=proc.returncode,
                elapsed_seconds=elapsed,
                status=status,
                log_path=log_path,
            )
            summary.update({"dataset": dataset, "seed": seed, "run_id": run_id, "matrix_id": matrix_id})
            summaries.append(summary)
            _write_csv(manifest_dir / "matrix_summary_partial.csv", summaries)
            if proc.returncode != 0 and args.stop_on_error:
                break
        else:
            continue
        break

    manifest = {
        "matrix_id": matrix_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "invariant_epochs": args.invariant_epochs,
        "max_train_batches": args.max_train_batches,
        "max_generator_batches": args.max_generator_batches,
        "max_eval_batches": args.max_eval_batches,
        "validation_windows": args.validation_windows,
        "teacher_only": True,
        "dry_run": bool(args.dry_run),
        "planned_runs": planned_runs,
        "summary_csv": str(manifest_dir / "matrix_summary.csv"),
    }
    _write_json(manifest_dir / "matrix_manifest.json", manifest)
    _write_csv(manifest_dir / "matrix_summary.csv", summaries)
    print(json.dumps({"matrix_id": matrix_id, "manifest_dir": str(manifest_dir), "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()



