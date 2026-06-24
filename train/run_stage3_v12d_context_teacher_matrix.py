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
DEFAULT_MODES = ("real", "zero", "shuffled", "sign_flipped", "amplified_2x", "random_matched_norm")
DEFAULT_FEATURE_MODES = ("drift_only", "history_only", "drift_plus_history")


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


def _by_feature(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("feature_mode", "")): row for row in rows if row.get("feature_mode")}


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
    rolling = _by_feature(_read_csv_rows(out_dir / "teacher_only_rolling_validation_aggregate.csv"))
    validation = _by_feature(_read_csv_rows(out_dir / "teacher_only_validation_diagnostic_summary.csv"))
    test = _by_feature(_read_csv_rows(out_dir / "teacher_only_test_diagnostic_summary.csv"))
    teacher = _by_feature(_read_csv_rows(out_dir / "residual_teacher_v12d_diagnostics.csv"))
    partial_validation_rows = _read_csv_rows(out_dir / "teacher_only_validation_partial_gain_vs_history.csv")
    partial_test_rows = _read_csv_rows(out_dir / "teacher_only_test_partial_gain_vs_history.csv")
    partial_rolling_rows = _read_csv_rows(out_dir / "teacher_only_rolling_partial_gain_vs_history.csv")
    abort_payload = _read_json(out_dir / "teacher_only_abort.json")
    complete_payload = _read_json(out_dir / "teacher_only_complete.json")

    partial_validation = partial_validation_rows[-1] if partial_validation_rows else {}
    partial_test = partial_test_rows[-1] if partial_test_rows else {}
    partial_rolling_min_mse = ""
    partial_rolling_min_mae = ""
    if partial_rolling_rows:
        mse_values = [float(row["partial_mse_gain_vs_history"]) for row in partial_rolling_rows]
        mae_values = [float(row["partial_mae_gain_vs_history"]) for row in partial_rolling_rows]
        partial_rolling_min_mse = min(mse_values)
        partial_rolling_min_mae = min(mae_values)

    rows: list[dict[str, Any]] = []
    feature_modes = sorted(set(rolling) | set(validation) | set(test) | set(teacher) | set(DEFAULT_FEATURE_MODES))
    for feature_mode in feature_modes:
        rolling_row = rolling.get(feature_mode, {})
        validation_row = validation.get(feature_mode, {})
        test_row = test.get(feature_mode, {})
        teacher_row = teacher.get(feature_mode, {})
        rows.append(
            {
                "status": status,
                "returncode": "" if returncode is None else returncode,
                "elapsed_seconds": "" if elapsed_seconds is None else f"{elapsed_seconds:.3f}",
                "feature_mode": feature_mode,
                "out_dir": str(out_dir),
                "ckpt_dir": str(ckpt_dir),
                "log_path": str(log_path),
                "command": " ".join(command),
                "teacher_abort_written": bool(abort_payload),
                "teacher_complete_written": bool(complete_payload),
                "teacher_gate_pass": rolling_row.get("teacher_diagnostic_gate_pass", ""),
                "teacher_gate_failure_reasons": rolling_row.get("teacher_diagnostic_failure_reasons", ""),
                "teacher_window_pass_count": rolling_row.get("teacher_window_pass_count", ""),
                "teacher_validation_window_count": rolling_row.get("teacher_window_count", ""),
                "teacher_worst_real_mse_gain": rolling_row.get("teacher_worst_real_mse_gain", ""),
                "teacher_worst_real_mae_gain": rolling_row.get("teacher_worst_real_mae_gain", ""),
                "teacher_worst_real_vs_best_cf_margin": rolling_row.get(
                    "teacher_worst_real_vs_best_counterfactual_mse_gain_margin", ""
                ),
                "teacher_worst_real_sign_agreement": rolling_row.get("teacher_worst_sign_agreement_rate", ""),
                "teacher_validation_mse_gain": validation_row.get("teacher_real_mse_gain", ""),
                "teacher_validation_mae_gain": validation_row.get("teacher_real_mae_gain", ""),
                "teacher_test_mse_gain": test_row.get("teacher_real_mse_gain", ""),
                "teacher_test_mae_gain": test_row.get("teacher_real_mae_gain", ""),
                "teacher_test_sign_agreement": test_row.get("teacher_real_sign_agreement_rate", ""),
                "teacher_train_r2": teacher_row.get("teacher_residual_r2", ""),
                "teacher_oof_mse_gain": teacher_row.get("teacher_oof_mse_gain", ""),
                "teacher_oof_mae_gain": teacher_row.get("teacher_oof_mae_gain", ""),
                "teacher_oof_sign_agreement": teacher_row.get("teacher_oof_sign_agreement", ""),
                "teacher_output_pass_count": teacher_row.get("teacher_output_pass_count", ""),
                "teacher_output_count": teacher_row.get("teacher_output_count", ""),
                "teacher_shrinkage_alpha": teacher_row.get("teacher_shrinkage_alpha", ""),
                "teacher_alpha_source": teacher_row.get("teacher_alpha_source", ""),
                "validation_partial_mse_gain_vs_history": partial_validation.get("partial_mse_gain_vs_history", ""),
                "validation_partial_mae_gain_vs_history": partial_validation.get("partial_mae_gain_vs_history", ""),
                "rolling_min_partial_mse_gain_vs_history": partial_rolling_min_mse,
                "rolling_min_partial_mae_gain_vs_history": partial_rolling_min_mae,
                "test_partial_mse_gain_vs_history": partial_test.get("partial_mse_gain_vs_history", ""),
                "test_partial_mae_gain_vs_history": partial_test.get("partial_mae_gain_vs_history", ""),
                "generator_artifact_exists": bool(list(ckpt_dir.glob("generator*.pt"))),
            }
        )
    return rows


def _build_command(args: argparse.Namespace, *, dataset: str, seed: int, run_id: str) -> list[str]:
    command = [
        args.python_executable,
        "train\\run_stage3_v12d_context_teacher_probe.py",
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
    ]
    if args.history_protocol is not None:
        command.extend(["--history-protocol", args.history_protocol])
    if args.history_label_delay is not None:
        command.extend(["--history-label-delay", str(args.history_label_delay)])
    if args.history_feedback_delay is not None:
        command.extend(["--history-feedback-delay", str(args.history_feedback_delay)])
    if args.history_lags is not None:
        command.extend(["--history-lags", *[str(value) for value in args.history_lags]])
    if args.history_horizon_offsets is not None:
        command.extend(["--history-horizon-offsets", *[str(value) for value in args.history_horizon_offsets]])
    if args.diagnostic_include_offline_counterfactuals is not None:
        command.append(
            "--diagnostic-include-offline-counterfactuals"
            if args.diagnostic_include_offline_counterfactuals
            else "--no-diagnostic-include-offline-counterfactuals"
        )
    command.append("--normalize-vector" if args.normalize_vector else "--no-normalize-vector")
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-overwriting V12d context teacher-only audit matrix.")
    parser.add_argument("--config", default="configs/stage3_v12d_context_teacher_probe.yaml")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--invariant-epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=480)
    parser.add_argument("--max-eval-batches", type=int, default=480)
    parser.add_argument("--validation-windows", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--drift-source", choices=["phi_gap", "delta_phi_v"], default="delta_phi_v")
    parser.add_argument("--drift-polarity", type=float, default=1.0)
    parser.add_argument("--signal-mode", choices=["level_vector", "delta_vector"], default="delta_vector")
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--feature-modes", nargs="*", default=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--history-protocol", default=None)
    parser.add_argument("--history-label-delay", type=int, default=None)
    parser.add_argument("--history-feedback-delay", type=int, default=None)
    parser.add_argument("--history-lags", nargs="*", type=int, default=None)
    parser.add_argument("--history-horizon-offsets", nargs="*", type=int, default=None)
    parser.add_argument("--diagnostic-include-offline-counterfactuals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out-root", default="logs/s3v12dctxm")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v12dctxm")
    parser.add_argument("--run-prefix", default="v12dctxm")
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
            if proc.returncode != 0 and args.stop_on_error:
                break

    manifest = {
        "matrix_id": matrix_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "feature_modes": args.feature_modes,
        "modes": args.modes,
        "out_root": args.out_root,
        "checkpoint_root": args.checkpoint_root,
        "planned_runs": planned_runs,
        "summary_csv": str(manifest_dir / "matrix_summary.csv"),
    }
    _write_json(manifest_dir / "matrix_manifest.json", manifest)
    _write_json(manifest_dir / "planned_runs.json", planned_runs)
    _write_csv(manifest_dir / "matrix_summary.csv", summaries)
    print(manifest_dir)


if __name__ == "__main__":
    main()
