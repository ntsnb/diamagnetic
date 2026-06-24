from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _read_csv(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required diagnostic input does not exist: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _best_counterfactual(rows: list[dict[str, str]]) -> dict[str, str] | None:
    counterfactuals = [row for row in rows if row.get("mode") != "real"]
    if not counterfactuals:
        return None
    return max(counterfactuals, key=lambda row: _to_float(row.get("teacher_mse_gain"), -float("inf")))


def _partial_by_window(partial_rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row in partial_rows:
        window_id = str(row.get("validation_window_id", ""))
        if window_id:
            mapped[window_id] = row
    return mapped


def build_window_failure_diagnostics(
    rows: list[dict[str, str]],
    *,
    partial_rows: list[dict[str, str]] | None = None,
    feature_mode: str = "fluctuation_plus_history",
    min_real_mse_gain: float = 0.005,
    min_real_mae_gain: float = 0.002,
    counterfactual_margin: float = 0.002,
    min_sign_agreement: float = 0.55,
    min_partial_mse_gain_vs_history: float = 0.001,
    min_partial_mae_gain_vs_history: float = 0.0005,
) -> list[dict[str, str]]:
    by_window: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("feature_mode") != feature_mode:
            continue
        window_id = str(row.get("validation_window_id", ""))
        if window_id == "":
            split = str(row.get("split", ""))
            window_id = split.rsplit("_", 1)[-1] if split.startswith("validation_window_") else split
        by_window[window_id].append(row)

    partial_map = _partial_by_window(partial_rows or [])
    diagnostics: list[dict[str, str]] = []
    for window_id in sorted(by_window, key=lambda value: int(value) if value.isdigit() else value):
        window_rows = by_window[window_id]
        real_rows = [row for row in window_rows if row.get("mode") == "real"]
        if not real_rows:
            continue
        real = real_rows[0]
        best_cf = _best_counterfactual(window_rows)
        real_mse = _to_float(real.get("teacher_mse_gain"))
        real_mae = _to_float(real.get("teacher_mae_gain"))
        real_sign = _to_float(real.get("teacher_sign_agreement_rate"))
        best_cf_mse = _to_float(best_cf.get("teacher_mse_gain")) if best_cf else float("nan")
        margin = real_mse - best_cf_mse if best_cf else float("nan")
        partial = partial_map.get(window_id, {})
        partial_mse = _to_float(partial.get("partial_mse_gain_vs_history"))
        partial_mae = _to_float(partial.get("partial_mae_gain_vs_history"))

        reasons: list[str] = []
        families: list[str] = []
        if real_mse < min_real_mse_gain:
            reasons.append("real_mse_gain")
        if real_mae < min_real_mae_gain:
            reasons.append("real_mae_gain")
        if real_sign < min_sign_agreement:
            reasons.append("real_sign_agreement")
        if reasons:
            families.append("real_behavior")
        if best_cf is not None and margin < counterfactual_margin:
            reasons.append("counterfactual")
            families.insert(0, "attribution")
        if partial and partial_mse < min_partial_mse_gain_vs_history:
            reasons.append("partial_mse_gain_vs_history")
        if partial and partial_mae < min_partial_mae_gain_vs_history:
            reasons.append("partial_mae_gain_vs_history")
        if any(reason.startswith("partial_") for reason in reasons):
            families.append("incremental")

        ordered_families = []
        for family in families:
            if family not in ordered_families:
                ordered_families.append(family)

        diagnostics.append(
            {
                "feature_mode": feature_mode,
                "validation_window_id": window_id,
                "real_mse_gain": _format_float(real_mse),
                "real_mae_gain": _format_float(real_mae),
                "real_sign_agreement": _format_float(real_sign),
                "best_counterfactual_mode": str(best_cf.get("mode", "")) if best_cf else "",
                "best_counterfactual_mse_gain": _format_float(best_cf_mse),
                "real_vs_best_counterfactual_mse_gain_margin": _format_float(margin),
                "partial_mse_gain_vs_history": _format_float(partial_mse),
                "partial_mae_gain_vs_history": _format_float(partial_mae),
                "gate_pass": str(not reasons),
                "failure_family": ";".join(ordered_families),
                "failure_reasons": ";".join(reasons),
            }
        )
    return diagnostics


def _real_and_best(rows: list[dict[str, str]], feature_mode: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    feature_rows = [row for row in rows if row.get("feature_mode") == feature_mode]
    real_rows = [row for row in feature_rows if row.get("mode") == "real"]
    return (real_rows[0] if real_rows else None, _best_counterfactual(feature_rows))


def _first_partial(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def build_split_generalization_diagnostics(
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    *,
    validation_partial_rows: list[dict[str, str]] | None = None,
    test_partial_rows: list[dict[str, str]] | None = None,
    feature_mode: str = "fluctuation_plus_history",
) -> list[dict[str, str]]:
    val_real, val_best = _real_and_best(validation_rows, feature_mode)
    test_real, test_best = _real_and_best(test_rows, feature_mode)
    if val_real is None or test_real is None:
        return []
    val_partial = _first_partial(validation_partial_rows or [])
    test_partial = _first_partial(test_partial_rows or [])

    val_mse = _to_float(val_real.get("teacher_mse_gain"))
    test_mse = _to_float(test_real.get("teacher_mse_gain"))
    val_mae = _to_float(val_real.get("teacher_mae_gain"))
    test_mae = _to_float(test_real.get("teacher_mae_gain"))
    val_partial_mse = _to_float(val_partial.get("partial_mse_gain_vs_history"))
    test_partial_mse = _to_float(test_partial.get("partial_mse_gain_vs_history"))
    val_partial_mae = _to_float(val_partial.get("partial_mae_gain_vs_history"))
    test_partial_mae = _to_float(test_partial.get("partial_mae_gain_vs_history"))
    val_best_mse = _to_float(val_best.get("teacher_mse_gain")) if val_best else float("nan")
    test_best_mse = _to_float(test_best.get("teacher_mse_gain")) if test_best else float("nan")
    reversal = (val_partial_mse > 0.0 and test_partial_mse < 0.0) or (
        val_partial_mae > 0.0 and test_partial_mae < 0.0
    )

    return [
        {
            "feature_mode": feature_mode,
            "validation_real_mse_gain": _format_float(val_mse),
            "test_real_mse_gain": _format_float(test_mse),
            "validation_to_test_real_mse_gain_change": _format_float(test_mse - val_mse),
            "validation_real_mae_gain": _format_float(val_mae),
            "test_real_mae_gain": _format_float(test_mae),
            "validation_to_test_real_mae_gain_change": _format_float(test_mae - val_mae),
            "validation_best_counterfactual_mode": str(val_best.get("mode", "")) if val_best else "",
            "validation_real_vs_best_counterfactual_mse_gain_margin": _format_float(val_mse - val_best_mse),
            "test_best_counterfactual_mode": str(test_best.get("mode", "")) if test_best else "",
            "test_real_vs_best_counterfactual_mse_gain_margin": _format_float(test_mse - test_best_mse),
            "validation_partial_mse_gain_vs_history": _format_float(val_partial_mse),
            "test_partial_mse_gain_vs_history": _format_float(test_partial_mse),
            "validation_partial_mae_gain_vs_history": _format_float(val_partial_mae),
            "test_partial_mae_gain_vs_history": _format_float(test_partial_mae),
            "incremental_value_reversal": str(bool(reversal)),
        }
    ]


def build_alpha_candidate_stability(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("ridge", "")), str(row.get("output_idx", "")), str(row.get("alpha", "")))].append(row)

    diagnostics: list[dict[str, str]] = []
    for (ridge, output_idx, alpha), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            int(item[0][1]) if item[0][1].isdigit() else item[0][1],
            _to_float(item[0][0]),
            _to_float(item[0][2]),
        ),
    ):
        mse_values = [_to_float(row.get("mse_gain")) for row in group_rows]
        mae_values = [_to_float(row.get("mae_gain")) for row in group_rows]
        sign_values = [_to_float(row.get("sign_agreement")) for row in group_rows]
        pass_count = sum(1 for row in group_rows if _truthy(row.get("block_pass")))
        fold_count = len(group_rows)
        min_mse = min(mse_values) if mse_values else float("nan")
        min_mae = min(mae_values) if mae_values else float("nan")
        min_sign = min(sign_values) if sign_values else float("nan")
        pass_rate = pass_count / fold_count if fold_count else 0.0
        warning = ""
        if min_mse < 0.0 or min_mae < 0.0:
            warning = "negative_block_gain"
        elif pass_rate < 1.0:
            warning = "partial_block_pass"
        diagnostics.append(
            {
                "ridge": ridge,
                "output_idx": output_idx,
                "alpha": alpha,
                "fold_count": str(fold_count),
                "block_pass_count": str(pass_count),
                "block_pass_rate": _format_float(pass_rate),
                "mean_mse_gain": _format_float(mean(mse_values) if mse_values else float("nan")),
                "min_mse_gain": _format_float(min_mse),
                "mean_mae_gain": _format_float(mean(mae_values) if mae_values else float("nan")),
                "min_mae_gain": _format_float(min_mae),
                "mean_sign_agreement": _format_float(mean(sign_values) if sign_values else float("nan")),
                "min_sign_agreement": _format_float(min_sign),
                "stability_warning": warning,
            }
        )
    return diagnostics


def write_csv_no_overwrite(path: Path, rows: list[dict[str, str]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing diagnostic output: {path}")
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


def write_text_no_overwrite(path: Path, text: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing diagnostic output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report_text(
    *,
    run_dir: Path,
    window_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    alpha_rows: list[dict[str, str]],
) -> str:
    failed_windows = [row for row in window_rows if row.get("gate_pass") != "True"]
    split = split_rows[0] if split_rows else {}
    unstable_alpha = [row for row in alpha_rows if row.get("stability_warning")]
    lines = [
        "# Stage III V14 Failure Diagnostics",
        "",
        f"Input run directory: `{run_dir}`",
        "",
        "## Window Gate Summary",
        "",
        f"- Diagnosed windows: {len(window_rows)}.",
        f"- Failed windows: {len(failed_windows)}.",
    ]
    for row in failed_windows:
        lines.append(
            "- Window {window}: family `{family}`, reasons `{reasons}`, real MSE gain `{mse}`, "
            "real MAE gain `{mae}`, best counterfactual `{cf}`, margin `{margin}`.".format(
                window=row.get("validation_window_id", ""),
                family=row.get("failure_family", ""),
                reasons=row.get("failure_reasons", ""),
                mse=row.get("real_mse_gain", ""),
                mae=row.get("real_mae_gain", ""),
                cf=row.get("best_counterfactual_mode", ""),
                margin=row.get("real_vs_best_counterfactual_mse_gain_margin", ""),
            )
        )
    if split:
        lines.extend(
            [
                "",
                "## Split Generalization",
                "",
                "- Validation-to-test real MSE gain change: `{}`.".format(
                    split.get("validation_to_test_real_mse_gain_change", "")
                ),
                "- Validation-to-test real MAE gain change: `{}`.".format(
                    split.get("validation_to_test_real_mae_gain_change", "")
                ),
                "- Incremental value reversal versus history: `{}`.".format(
                    split.get("incremental_value_reversal", "")
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Alpha Candidate Stability",
            "",
            f"- Candidate rows: {len(alpha_rows)}.",
            f"- Rows with stability warnings: {len(unstable_alpha)}.",
            "",
            "Interpretation: this is a derived diagnostic only. It does not validate V14 as a manuscript result.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_run(
    *,
    run_dir: Path,
    out_dir: Path,
    feature_mode: str = "fluctuation_plus_history",
    overwrite: bool = False,
    min_real_mse_gain: float = 0.005,
    min_real_mae_gain: float = 0.002,
    counterfactual_margin: float = 0.002,
    min_sign_agreement: float = 0.55,
    min_partial_mse_gain_vs_history: float = 0.001,
    min_partial_mae_gain_vs_history: float = 0.0005,
) -> None:
    rolling_rows = _read_csv(run_dir / "teacher_only_rolling_validation_summary.csv")
    rolling_partial = _read_csv(run_dir / "teacher_only_rolling_partial_gain_vs_history.csv", required=False)
    validation_rows = _read_csv(run_dir / "teacher_only_validation_summary.csv")
    test_rows = _read_csv(run_dir / "teacher_only_test_summary.csv")
    validation_partial = _read_csv(run_dir / "teacher_only_validation_partial_gain_vs_history.csv", required=False)
    test_partial = _read_csv(run_dir / "teacher_only_test_partial_gain_vs_history.csv", required=False)
    alpha_rows = _read_csv(run_dir / "v14_alpha_blocks_fh.csv", required=False)

    window_diagnostics = build_window_failure_diagnostics(
        rolling_rows,
        partial_rows=rolling_partial,
        feature_mode=feature_mode,
        min_real_mse_gain=min_real_mse_gain,
        min_real_mae_gain=min_real_mae_gain,
        counterfactual_margin=counterfactual_margin,
        min_sign_agreement=min_sign_agreement,
        min_partial_mse_gain_vs_history=min_partial_mse_gain_vs_history,
        min_partial_mae_gain_vs_history=min_partial_mae_gain_vs_history,
    )
    split_diagnostics = build_split_generalization_diagnostics(
        validation_rows,
        test_rows,
        validation_partial_rows=validation_partial,
        test_partial_rows=test_partial,
        feature_mode=feature_mode,
    )
    alpha_diagnostics = build_alpha_candidate_stability(alpha_rows)

    write_csv_no_overwrite(out_dir / "v14_window_failure_diagnostics.csv", window_diagnostics, overwrite=overwrite)
    write_csv_no_overwrite(
        out_dir / "v14_split_generalization_diagnostics.csv",
        split_diagnostics,
        overwrite=overwrite,
    )
    write_csv_no_overwrite(
        out_dir / "v14_alpha_candidate_stability_fh.csv",
        alpha_diagnostics,
        overwrite=overwrite,
    )
    write_text_no_overwrite(
        out_dir / "v14_failure_diagnostics_report.md",
        _report_text(
            run_dir=run_dir,
            window_rows=window_diagnostics,
            split_rows=split_diagnostics,
            alpha_rows=alpha_diagnostics,
        ),
        overwrite=overwrite,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage III V14 teacher-only failure diagnostics.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--feature-mode", default="fluctuation_plus_history")
    parser.add_argument("--min-real-mse-gain", type=float, default=0.005)
    parser.add_argument("--min-real-mae-gain", type=float, default=0.002)
    parser.add_argument("--counterfactual-margin", type=float, default=0.002)
    parser.add_argument("--min-sign-agreement", type=float, default=0.55)
    parser.add_argument("--min-partial-mse-gain-vs-history", type=float, default=0.001)
    parser.add_argument("--min-partial-mae-gain-vs-history", type=float, default=0.0005)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze_run(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        feature_mode=args.feature_mode,
        overwrite=args.overwrite,
        min_real_mse_gain=args.min_real_mse_gain,
        min_real_mae_gain=args.min_real_mae_gain,
        counterfactual_margin=args.counterfactual_margin,
        min_sign_agreement=args.min_sign_agreement,
        min_partial_mse_gain_vs_history=args.min_partial_mse_gain_vs_history,
        min_partial_mae_gain_vs_history=args.min_partial_mae_gain_vs_history,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
