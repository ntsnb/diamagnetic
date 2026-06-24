from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.run_experiment import run_single_experiment
from utils.config import load_yaml
from utils.paths import resolve_project_path


VARIANTS = [
    ("baseline", False, False, False),
    ("stage1", True, False, False),
    ("stage2", False, True, False),
    ("stage3", False, False, True),
    ("stage1_stage2", True, True, False),
    ("stage1_stage3", True, False, True),
    ("stage2_stage3", False, True, True),
    ("stage1_stage2_stage3", True, True, True),
]


def append_result(path: str, row: dict) -> None:
    resolved = resolve_project_path(path)
    os.makedirs(resolved.parent, exist_ok=True)
    exists = resolved.exists()
    with open(resolved, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def adapted_metric(result: dict, stage3_enabled: bool) -> float:
    return result["online_stage3_mse"] if stage3_enabled else result["online_stage2_mse"]


def summarize_suite(rows: list[dict]) -> dict:
    by_variant = {row["variant"]: row for row in rows}
    baseline = by_variant["baseline"]["adapted_online_mse"]
    single_variants = ["stage1", "stage2", "stage3"]
    singles_improved = all(by_variant[name]["adapted_online_mse"] < baseline for name in single_variants)
    full_combo = by_variant["stage1_stage2_stage3"]["adapted_online_mse"]
    best_variant = min(rows, key=lambda row: row["adapted_online_mse"])["variant"]
    combo_best = best_variant == "stage1_stage2_stage3"

    # Lower is better. Negative score means better than baseline.
    score = 0.0
    for name in single_variants:
        score += by_variant[name]["adapted_online_mse"] - baseline
    score += 2.0 * (full_combo - baseline)
    if singles_improved:
        score -= 1.0
    if combo_best:
        score -= 1.0

    return {
        "baseline_online_mse": baseline,
        "full_combo_online_mse": full_combo,
        "singles_improved": singles_improved,
        "combo_best": combo_best,
        "best_variant": best_variant,
        "objective_score": score,
    }


def run_ablation_suite(
    base_cfg: dict,
    results_path: str = "artifacts/autoresearch/main_results.tsv",
    trial_tag: str = "manual",
) -> tuple[list[dict], dict]:
    rows = []
    for name, stage1, stage2, stage3 in VARIANTS:
        cfg = copy.deepcopy(base_cfg)
        cfg["online"]["stage1"]["enable"] = stage1
        cfg["online"]["stage2"]["enable"] = stage2
        cfg["online"]["stage3"]["enable"] = stage3

        result = run_single_experiment(cfg, save_ckpt=False, variant_name=name)
        adapted_online = adapted_metric(result, stage3_enabled=stage3)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_tag": trial_tag,
            "variant": name,
            "offline_val_mse": f"{result['offline_val_mse']:.6f}",
            "offline_test_mse": f"{result['offline_test_mse']:.6f}",
            "online_base_mse": f"{result['online_base_mse']:.6f}",
            "online_stage2_mse": f"{result['online_stage2_mse']:.6f}",
            "online_stage3_mse": f"{result['online_stage3_mse']:.6f}",
            "adapted_online_mse": f"{adapted_online:.6f}",
            "status": "ok",
            "description": f"stage1={stage1}, stage2={stage2}, stage3={stage3}",
        }
        append_result(results_path, row)
        rows.append(
            {
                "variant": name,
                "offline_val_mse": result["offline_val_mse"],
                "offline_test_mse": result["offline_test_mse"],
                "online_base_mse": result["online_base_mse"],
                "online_stage2_mse": result["online_stage2_mse"],
                "online_stage3_mse": result["online_stage3_mse"],
                "adapted_online_mse": adapted_online,
            }
        )
        print(
            f"[{trial_tag}:{name}] offline_val={result['offline_val_mse']:.6f} "
            f"offline_test={result['offline_test_mse']:.6f} "
            f"online_base={result['online_base_mse']:.6f} "
            f"online_stage2={result['online_stage2_mse']:.6f} "
            f"online_stage3={result['online_stage3_mse']:.6f}"
        )

    summary = summarize_suite(rows)
    print(
        f"[{trial_tag}:summary] baseline={summary['baseline_online_mse']:.6f} "
        f"full_combo={summary['full_combo_online_mse']:.6f} "
        f"best={summary['best_variant']} "
        f"singles_improved={summary['singles_improved']} "
        f"combo_best={summary['combo_best']} "
        f"score={summary['objective_score']:.6f}"
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--results", default="artifacts/autoresearch/main_results.tsv")
    parser.add_argument("--trial-tag", default="manual")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    run_ablation_suite(base_cfg, results_path=args.results, trial_tag=args.trial_tag)


if __name__ == "__main__":
    main()
