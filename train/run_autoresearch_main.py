from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.run_ablation_suite import run_ablation_suite
from utils.config import load_yaml, save_yaml
from utils.paths import resolve_project_path


@dataclass
class SearchConfig:
    max_trials: int = 12
    time_budget_hours: float = 8.0
    seed: int = 123
    proxy_epochs: int = 4
    proxy_invariant_epochs: int = 1
    keep_top_k: int = 3
    mutation_scale: float = 1.0
    early_stop_patience: int = 4


def ensure_dir(path: str) -> None:
    os.makedirs(resolve_project_path(path), exist_ok=True)


def read_tsv_rows(path: str) -> list[dict]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return []
    with open(resolved, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def append_tsv_row(path: str, row: dict) -> None:
    resolved = resolve_project_path(path)
    os.makedirs(resolved.parent, exist_ok=True)
    exists = resolved.exists()
    with open(resolved, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round6(value: float) -> float:
    return float(f"{value:.6g}")


def apply_proxy_overrides(cfg: dict, search: SearchConfig) -> dict:
    out = copy.deepcopy(cfg)
    out["train"]["epochs"] = min(out["train"]["epochs"], search.proxy_epochs)
    out["invariant"]["epochs"] = min(out["invariant"]["epochs"], search.proxy_invariant_epochs)
    return out


def make_mutation(base_cfg: dict, incumbent_cfg: dict, rng: random.Random, scale: float) -> tuple[str, dict]:
    cfg = copy.deepcopy(incumbent_cfg)
    parts = []

    # Offline schedule / backbone stability.
    if rng.random() < 0.4:
        lr = cfg["train"]["lr"]
        factor = 10 ** rng.uniform(-0.25, 0.15)
        cfg["train"]["lr"] = round6(clip(lr * factor, 1e-4, 3e-3))
        parts.append(f"train.lr={cfg['train']['lr']}")

    if rng.random() < 0.35:
        sched_type = rng.choice(["warmup_cosine", "step", "none"])
        cfg["train"]["lr_schedule"]["type"] = sched_type
        if sched_type == "step":
            cfg["train"]["lr_schedule"]["step_size"] = rng.choice([2, 3, 4, 5])
            cfg["train"]["lr_schedule"]["gamma"] = round6(rng.choice([0.3, 0.5, 0.7, 0.8]))
        if sched_type == "warmup_cosine":
            cfg["train"]["lr_schedule"]["warmup_epochs"] = rng.choice([1, 2])
            cfg["train"]["lr_schedule"]["min_lr"] = round6(rng.choice([5e-5, 1e-4, 2e-4]))
        parts.append(f"sched={sched_type}")

    # Stage I.
    if rng.random() < 0.5:
        lr = cfg["online"]["stage1"]["lr"]
        factor = 10 ** rng.uniform(-0.35, 0.1) * scale
        cfg["online"]["stage1"]["lr"] = round6(clip(lr * factor, 1e-5, 5e-4))
        cfg["online"]["stage1"]["lr_decay"] = round6(clip(
            cfg["online"]["stage1"]["lr_decay"] + rng.uniform(-0.01, 0.005), 0.97, 0.9995
        ))
        cfg["online"]["stage1"]["alpha_kl_weight"] = round6(clip(
            cfg["online"]["stage1"]["alpha_kl_weight"] + rng.uniform(-0.03, 0.04), 0.0, 0.25
        ))
        cfg["online"]["stage1"]["bias_l2_weight"] = round6(clip(
            cfg["online"]["stage1"]["bias_l2_weight"] * (10 ** rng.uniform(-0.4, 0.4)), 1e-5, 1e-1
        ))
        cfg["online"]["stage1"]["bias_clip"] = round6(clip(
            cfg["online"]["stage1"]["bias_clip"] + rng.uniform(-0.1, 0.1), 0.05, 0.6
        ))
        cfg["online"]["stage1"]["update_interval"] = int(clip(
            cfg["online"]["stage1"].get("update_interval", 1) + rng.choice([-2, -1, 0, 1, 2]), 1, 8
        ))
        cfg["online"]["stage1"]["min_trigger_loss"] = round6(clip(
            cfg["online"]["stage1"].get("min_trigger_loss", 0.0) + rng.uniform(-0.5, 0.6), 0.0, 4.0
        ))
        cfg["online"]["stage1"]["schedule"] = rng.choice(["invsqrt", "exp"])
        parts.append(f"s1.lr={cfg['online']['stage1']['lr']}")

    # Stability trigger.
    if rng.random() < 0.55:
        cfg["online"]["stability"]["gamma"] = round6(clip(
            cfg["online"]["stability"]["gamma"] + rng.uniform(-0.06, 0.05), 0.45, 0.8
        ))
        cfg["online"]["stability"]["window_size"] = int(clip(
            cfg["online"]["stability"]["window_size"] + rng.choice([-4, -2, 0, 2, 4]), 6, 30
        ))
        parts.append(
            f"stable=({cfg['online']['stability']['gamma']},{cfg['online']['stability']['window_size']})"
        )

    # Stage II.
    if rng.random() < 0.75:
        factor = 10 ** rng.uniform(-0.3, 0.15) * scale
        cfg["online"]["stage2"]["lr_scale"] = round6(clip(
            cfg["online"]["stage2"]["lr_scale"] * factor, 1e-5, 2e-2
        ))
        cfg["online"]["stage2"]["lr_decay"] = round6(clip(
            cfg["online"]["stage2"]["lr_decay"] + rng.uniform(-0.004, 0.003), 0.985, 0.9999
        ))
        cfg["online"]["stage2"]["mask_threshold"] = round6(clip(
            cfg["online"]["stage2"]["mask_threshold"] + rng.uniform(-0.4, 0.5), 0.8, 3.5
        ))
        cfg["online"]["stage2"]["min_instability_gap"] = round6(clip(
            cfg["online"]["stage2"]["min_instability_gap"] + rng.uniform(-0.03, 0.03), 0.0, 0.2
        ))
        cfg["online"]["stage2"]["min_drift_abs"] = round6(clip(
            cfg["online"]["stage2"]["min_drift_abs"] + rng.uniform(-0.05, 0.06), 0.0, 0.4
        ))
        cfg["online"]["stage2"]["min_recon_error"] = round6(clip(
            cfg["online"]["stage2"]["min_recon_error"] + rng.uniform(-0.03, 0.04), 0.02, 0.3
        ))
        cfg["online"]["stage2"]["with_stage1_scale"] = round6(clip(
            cfg["online"]["stage2"]["with_stage1_scale"] + rng.uniform(-0.15, 0.15), 0.2, 1.0
        ))
        cfg["online"]["stage2"]["min_improve_abs"] = round6(clip(
            cfg["online"]["stage2"]["min_improve_abs"] + rng.uniform(-5e-5, 8e-5), 0.0, 5e-3
        ))
        cfg["online"]["stage2"]["with_stage1_min_improve_abs"] = round6(clip(
            cfg["online"]["stage2"]["with_stage1_min_improve_abs"] + rng.uniform(-8e-5, 1e-4), 0.0, 5e-3
        ))
        cfg["online"]["stage2"]["no_stage1_lr_scale"] = round6(clip(
            cfg["online"]["stage2"].get("no_stage1_lr_scale", cfg["online"]["stage2"]["lr_scale"]) * (10 ** rng.uniform(-0.2, 0.2)),
            1e-5,
            3e-2,
        ))
        cfg["online"]["stage2"]["no_stage1_min_drift_abs"] = round6(clip(
            cfg["online"]["stage2"].get("no_stage1_min_drift_abs", cfg["online"]["stage2"]["min_drift_abs"]) + rng.uniform(-0.05, 0.06),
            0.0,
            0.4,
        ))
        cfg["online"]["stage2"]["no_stage1_min_recon_error"] = round6(clip(
            cfg["online"]["stage2"].get("no_stage1_min_recon_error", cfg["online"]["stage2"]["min_recon_error"]) + rng.uniform(-0.04, 0.05),
            0.02,
            0.3,
        ))
        cfg["online"]["stage2"]["no_stage1_min_improve_abs"] = round6(clip(
            cfg["online"]["stage2"].get("no_stage1_min_improve_abs", cfg["online"]["stage2"]["min_improve_abs"]) + rng.uniform(-5e-5, 8e-5),
            0.0,
            5e-3,
        ))
        cfg["online"]["stage2"]["no_stage1_update_interval"] = int(rng.choice([1, 2, 3, 4]))
        cfg["online"]["stage2"]["gate_by_recon"] = bool(rng.choice([True, True, False]))
        cfg["online"]["stage2"]["eval_with_stage3"] = bool(rng.choice([True, True, False]))
        cfg["online"]["stage2"]["update_interval"] = int(rng.choice([1, 2, 3, 4]))
        parts.append(
            "s2=("
            f"{cfg['online']['stage2']['lr_scale']},"
            f"{cfg['online']['stage2']['mask_threshold']},"
            f"{cfg['online']['stage2']['update_interval']},"
            f"{cfg['online']['stage2']['min_drift_abs']},"
            f"{cfg['online']['stage2']['min_recon_error']},"
            f"{cfg['online']['stage2']['with_stage1_scale']},"
            f"{cfg['online']['stage2']['min_improve_abs']},"
            f"{cfg['online']['stage2']['with_stage1_min_improve_abs']},"
            f"{cfg['online']['stage2']['gate_by_recon']},"
            f"{cfg['online']['stage2']['eval_with_stage3']})"
        )

    # Stage III.
    if rng.random() < 0.65:
        factor = 10 ** rng.uniform(-0.35, 0.05) * scale
        cfg["online"]["stage3"]["lr_scale"] = round6(clip(
            cfg["online"]["stage3"]["lr_scale"] * factor, 1e-6, 5e-3
        ))
        cfg["online"]["stage3"]["lr_decay"] = round6(clip(
            cfg["online"]["stage3"]["lr_decay"] + rng.uniform(-0.004, 0.002), 0.985, 0.9999
        ))
        cfg["online"]["stage3"]["max_delta"] = round6(clip(
            cfg["online"]["stage3"]["max_delta"] + rng.uniform(-0.015, 0.01), 0.005, 0.08
        ))
        cfg["online"]["stage3"]["blend_alpha"] = round6(clip(
            cfg["online"]["stage3"]["blend_alpha"] + rng.uniform(-0.15, 0.15), 0.05, 0.8
        ))
        cfg["online"]["stage3"]["blend_alpha_min"] = round6(clip(
            cfg["online"]["stage3"].get("blend_alpha_min", cfg["online"]["stage3"]["blend_alpha"]) + rng.uniform(-0.1, 0.1),
            0.05,
            cfg["online"]["stage3"].get("blend_alpha_max", 0.9),
        ))
        cfg["online"]["stage3"]["blend_alpha_max"] = round6(clip(
            cfg["online"]["stage3"].get("blend_alpha_max", cfg["online"]["stage3"]["blend_alpha"]) + rng.uniform(-0.1, 0.1),
            max(cfg["online"]["stage3"].get("blend_alpha_min", 0.05), 0.1),
            0.9,
        ))
        cfg["online"]["stage3"]["with_stage2_blend_scale"] = round6(clip(
            cfg["online"]["stage3"]["with_stage2_blend_scale"] + rng.uniform(-0.2, 0.15), 0.4, 1.0
        ))
        cfg["online"]["stage3"]["with_stage1_stage2_blend_scale"] = round6(clip(
            cfg["online"]["stage3"].get("with_stage1_stage2_blend_scale", cfg["online"]["stage3"]["with_stage2_blend_scale"]) + rng.uniform(-0.1, 0.1),
            0.5,
            1.2,
        ))
        parts.append(
            "s3=("
            f"{cfg['online']['stage3']['lr_scale']},"
            f"{cfg['online']['stage3']['max_delta']},"
            f"{cfg['online']['stage3']['blend_alpha']},"
            f"{cfg['online']['stage3']['with_stage2_blend_scale']})"
        )

    # Optionally toggle LRP in offline loss; online Stage II still uses LRP.
    if rng.random() < 0.25:
        cfg["train"]["use_lrp"] = bool(rng.choice([True, False]))
        parts.append(f"offline_lrp={cfg['train']['use_lrp']}")

    desc = ", ".join(parts) if parts else "no-op"
    return desc, cfg


def load_search_config(path: str | None) -> SearchConfig:
    if path is None or not os.path.exists(path):
        return SearchConfig()
    raw = load_yaml(path)
    section = raw.get("autoresearch", raw)
    return SearchConfig(
        max_trials=section.get("max_trials", 12),
        time_budget_hours=section.get("time_budget_hours", 8.0),
        seed=section.get("seed", 123),
        proxy_epochs=section.get("proxy_epochs", 4),
        proxy_invariant_epochs=section.get("proxy_invariant_epochs", 1),
        keep_top_k=section.get("keep_top_k", 3),
        mutation_scale=section.get("mutation_scale", 1.0),
        early_stop_patience=section.get("early_stop_patience", 4),
    )


def score_from_summary(summary: dict) -> float:
    return float(summary["objective_score"])


def write_journal(path: str, message: str) -> None:
    resolved = resolve_project_path(path)
    os.makedirs(resolved.parent, exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_detailed_log(
    path: str,
    title: str,
    desc: str,
    cfg: dict,
    summary: dict,
    status: str,
    elapsed: float,
) -> None:
    online = cfg["online"]
    lines = [
        "",
        f"## {datetime.now().isoformat(timespec='seconds')} {title}",
        f"- status: {status}",
        f"- elapsed_sec: {elapsed:.1f}",
        f"- description: {desc}",
        "- summary:",
        f"  - baseline_online_mse: {summary['baseline_online_mse']}",
        f"  - full_combo_online_mse: {summary['full_combo_online_mse']}",
        f"  - best_variant: {summary['best_variant']}",
        f"  - singles_improved: {summary['singles_improved']}",
        f"  - combo_best: {summary['combo_best']}",
        f"  - objective_score: {summary['objective_score']}",
        "- config snapshot:",
        f"  - data.train_ratio: {cfg['data']['train_ratio']}",
        f"  - train.epochs: {cfg['train']['epochs']}",
        f"  - invariant.epochs: {cfg['invariant']['epochs']}",
        f"  - stage1.lr: {online['stage1']['lr']}",
        f"  - stage1.update_interval: {online['stage1'].get('update_interval', 1)}",
        f"  - stage1.min_trigger_loss: {online['stage1'].get('min_trigger_loss', 0.0)}",
        f"  - stage1.alpha_kl_weight: {online['stage1'].get('alpha_kl_weight', 0.0)}",
        f"  - stage1.bias_clip: {online['stage1'].get('bias_clip', 0.0)}",
        f"  - stage2.lr_scale: {online['stage2']['lr_scale']}",
        f"  - stage2.update_interval: {online['stage2']['update_interval']}",
        f"  - stage2.min_drift_abs: {online['stage2'].get('min_drift_abs', 0.0)}",
        f"  - stage2.min_recon_error: {online['stage2'].get('min_recon_error', 0.0)}",
        f"  - stage2.with_stage1_scale: {online['stage2'].get('with_stage1_scale', 1.0)}",
        f"  - stage2.with_stage1_min_improve_abs: {online['stage2'].get('with_stage1_min_improve_abs', 0.0)}",
        f"  - stage2.no_stage1_lr_scale: {online['stage2'].get('no_stage1_lr_scale', online['stage2']['lr_scale'])}",
        f"  - stage2.no_stage1_update_interval: {online['stage2'].get('no_stage1_update_interval', online['stage2']['update_interval'])}",
        f"  - stage2.no_stage1_min_drift_abs: {online['stage2'].get('no_stage1_min_drift_abs', online['stage2'].get('min_drift_abs', 0.0))}",
        f"  - stage2.no_stage1_min_recon_error: {online['stage2'].get('no_stage1_min_recon_error', online['stage2'].get('min_recon_error', 0.0))}",
        f"  - stage3.lr_scale: {online['stage3']['lr_scale']}",
        f"  - stage3.blend_alpha: {online['stage3'].get('blend_alpha', 0.0)}",
        f"  - stage3.blend_alpha_min: {online['stage3'].get('blend_alpha_min', online['stage3'].get('blend_alpha', 0.0))}",
        f"  - stage3.blend_alpha_max: {online['stage3'].get('blend_alpha_max', online['stage3'].get('blend_alpha', 0.0))}",
        f"  - stage3.with_stage2_blend_scale: {online['stage3'].get('with_stage2_blend_scale', 1.0)}",
        f"  - stage3.with_stage1_stage2_blend_scale: {online['stage3'].get('with_stage1_stage2_blend_scale', online['stage3'].get('with_stage2_blend_scale', 1.0))}",
    ]
    resolved = resolve_project_path(path)
    os.makedirs(resolved.parent, exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--search-config", default="configs/autoresearch_main.yaml")
    parser.add_argument("--results", default="artifacts/autoresearch/main_search_results.tsv")
    parser.add_argument("--best-config-out", default="configs/best_online.yaml")
    parser.add_argument("--journal", default="logs/autoresearch_main/journal.md")
    parser.add_argument("--detailed-log", default="logs/autoresearch_main/detailed_log.md")
    parser.add_argument("--trial-config-dir", default="logs/autoresearch_main/trial_configs")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    args = parser.parse_args()

    search = load_search_config(args.search_config)
    if args.max_trials is not None:
        search.max_trials = args.max_trials
    if args.time_budget_hours is not None:
        search.time_budget_hours = args.time_budget_hours

    ensure_dir(os.path.dirname(args.journal))
    ensure_dir(os.path.dirname(args.detailed_log))
    ensure_dir(args.trial_config_dir)

    base_cfg = load_yaml(args.config)
    rng = random.Random(search.seed)
    deadline = time.time() + search.time_budget_hours * 3600.0
    prior_rows = read_tsv_rows(args.results)
    completed_trials = {row["trial"] for row in prior_rows if "trial" in row}
    completed_count = len(completed_trials)

    incumbent_cfg = apply_proxy_overrides(base_cfg, search)
    incumbent_score = math.inf
    incumbent_name = "base_proxy"
    no_improve = 0
    leaderboard: list[tuple[float, str, dict]] = []

    write_journal(
        args.journal,
        f"\n## {datetime.now().isoformat(timespec='seconds')} start\n"
        f"- config: `{args.config}`\n"
        f"- search_config: `{args.search_config}`\n"
        f"- max_trials: {search.max_trials}\n"
        f"- time_budget_hours: {search.time_budget_hours}\n"
        f"- proxy_epochs: {search.proxy_epochs}\n",
    )
    write_detailed_log(
        args.detailed_log,
        "search start",
        f"config={args.config}, search_config={args.search_config}, max_trials={search.max_trials}, time_budget_hours={search.time_budget_hours}, proxy_epochs={search.proxy_epochs}",
        incumbent_cfg,
        {
            "baseline_online_mse": "n/a",
            "full_combo_online_mse": "n/a",
            "best_variant": "n/a",
            "singles_improved": "n/a",
            "combo_best": "n/a",
            "objective_score": "n/a",
        },
        "info",
        0.0,
    )

    trial_idx = 0
    next_trial_number = completed_count + 1
    while trial_idx < search.max_trials and time.time() < deadline:
        trial_idx += 1
        if trial_idx == 1 and not prior_rows:
            trial_name = "trial001_seed_base"
            candidate_cfg = copy.deepcopy(incumbent_cfg)
            desc = "seed base proxy config"
        else:
            desc, candidate_cfg = make_mutation(base_cfg, incumbent_cfg, rng, search.mutation_scale)
            trial_name = f"trial{next_trial_number:03d}"
            next_trial_number += 1

        if trial_name in completed_trials:
            continue

        cfg_path = os.path.join(args.trial_config_dir, f"{trial_name}.yaml")
        save_yaml(cfg_path, candidate_cfg)
        start = time.time()
        status = "ok"
        try:
            _, summary = run_ablation_suite(
                candidate_cfg,
                results_path="artifacts/autoresearch/main_results.tsv",
                trial_tag=trial_name,
            )
            score = score_from_summary(summary)
        except Exception as exc:
            status = "error"
            summary = {
                "baseline_online_mse": float("nan"),
                "full_combo_online_mse": float("nan"),
                "best_variant": "error",
                "singles_improved": False,
                "combo_best": False,
                "objective_score": float("inf"),
            }
            score = float("inf")
            desc = f"{desc} | error={type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial": trial_name,
            "objective_score": f"{score:.6f}" if math.isfinite(score) else "inf",
            "baseline_online_mse": f"{summary['baseline_online_mse']:.6f}" if math.isfinite(summary["baseline_online_mse"]) else "nan",
            "full_combo_online_mse": f"{summary['full_combo_online_mse']:.6f}" if math.isfinite(summary["full_combo_online_mse"]) else "nan",
            "best_variant": summary["best_variant"],
            "singles_improved": str(summary["singles_improved"]),
            "combo_best": str(summary["combo_best"]),
            "elapsed_sec": f"{elapsed:.1f}",
            "status": status,
            "config_path": cfg_path,
            "description": desc,
        }
        append_tsv_row(args.results, row)
        completed_trials.add(trial_name)

        if score < incumbent_score:
            incumbent_score = score
            incumbent_cfg = copy.deepcopy(candidate_cfg)
            incumbent_name = trial_name
            save_yaml(args.best_config_out, incumbent_cfg)
            no_improve = 0
        else:
            no_improve += 1

        leaderboard.append((score, trial_name, copy.deepcopy(candidate_cfg)))
        leaderboard.sort(key=lambda item: item[0])
        leaderboard = leaderboard[: search.keep_top_k]

        write_journal(
            args.journal,
            f"- trial `{trial_name}` status={status} score={row['objective_score']} "
            f"best_variant={summary['best_variant']} combo_best={summary['combo_best']} "
            f"elapsed={elapsed:.1f}s desc={desc}",
        )
        write_detailed_log(args.detailed_log, f"trial {trial_name}", desc, candidate_cfg, summary, status, elapsed)

        if no_improve >= search.early_stop_patience:
            write_journal(args.journal, f"- early stop triggered after {no_improve} non-improving trials")
            break

    # Final verification on top candidates with full training budget.
    finalists = leaderboard[: search.keep_top_k]
    write_journal(args.journal, "\n## final verification")
    final_best_score = math.inf
    final_best_name = None
    final_best_cfg = None
    for rank, (_, trial_name, proxy_cfg) in enumerate(finalists, start=1):
        if time.time() >= deadline:
            break
        full_cfg = copy.deepcopy(proxy_cfg)
        full_cfg["train"]["epochs"] = base_cfg["train"]["epochs"]
        full_cfg["invariant"]["epochs"] = base_cfg["invariant"]["epochs"]
        final_name = f"{trial_name}_final"
        cfg_path = os.path.join(args.trial_config_dir, f"{final_name}.yaml")
        save_yaml(cfg_path, full_cfg)
        start = time.time()
        status = "ok"
        try:
            _, summary = run_ablation_suite(
                full_cfg,
                results_path="artifacts/autoresearch/main_results.tsv",
                trial_tag=final_name,
            )
            score = score_from_summary(summary)
        except Exception as exc:
            status = "error"
            summary = {
                "baseline_online_mse": float("nan"),
                "full_combo_online_mse": float("nan"),
                "best_variant": "error",
                "singles_improved": False,
                "combo_best": False,
                "objective_score": float("inf"),
            }
            score = float("inf")
        elapsed = time.time() - start
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial": final_name,
            "objective_score": f"{score:.6f}" if math.isfinite(score) else "inf",
            "baseline_online_mse": f"{summary['baseline_online_mse']:.6f}" if math.isfinite(summary["baseline_online_mse"]) else "nan",
            "full_combo_online_mse": f"{summary['full_combo_online_mse']:.6f}" if math.isfinite(summary["full_combo_online_mse"]) else "nan",
            "best_variant": summary["best_variant"],
            "singles_improved": str(summary["singles_improved"]),
            "combo_best": str(summary["combo_best"]),
            "elapsed_sec": f"{elapsed:.1f}",
            "status": status,
            "config_path": cfg_path,
            "description": f"final verification rank={rank} from {trial_name}",
        }
        append_tsv_row(args.results, row)
        write_journal(
            args.journal,
            f"- final `{final_name}` status={status} score={row['objective_score']} "
            f"best_variant={summary['best_variant']} combo_best={summary['combo_best']} elapsed={elapsed:.1f}s",
        )
        write_detailed_log(
            args.detailed_log,
            f"final verification {final_name}",
            f"final verification rank={rank} from {trial_name}",
            full_cfg,
            summary,
            status,
            elapsed,
        )
        if score < final_best_score:
            final_best_score = score
            final_best_name = final_name
            final_best_cfg = full_cfg

    if final_best_cfg is not None:
        save_yaml(args.best_config_out, final_best_cfg)
        write_journal(
            args.journal,
            f"\n## best\n- final_best: `{final_best_name}`\n- score: {final_best_score:.6f}\n- config: `{args.best_config_out}`",
        )
        print(f"[best] trial={final_best_name} score={final_best_score:.6f} saved_config={args.best_config_out}")
    elif math.isfinite(incumbent_score):
        save_yaml(args.best_config_out, incumbent_cfg)
        write_journal(
            args.journal,
            f"\n## best\n- proxy_best: `{incumbent_name}`\n- score: {incumbent_score:.6f}\n- config: `{args.best_config_out}`",
        )
        print(f"[best] trial={incumbent_name} score={incumbent_score:.6f} saved_config={args.best_config_out}")
    else:
        raise RuntimeError("no successful trials ran")


if __name__ == "__main__":
    main()
