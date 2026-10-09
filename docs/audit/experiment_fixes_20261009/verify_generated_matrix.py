"""只检查双种子配置、任务清单和真实数据时间索引，不运行训练。"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper_impl.data import DataConfig, prepare_data
from paper_impl.experiment import plan_matrix
from paper_impl.run import _normalize


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    matrix = load(matrix_path)
    plan = plan_matrix(matrix_path, seeds=[42, 43])
    save(output / "plan.json", plan)
    counts = Counter()
    runnable, evaluation = [], []
    for family, field, variant_key in (("framework", "jobs", "ablation"), ("baseline", "baseline_jobs", "method")):
        for job in matrix[field]:
            config = matrix_path.parent / job["config"]
            assert config.is_file(), config
            assert digest(config) == job["config_sha256"], config
            key = (family, job["dataset"], job["backbone"], job[variant_key], job["seed"])
            counts[key] += 1
            record = {"family": family, "dataset": job["dataset"], "backbone": job["backbone"],
                      "variant": job[variant_key], "seed": job["seed"], "config": str(config),
                      "config_sha256": job["config_sha256"], "status": "planned_not_executed"}
            variants = job["evaluation_variants"] if family == "framework" else [job["method"]]
            runnable.append({**record, "evaluation_count": len(variants)})
            evaluation.extend({**record, "evaluation_variant": variant} for variant in variants)
    assert all(count == 1 for count in counts.values()), "duplicate runnable tasks"
    assert len(counts) == plan["job_count"]
    for name, rows in (("training_tasks.csv", runnable), ("evaluation_tasks.csv", evaluation)):
        with (output / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    datasets = []
    for dataset in matrix["datasets"]:
        job = next(j for j in matrix["jobs"] if j["dataset"] == dataset["id"]
                   and j["ablation"] == "full" and j["seed"] == 42)
        config = _normalize(load(matrix_path.parent / job["config"]))
        source = Path(config["data"]["path"])
        assert digest(source) == dataset["source_sha256"], source
        data = prepare_data(DataConfig(**config["data"]))
        ids = data.train.issue_times
        assert len(ids) == len(data.train) and np.all(np.diff(ids) > 0)
        assert torch.isfinite(data.train.targets).all()
        horizon = config["data"]["horizon"]
        # Training labels must still equal the original target slice at each issue time.
        for row, issue in enumerate(ids):
            assert torch.equal(data.train.targets[row], data.targets[int(issue) + 1:int(issue) + 1 + horizon])
        gaps = [{"left_issue": int(ids[row]), "right_issue": int(ids[row + 1]),
                 "distance_steps": int(ids[row + 1] - ids[row])}
                for row in np.flatnonzero(np.diff(ids) > 1)]
        datasets.append({"dataset": dataset["id"], "source_sha256": dataset["source_sha256"],
                         "train_windows": len(ids), "interval_seconds": config["data"]["expected_interval_seconds"],
                         "original_training_issue_times_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                         "excluded_incomplete_supervised_horizons": data.metadata["excluded_incomplete_supervised_horizons"],
                         "missing_target_elements": int(torch.isnan(data.targets).sum()),
                         "train_gaps": gaps, "labels_match_original_time_slices": True})
        del data
        gc.collect()
    result = {"status": "passed_no_training_started", "matrix_sha256": digest(matrix_path),
              "seed_coverage_validated": plan["seed_coverage_validated"], "seeds": plan["seeds"],
              "training_job_count": len(runnable), "evaluation_task_count": len(evaluation),
              "framework_training_configs": len(matrix["jobs"]), "baseline_configs": len(matrix["baseline_jobs"]),
              "framework_evaluation_count": matrix["evaluation_count"],
              "jobs_per_seed": {str(seed): sum(row["seed"] == seed for row in runnable) for seed in (42, 43)},
              "all_configuration_hashes_verified": True, "all_data_source_hashes_verified": True,
              "datasets": datasets}
    save(output / "validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "datasets"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
