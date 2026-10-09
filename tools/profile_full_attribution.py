"""八个完整官方骨干的有界单样本归因成本记录；不训练，不做预测性能实验。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from paper_impl.discrepancy import specialization_loss
from paper_impl.model import ModelConfig, PaperForecaster
from paper_impl.official_backbones import OFFICIAL_MODEL_NAMES, official_source_info
from paper_impl.training import seed_everything


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def matrix_configuration(name):
    """直接读取本轮 Full 配置的默认结构，不为通过成本检查缩小任何尺寸。"""
    path = ROOT / "configs/full_experiment/structural/lishui_city" / name / "full_seed42.json"
    stored = json.loads(path.read_text(encoding="utf8"))
    values = {**stored["model"], "n_variables": 8, "lookback": 24, "horizon": 12,
              "targets": 1, "target_indices": (0,)}
    config = ModelConfig(**values)
    assert (config.n_experts, config.hidden, config.heads, config.depth) == (3, 64, 8, 2)
    assert stored["data"]["lookback"] == 24 and stored["data"]["horizon"] == 12
    return config, {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def source_evidence(name):
    source = official_source_info(name)
    runtime = ROOT / "third_party/official_forecasters/runtime" / name
    actual = {relative: sha256(runtime / relative) for relative in source["runtime_files"]}
    paths = ["paper_impl/model.py", "paper_impl/backbones.py", "paper_impl/lrp.py",
             "paper_impl/discrepancy.py", "paper_impl/official_backbones.py", "paper_impl/official_lrp.py",
             "paper_impl/training.py", "third_party/official_forecasters/SOURCE_MANIFEST.json",
             "third_party/official_forecasters/RUNTIME_MANIFEST.json", "tools/profile_full_attribution.py"]
    return {"repository": source["repository"], "commit": source["commit"],
            "implementation_kind": source["implementation_kind"], "runtime_files_sha256": actual,
            "runtime_matches_pinned_manifest": actual == source["runtime_files"],
            "patch_sha256": sha256(source["patch_path"]),
            "source_hashes": {path: sha256(ROOT / path) for path in paths}}


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"query": "name,memory.total,memory.used,memory.free,utilization.gpu",
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode}
    except Exception as error:
        return {"unavailable": repr(error)}


def run_worker(name, output, device_name, seed):
    """一个独立顺序子进程只实例化一个模型，成功或失败后仅释放自己的资源。"""
    started = time.perf_counter()
    result = {"model": name, "status": "running", "phase": "configuration", "started_utc": utc_now(),
              "seed": seed, "dtype": "torch.float32", "device_requested": device_name,
              "optimizer_steps": 0, "real_data_used": False, "repetitions": 1,
              "shape": {"batch": 1, "lookback": 24, "horizon": 12, "variables": 8,
                        "targets": 1, "experts": 3, "hidden": 64, "heads": 8, "depth": 2},
              "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "margin": .3, "freeze_head": True, "model_training_mode": True,
              "attribution_mode": "existing official per-sample eval-mode relevance rules",
              "timing_note": "one cold attribution, specialization forward and backward; no warmup or optimizer; synchronized CUDA phase timing"}
    model = x = attribution = spec = None
    device = torch.device(device_name)
    try:
        torch.set_num_threads(1)
        seed_everything(seed)
        gc.collect()
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            result["gpu_name"] = torch.cuda.get_device_name(device)
            result["cuda_memory_before_model"] = dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info(device)))
        config, config_source = matrix_configuration(name)
        result["model_config"] = asdict(config)
        result["matrix_config_source"] = config_source
        result["source"] = source_evidence(name)
        if not result["source"]["runtime_matches_pinned_manifest"]:
            raise RuntimeError("runtime source differs from its pinned manifest")
        result["phase"] = "construction"
        write_json(output / "result.json", result)
        construction_start = time.perf_counter()
        model = PaperForecaster(config).to(device).train()
        # 单独的 CPU RNG 确保所有骨干接收同一个样本，不受模型初始化耗随机数影响。
        x = torch.randn(1, 24, 8, generator=torch.Generator().manual_seed(seed + 1))
        result["input_sha256"] = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        x = x.to(device)
        result["official_config"] = asdict(model.experts[0].config)
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        result["trainable_parameter_count"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        result["parameter_tensor_count"] = len(list(model.parameters()))
        result["expert_parameter_counts"] = [sum(parameter.numel() for parameter in expert.parameters()) for expert in model.experts]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["initial_allocated_bytes"] = torch.cuda.memory_allocated(device)
            result["initial_reserved_bytes"] = torch.cuda.memory_reserved(device)
        result["construction_seconds"] = time.perf_counter() - construction_start
        result["phase"] = "attribution"
        write_json(output / "result.json", result)
        phase_start = time.perf_counter()
        attribution = model.attribute(x, freeze_head=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["attribution_seconds"] = time.perf_counter() - phase_start
        result["phase"] = "specialization_forward"
        write_json(output / "result.json", result)
        phase_start = time.perf_counter()
        spec = specialization_loss(attribution.maps, .3)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["specialization_forward_seconds"] = time.perf_counter() - phase_start
        result["phase"] = "specialization_backward"
        write_json(output / "result.json", result)
        phase_start = time.perf_counter()
        spec.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["specialization_backward_seconds"] = time.perf_counter() - phase_start
        result["attribution_and_spec_seconds"] = sum(result[key] for key in
            ("attribution_seconds", "specialization_forward_seconds", "specialization_backward_seconds"))
        if device.type == "cuda":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            result["incremental_peak_allocated_bytes"] = result["peak_allocated_bytes"] - result["initial_allocated_bytes"]
        result["phase"] = "validation"
        write_json(output / "result.json", result)
        result["maps_shape"] = list(attribution.maps.shape)
        result["finite_maps"] = bool(torch.isfinite(attribution.maps).all())
        result["finite_specialization_loss"] = bool(torch.isfinite(spec))
        result["specialization_loss"] = float(spec.detach()) if result["finite_specialization_loss"] else None
        result["finite_gradients"] = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        result["gradient_tensor_count"] = sum(parameter.grad is not None for parameter in model.parameters())
        result["nonzero_spec_gradient_tensor_count"] = sum(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.parameters())
        result["nonzero_spec_gradient_element_count"] = sum(int(parameter.grad.count_nonzero()) for parameter in model.parameters() if parameter.grad is not None)
        result["nonzero_spec_gradient"] = result["nonzero_spec_gradient_tensor_count"] > 0
        error = attribution.conservation_error.detach().max()
        result["max_signed_conservation_error"] = float(error) if bool(torch.isfinite(error)) else None
        if not all(result[key] for key in ("finite_maps", "finite_specialization_loss", "finite_gradients")):
            raise FloatingPointError("nonfinite attribution, specialization loss, or parameter gradient")
        result["status"], result["phase"] = "passed", "complete"
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if device.type == "cuda" and torch.cuda.is_initialized():
            try:
                result["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                result["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            except Exception:
                pass
    finally:
        model = x = attribution = spec = error = None
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                result["allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated(device)
                result["reserved_after_cleanup_bytes"] = torch.cuda.memory_reserved(device)
            except Exception as error:
                result["cleanup_error"] = repr(error)
        result["worker_wall_seconds"] = time.perf_counter() - started
        result["finished_utc"] = utc_now()
        write_json(output / "result.json", result)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-wall-seconds", type=int, default=570)
    parser.add_argument("--per-model-seconds", type=int, default=90)
    parser.add_argument("--worker", choices=OFFICIAL_MODEL_NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(run_worker(args.worker, args.output, args.device, args.seed))
    if not 1 <= args.max_wall_seconds <= 570 or not 1 <= args.per_model_seconds <= 90:
        raise ValueError("keep total budget <=570 seconds and each model <=90 seconds")
    start = time.perf_counter()
    deadline = start + args.max_wall_seconds
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"run_id": args.output.name, "started_utc": utc_now(), "device": args.device, "seed": args.seed,
               "max_wall_seconds": args.max_wall_seconds, "per_model_seconds": args.per_model_seconds,
               "requested_models": list(OFFICIAL_MODEL_NAMES), "script_sha256": sha256(__file__),
               "isolation": "sequential child processes owned by this script; gc/empty_cache then exit; no external process control",
               "gpu_before": gpu_snapshot(), "status": "running", "records": [],
               "scope": "one synthetic sample, default full matrix structure, no optimizer, no real-data accuracy, no linear extrapolation"}
    write_json(args.output / "summary.json", summary)
    for name in OFFICIAL_MODEL_NAMES:
        directory = args.output / name
        directory.mkdir()
        remaining = deadline - time.perf_counter() - 5
        if remaining <= 0:
            result = {"model": name, "status": "not_run_budget_exhausted", "failure": {"type": "TotalBudgetExpired"}}
            write_json(directory / "result.json", result)
        else:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--output", str(directory.resolve()),
                       "--device", args.device, "--seed", str(args.seed)]
            timed_out, return_code = False, None
            worker_start = time.perf_counter()
            with (directory / "stdout.txt").open("w", encoding="utf8") as stdout, (directory / "stderr.txt").open("w", encoding="utf8") as stderr:
                try:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=min(args.per_model_seconds, remaining),
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), cwd=ROOT)
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True  # subprocess.run terminates only the child it created.
            path = directory / "result.json"
            try:
                result = json.loads(path.read_text(encoding="utf8"))
            except (FileNotFoundError, json.JSONDecodeError):
                result = {"model": name, "status": "failed", "phase": "worker_startup"}
            if timed_out:
                result.update(status="failed", failure={"type": "OwnedWorkerTimeout", "message": "owned worker exceeded bounded time"})
            elif return_code != 0 and result.get("status") != "failed":
                result.update(status="failed", failure={"type": "OwnedWorkerExit", "returncode": return_code})
            result["subprocess_wall_seconds"] = time.perf_counter() - worker_start
            result["subprocess_returncode"] = return_code
            write_json(path, result)
        summary["records"].append(result)
        write_json(args.output / "progress.json", [{"model": item["model"], "status": item["status"],
                   "seconds": item.get("attribution_and_spec_seconds"), "peak_allocated_bytes": item.get("peak_allocated_bytes"),
                   "failure": item.get("failure", {}).get("type")} for item in summary["records"]])
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"model": name, "status": result["status"], "seconds": result.get("attribution_and_spec_seconds"),
                          "peak_allocated_bytes": result.get("peak_allocated_bytes"), "failure": result.get("failure", {}).get("type")}), flush=True)
    summary["status"] = "completed" if all(item["status"] == "passed" for item in summary["records"]) else "completed_with_failures"
    summary["finished_utc"] = utc_now()
    summary["gpu_after"] = gpu_snapshot()
    summary["wall_seconds"] = time.perf_counter() - start
    write_json(args.output / "summary.json", summary)
    raise SystemExit(0 if summary["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
