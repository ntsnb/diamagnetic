"""固定发行预测的在线评估；逐观测释放标签，记录适应成本与完整消融。"""
from dataclasses import asdict, replace
import copy
from itertools import product
import time
import numpy as np
import torch
from .adaptation import OnlineController
from .stream import DelayedFeedbackStream
from .diagnostics import ExpertDiagnostics, ExpertDiagnosticsConfig


def stage_variants():
    return {"stages_" + "".join(str(int(v)) for v in flags): flags for flags in product((False, True), repeat=3)}


def selection_variants():
    """同一离线模型的定位对照；统一仅启用Stage II，不改变训练。"""
    common = {"stage_i": False, "stage_ii": True, "stage_iii": False,
              "selection_protocol": "fixed_budget", "stage_ii_scope": "sparse_experts"}
    return {"matched_lrp": {**common, "selector": "relevance_only"},
            "matched_gradient": {**common, "selector": "gradient_only"},
            "matched_random": {**common, "selector": "random"},
            "matched_dense": {**common, "selector": "gradient_only", "stage_ii_scope": "dense_eligible"}}


def selection_comparison_summary(records):
    """核验相同释放事件上的精确稀疏预算；非零改变量与预算分开报告。"""
    sparse = [name for name in ("matched_lrp", "matched_gradient", "matched_random") if name in records]
    equal = None
    if len(sparse) > 1:
        def budget_signature(events):
            return [(item["time"], tuple(item["issue_times"]), item["experts"]["selected_parameter_count"],
                     tuple(sorted((layer, budget["selected_parameter_count"], budget["input_width"])
                                  for layer, budget in item["experts"]["layer_budgets"].items()))) for item in events]
        reference = budget_signature(records[sparse[0]])
        for name in sparse[1:]:
            actual = budget_signature(records[name])
            if actual != reference:
                raise ValueError(f"matched parameter budgets or release schedules differ: {sparse[0]} vs {name}")
        equal = bool(reference)
    summaries = {}
    for name, events in records.items():
        experts = [event["experts"] for event in events]
        summaries[name] = {"released_events": len(events), "accepted_events": sum(event["accepted"] for event in experts),
            "selected_parameter_counts": sorted({event["selected_parameter_count"] for event in experts}),
            "total_selected_parameter_events": sum(event["selected_parameter_count"] for event in experts),
            "total_changed_parameter_events": sum(event["changed_parameter_count"] for event in experts),
            "eligible_parameter_counts": sorted({event["eligible_parameter_count"] for event in experts})}
    return {"same_offline_checkpoint": True, "stages": [False, True, False],
            "candidate_scope": "expert linear weight matrices excluding prediction heads",
            "equal_selected_parameters_per_sparse_event": equal,
            "all_three_sparse_controls_present": len(sparse) == 3,
            "actual_nonzero_changes_may_differ": "zero gradients, rounding or rejected line searches; reported separately",
            "controls": summaries}


def evaluate_stream(model, branches, generator, data, split, online_config, *, max_windows=None,
                    diagnostics_config=None):
    """split仅指定发行起点；监督标签由data.targets逐时刻读取。"""
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    working = copy.deepcopy(model)
    diagnostics_config = diagnostics_config or ExpertDiagnosticsConfig()
    diagnostics = ExpertDiagnostics(diagnostics_config, working) if diagnostics_config.enabled else None
    device = next(working.parameters()).device
    selected = split.issue_times if max_windows is None else split.issue_times[:max_windows]
    # 超出文件时间范围的额外延迟不虚构标签；该类样本仍可离线评分。
    if not len(selected):
        raise ValueError("no evaluation forecasts")
    first, last = int(selected[0]), int(selected[-1])
    if not np.all(np.diff(selected) > 0):
        raise ValueError("issue times must be strictly increasing")
    selected_set = {int(t) for t in selected}
    previous = []
    for t in range(max(model.config.lookback - 1, first - online_config.reference_windows), first):
        previous.append(data.values[t - model.config.lookback + 1:t + 1])
    reference = torch.stack(previous) if len(previous) == online_config.reference_windows else None
    controller = OnlineController(working, online_config, generator=generator, contrast_fn=branches, reference_inputs=reference)
    stream = DelayedFeedbackStream(controller, allow_missing=bool(data.metadata.get("config", {}).get("target_mask_columns")))
    issued, updates = [], []
    algorithm_peak, diagnostic_peak = 0, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # 只为全部已发行预测读入已有未来观测；不超出数据末尾。
    end = min(len(data.values) - 1, int(split.issue_times[-1]) + model.config.horizon,
              last + model.config.horizon + online_config.extra_delay)
    for t in range(first, end + 1):
        x = data.values[t - model.config.lookback + 1:t + 1].unsqueeze(0).to(device)
        if t in selected_set:
            forecast, update = stream.step(t, x, data.targets[t])
            issued.append(forecast)
            if diagnostics is not None:
                if device.type == "cuda":
                    algorithm_peak = max(algorithm_peak, torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                diagnostics.capture(working, x, t)
                if device.type == "cuda":
                    diagnostic_peak = max(diagnostic_peak, torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
        else:
            update = stream.observe(t, data.targets[t], current_x=x)
        if update.issue_times:
            updates.append(asdict(update))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_with_diagnostics = time.perf_counter() - started
    diagnostic_seconds = diagnostics.seconds if diagnostics is not None else 0.0
    elapsed = elapsed_with_diagnostics - diagnostic_seconds
    prediction = torch.cat([r.prediction for r in issued]).numpy()
    truth = torch.stack([data.targets[r.issue_time + 1:r.issue_time + model.config.horizon + 1] for r in issued]).numpy()
    error = prediction - truth
    scale = np.array(data.metadata["std"])[data.metadata["target_indices"]]
    native_error = error * scale[None, None, :]
    observed = np.isfinite(truth)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite issued prediction")
    mean_observed = lambda values: float(values[observed].mean()) if observed.any() else None
    metrics = {"normalized_mse": mean_observed(np.square(error)), "normalized_mae": mean_observed(np.abs(error)),
               "native_mse": mean_observed(np.square(native_error)), "native_mae": mean_observed(np.abs(native_error)),
               "observed_target_values": int(observed.sum()), "missing_target_values": int((~observed).sum()),
               "skipped_incomplete_label_updates": len(stream.skipped_forecasts),
               "issued_forecasts": len(issued), "released_updates": len(updates), "unreleased_forecasts": len(controller.pending_records),
               "seconds": elapsed, "milliseconds_per_forecast": elapsed * 1000 / len(issued),
               "diagnostic_seconds": diagnostic_seconds, "total_seconds_including_diagnostics": elapsed_with_diagnostics,
               "peak_cuda_bytes": max(algorithm_peak, torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
               "diagnostic_peak_cuda_bytes": diagnostic_peak,
               "accepted_router_updates": sum(u["routing"]["accepted"] for u in updates),
               "accepted_expert_updates": sum(u["experts"]["accepted"] for u in updates)}
    issue_times = np.array([r.issue_time for r in issued])
    result = {"metrics": metrics, "updates": updates, "skipped_feedback": stream.skipped_forecasts,
              "issue_times": issue_times, "predictions": prediction, "targets": truth}
    if diagnostics is not None:
        result["expert_diagnostics"] = diagnostics.finish(issue_times, truth, prediction, scale)
    return result


@torch.no_grad()
def validation_error_threshold(model, validation_x, validation_y, quantile=.9, batch_size=32,
                               generator=None, branches=None):
    if not 0 < quantile < 1:
        raise ValueError("error quantile must lie in (0,1)")
    device = next(model.parameters()).device
    errors = []
    for start in range(0, len(validation_x), batch_size):
        x = validation_x[start:start + batch_size].to(device)
        gains = generator(branches(x)) if generator is not None else None
        pred = model(x, modulations=gains).prediction
        errors.append((pred - validation_y[start:start + batch_size].to(device)).square().mean((1, 2)).cpu())
    return float(torch.quantile(torch.cat(errors), quantile))
