"""发行时专家诊断；只读取当前模型，真实目标仅用于事后评分。"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .discrepancy import pairwise_discrepancies


@dataclass(frozen=True)
class ExpertDiagnosticsConfig:
    enabled: bool = False
    distance_interval: int = 1

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("expert_diagnostics.enabled must be boolean")
        if (isinstance(self.distance_interval, bool) or not isinstance(self.distance_interval, int)
                or self.distance_interval < 1):
            raise ValueError("expert_diagnostics.distance_interval must be a positive integer")


class ExpertDiagnostics:
    """保存每窗路由/专家预测及固定间隔的未加路由权重的LRP分工距离。

    使用持久专家参数，不包含Stage III临时调制；与训练分工损失定义一致。
    enabled由调用方控制。间隔按已发行窗口计，采样不依赖误差或未来标签。
    """

    def __init__(self, config: ExpertDiagnosticsConfig, model):
        self.config = config
        self.n_experts = model.config.n_experts
        self.n_targets = model.config.targets
        self.horizon = model.config.horizon
        self.issue_times, self.gates, self.predictions = [], [], []
        self.pair_times, self.distances, self.valid, self.total_relevance = [], [], [], []
        self.pair_indices = torch.triu_indices(self.n_experts, self.n_experts, 1).numpy()
        self.seconds = 0.0

    def capture(self, model, x, issue_time):
        """在本窗口更新和预测完成后、下一窗口更新之前取样，不接收目标。"""
        device = next(model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        if devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        # 不消耗在线预测/更新使用的随机序列，也不调用backward或optimizer。
        with torch.random.fork_rng(devices=devices):
            with torch.no_grad():
                output = model(x, stochastic=False)
                gates = output.gates.detach().cpu().numpy()[0]
                predictions = torch.stack([model.expert_prediction(x, m)
                                           for m in range(self.n_experts)], dim=1).detach().cpu().numpy()[0]
            if gates.shape != (self.n_experts,) or not np.isfinite(gates).all() or np.any(gates < 0):
                raise ValueError("invalid diagnostic routing weights")
            if not np.isclose(gates.sum(), 1.0) or predictions.shape != (self.n_experts, self.horizon, self.n_targets):
                raise ValueError("invalid diagnostic expert output shape or routing sum")
            if not np.isfinite(predictions).all():
                raise FloatingPointError("nonfinite diagnostic expert prediction")
            if len(self.issue_times) % self.config.distance_interval == 0:
                # Official LRP enables the local gradients it needs internally;
                # diagnostics do not need a higher-order parameter-gradient graph.
                with torch.no_grad():
                    maps = model.attribute(x, freeze_head=True).maps.detach()
                expected = (1, self.n_targets, self.n_experts, model.config.n_variables, model.config.lookback)
                if tuple(maps.shape) != expected:
                    raise ValueError("diagnostic relevance map dimensions differ from the model")
                if not bool(torch.isfinite(maps).all()) or bool((maps < 0).any()):
                    raise ValueError("diagnostic relevance must be finite and nonnegative")
                if self.n_experts > 1:
                    pairs = pairwise_discrepancies(maps)
                    self.distances.append(pairs.distances[0].cpu().numpy())
                    self.valid.append(pairs.valid[0].cpu().numpy())
                else:
                    self.distances.append(np.empty((self.n_targets, 0), dtype=np.float64))
                    self.valid.append(np.empty((self.n_targets, 0), dtype=bool))
                self.total_relevance.append(maps[0].double().sum((-2, -1)).cpu().numpy())
                self.pair_times.append(issue_time)
        self.issue_times.append(issue_time)
        self.gates.append(gates)
        self.predictions.append(predictions)
        if devices:
            torch.cuda.synchronize(device)
        self.seconds += time.perf_counter() - started

    def finish(self, issue_times, targets, served_predictions, native_scale):
        """目标只在全部发行完成后用于误差计算，NaN不参与统计。"""
        if not np.array_equal(np.asarray(self.issue_times), issue_times):
            raise ValueError("diagnostics do not align with issued forecast times")
        gates, predictions = np.stack(self.gates), np.stack(self.predictions)
        error = predictions - targets[:, None, :, :]
        native_error = error * native_scale[None, None, None, :]
        observed = np.isfinite(targets)
        count = int(observed.sum())
        per_expert = []
        for m in range(self.n_experts):
            per_expert.append({"expert_index": m,
                "normalized_mse": float(np.square(error[:, m][observed]).mean()) if count else None,
                "normalized_mae": float(np.abs(error[:, m][observed]).mean()) if count else None,
                "native_mse": float(np.square(native_error[:, m][observed]).mean()) if count else None,
                "native_mae": float(np.abs(native_error[:, m][observed]).mean()) if count else None,
                "observed_target_values": count})
        safe_gates = np.clip(gates.astype(np.float64), np.finfo(np.float64).tiny, 1.)
        entropy = -(gates * np.log(safe_gates)).sum(-1)
        winners = gates.argmax(-1)
        distances, valid = np.stack(self.distances), np.stack(self.valid)
        pair_summaries = []
        for index, (m, n) in enumerate(self.pair_indices.T):
            selected = distances[:, :, index][valid[:, :, index]]
            pair_summaries.append({"experts": [int(m), int(n)], "valid_target_windows": int(selected.size),
                "mean_distance": float(selected.mean()) if selected.size else None})
        summary = {"status": "recorded", "scope": "persistent experts without Stage III modulation; ungated expert relevance",
            "issued_windows": len(gates), "distance_windows": len(self.pair_times),
            "distance_interval": self.config.distance_interval, "distance_sampling": "every fixed number of issued windows, starting at the first",
            "diagnostic_seconds": self.seconds,
            "mean_gate_weight": gates.mean(0).tolist(),
            "top1_count": np.bincount(winners, minlength=self.n_experts).tolist(),
            "top1_share": (np.bincount(winners, minlength=self.n_experts) / len(gates)).tolist(),
            "top1_tie_rule": "smallest expert index", "utilization_definition": "mean routing mass and top-1 routing share; not compute sparsity",
            "mean_routing_entropy": float(entropy.mean()),
            "mean_effective_experts": float(np.exp(entropy).mean()),
            "per_expert_error": per_expert, "pair_distances": pair_summaries,
            "mean_specialization_distance": float(distances[valid].mean()) if valid.any() else None,
            "invalid_pair_target_windows": int((~valid).sum()),
            "single_expert": self.n_experts == 1}
        arrays = {"issue_times": np.asarray(self.issue_times), "routing_weights": gates,
                  "routing_entropy": entropy, "expert_predictions": predictions,
                  "expert_errors_normalized": error, "expert_errors_native": native_error,
                  "served_errors_normalized": served_predictions - targets,
                  "served_errors_native": (served_predictions - targets) * native_scale[None, None, :],
                  "observed_target_mask": observed, "pair_issue_times": np.asarray(self.pair_times),
                  "pair_indices": self.pair_indices, "pair_distances": distances, "pair_valid": valid,
                  "total_relevance": np.stack(self.total_relevance)}
        return {"summary": summary, "arrays": arrays}
