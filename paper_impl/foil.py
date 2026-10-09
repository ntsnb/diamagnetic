"""FOIL 离线训练：原文式 (6)--(10) 与 Algorithm 1 的交替过程。

依据 https://arxiv.org/html/2406.09130v1 与
https://proceedings.mlr.press/v235/liu24ae.html。双分支、逐目标归一化和
预测差诊断是本文扩展；较小的 variant 环境惩罚不保证表示或预测分离。

MIL 同时优化编码器和共享输出头，MEI 只优化独立的环境输出头。
每个 epoch 为一次全训练集优化步，计算图仅保留一个 batch。MIL 先
统计环境风险，再用解析链式系数累计精确全局目标的梯度。所有阶段用
eval 前向避免 dropout/BatchNorm 改变两遍扫描的函数；requires_grad
独立控制可训练参数。此精确等价要求前向确定且样本间独立。

sample_ids 与 train_x/train_y 逐行对应，保存过滤前的原始观测时间索引；
时间平滑按 id 差值选邻居，batch shuffle 不改变对应关系。未传 id 时仅为
兼容旧调用而使用连续行号。只在训练集推断环境，验证集仅选共享预测器
的最佳 checkpoint。输入应已用同一套训练统计量转换到相同目标单位。
训练不读取 test，也不写文件。
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FoilConfig:
    """轮数、全量优化步、损失权重及确定性设置。"""

    n_environments: int = 3
    rounds: int = 5
    mil_epochs: int = 10
    mei_epochs: int = 5
    batch_size: int = 128
    mil_learning_rate: float = 1e-3
    mei_learning_rate: float = 1e-3
    lambda_erm: float = 1.0
    lambda_invariant: float = 1.0
    lambda_variant: float = 0.1
    std_floor: float = 1e-5
    smoothing_radius: int = 2
    horizon_one: str = "error"
    max_grad_norm: float | None = None
    seed: int = 42
    shuffle: bool = True
    deterministic: bool = True
    mode: str = "foil"
    contrast: str = "difference"

    def __post_init__(self):
        if self.mode not in ("foil", "erm_pair", "variant_only", "disabled"):
            raise ValueError("mode must be foil, erm_pair, variant_only, or disabled")
        if self.contrast not in ("difference", "variant", "invariant"):
            raise ValueError("unknown context contrast")
        for name in ("n_environments", "rounds", "mil_epochs", "mei_epochs", "batch_size"):
            _positive_integer(getattr(self, name), name)
        for name in ("smoothing_radius", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**63 - 1009:
            raise ValueError("seed is too large")
        for name in ("mil_learning_rate", "mei_learning_rate", "std_floor"):
            _finite_number(getattr(self, name), name, positive=True)
        for name in ("lambda_erm", "lambda_invariant", "lambda_variant"):
            _finite_number(getattr(self, name), name)
        if self.max_grad_norm is not None:
            _finite_number(self.max_grad_norm, "max_grad_norm", positive=True)
        if self.horizon_one not in ("error", "zero_irn"):
            raise ValueError("horizon_one must be 'error' or 'zero_irn'")
        if not isinstance(self.shuffle, bool) or not isinstance(self.deterministic, bool):
            raise ValueError("shuffle and deterministic must be bool")


class MilLosses(NamedTuple):
    total: Tensor
    irn_mean: Tensor
    erm: Tensor
    environment_variance: Tensor
    environment_risks: Tensor


class EnvironmentAssignment(NamedTuple):
    labels: Tensor
    losses: Tensor
    counts: list[int]
    reassigned: int
    smoothed: int
    repaired: int


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def irn_loss(prediction: Tensor, target: Tensor, *, std_floor: float = 1e-5,
             horizon_one: str = "error", reduction: str = "mean") -> Tensor:
    """逐样本、逐目标沿 H 归一化；总体标准差及其梯度均保留。

输入 [..., H, T]，none 返回前导维度。floor 未激活时，对各目标的
独立正比例缩放和平移不变。H=1 默认报错；zero_irn 明确返回零形状
风险，只有 MIL 的 ERM 项可学习，环境分配没有 IRN 识别信息。
"""
    _finite_number(std_floor, "std_floor", positive=True)
    if horizon_one not in ("error", "zero_irn"):
        raise ValueError("invalid horizon_one policy")
    if reduction not in ("none", "mean"):
        raise ValueError("reduction must be 'none' or 'mean'")
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.shape != target.shape or prediction.ndim < 3 or min(prediction.shape) < 1:
        raise ValueError("expected matching nonempty [..., H, T] tensors")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("IRN requires floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("IRN inputs must be finite")
    if prediction.shape[-2] == 1 and horizon_one == "error":
        raise ValueError("IRN is uninformative for H=1; explicitly select zero_irn to use ERM only")
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    if torch.tensor(std_floor, dtype=dtype).item() == 0:
        raise ValueError("std_floor underflows the working dtype")

    def normalize(values):
        values = values.to(dtype=dtype)
        mean = values.mean(dim=-2, keepdim=True)
        std = values.std(dim=-2, correction=0, keepdim=True).clamp_min(std_floor)
        return (values - mean) / std

    per_sample = (normalize(prediction) - normalize(target)).square().mean(dim=(-2, -1))
    return per_sample if reduction == "none" else per_sample.mean()


def _labels(labels: Tensor, n: int, n_environments: int, *, nonempty=True) -> Tensor:
    _positive_integer(n_environments, "n_environments")
    if (not isinstance(labels, Tensor) or labels.ndim != 1 or labels.numel() != n
            or labels.dtype not in (torch.int32, torch.int64)):
        raise ValueError("labels must be an integer vector with one entry per sample")
    labels = labels.detach().to(device="cpu", dtype=torch.long)
    if n == 0 or (labels < 0).any() or (labels >= n_environments).any():
        raise ValueError("environment labels out of range or empty")
    if nonempty and (torch.bincount(labels, minlength=n_environments) == 0).any():
        raise ValueError("every environment must contain a sample")
    return labels


def initialize_environments(n_samples: int, n_environments: int, *, seed: int = 42) -> Tensor:
    """对平衡标签随机置换，保证初始环境非空；返回 CPU 全局标签。"""
    _positive_integer(n_samples, "n_samples")
    _positive_integer(n_environments, "n_environments")
    if n_samples < n_environments:
        raise ValueError("n_samples must be at least n_environments")
    generator = torch.Generator().manual_seed(seed)
    return (torch.arange(n_samples) % n_environments)[torch.randperm(n_samples, generator=generator)]


def _sample_ids(sample_ids: Tensor | None, n: int) -> Tensor:
    if sample_ids is None:
        return torch.arange(n)
    if (not isinstance(sample_ids, Tensor) or sample_ids.shape != (n,)
            or sample_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("sample_ids must be an integer vector")
    ids = sample_ids.detach().to(device="cpu", dtype=torch.long)
    if (ids < 0).any() or ids.unique().numel() != n:
        raise ValueError("sample_ids must be unique and nonnegative")
    return ids


def smooth_environment_labels(labels: Tensor, sample_ids: Tensor | None = None, *,
                              radius: int = 2, n_environments: int | None = None) -> Tensor:
    """按全局时间 id 同步多数投票；平票优先当前标签，其次较小编号。"""
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if not isinstance(labels, Tensor) or labels.numel() == 0:
        raise ValueError("labels must be nonempty")
    environments = int(labels.max()) + 1 if n_environments is None else n_environments
    labels = _labels(labels, labels.numel(), environments, nonempty=False)
    ids = _sample_ids(sample_ids, labels.numel())
    if radius > torch.iinfo(torch.long).max - int(ids.max()):
        raise ValueError("sample id plus radius overflows int64")
    order = ids.argsort(stable=True)
    ordered_ids, ordered_labels = ids[order], labels[order]
    one_hot = torch.nn.functional.one_hot(ordered_labels, environments)
    prefix = torch.cat((torch.zeros(1, environments, dtype=torch.long), one_hot.cumsum(0)))
    left = torch.searchsorted(ordered_ids, ordered_ids - radius)
    right = torch.searchsorted(ordered_ids, ordered_ids + radius, right=True)
    votes = prefix[right] - prefix[left]
    winners = votes.argmax(1)
    tied_current = votes.gather(1, ordered_labels[:, None]).squeeze(1) == votes.amax(1)
    result = torch.empty_like(labels)
    result[order] = torch.where(tied_current, ordered_labels, winners)
    return result


def _environment_scores(losses: Tensor) -> Tensor:
    if (not isinstance(losses, Tensor) or losses.ndim != 2 or min(losses.shape) < 1
            or not losses.is_floating_point() or not torch.isfinite(losses).all()
            or (losses < 0).any()):
        raise ValueError("losses must be finite nonnegative [N, E]")
    return losses.detach().to(device="cpu")


def assign_environments(losses: Tensor, previous_labels: Tensor) -> Tensor:
    """IRN argmin；精确平票保留旧标签，否则用最小环境编号。"""
    losses = _environment_scores(losses)
    previous = _labels(previous_labels, *losses.shape, nonempty=False)
    minima, chosen = losses.min(1)
    retain = losses.gather(1, previous[:, None]).squeeze(1) == minima
    return torch.where(retain, previous, chosen)


def repair_empty_environments(labels: Tensor, losses: Tensor,
                              sample_ids: Tensor | None = None) -> Tensor:
    """从至少有两个样本的环境迁移当前风险最高者；平票按全局 id。"""
    losses = _environment_scores(losses)
    n, environments = losses.shape
    if n < environments:
        raise ValueError("n_samples must be at least n_environments")
    result = _labels(labels, n, environments, nonempty=False).clone()
    ids = _sample_ids(sample_ids, n)
    order = ids.argsort(stable=True)
    counts = torch.bincount(result, minlength=environments)
    for destination in (counts == 0).nonzero().flatten().tolist():
        candidates = order[counts[result[order]] > 1]
        difficulty = losses[candidates, result[candidates]]
        selected = candidates[difficulty.argmax()]
        counts[result[selected]] -= 1
        result[selected] = destination
        counts[destination] += 1
    return result


def mil_objective(prediction: Tensor, target: Tensor, labels: Tensor, *,
                  n_environments: int, lambda_erm: float, lambda_env: float,
                  std_floor: float = 1e-5, horizon_one: str = "error") -> MilLosses:
    """式 (10)：环境等权 IRN 均值、全样本 ERM、环境总体风险方差。"""
    _finite_number(lambda_erm, "lambda_erm")
    _finite_number(lambda_env, "lambda_env")
    row_losses = irn_loss(prediction, target, std_floor=std_floor,
                          horizon_one=horizon_one, reduction="none")
    if row_losses.ndim != 1:
        raise ValueError("MIL expects [N, H, T]")
    labels = _labels(labels, prediction.shape[0], n_environments).to(prediction.device)
    risks = torch.stack([row_losses[labels == e].mean() for e in range(n_environments)])
    irn_mean = risks.mean()
    erm = (prediction - target).square().mean()
    variance = risks.var(correction=0)
    return MilLosses(irn_mean + lambda_erm * erm + lambda_env * variance,
                     irn_mean, erm, variance, risks)


def set_foil_phase(branch: nn.Module, environment_heads: nn.ModuleList | None, phase: str) -> None:
    """显式切换梯度所有权并清除旧梯度；eval 模式不阻止参数学习。"""
    if phase not in ("mil", "mei", "frozen"):
        raise ValueError("phase must be mil, mei or frozen")
    if phase == "mei" and environment_heads is None:
        raise ValueError("MEI requires environment_heads")
    branch.eval()
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if phase == "mil":
        for module in (branch.encoder, branch.head):
            module.requires_grad_(True)
    if environment_heads is not None:
        environment_heads.eval()
        environment_heads.requires_grad_(phase == "mei")
        for parameter in environment_heads.parameters():
            parameter.grad = None


def _placement(module: nn.Module):
    parameters = list(module.parameters())
    if not parameters:
        raise ValueError("trainable module must have parameters")
    device, dtype = parameters[0].device, parameters[0].dtype
    if any(p.device != device or p.dtype != dtype for p in parameters):
        raise ValueError("each branch must use one device and floating-point dtype")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("FOIL training requires float32 or float64 parameters")
    return device, dtype


def _batch_indices(n, config, generator, *, shuffle=None):
    shuffle = config.shuffle if shuffle is None else shuffle
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    yield from order.split(config.batch_size)


def _take(values: Tensor, indices: Tensor, device, dtype):
    return values.detach().index_select(0, indices.to(values.device)).to(device=device, dtype=dtype)


def _prediction(branch, x):
    return branch.head(branch.encoder(x)).reshape(x.shape[0], branch.config.horizon, branch.config.targets)


def _irn(prediction, target, config):
    return irn_loss(prediction, target, std_floor=config.std_floor,
                    horizon_one=config.horizon_one, reduction="none")


def _optimizer_parameters(optimizer, desired):
    desired = list(desired)
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in desired}:
        raise ValueError("optimizer parameters must exactly match the current phase")
    return desired


def _finish_step(optimizer, parameters, config):
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("nonfinite FOIL gradient; no optimizer step applied")
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()


def mil_step(branch: nn.Module, optimizer: torch.optim.Optimizer,
             train_x: Tensor, train_y: Tensor, labels: Tensor, config: FoilConfig, *,
             lambda_env: float, generator: torch.Generator | None = None) -> dict:
    """分两遍 batch 累计一次精确式 (10) 梯度；返回更新前的三项风险。"""
    _finite_number(lambda_env, "lambda_env")
    labels = _labels(labels, len(train_x), config.n_environments)
    counts = torch.bincount(labels, minlength=config.n_environments).double()
    set_foil_phase(branch, None, "mil")
    parameters = _optimizer_parameters(optimizer, (p for p in branch.parameters() if p.requires_grad))
    device, dtype = _placement(branch)
    risk_sums = torch.zeros(config.n_environments, dtype=torch.float64)
    mse_sum = 0.0
    with torch.no_grad():
        for indices in _batch_indices(len(train_x), config, generator, shuffle=False):
            x, y = (_take(values, indices, device, dtype) for values in (train_x, train_y))
            prediction = _prediction(branch, x)
            risk_sums.scatter_add_(0, labels[indices], _irn(prediction, y, config).double().cpu())
            mse_sum += float((prediction - y).square().mean((-2, -1)).double().sum())
    risks = risk_sums / counts
    mean, variance = risks.mean(), risks.var(correction=0)
    erm = mse_sum / len(train_x)
    total = mean + config.lambda_erm * erm + lambda_env * variance
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite MIL objective")
    # dL/d(loss_i) = [1/E + 2 lambda_env (R_e - mean(R))/E] / n_e.
    coefficients = (1 + 2 * lambda_env * (risks - mean)) / (config.n_environments * counts)
    optimizer.zero_grad(set_to_none=True)
    for indices in _batch_indices(len(train_x), config, generator):
        x, y = (_take(values, indices, device, dtype) for values in (train_x, train_y))
        prediction = _prediction(branch, x)
        irn = _irn(prediction, y, config)
        mse = (prediction - y).square().mean((-2, -1))
        weights = coefficients[labels[indices]].to(device=device, dtype=irn.dtype)
        surrogate = (weights * irn).sum() + config.lambda_erm * mse.sum() / len(train_x)
        surrogate.backward()
    _finish_step(optimizer, parameters, config)
    return {"total": float(total), "irn_mean": float(mean), "erm": erm,
            "environment_variance": float(variance), "environment_risks": risks.tolist(),
            "environment_counts": counts.long().tolist(), "loss_evaluation": "pre_update"}


def mei_step(branch: nn.Module, environment_heads: nn.ModuleList, optimizer: torch.optim.Optimizer,
             train_x: Tensor, train_y: Tensor, labels: Tensor, config: FoilConfig, *,
             generator: torch.Generator | None = None) -> dict:
    """固定编码器/共享头，只给环境回归头累计环境等权 IRN 梯度。"""
    if len(environment_heads) != config.n_environments:
        raise ValueError("wrong number of environment heads")
    labels = _labels(labels, len(train_x), config.n_environments)
    counts = torch.bincount(labels, minlength=config.n_environments)
    set_foil_phase(branch, environment_heads, "mei")
    parameters = _optimizer_parameters(optimizer, environment_heads.parameters())
    device, dtype = _placement(branch)
    risk_sums = torch.zeros(config.n_environments, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    for indices in _batch_indices(len(train_x), config, generator):
        x, y = (_take(values, indices, device, dtype) for values in (train_x, train_y))
        with torch.no_grad():
            features = branch.encoder(x)
        batch_labels = labels[indices].to(device)
        for e, head in enumerate(environment_heads):
            selected = batch_labels == e
            if not selected.any():
                continue
            prediction = head(features[selected]).reshape(-1, branch.config.horizon, branch.config.targets)
            row_risks = _irn(prediction, y[selected], config)
            risk_sums[e] += float(row_risks.detach().double().sum())
            (row_risks.sum() / (config.n_environments * int(counts[e]))).backward()
    _finish_step(optimizer, parameters, config)
    risks = risk_sums / counts
    return {"irn_mean": float(risks.mean()), "environment_risks": risks.tolist(),
            "environment_counts": counts.tolist(), "loss_evaluation": "pre_update"}


def infer_environments(branch: nn.Module, environment_heads: nn.ModuleList,
                       train_x: Tensor, train_y: Tensor, labels: Tensor, config: FoilConfig, *,
                       generator: torch.Generator | None = None,
                       sample_ids: Tensor | None = None) -> EnvironmentAssignment:
    """风险按输入行写回；平滑及修复按对应的原始时间 id，返回顺序不变。"""
    if len(environment_heads) != config.n_environments:
        raise ValueError("wrong number of environment heads")
    labels = _labels(labels, len(train_x), config.n_environments)
    ids = _sample_ids(sample_ids, len(train_x))
    set_foil_phase(branch, environment_heads, "frozen")
    device, dtype = _placement(branch)
    losses = torch.empty(len(train_x), config.n_environments, dtype=torch.float64)
    with torch.no_grad():
        for indices in _batch_indices(len(train_x), config, generator):
            x, y = (_take(values, indices, device, dtype) for values in (train_x, train_y))
            features = branch.encoder(x)
            for e, head in enumerate(environment_heads):
                prediction = head(features).reshape(-1, branch.config.horizon, branch.config.targets)
                losses[indices, e] = _irn(prediction, y, config).double().cpu()
    reassigned = assign_environments(losses, labels)
    smoothed = smooth_environment_labels(reassigned, ids, radius=config.smoothing_radius,
                                         n_environments=config.n_environments)
    repaired = repair_empty_environments(smoothed, losses, ids)
    return EnvironmentAssignment(repaired, losses,
                                 torch.bincount(repaired, minlength=config.n_environments).tolist(),
                                 int((reassigned != labels).sum()), int((smoothed != reassigned).sum()),
                                 int((repaired != smoothed).sum()))


def _validate_data(branch, x, y, name, config):
    expected = (branch.config.horizon, branch.config.targets)
    if (not isinstance(x, Tensor) or not isinstance(y, Tensor) or x.ndim < 2
            or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[0] < 1
            or y.shape[1:] != expected):
        raise ValueError(f"{name} must have matching inputs and targets [N, H, T]")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError(f"{name} inputs and targets must be floating point")
    for values in (x, y):
        for batch in values.split(config.batch_size):
            if not torch.isfinite(batch).all():
                raise ValueError(f"{name} contains nonfinite values")
    if expected[0] == 1 and config.horizon_one == "error" and config.mode == "foil":
        raise ValueError("H=1 requires the explicit zero_irn policy")


def _validation_mse(branch, x, y, config):
    branch.eval()
    device, dtype = _placement(branch)
    total = 0.0
    with torch.no_grad():
        for indices in _batch_indices(len(x), config, None, shuffle=False):
            target = _take(y, indices, device, dtype)
            prediction = _prediction(branch, _take(x, indices, device, dtype))
            total += float((prediction.double() - target.double()).square().sum())
    result = total / y.numel()
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite validation MSE")
    return result


def _checkpoint(branch):
    return {key: value.detach().cpu().clone() for key, value in branch.state_dict().items()}


def _new_environment_heads(branch, count):
    heads = nn.ModuleList([deepcopy(branch.head) for _ in range(count)])
    for head in heads:
        for layer in head.modules():
            reset = getattr(layer, "reset_parameters", None)
            if callable(reset):
                reset()
    return heads


@contextmanager
def _random_context(branches, seed, deterministic):
    devices = sorted({p.device.index for p in branches.parameters() if p.is_cuda})
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(deterministic)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic


def _train_branch(branch, train_x, train_y, validation_x, validation_y, config, lambda_env, seed,
                  sample_ids):
    labels = initialize_environments(len(train_x), config.n_environments, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    heads = _new_environment_heads(branch, config.n_environments)
    mil_optimizer = torch.optim.Adam(list(branch.encoder.parameters()) + list(branch.head.parameters()),
                                     lr=config.mil_learning_rate)
    mei_optimizer = torch.optim.Adam(heads.parameters(), lr=config.mei_learning_rate)
    baseline = _validation_mse(branch, validation_x, validation_y, config)
    best_mse, best_step, best_state = baseline, 0, _checkpoint(branch)
    records, round_records, step = [], [], 0
    initial_counts = torch.bincount(labels, minlength=config.n_environments).tolist()
    for round_index in range(config.rounds):
        for epoch in range(config.mil_epochs):
            set_foil_phase(branch, heads, "mil")
            metrics = mil_step(branch, mil_optimizer, train_x, train_y, labels, config,
                               lambda_env=lambda_env, generator=generator)
            step += 1
            validation_mse = _validation_mse(branch, validation_x, validation_y, config)
            records.append({"round": round_index + 1, "epoch": epoch + 1, "step": step,
                            **metrics, "validation_mse": validation_mse,
                            "trainable": ["encoder", "shared_head"]})
            if validation_mse < best_mse:
                best_mse, best_step, best_state = validation_mse, step, _checkpoint(branch)
        mei_records = []
        for epoch in range(config.mei_epochs):
            metrics = mei_step(branch, heads, mei_optimizer, train_x, train_y, labels, config,
                               generator=generator)
            mei_records.append({"epoch": epoch + 1, **metrics, "trainable": ["environment_heads"]})
        assignment = infer_environments(branch, heads, train_x, train_y, labels, config,
                                         generator=generator, sample_ids=sample_ids)
        labels = assignment.labels
        round_records.append({"round": round_index + 1, "mei": mei_records,
                              "environment_counts": assignment.counts, "argmin_changes": assignment.reassigned,
                              "smoothing_changes": assignment.smoothed, "empty_environment_repairs": assignment.repaired})
    branch.load_state_dict(best_state)
    set_foil_phase(branch, heads, "frozen")
    return {"seed": seed, "lambda_env": lambda_env, "initial_validation_mse": baseline,
            "best_validation_mse": best_mse, "best_step": best_step,
            "restored_validation_mse": _validation_mse(branch, validation_x, validation_y, config),
            "initial_environment_counts": initial_counts, "mil": records, "rounds": round_records,
            "final_environment_labels": labels.tolist(), "final_environment_counts": assignment.counts,
            "environment_labels_describe": "last training partition; not the restored checkpoint"}


def _contrast_diagnostics(branches, x, config):
    sum_square, elements, zeros, zero_vectors = 0.0, 0, 0, 0
    with torch.no_grad():
        for indices in _batch_indices(len(x), config, None, shuffle=False):
            predictions = []
            for branch in (branches.invariant, branches.variant):
                if branch is not None:
                    device, dtype = _placement(branch)
                    predictions.append(_prediction(branch, _take(x, indices, device, dtype)).double().cpu())
            if getattr(branches, "contrast", "difference") == "variant":
                contrast = predictions[-1]
            elif getattr(branches, "contrast", "difference") == "invariant":
                contrast = predictions[0]
            else:
                contrast = predictions[-1] - predictions[0] if len(predictions) == 2 else predictions[0]
            sum_square += float(contrast.square().sum())
            elements += contrast.numel()
            zeros += int((contrast == 0).sum())
            zero_vectors += int((contrast == 0).flatten(1).all(1).sum())
    return {"split": "validation", "rms": math.sqrt(sum_square / elements),
            "zero_fraction": zeros / elements, "zero_vector_fraction": zero_vectors / len(x),
            "zero_definition": "exact equality", "separation_guaranteed": False}


def _train_erm_branch(branch, train_x, train_y, validation_x, validation_y, config, seed):
    """Capacity-controlled ERM: one full-data optimizer step per MIL epoch."""
    branch.requires_grad_(True)
    device, dtype = _placement(branch)
    optimizer = torch.optim.Adam(branch.parameters(), lr=config.mil_learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best = _validation_mse(branch, validation_x, validation_y, config)
    best_state, best_step = _checkpoint(branch), 0
    records = []
    for epoch in range(config.rounds * config.mil_epochs):
        branch.eval()
        optimizer.zero_grad(set_to_none=True)
        squared = 0.
        for indices in _batch_indices(len(train_x), config, generator):
            y = _take(train_y, indices, device, dtype)
            prediction = _prediction(branch, _take(train_x, indices, device, dtype))
            loss = (prediction - y).square().sum() / train_y.numel()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite branch ERM loss")
            loss.backward()
            squared += float(loss.detach())
        _finish_step(optimizer, list(branch.parameters()), config)
        score = _validation_mse(branch, validation_x, validation_y, config)
        records.append({"step": epoch + 1, "mse": squared, "validation_mse": score})
        if score < best:
            best, best_step, best_state = score, epoch + 1, _checkpoint(branch)
    branch.load_state_dict(best_state)
    return {"objective": "mean_squared_error", "seed": seed, "history": records,
            "best_validation_mse": best, "best_step": best_step,
            "environment_inference": False, "optimizer_step_unit": "full training set per epoch"}


def _train_ablation_branches(branches, train_x, train_y, validation_x, validation_y, config):
    history = {"method": config.mode, "config": asdict(config)}
    if config.mode == "disabled":
        if any(True for _ in branches.parameters()):
            raise ValueError("disabled invariant module must contain no parameters")
        return {**history, "status": "skipped", "reason": "invariant_and_variant_branches_removed"}
    for offset, name in enumerate(("invariant", "variant")):
        branch = getattr(branches, name)
        if branch is None:
            continue
        _validate_data(branch, train_x, train_y, "train", config)
        _validate_data(branch, validation_x, validation_y, "validation", config)
        seed = config.seed + offset * 1009
        with _random_context(branches, seed, config.deterministic):
            history[name] = _train_erm_branch(branch, train_x, train_y, validation_x, validation_y, config, seed)
    branches.freeze()
    history["contrast"] = _contrast_diagnostics(branches, validation_x, config)
    return history


def train_foil_branches(branches: nn.Module, train_x: Tensor, train_y: Tensor,
                        validation_x: Tensor, validation_y: Tensor, config: FoilConfig, *,
                        sample_ids: Tensor | None = None) -> dict:
    """独立交替训练双分支，按验证 MSE 恢复最佳参数并冻结；返回 JSON 日志。

    复现要求调用方也固定初始模型参数。H=1 的 zero_irn 仅是显式 ERM
    退化模式。严格确定性 CUDA 可能要求启动前设置 CUBLAS_WORKSPACE_CONFIG。
    lambda_variant < lambda_invariant 是设计假设，日志只测量预测差。
    sample_ids 必须逐行对应训练窗口的原始观测索引；平滑半径以索引差计，
    不填补已剔除的窗口。None 保持旧调用的连续行号语义并在日志中标明。
    """
    if not isinstance(config, FoilConfig):
        raise TypeError("config must be FoilConfig")
    if getattr(branches, "mode", "foil") != config.mode:
        raise ValueError("branch architecture does not match the requested invariant-module ablation")
    if getattr(branches, "contrast", "difference") != config.contrast:
        raise ValueError("branch contrast differs from the requested ablation")
    ids = _sample_ids(sample_ids, len(train_x)).clone()
    sample_metadata = {
        "training_sample_ids": ids.tolist(),
        "sample_id_source": "caller_supplied" if sample_ids is not None else "contiguous_row_fallback",
        "smoothing_radius_unit": "sample_id_difference",
        "environment_label_order": "training_sample_ids row order",
    }
    if config.mode != "foil":
        return {**_train_ablation_branches(branches, train_x, train_y, validation_x, validation_y, config),
                **sample_metadata}
    pair = (branches.invariant, branches.variant)
    if {id(p) for p in pair[0].parameters()} & {id(p) for p in pair[1].parameters()}:
        raise ValueError("FOIL branches must not share parameters")
    for branch in pair:
        if {id(p) for p in branch.encoder.parameters()} & {id(p) for p in branch.head.parameters()}:
            raise ValueError("encoder and shared head must not share parameters")
        _placement(branch)
        _validate_data(branch, train_x, train_y, "train", config)
        _validate_data(branch, validation_x, validation_y, "validation", config)
    if len(train_x) < config.n_environments:
        raise ValueError("n_samples must be at least n_environments")
    history = {"method": "two independently trained FOIL branches (manuscript extension)",
               "source": "https://arxiv.org/html/2406.09130v1",
               **sample_metadata,
               "config": asdict(config), "optimizer_step_unit": "full training set per epoch",
               "normalization_axis": "horizon separately for each sample and target",
               "horizon_one_degenerate": train_y.shape[1] == 1,
               "initial_model_parameters": "supplied by caller; not reinitialized",
               "lambda_variant_lt_invariant": config.lambda_variant < config.lambda_invariant}
    for offset, (name, branch, weight) in enumerate(zip(
            ("invariant", "variant"), pair, (config.lambda_invariant, config.lambda_variant))):
        seed = config.seed + offset * 1009
        with _random_context(branches, seed, config.deterministic):
            history[name] = _train_branch(branch, train_x, train_y, validation_x, validation_y,
                                          config, weight, seed, ids)
    branches.eval()
    branches.requires_grad_(False)
    history["contrast"] = _contrast_diagnostics(branches, validation_x, config)
    return history
