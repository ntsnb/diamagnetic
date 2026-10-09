"""Source-audited online baselines with an explicit delayed-label boundary.

The algorithms in this module are ports of the cited authors' implementations,
not the legacy proxy implementations in ``train/``.  See ONLINE_BASELINES.md
for commit identifiers, source/code differences, and the common-stream contract.

``forecast(x, time)`` receives observed inputs only. ``update(ReleasedBatch,
time=...)`` is the *only* entrance for online targets. SOLID can additionally
adapt to a current input using its already released memory. All persistent
algorithm state, optimizers, buffers and local random counters are checkpointed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OnlineBaselineConfig:
    """Common protocol and explicitly recorded method hyperparameters."""

    lookback: int
    horizon: int
    target_indices: tuple[int, ...]
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    update_steps: int = 1
    extra_delay: int = 0
    seed: int = 2026
    max_grad_norm: float | None = None
    memory_size: int = 500
    selected_data_num: int = 5
    period: float = 1.0
    lambda_period: float = 0.1
    # Original SOLID sums sample-wise gradients; OnlineTSF's port averages.
    solid_gradient_reduction: str = "sum"
    # Original SOLID adapts in eval mode; OnlineTSF uses train mode with mean.
    solid_adaptation_mode: str = "eval"
    learning_rate_w: float = 1e-3
    learning_rate_bias: float = 1e-3
    decision_dropout: float = 0.1
    concept_dim: int = 200
    bottleneck_dim: int = 32
    clip_drift: bool = True
    ema: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "target_indices", tuple(self.target_indices))
        for name in ("lookback", "horizon", "update_steps", "memory_size",
                     "selected_data_num", "concept_dim", "bottleneck_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.target_indices or len(set(self.target_indices)) != len(self.target_indices):
            raise ValueError("target_indices must be a nonempty set of ordered channel indices")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.target_indices):
            raise ValueError("target_indices must contain nonnegative integers")
        if self.extra_delay < 0 or int(self.extra_delay) != self.extra_delay:
            raise ValueError("extra_delay must be a nonnegative integer")
        for name in ("learning_rate", "learning_rate_w", "learning_rate_bias", "period"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.optimizer not in ("Adam", "AdamW", "SGD"):
            raise ValueError("optimizer must be Adam, AdamW, or SGD")
        if not 0 <= self.lambda_period <= 1 or not 0 <= self.ema < 1:
            raise ValueError("lambda_period must be in [0,1] and ema in [0,1)")
        if not 0 <= self.decision_dropout < 1:
            raise ValueError("decision_dropout must be in [0,1)")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.solid_gradient_reduction not in ("sum", "mean"):
            raise ValueError("solid_gradient_reduction must be sum or mean")
        if self.solid_adaptation_mode not in ("train", "eval"):
            raise ValueError("solid_adaptation_mode must be train or eval")


@dataclass(frozen=True)
class ReleasedBatch:
    """Fully observed historical samples, with their actual release times.

    X is [B,L,N], Y is [B,H,T]. ``prediction`` is optional immutable issued
    prediction evidence, not a substitute for rerunning X during gradient GD.
    Times use the parent's common integer observation clock.
    """

    x: Tensor
    y: Tensor
    issue_times: Sequence[int] | Tensor
    available_at: Sequence[int] | Tensor | int
    prediction: Tensor | None = None


@dataclass(frozen=True)
class OnlineUpdate:
    method: str
    issue_times: tuple[int, ...] = ()
    steps: int = 0
    loss: float | None = None
    selected_issue_times: tuple[int, ...] = ()

    def to_dict(self):
        return asdict(self)


def _prediction(output):
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "prediction"):
        return output.prediction
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("forecaster must return a Tensor or an object with .prediction")


def _cpu_clone(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def _times(value, size, name):
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value] * size
    if not isinstance(value, (list, tuple)):
        value = list(value)
    if len(value) != size or any(isinstance(t, bool) or int(t) != t for t in value):
        raise ValueError(f"{name} must contain {size} integer times")
    return tuple(int(t) for t in value)


class OnlineBaseline:
    """Stateful controller; clones the supplied pretrained model on construction."""

    name = "base"
    checkpoint_version = 1
    online_updates_enabled = True

    def __init__(self, model: nn.Module, config: OnlineBaselineConfig):
        self.config = config
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(True)
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self._last_release_issue: int | None = None
        self._last_time: int | None = None
        self._rng_step = 0
        self._shuffle = torch.Generator(device="cpu").manual_seed(config.seed)
        self.total_update_steps = 0
        self.total_released_samples = 0
        self.last_update = OnlineUpdate(self.name)
        self._initial_state = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _optimizer(self, parameters: Iterable[nn.Parameter], *, kind=None, lr=None):
        params = [p for p in parameters if p.requires_grad]
        if not params:
            raise ValueError("method has no trainable parameters")
        return getattr(torch.optim, kind or self.config.optimizer)(
            params, lr=self.config.learning_rate if lr is None else lr)

    @contextmanager
    def _random_context(self):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            event_seed = self.config.seed + self._rng_step
            numpy_state = np.random.get_state()
            torch.manual_seed(event_seed)
            np.random.seed(event_seed % (2 ** 32))
            self._rng_step += 1
            try:
                yield
            finally:
                np.random.set_state(numpy_state)

    def _validate_x(self, x):
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[1] != self.config.lookback:
            raise ValueError("x must have shape [B,lookback,N]")
        if x.shape[0] < 1 or max(self.config.target_indices) >= x.shape[-1]:
            raise ValueError("empty batch or target channel index outside x")
        if not torch.isfinite(x).all():
            raise ValueError("x contains nonfinite values")

    def _validate_time(self, time):
        if isinstance(time, bool) or int(time) != time:
            raise ValueError("time must be an integer observation index")
        if self._last_time is not None and time < self._last_time:
            raise ValueError("online time may not move backwards")

    def _validate_released(self, batch: ReleasedBatch, time, *, new=True):
        self._validate_time(time)
        self._validate_x(batch.x)
        n = batch.x.shape[0]
        if tuple(batch.y.shape) != (n, self.config.horizon, len(self.config.target_indices)):
            raise ValueError("released y must have shape [B,horizon,target_channels]")
        if not torch.isfinite(batch.y).all():
            raise ValueError("released y contains nonfinite values")
        issues = _times(batch.issue_times, n, "issue_times")
        available = _times(batch.available_at, n, "available_at")
        if any(b <= a for a, b in zip(issues, issues[1:])):
            raise ValueError("released issue_times must be strictly increasing")
        if new and self._last_release_issue is not None and issues[0] <= self._last_release_issue:
            raise ValueError("duplicate or out-of-order label release")
        for issue, at in zip(issues, available):
            if at < issue + self.config.horizon + self.config.extra_delay or at > time:
                raise ValueError("future/unreleased target: horizon or feedback delay has not elapsed")
        if batch.prediction is not None and tuple(batch.prediction.shape) != tuple(batch.y.shape):
            raise ValueError("issued prediction shape does not match released y")
        return issues, available

    def _forward(self, x):
        pred = _prediction(self.model(x))
        expected = (x.shape[0], self.config.horizon, len(self.config.target_indices))
        if tuple(pred.shape) != expected:
            raise ValueError(f"forecaster output {tuple(pred.shape)} does not match {expected}")
        return pred

    def forecast(self, x: Tensor, time: int) -> Tensor:
        self._validate_x(x)
        self._validate_time(time)
        self.model.eval()
        with self._random_context(), torch.no_grad():
            result = self._forward(x.to(self.device)).detach().clone()
        self._last_time = int(time)
        return result

    def predict(self, x: Tensor, time: int) -> Tensor:
        return self.forecast(x, time)

    def _step(self, loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("online loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("online gradient is nonfinite")
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
        optimizer.step()
        # FSNet stores the gradients after parameter update, before clearing them.
        for model in self._gradient_models():
            if hasattr(model, "store_grad"):
                model.store_grad()
        optimizer.zero_grad(set_to_none=True)

    def _gradient_models(self):
        return (self.model,)

    def update(self, released_batch: ReleasedBatch, *, time: int,
               current_context: Tensor | None = None) -> OnlineUpdate:
        issues, _ = self._validate_released(released_batch, time)
        losses = []
        for i in range(len(issues)):
            x = released_batch.x[i:i + 1].detach().to(self.device)
            y = released_batch.y[i:i + 1].detach().to(self.device)
            for _ in range(self.config.update_steps):
                with self._random_context():
                    self.model.train()
                    loss = F.mse_loss(self._forward(x), y)
                    self._step(loss, self.optimizers["model"])
                losses.append(float(loss.detach()))
        self._last_release_issue = issues[-1]
        self._last_time = int(time)
        self.total_update_steps += len(losses)
        self.total_released_samples += len(issues)
        self.last_update = OnlineUpdate(self.name, issues, len(losses), sum(losses) / len(losses))
        return self.last_update

    def observe(self, released_batch: ReleasedBatch, *, time: int,
                current_context: Tensor | None = None):
        return self.update(released_batch, time=time, current_context=current_context)

    release = observe

    def seed_history(self, released_batch: ReleasedBatch, *, time: int):
        """Validate an observed pre-online reference without doing any gradient step."""
        issues, _ = self._validate_released(released_batch, time)
        self._last_release_issue = issues[-1]
        self._last_time = int(time)

    def _extra_state(self):
        return {}

    def _load_extra_state(self, state):
        if state:
            raise ValueError("unexpected algorithm state")

    def state_dict(self):
        return _cpu_clone({
            "version": self.checkpoint_version, "method": self.name,
            "config": asdict(self.config), "model": self.model.state_dict(),
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
            "last_release_issue": self._last_release_issue, "last_time": self._last_time,
            "rng_step": self._rng_step, "shuffle_state": self._shuffle.get_state(),
            "total_update_steps": self.total_update_steps,
            "total_released_samples": self.total_released_samples,
            "last_update": asdict(self.last_update), "extra": self._extra_state(),
        })

    def load_state_dict(self, state: Mapping):
        if state["version"] != self.checkpoint_version or state["method"] != self.name:
            raise ValueError("incompatible online method checkpoint")
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint protocol/hyperparameters differ from current config")
        self.model.load_state_dict(state["model"])
        self._load_extra_state(state["extra"])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
        self._last_release_issue, self._last_time = state["last_release_issue"], state["last_time"]
        self._rng_step = state["rng_step"]
        self._shuffle.set_state(state["shuffle_state"])
        self.total_update_steps = state["total_update_steps"]
        self.total_released_samples = state["total_released_samples"]
        self.last_update = OnlineUpdate(**state["last_update"])

    def set_reset_point(self):
        """Define reset as the fully pretrained, optional history-seeded state."""
        self._initial_state = self.state_dict()

    def reset(self):
        if self._initial_state is None:
            raise RuntimeError("no reset point")
        self.load_state_dict(self._initial_state)

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "reset_state": self._initial_state}, path)

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["state"])
        self._initial_state = checkpoint["reset_state"]


class Frozen(OnlineBaseline):
    """One pretrained forecaster with no test-time parameter or buffer updates.

    Released labels are validated and counted only for the common stream audit.
    The model stays in evaluation mode, including BatchNorm and dropout, and no
    optimizer is constructed. This is separate from a frozen mixture of experts.
    """

    name = "Frozen"
    online_updates_enabled = False

    def __init__(self, model, config):
        super().__init__(model, config)
        self.model.requires_grad_(False)
        self.model.eval()
        self.set_reset_point()

    def update(self, released_batch, *, time, current_context=None):
        issues, _ = self._validate_released(released_batch, time)
        # Never run a training forward pass, even when feedback is available.
        self.model.eval()
        self._last_release_issue = issues[-1]
        self._last_time = int(time)
        self.total_released_samples += len(issues)
        self.last_update = OnlineUpdate(self.name, issues)
        return self.last_update


class GradientDescent(OnlineBaseline):
    """GD: persistent full-model gradients on each fully released sample."""

    name = "GD"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.optimizers["model"] = self._optimizer(self.model.parameters())
        self.set_reset_point()


class HeadGradientDescent(OnlineBaseline):
    """Head-GD: persistent supervised update of the actual prediction head."""

    name = "Head-GD"

    def __init__(self, model, config):
        super().__init__(model, config)
        if not hasattr(self.model, "head_parameters"):
            raise TypeError("Head-GD needs forecaster.head_parameters()")
        self.model.requires_grad_(False)
        params = list(self.model.head_parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        self.optimizers["model"] = self._optimizer(params)
        self.set_reset_point()

    def _forward(self, x):
        # Frozen representations must not mutate BatchNorm state through training.
        self.model.eval()
        return super()._forward(x)


class SOLID(OnlineBaseline):
    """Periodic, nearest-context sample selection followed by head-only probing.

    Official SOLID starts from pretrained parameters for every test input. The
    original implementation sums gradients of selected examples before one SGD
    step in eval mode. ``solid_gradient_reduction='mean'`` together with
    ``solid_adaptation_mode='train'`` exposes the OnlineTSF adaptation-mode and
    reduction variant. All episodic modes restore the complete pretrained state.
    """

    name = "SOLID"
    continual = False

    def __init__(self, model, config):
        super().__init__(model, config)
        self.memory = []
        self._prepared_time = None
        self._prepared_x = None
        self._pretrained = _cpu_clone(self.model.state_dict())
        if self.continual:
            params = list(self.model.parameters())
            self.optimizers["model"] = self._optimizer(params)
        else:
            if not hasattr(self.model, "head_parameters"):
                raise TypeError("SOLID needs forecaster.head_parameters()")
            self.model.requires_grad_(False)
            params = list(self.model.head_parameters())
            for parameter in params:
                parameter.requires_grad_(True)
            self.optimizers["model"] = self._optimizer(params, kind="SGD")
        self.set_reset_point()

    @property
    def adaptation_mode(self):
        return "train" if self.continual else self.config.solid_adaptation_mode

    @property
    def gradient_reduction(self):
        return "mean" if self.continual else self.config.solid_gradient_reduction

    @property
    def source_variant(self):
        if self.continual:
            return "onlinetsf_continual_train_mean"
        if self.adaptation_mode == "eval" and self.gradient_reduction == "sum":
            return "solid_original_eval_sum"
        if self.adaptation_mode == "train" and self.gradient_reduction == "mean":
            return "onlinetsf_train_mean_full_state_reset"
        return "explicit_solid_mode_reduction_variant"

    def _append(self, batch, issues, available):
        for i, (issue, at) in enumerate(zip(issues, available)):
            self.memory.append({"x": _cpu_clone(batch.x[i:i + 1]),
                                "y": _cpu_clone(batch.y[i:i + 1]),
                                "issue_time": issue, "available_at": at})
        self.memory = self.memory[-self.config.memory_size:]
        self._prepared_time, self._prepared_x = None, None

    def seed_history(self, released_batch, *, time):
        issues, available = self._validate_released(released_batch, time)
        self._append(released_batch, issues, available)
        self._last_release_issue, self._last_time = issues[-1], int(time)

    def update(self, released_batch, *, time, current_context=None):
        issues, available = self._validate_released(released_batch, time)
        self._append(released_batch, issues, available)
        self.total_released_samples += len(issues)
        self._last_release_issue, self._last_time = issues[-1], int(time)
        self.last_update = OnlineUpdate(self.name, issues)
        if current_context is not None:
            self.prepare_context(current_context, time)
            self.last_update = OnlineUpdate(self.name, issues, self.last_update.steps,
                                            self.last_update.loss, self.last_update.selected_issue_times)
        return self.last_update

    def prepare_context(self, x, time):
        self._validate_time(time)
        self._validate_x(x)
        if x.shape[0] != 1:
            raise ValueError("SOLID's context adaptation requires one online input at a time")
        if self._prepared_time == time:
            if not torch.equal(self._prepared_x, x.detach().cpu()):
                raise ValueError("different current inputs share one online timestamp")
            return self.last_update
        if not self.continual:
            self.model.load_state_dict(self._pretrained)
            self.optimizers["model"].state.clear()
        # Exact one-sided modular-period test in both author repositories.
        candidates = [r for r in self.memory if r["available_at"] <= time and
                      time - r["issue_time"] <= self.config.horizon + self.config.extra_delay + self.config.memory_size - 1 and
                      (time - r["issue_time"]) % self.config.period <= self.config.period * self.config.lambda_period]
        current = x.detach().cpu().reshape(-1)
        candidates.sort(key=lambda r: (float(F.pairwise_distance(current,
                          r["x"].reshape(-1), p=2)), r["issue_time"]))
        selected = candidates[:self.config.selected_data_num]
        losses = []
        if selected:
            train_x = torch.cat([r["x"] for r in selected]).to(self.device)
            train_y = torch.cat([r["y"] for r in selected]).to(self.device)
            for _ in range(self.config.update_steps):
                with self._random_context():
                    # Gradients remain restricted to the selected parameter
                    # scope. Train mode additionally activates dropout/BN; the
                    # full episodic snapshot resets these buffers next input.
                    self.model.train(self.adaptation_mode == "train")
                    per_example = (self._forward(train_x) - train_y).square().mean((1, 2))
                    # SOLID++ follows the OnlineTSF full-model mean-loss update.
                    loss = per_example.mean() if self.gradient_reduction == "mean" else per_example.sum()
                    self._step(loss, self.optimizers["model"])
                losses.append(float(per_example.mean().detach()))
        self.total_update_steps += len(losses)
        self._prepared_time, self._prepared_x = int(time), _cpu_clone(x)
        self.last_update = OnlineUpdate(self.name, (), len(losses),
                                       sum(losses) / len(losses) if losses else None,
                                       tuple(r["issue_time"] for r in selected))
        return self.last_update

    def forecast(self, x, time):
        self.prepare_context(x, time)
        return super().forecast(x, time)

    def _extra_state(self):
        return {"memory": self.memory, "pretrained": self._pretrained,
                "prepared_time": self._prepared_time, "prepared_x": self._prepared_x}

    def _load_extra_state(self, state):
        self.memory = _cpu_clone(state["memory"])
        self._pretrained = _cpu_clone(state["pretrained"])
        self._prepared_time, self._prepared_x = state["prepared_time"], _cpu_clone(state["prepared_x"])


class SOLIDPlusPlus(SOLID):
    """SOLID++ from PROCEED: continual updates of all forecaster parameters."""

    name = "SOLID++"
    continual = True


class _DecisionNetwork(nn.Module):
    def __init__(self, horizon, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3 * horizon, 32), nn.Dropout(dropout), nn.Tanh(),
                                    nn.Linear(32, 32), nn.Dropout(dropout), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x)


class OneNet(OnlineBaseline):
    """Author OneNet optimizer/decision logic over two explicit pretrained branches.

    The second branch cannot be fabricated by silently cloning the first one.
    The experiment manifest must identify both branch architectures/checkpoints.
    """

    name = "OneNet"

    def __init__(self, model, config, secondary_model: nn.Module):
        if secondary_model is None or secondary_model is model:
            raise ValueError("OneNet requires a separate, explicitly pretrained secondary_model")
        super().__init__(model, config)
        self.secondary_model = copy.deepcopy(secondary_model).to(self.device)
        self.secondary_model.requires_grad_(True)
        self.weight = nn.Parameter(torch.zeros(len(config.target_indices), device=self.device))
        with self._random_context():
            self.decision = _DecisionNetwork(config.horizon, config.decision_dropout).to(self.device)
        self.bias = torch.zeros(1, 1, len(config.target_indices), device=self.device)
        self.optimizers["model"] = self._optimizer(list(self.model.parameters()) + list(self.secondary_model.parameters()))
        self.optimizers["weight"] = torch.optim.Adam([self.weight], lr=config.learning_rate_w)
        self.optimizers["decision"] = torch.optim.Adam(self.decision.parameters(), lr=config.learning_rate_bias)
        self.offline_epochs = 0
        self.set_reset_point()

    def _gradient_models(self):
        return self.model, self.secondary_model

    def _branches(self, x):
        a, b = _prediction(self.model(x)), _prediction(self.secondary_model(x))
        expected = (x.shape[0], self.config.horizon, len(self.config.target_indices))
        if tuple(a.shape) != expected or a.shape != b.shape:
            raise ValueError("OneNet branches must have identical [B,H,T] output shapes")
        return a, b

    def _forward(self, x):
        a, b = self._branches(x)
        weight = torch.sigmoid(self.weight.view(1, 1, -1) + self.bias)
        return weight * a + (1 - weight) * b

    def forecast(self, x, time):
        self.secondary_model.eval()
        self.decision.eval()
        for branch in self._gradient_models():
            if hasattr(branch, "try_trigger_"):
                branch.try_trigger_(True)
        return super().forecast(x, time)

    def _weight_step(self, a, b, y):
        weight = torch.sigmoid(self.weight).view(1, 1, -1)
        loss = F.mse_loss(weight * a + (1 - weight) * b, y)
        # Upstream retains weight gradients from the decision loss before this
        # step. This is deliberately reproduced; only the decision optimizer was
        # zeroed after the preceding online decision step.
        loss.backward()
        self.optimizers["weight"].step()
        self.optimizers["weight"].zero_grad(set_to_none=True)
        return float(loss.detach())

    def _decision_step(self, a, b, y):
        weight = torch.sigmoid(self.weight).view(1, 1, -1)
        features = torch.cat((weight * a, (1 - weight) * b, y), dim=1).transpose(1, 2)
        bias = self.decision(features).transpose(1, 2)
        adjusted = torch.sigmoid(self.weight.view(1, 1, -1) + bias)
        loss = F.mse_loss(adjusted * a + (1 - adjusted) * b, y)
        self.optimizers["decision"].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers["decision"].step()
        self.optimizers["decision"].zero_grad(set_to_none=True)
        # The original current-batch bias becomes the next step's correction.
        self.bias = bias[-1:].detach().clone()
        return float(loss.detach())

    def _learn(self, x, y, *, offline=False):
        for branch in self._gradient_models():
            branch.train(True if offline else getattr(branch, "online_training_mode", True))
            if hasattr(branch, "try_trigger_"):
                branch.try_trigger_(not offline)
        self.decision.train()
        a, b = self._branches(x)
        loss = F.mse_loss(a, y) + F.mse_loss(b, y)
        self._step(loss, self.optimizers["model"])
        a, b = a.detach(), b.detach()
        if offline:
            self._weight_step(a, b, y)
            self._decision_step(a, b, y)
        else:
            self._decision_step(a, b, y)
            self._weight_step(a, b, y)
        return float(loss.detach()) / 2

    def update(self, released_batch, *, time, current_context=None):
        issues, _ = self._validate_released(released_batch, time)
        losses = []
        for i in range(len(issues)):
            for _ in range(self.config.update_steps):
                with self._random_context():
                    losses.append(self._learn(released_batch.x[i:i + 1].detach().to(self.device),
                                              released_batch.y[i:i + 1].detach().to(self.device)))
        self.total_update_steps += len(losses)
        self.total_released_samples += len(issues)
        self._last_release_issue, self._last_time = issues[-1], int(time)
        self.last_update = OnlineUpdate(self.name, issues, len(losses), sum(losses) / len(losses))
        return self.last_update

    def fit_offline(self, x, y, *, epochs=1, batch_size=32, learning_rate=None,
                    issue_times=None, cutoff=None):
        _validate_offline(self, x, y, epochs, batch_size, issue_times, cutoff)
        old_lrs = [g["lr"] for g in self.optimizers["model"].param_groups]
        if learning_rate is not None:
            for group in self.optimizers["model"].param_groups:
                group["lr"] = learning_rate
        history = []
        try:
            for _ in range(epochs):
                order = torch.randperm(len(x), generator=self._shuffle)
                losses = []
                for indices in order.split(batch_size):
                    with self._random_context():
                        losses.append(self._learn(x[indices].detach().to(self.device),
                                                  y[indices].detach().to(self.device), offline=True))
                history.append(sum(losses) / len(losses))
                self.offline_epochs += 1
        finally:
            for group, lr in zip(self.optimizers["model"].param_groups, old_lrs):
                group["lr"] = lr
        self.bias.zero_()
        self.set_reset_point()
        return {"method": self.name, "epochs": epochs, "losses": history,
                "samples": len(x), "cutoff": cutoff}

    def _extra_state(self):
        return {"secondary_model": self.secondary_model.state_dict(), "weight": self.weight,
                "weight_grad": self.weight.grad, "decision": self.decision.state_dict(),
                "bias": self.bias, "offline_epochs": self.offline_epochs}

    def _load_extra_state(self, state):
        self.secondary_model.load_state_dict(state["secondary_model"])
        self.weight.data.copy_(state["weight"].to(self.device))
        self.weight.grad = None if state["weight_grad"] is None else state["weight_grad"].to(self.device).clone()
        self.decision.load_state_dict(state["decision"])
        self.bias = state["bias"].to(self.device).clone()
        self.offline_epochs = state["offline_epochs"]


def _validate_offline(controller, x, y, epochs, batch_size, issue_times, cutoff):
    controller._validate_x(x)
    if tuple(y.shape) != (len(x), controller.config.horizon, len(controller.config.target_indices)):
        raise ValueError("offline target shape mismatch")
    if not torch.isfinite(y).all() or epochs < 1 or batch_size < 1:
        raise ValueError("offline training needs finite targets and positive epochs/batch_size")
    if (issue_times is None) != (cutoff is None):
        raise ValueError("supply both offline issue_times and training cutoff, or neither")
    if issue_times is not None:
        times = _times(issue_times, len(x), "offline issue_times")
        if any(t + controller.config.horizon + controller.config.extra_delay > cutoff for t in times):
            raise ValueError("offline training target release crosses training cutoff")
    if controller.total_released_samples:
        raise RuntimeError("offline fitting after online updates is forbidden")


class _ProceedLinear(nn.Linear):
    """Efficient per-sample row/column scaling, matching down_up.Linear."""

    adaptation = None

    def forward(self, x):
        if self.adaptation is None:
            return F.linear(x, self.weight, self.bias)
        output_scale, input_scale, shift = self.adaptation
        if x.shape[0] % input_scale.shape[0]:
            raise ValueError("PROCEED cannot align this layer's batch layout")
        shape = x.shape
        adapted_x = x.reshape(input_scale.shape[0], -1, shape[-1]) * input_scale[:, None, :]
        result = F.linear(adapted_x, self.weight, None) * output_scale[:, None, :]
        if self.bias is not None:
            result = result + self.bias + shift[:, None, :]
        return result.reshape(*shape[:-1], self.out_features)


class _ProceedConv1d(nn.Conv1d):
    adaptation = None

    def forward(self, x):
        if self.adaptation is None:
            return self._conv_forward(x, self.weight, self.bias)
        output_scale, input_scale, shift = self.adaptation
        if x.ndim != 3 or x.shape[0] % input_scale.shape[0]:
            raise ValueError("PROCEED cannot align this convolution's batch layout")
        # Scaling activations also handles grouped convolutions correctly.
        inp = x.reshape(input_scale.shape[0], -1, x.shape[-2], x.shape[-1])
        inp = (inp * input_scale[:, None, :, None]).reshape_as(x)
        result = self._conv_forward(inp, self.weight, None)
        shape = result.shape
        result = result.reshape(output_scale.shape[0], -1, *shape[-2:]) * output_scale[:, None, :, None]
        if self.bias is not None:
            result = result + (self.bias + shift)[:, None, :, None]
        return result.reshape(shape)


class _ProceedBottleneck(nn.Module):
    def __init__(self, concept_dim, out_dim, layer_count, rank):
        super().__init__()
        self.down = nn.Linear(concept_dim, rank, bias=False)
        self.layer_bias = nn.Parameter(torch.empty(layer_count, 1, rank))
        nn.init.uniform_(self.layer_bias, -1 / math.sqrt(concept_dim), 1 / math.sqrt(concept_dim))
        self.up = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        # Official code's trainable final biases stay active during online GD.
        self.output_bias = nn.Parameter(torch.zeros(layer_count, 1, out_dim))

    def forward(self, drift):
        return self.up(torch.sigmoid(self.down(drift).unsqueeze(0) + self.layer_bias)) + self.output_bias


class ProceedAdapter(nn.Module):
    """Two concept MLPs and layer-type-shared adaptation generators."""

    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.current_encoder = nn.Sequential(nn.Linear(config.lookback, config.concept_dim), nn.GELU(),
                                             nn.Linear(config.concept_dim, config.concept_dim))
        self.history_encoder = nn.Sequential(nn.Linear(config.lookback + config.horizon, config.concept_dim), nn.GELU(),
                                             nn.Linear(config.concept_dim, config.concept_dim))
        self.groups = {}
        self.dimensions = {}
        # The official default tune_mode=down_up covers Linear and Conv1d.
        for name, module in list(model.named_modules()):
            if not name or not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            if isinstance(module, nn.Linear):
                replacement = _ProceedLinear(module.in_features, module.out_features, bias=module.bias is not None,
                                             device=module.weight.device, dtype=module.weight.dtype)
                n_in, n_out = module.in_features, module.out_features
            else:
                replacement = _ProceedConv1d(module.in_channels, module.out_channels, module.kernel_size,
                                              stride=module.stride, padding=module.padding,
                                              dilation=module.dilation, groups=module.groups,
                                              bias=module.bias is not None, padding_mode=module.padding_mode,
                                              device=module.weight.device, dtype=module.weight.dtype)
                n_in, n_out = module.in_channels, module.out_channels
            replacement.load_state_dict(module.state_dict())
            parent_name, _, leaf_name = name.rpartition(".")
            setattr(model.get_submodule(parent_name), leaf_name, replacement)
            out_dim = n_in + n_out + (n_out if module.bias is not None else 0)
            key = leaf_name + "_" + str(out_dim)
            self.groups.setdefault(key, []).append(name)
            self.dimensions[name] = (n_in, n_out, module.bias is not None)
        if not self.groups:
            raise ValueError("PROCEED needs at least one supported Linear or Conv1d layer")
        self.generators = nn.ModuleDict({key: _ProceedBottleneck(config.concept_dim, int(key.rsplit("_", 1)[1]),
                                                                 len(names), config.bottleneck_dim)
                                         for key, names in self.groups.items()})

    def online_mode(self):
        self.requires_grad_(False)
        for generator in self.generators.values():
            generator.output_bias.requires_grad_(True)

    def online_parameters(self):
        return [g.output_bias for g in self.generators.values()]

    def concepts(self, current_x, previous_x, previous_y):
        # With exogenous inputs and a target subset, use observed target channels
        # on both sides. No unobserved future exogenous values are fabricated.
        current = current_x[..., list(self.config.target_indices)].transpose(-1, -2)
        previous = torch.cat((previous_x[..., list(self.config.target_indices)], previous_y), dim=1)
        return self.current_encoder(current).mean(-2), self.history_encoder(previous.transpose(-1, -2)).mean((0, 1))

    def assign(self, model, drift):
        if self.config.clip_drift:
            drift = drift / torch.linalg.vector_norm(drift, dim=-1, keepdim=True).clamp_min(1)
        for key, names in self.groups.items():
            coefficients = self.generators[key](drift)
            for index, name in enumerate(names):
                n_in, n_out, has_bias = self.dimensions[name]
                coefs = coefficients[index]
                output_scale = coefs[:, :n_out] + 1
                input_scale = coefs[:, n_out:n_out + n_in] + 1
                shift = coefs[:, -n_out:] if has_bias else None
                model.get_submodule(name).adaptation = (output_scale, input_scale, shift)

    def clear(self, model):
        for names in self.groups.values():
            for name in names:
                model.get_submodule(name).adaptation = None


class PROCEED(OnlineBaseline):
    """PROCEED with mandatory synthetic-drift offline training and frozen encoders.

    Official default down_up mapping is reproduced including zero-initialized
    output weights, drift clipping, shared generators and additive bias shifts.
    A forecast's generated changes are temporary; they never accumulate into
    persistent backbone weights. Online updates retain only released feedback.
    """

    name = "PROCEED"

    def __init__(self, model, config):
        super().__init__(model, config)
        with self._random_context():
            self.adapter = ProceedAdapter(self.model, config).to(self.device)
        self.adapter.online_mode()
        self.recent_x = self.recent_y = self.recent_concept = None
        self.offline_fitted = False
        self.offline_epochs = 0
        self.offline_training_steps = 0
        self.optimizers["model"] = self._optimizer(list(self.model.parameters()) + self.adapter.online_parameters())
        self.optimizers["offline"] = torch.optim.Adam(list(self.model.parameters()) + list(self.adapter.parameters()),
                                                     lr=config.learning_rate)
        self.set_reset_point()

    def _adapted_forward(self, x, *, update_reference=False):
        if self.recent_x is None:
            previous_x = torch.zeros_like(x[:1])
            previous_y = x.new_zeros(1, self.config.horizon, len(self.config.target_indices))
        else:
            previous_x, previous_y = self.recent_x.to(self.device), self.recent_y.to(self.device)
        current, recent = self.adapter.concepts(x, previous_x, previous_y)
        if self.config.ema and self.recent_concept is not None:
            recent = self.config.ema * self.recent_concept.to(self.device) + (1 - self.config.ema) * recent
        if self.config.ema and update_reference:
            self.recent_concept = _cpu_clone(recent)
        self.adapter.assign(self.model, current - recent)
        try:
            return super()._forward(x)
        finally:
            self.adapter.clear(self.model)

    def forecast(self, x, time):
        if not self.offline_fitted:
            raise RuntimeError("PROCEED requires fit_offline() or a trained adapter checkpoint before forecasting")
        self._validate_x(x)
        self._validate_time(time)
        self.model.eval()
        self.adapter.eval()
        with self._random_context(), torch.no_grad():
            prediction = self._adapted_forward(x.to(self.device)).detach().clone()
        self._last_time = int(time)
        return prediction

    def seed_history(self, released_batch, *, time):
        issues, _ = self._validate_released(released_batch, time)
        self.recent_x, self.recent_y = _cpu_clone(released_batch.x[-1:]), _cpu_clone(released_batch.y[-1:])
        self._last_release_issue, self._last_time = issues[-1], int(time)

    def update(self, released_batch, *, time, current_context=None):
        if not self.offline_fitted:
            raise RuntimeError("PROCEED adapter has not been fitted offline")
        issues, _ = self._validate_released(released_batch, time)
        losses = []
        self.adapter.online_mode()
        self.adapter.eval()
        for i in range(len(issues)):
            x, y = released_batch.x[i:i + 1].detach().to(self.device), released_batch.y[i:i + 1].detach().to(self.device)
            for _ in range(self.config.update_steps):
                with self._random_context():
                    self.model.train()
                    loss = F.mse_loss(self._adapted_forward(x, update_reference=True), y)
                    self._step(loss, self.optimizers["model"])
                losses.append(float(loss.detach()))
                self.recent_x, self.recent_y = _cpu_clone(x), _cpu_clone(y)
        self.total_update_steps += len(losses)
        self.total_released_samples += len(issues)
        self._last_release_issue, self._last_time = issues[-1], int(time)
        self.last_update = OnlineUpdate(self.name, issues, len(losses), sum(losses) / len(losses))
        return self.last_update

    def fit_offline(self, x, y, *, epochs=1, batch_size=32, learning_rate=None,
                    issue_times=None, cutoff=None):
        """Jointly train backbone/encoders/generator on shuffled historical batches.

        The previous random mini-batch supplies the historical concept and each
        current sample receives its own drift-conditioned coefficients. This is
        the paper's synthetic-drift stage; it is not an empty adapter warm-up.
        ``issue_times`` and ``cutoff`` permit auditable train-only label checks.
        """
        _validate_offline(self, x, y, epochs, batch_size, issue_times, cutoff)
        if len(x) < 2:
            raise ValueError("synthetic concept drift training needs at least two historical samples")
        # Ensure at least two batches: a single full-dataset batch has no diverse
        # transitions across independently shuffled historical mini-batches.
        batch_size = min(batch_size, max(1, len(x) // 2))
        optimizer = self.optimizers["offline"]
        if learning_rate is not None:
            if learning_rate <= 0 or not math.isfinite(learning_rate):
                raise ValueError("offline learning_rate must be finite and positive")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        self.adapter.requires_grad_(True)
        history = []
        try:
            for _ in range(epochs):
                self.model.train()
                self.adapter.train()
                losses = []
                order = torch.randperm(len(x), generator=self._shuffle)
                for indices in order.split(batch_size):
                    current_x, current_y = x[indices].detach().to(self.device), y[indices].detach().to(self.device)
                    with self._random_context():
                        prediction = self._adapted_forward(current_x, update_reference=True)
                        loss = F.mse_loss(prediction, current_y)
                        self._step(loss, optimizer)
                    losses.append(float(loss.detach()))
                    self.recent_x, self.recent_y = _cpu_clone(current_x), _cpu_clone(current_y)
                    self.offline_training_steps += 1
                self.offline_epochs += 1
                history.append(sum(losses) / len(losses))
        finally:
            self.adapter.online_mode()
        self.offline_fitted = self.offline_training_steps > 0
        self.optimizers["model"].state.clear()
        self.set_reset_point()
        return {"method": self.name, "epochs": epochs, "losses": history, "samples": len(x),
                "training_steps": self.offline_training_steps, "batch_size": batch_size, "cutoff": cutoff}

    def _extra_state(self):
        return {"adapter": self.adapter.state_dict(), "recent_x": self.recent_x,
                "recent_y": self.recent_y, "recent_concept": self.recent_concept,
                "offline_fitted": self.offline_fitted, "offline_epochs": self.offline_epochs,
                "offline_training_steps": self.offline_training_steps}

    def _load_extra_state(self, state):
        self.adapter.load_state_dict(state["adapter"])
        self.adapter.clear(self.model)
        self.adapter.online_mode()
        self.recent_x, self.recent_y = _cpu_clone(state["recent_x"]), _cpu_clone(state["recent_y"])
        self.recent_concept = _cpu_clone(state["recent_concept"])
        self.offline_fitted, self.offline_epochs = state["offline_fitted"], state["offline_epochs"]
        self.offline_training_steps = state["offline_training_steps"]


def make_online_baseline(name: str, model: nn.Module, config: OnlineBaselineConfig | Mapping,
                         *, secondary_model: nn.Module | None = None) -> OnlineBaseline:
    """Build a source-defined controller; reject unknown names and OneNet proxies."""
    if not isinstance(config, OnlineBaselineConfig):
        config = OnlineBaselineConfig(**config)
    normalized = name.lower().replace("_", "-")
    choices = {"frozen": Frozen, "gd": GradientDescent, "head-gd": HeadGradientDescent, "solid": SOLID,
               "solid++": SOLIDPlusPlus, "solid-plus-plus": SOLIDPlusPlus, "proceed": PROCEED}
    if normalized == "onenet":
        return OneNet(model, config, secondary_model)
    if normalized not in choices:
        raise ValueError(f"unsupported online baseline: {name}")
    return choices[normalized](model, config)


__all__ = ["OnlineBaselineConfig", "ReleasedBatch", "OnlineUpdate", "OnlineBaseline", "Frozen", "GradientDescent",
           "HeadGradientDescent", "OneNet", "SOLID", "SOLIDPlusPlus", "PROCEED", "ProceedAdapter", "make_online_baseline"]
