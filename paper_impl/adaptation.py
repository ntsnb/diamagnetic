"""Deterministic, delayed-feedback adaptation for the paper's three stages.

PyTorch stores a linear weight as ``[out_features, in_features]``.  Consequently
the paper's output-channel columns are *rows* of the tensors masked here.
Temporary gains are passed to the model's functional forward; they are never
copied into a persistent parameter.

Minimal use (``contrast_fn`` returns the frozen branch-forecast difference)::

    generator = CenteredBoundedGenerator(
        {n: tuple(p.shape) for n, p in model.expert_weights().items()}, H * T)
    controller = OnlineController(model, OnlineConfig(), generator=generator,
                                  contrast_fn=contrast_fn)
    forecast = controller.predict(x_t, time=t)
    # Only after the whole horizon and the extra delay have elapsed:
    update = controller.release(t + H + delay, {t: released_y}, current_x=x_now)

``DelayedFeedbackStream`` provides the safer single-observation event API.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn


class _GainBlock(nn.Module):
    def __init__(self, contrast_dim: int, in_features: int, out_features: int,
                 hidden_dim: int, epsilon: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon
        self.w1 = nn.Linear(contrast_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, in_features + out_features, bias=False)
        # A random first layer is necessary: zeroing both layers kills learning.
        nn.init.zeros_(self.w2.weight)

    def forward(self, contrast: Tensor) -> tuple[Tensor, Tensor]:
        centered = torch.sigmoid(self.w1(contrast)) - torch.sigmoid(self.w1.bias)
        gains = 1.0 + self.epsilon * torch.tanh(self.w2(centered))
        return gains.split((self.in_features, self.out_features), dim=-1)


class CenteredBoundedGenerator(nn.Module):
    """Input/output gains with exact zero-contrast identity and bounded strength.

    ``expert_shapes`` maps the model's full parameter names to ``(Cout, Cin)``.
    Calling this module is differentiable for offline replay.  The controller
    detaches its output during online persistent updates.
    """

    def __init__(self, expert_shapes: Mapping[str, tuple[int, int]],
                 contrast_dim: int, hidden_dim: int = 16, epsilon: float = 0.1,
                 mode: str = "centered_bounded", share_matching_shapes: bool = False):
        super().__init__()
        if contrast_dim < 1 or hidden_dim < 1 or not 0.0 < epsilon < 1.0:
            raise ValueError("Positive dimensions and 0 < epsilon < 1 are required")
        if mode not in {"centered_bounded", "identity"}:
            raise ValueError("Unknown generator mode")
        if not expert_shapes:
            raise ValueError("At least one expert weight is required")
        self.contrast_dim, self.epsilon, self.mode = contrast_dim, epsilon, mode
        self.expert_shapes = {name: tuple(shape) for name, shape in expert_shapes.items()}
        self.blocks = nn.ModuleDict()
        self._block_names: dict[str, str] = {}
        shared: dict[tuple[int, int], str] = {}
        for name, shape in self.expert_shapes.items():
            if len(shape) != 2 or min(shape) < 1:
                raise ValueError(f"{name} must be a nonempty linear weight")
            if mode == "identity":
                continue
            if share_matching_shapes and shape in shared:
                block_name = shared[shape]
            else:
                block_name = f"layer_{len(self.blocks)}"
                self.blocks[block_name] = _GainBlock(contrast_dim, shape[1], shape[0],
                                                       hidden_dim, epsilon)
                shared[shape] = block_name
            self._block_names[name] = block_name

    def gain_vectors(self, contrast: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
        if contrast.ndim != 2 or contrast.shape[1] != self.contrast_dim:
            raise ValueError("contrast must have shape [batch, horizon * targets]")
        if not torch.isfinite(contrast).all():
            raise ValueError("contrast contains nonfinite values")
        result = {}
        for name, (out_features, in_features) in self.expert_shapes.items():
            if self.mode == "identity":
                result[name] = (contrast.new_ones((len(contrast), in_features)),
                                contrast.new_ones((len(contrast), out_features)))
            else:
                result[name] = self.blocks[self._block_names[name]](contrast)
        return result

    def forward(self, contrast: Tensor) -> dict[str, Tensor]:
        return {name: out_gain.unsqueeze(-1) * in_gain.unsqueeze(-2)
                for name, (in_gain, out_gain) in self.gain_vectors(contrast).items()}


@dataclass(frozen=True)
class OnlineConfig:
    stage_i: bool = True
    stage_ii: bool = True
    stage_iii: bool = True
    selector: str = "relevance_error"
    selection_protocol: str = "adaptive"
    selection_seed: int = 42
    learning_rate_i: float = 0.01
    learning_rate_ii: float = 0.001
    radius_i: float = 0.05
    radius_ii: float = 0.05
    armijo_c: float = 1e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 12
    stability_threshold: float = 0.9
    relevance_floor: float = 1e-8
    error_threshold: float = math.inf
    channel_budget: int | Mapping[str, int] = 1
    reference_windows: int = 4
    reference_interval: int = 16
    reference_ema: float = 0.1
    history_capacity: int = 64
    extra_delay: int = 0
    pending_capacity: int = 4096
    record_capacity: int = 512
    stage_ii_scope: str = "sparse_experts"
    rollback_on_worse: bool = True
    reference_mode: str = "ema"

    def __post_init__(self):
        if self.stage_ii_scope not in ("sparse_experts", "dense_eligible", "dense_experts", "all_forecaster", "experts_router_head"):
            raise ValueError("unknown Stage II update scope")
        if not isinstance(self.rollback_on_worse, bool):
            raise ValueError("rollback_on_worse must be boolean")
        if self.reference_mode not in ("ema", "current", "frozen"):
            raise ValueError("reference_mode must be ema, current, or frozen")
        if self.selection_protocol not in {"adaptive", "fixed_budget"}:
            raise ValueError("selection_protocol must be adaptive or fixed_budget")
        selectors = ({"relevance_error", "gradient_only"} if self.selection_protocol == "adaptive"
                     else {"relevance_only", "gradient_only", "random"})
        if self.selector not in selectors:
            raise ValueError(f"selector is incompatible with {self.selection_protocol} selection_protocol")
        if self.selection_protocol == "fixed_budget" and self.stage_ii_scope not in {"sparse_experts", "dense_eligible"}:
            raise ValueError("fixed_budget comparisons require sparse_experts or dense_eligible")
        if (isinstance(self.selection_seed, bool) or not isinstance(self.selection_seed, int)
                or not 0 <= self.selection_seed < 2**32):
            raise ValueError("selection_seed must be an integer in [0, 2**32)")
        for name in ("learning_rate_i", "learning_rate_ii", "radius_i", "radius_ii",
                     "relevance_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.armijo_c < 1 or not 0 < self.backtrack_factor < 1:
            raise ValueError("Armijo and backtracking factors must be in (0, 1)")
        if not 0 <= self.stability_threshold <= 1 or not 0 < self.reference_ema <= 1:
            raise ValueError("Invalid stability threshold or reference EMA")
        if math.isnan(self.error_threshold) or self.error_threshold < 0:
            raise ValueError("error_threshold must be nonnegative")
        for name in ("max_backtracks", "reference_windows", "reference_interval",
                     "history_capacity", "pending_capacity", "record_capacity"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.extra_delay, int) or self.extra_delay < 0:
            raise ValueError("extra_delay must be a nonnegative integer")
        if self.history_capacity < self.reference_windows:
            raise ValueError("history_capacity must hold the full reference set")
        budgets = (self.channel_budget.values() if isinstance(self.channel_budget, Mapping)
                   else (self.channel_budget,))
        if any(not isinstance(k, int) or k < 0 for k in budgets):
            raise ValueError("Channel budgets must be nonnegative integers")


@dataclass(frozen=True)
class ForecastRecord:
    issue_time: int
    release_time: int
    inputs: Tensor
    prediction: Tensor

    def clone(self) -> "ForecastRecord":
        return ForecastRecord(self.issue_time, self.release_time,
                              self.inputs.clone(), self.prediction.clone())


@dataclass(frozen=True)
class ReleasedBatch:
    issue_times: tuple[int, ...]
    inputs: Tensor
    targets: Tensor
    stored_predictions: Tensor


@dataclass
class StageUpdate:
    stage: str
    reason: str
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    gradient_norm: float = 0.0
    update_norm: float = 0.0
    step_size: float = 0.0
    trials: int = 0
    accepted: bool = False
    # Each listed output channel represents a full row of a PyTorch weight.
    selected_channels: dict[str, tuple[int, ...]] = field(default_factory=dict)
    selection_protocol: str = "adaptive"
    selector: Optional[str] = None
    selection_event_seed: Optional[int] = None
    # Selected counts measure mask capacity; changed counts measure the exact
    # nonzero persistent differences after an accepted step, and remain zero
    # for zero gradients or a rejected/backtracked step.
    selected_parameter_count: int = 0
    changed_parameter_count: int = 0
    eligible_parameter_count: int = 0
    layer_budgets: dict[str, dict[str, int]] = field(default_factory=dict)


class SelectionUnavailableError(RuntimeError):
    """A fixed-budget ranking is unavailable; the release is rolled back.

    No undersized mask or substitute ranking is used. The failing method/run
    must be reported as unavailable, rather than pooled with completed controls.
    """


@dataclass
class UpdateRecord:
    time: int
    issue_times: tuple[int, ...]
    stored_error: Optional[float]
    routing: StageUpdate
    experts: StageUpdate
    stability: tuple[Optional[float], ...]
    reference_version: int
    reference_refreshed: bool = False


class OnlineController:
    """Bounded-state controller; no target is accepted before its release time.

    ``predict`` issues exactly one chronological window.  ``release`` is called
    before ``predict`` at the same time and accepts only previously issued,
    fully released horizons.  Empty releases never update persistent parameters.
    """

    def __init__(self, model: nn.Module, config: OnlineConfig | None = None, *,
                 generator: CenteredBoundedGenerator | None = None,
                 contrast_fn: Callable[[Tensor], Tensor] | None = None,
                 reference_inputs: Tensor | None = None,
                 weight_layout: Mapping[str, tuple[int, str]] | None = None,
                 discrepancy_fn: Callable[[Tensor, Tensor], Tensor] | None = None):
        self.model, self.config = model, config or OnlineConfig()
        self.generator, self.contrast_fn = generator, contrast_fn
        self._discrepancy_fn = discrepancy_fn
        self.horizon = int(model.config.horizon)
        self.n_targets = (len(model.config.targets) if not isinstance(model.config.targets, int)
                          else model.config.targets)
        self.n_experts = int(model.config.n_experts)
        if self.config.pending_capacity < self.horizon + self.config.extra_delay + 1:
            raise ValueError("pending_capacity must cover horizon + delay + 1")
        if self.config.stage_iii and (generator is None or contrast_fn is None):
            raise ValueError("Stage III requires a generator and frozen branch contrast_fn")
        self.weights: dict[str, nn.Parameter] = dict(model.expert_weights())
        self.update_parameters = dict(self.weights)
        if self.config.stage_ii_scope in ("dense_experts", "experts_router_head", "all_forecaster"):
            prefixes = (("experts.", "router.", "output_head.") if self.config.stage_ii_scope == "experts_router_head"
                        else ("experts.",))
            self.update_parameters = {name: parameter for name, parameter in model.named_parameters()
                if self.config.stage_ii_scope == "all_forecaster" or name.startswith(prefixes)}
        if not self.weights or any(p.ndim != 2 for p in self.weights.values()):
            raise ValueError("expert_weights() must return nonempty linear weight tensors")
        named = dict(model.named_parameters())
        if any(named.get(name) is not parameter for name, parameter in self.weights.items()):
            raise ValueError("Expert weight names must match model.named_parameters()")
        self.layout = self._resolve_layout(weight_layout)
        self.layers: dict[str, list[str]] = defaultdict(list)
        for name, (_, layer) in self.layout.items():
            self.layers[layer].append(name)
        for layer, names in self.layers.items():
            names.sort(key=lambda name: self.layout[name][0])
            if len({tuple(self.weights[n].shape) for n in names}) != 1:
                raise ValueError(f"Corresponding layers must have equal shapes: {layer}")
            if len({self.layout[n][0] for n in names}) != len(names):
                raise ValueError(f"Duplicate expert-layer entry: {layer}")
        if isinstance(self.config.channel_budget, Mapping):
            if set(self.config.channel_budget) != set(self.layers):
                raise ValueError("Provide a channel budget for every corresponding layer")
        if self.config.selection_protocol == "fixed_budget":
            for layer, names in self.layers.items():
                available = sum(self.weights[name].shape[0] for name in names)
                if self._layer_budget(layer) > available:
                    raise ValueError(f"fixed_budget for {layer} exceeds its {available} eligible channels")
        if generator is not None:
            if generator.expert_shapes != {n: tuple(p.shape) for n, p in self.weights.items()}:
                raise ValueError("Generator weights must match expert_weights() exactly")
            # Online forwards use no_grad; do not disable offline replay training.
            generator.eval()
        self.model.eval().requires_grad_(False)
        self.router_bias = model.router.bias
        if self.router_bias is None:
            raise ValueError("The router must have an output bias")
        self.router_bias.requires_grad_(self.config.stage_i)
        for parameter in self.update_parameters.values():
            parameter.requires_grad_(self.config.stage_ii)
        self.router_bias.requires_grad_(self.config.stage_i or
            (self.config.stage_ii and any(p is self.router_bias for p in self.update_parameters.values())))
        self.reference = copy.deepcopy(model).eval().requires_grad_(False)
        self._initial_state = copy.deepcopy(model.state_dict())
        self._initial_inputs = None if reference_inputs is None else reference_inputs.detach().cpu().clone()
        self.pending: OrderedDict[int, ForecastRecord] = OrderedDict()
        self.history: deque[Tensor] = deque(maxlen=self.config.history_capacity)
        self.records: deque[UpdateRecord] = deque(maxlen=self.config.record_capacity)
        self.reset()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _resolve_layout(self, explicit):
        if explicit is not None:
            if set(explicit) != set(self.weights):
                raise ValueError("weight_layout must cover each expert weight exactly")
            result = dict(explicit)
        else:
            result = {}
            for name in self.weights:
                match = re.match(r"^(?:experts|expert_bank)\.(\d+)\.(.+)$", name)
                if match is None:
                    raise ValueError(f"Supply weight_layout for nonstandard name {name}")
                result[name] = (int(match.group(1)), match.group(2))
        if any(not 0 <= m < self.n_experts for m, _ in result.values()):
            raise ValueError("Expert indices must match model.config.n_experts")
        return result

    def reset(self):
        """Restore the episode's initial predictor/reference; retain learned generator."""
        self.model.load_state_dict(self._initial_state)
        self.reference.load_state_dict(self._initial_state)
        self.model.eval()
        self.reference.eval()
        self.pending.clear()
        self.history.clear()
        self.records.clear()
        self.reference_windows: Optional[Tensor] = None
        self.reference_maps: Optional[Tensor] = None
        self.reference_channels: dict[str, Tensor] = {}
        self.reference_version = 0
        self.update_occasions = 0
        self._clock: Optional[int] = None
        self._last_issue: Optional[int] = None
        self._squared_error = self._absolute_error = 0.0
        self._evaluated_values = 0
        if self._initial_inputs is not None:
            data = self._initial_inputs
            if data.ndim != 3:
                raise ValueError("reference_inputs must have shape [windows, lookback, variables]")
            for sample in data:
                self.history.append(self._input(sample).detach().cpu().clone())
            self._initialize_reference_windows()

    def _input(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if tuple(x.shape) != (1, int(self.model.config.lookback), int(self.model.config.n_variables)):
            raise ValueError("An event must contain one [lookback, variables] input window")
        if not torch.isfinite(x).all():
            raise ValueError("Input window contains nonfinite values")
        return x.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _time(time: int):
        if not isinstance(time, int) or isinstance(time, bool):
            raise ValueError("time must be an integer observation index")

    def _check_time(self, time: int, releasing=False):
        self._time(time)
        if self._clock is not None and time < self._clock:
            raise ValueError("Events must be chronological")
        if releasing and self._last_issue is not None and time <= self._last_issue:
            raise ValueError("Release labels before issuing the forecast at that time")

    def modulations(self, x: Tensor) -> dict[str, Tensor] | None:
        """Frozen gains used for deterministic online prediction/update evaluation."""
        if not self.config.stage_iii:
            return None
        self.generator.eval()
        if isinstance(self.contrast_fn, nn.Module):
            self.contrast_fn.eval()
        with torch.no_grad():
            contrast = self.contrast_fn(x)
            if contrast.ndim > 2:
                contrast = contrast.flatten(1)
            return {name: value.detach() for name, value in self.generator(contrast).items()}

    def _prediction(self, x: Tensor, gains=None) -> Tensor:
        result = self.model(x, stochastic=False, modulations=gains)
        prediction = result.prediction
        if prediction.shape != (len(x), self.horizon, self.n_targets):
            raise ValueError("Model prediction must have shape [batch, horizon, targets]")
        return prediction

    def predict(self, x: Tensor, time: int) -> ForecastRecord:
        self._check_time(time)
        if self._last_issue is not None and time <= self._last_issue:
            raise ValueError("Forecast issue times must be strictly increasing")
        if len(self.pending) >= self.config.pending_capacity:
            raise BufferError("Pending forecast buffer is full; release available labels first")
        x = self._input(x)
        self.model.eval()
        with torch.no_grad():
            prediction = self._prediction(x, self.modulations(x))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite forecast; no record was issued")
        record = ForecastRecord(time, time + self.horizon + self.config.extra_delay,
                                x.detach().cpu().clone(), prediction.detach().cpu().clone())
        self.pending[time] = record
        self._clock = self._last_issue = time
        self.history.append(record.inputs.clone())
        self._initialize_reference_windows()
        return record.clone()

    def due_forecasts(self, time: int) -> tuple[ForecastRecord, ...]:
        self._time(time)
        return tuple(record.clone() for record in self.pending.values() if record.release_time <= time)

    @property
    def pending_records(self) -> tuple[ForecastRecord, ...]:
        """Detached copies for evaluation or chronological generator replay."""
        return tuple(record.clone() for record in self.pending.values())

    def discard_unobserved(self, time: int, issue_times):
        """Retire due horizons whose observed truth is incomplete, without updating."""
        self._check_time(time, releasing=True)
        issues = tuple(issue_times)
        if len(set(issues)) != len(issues):
            raise ValueError("duplicate issue times")
        for issue in issues:
            if issue not in self.pending or self.pending[issue].release_time > time:
                raise ValueError("only due, pending forecasts may be retired for missing observations")
        for issue in issues:
            del self.pending[issue]

    def prepare_release(self, time: int, targets: Mapping[int, Tensor]) -> ReleasedBatch | None:
        """Validate released horizons without mutating state (also used by replay)."""
        self._check_time(time, releasing=True)
        if not targets:
            return None
        inputs, labels, predictions, times = [], [], [], []
        for issue_time in sorted(targets):
            if issue_time not in self.pending:
                raise KeyError(f"No unreleased forecast at issue time {issue_time}")
            record = self.pending[issue_time]
            if time < record.release_time:
                raise ValueError(f"Forecast {issue_time} is unavailable until {record.release_time}")
            label = targets[issue_time]
            if label.ndim == 2:
                label = label.unsqueeze(0)
            if label.shape != record.prediction.shape or not torch.isfinite(label).all():
                raise ValueError("Each released target must be finite and match [1, horizon, targets]")
            inputs.append(record.inputs)
            labels.append(label.detach().cpu())
            predictions.append(record.prediction)
            times.append(issue_time)
        convert = lambda values: torch.cat(values).to(device=self.device, dtype=self.dtype)
        return ReleasedBatch(tuple(times), convert(inputs), convert(labels), convert(predictions))

    def _loss(self, batch: ReleasedBatch, gains):
        return (self._prediction(batch.inputs, gains) - batch.targets).square().mean()

    @staticmethod
    def _norm(tensors) -> float:
        return math.sqrt(sum(float(t.detach().abs().double().square().sum()) for t in tensors))

    def _gradients(self, loss: Tensor, parameters):
        if not loss.requires_grad:
            return [torch.zeros_like(p) for p in parameters]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return [torch.zeros_like(p) if gradient is None else gradient.detach()
                for p, gradient in zip(parameters, gradients)]

    def _step(self, stage: str, reason: str, batch: ReleasedBatch, gains,
              parameters, learning_rate: float, radius: float, *, gradients=None,
              masks=None, selected=None, baseline=None, layer_budgets=None,
              selection_event_seed=None) -> StageUpdate:
        loss = self._loss(batch, gains) if baseline is None else baseline
        before = float(loss.detach())
        eligible_count = sum(parameter.numel() for parameter in parameters)
        selected_count = (sum(int(torch.count_nonzero(mask)) for mask in masks)
                          if masks is not None else eligible_count)
        result = StageUpdate(
            stage, reason, before, before, selected_channels=selected or {},
            selection_protocol=self.config.selection_protocol,
            selector=self.config.selector if stage == "II" else None,
            selection_event_seed=selection_event_seed,
            selected_parameter_count=selected_count, eligible_parameter_count=eligible_count,
            layer_budgets=layer_budgets or {},
        )
        if not math.isfinite(before):
            result.loss_before = result.loss_after = None
            result.reason += ":nonfinite_loss"
            return result
        if gradients is None:
            gradients = self._gradients(loss, parameters)
        if masks is not None:
            gradients = [g * mask for g, mask in zip(gradients, masks)]
        if any(not torch.isfinite(g).all() for g in gradients):
            result.reason += ":nonfinite_gradient"
            return result
        norm = self._norm(gradients)
        result.gradient_norm = norm
        if norm == 0:
            result.reason += ":zero_gradient"
            return result
        step = min(learning_rate, radius / norm)
        originals = [p.detach().clone() for p in parameters]
        try:
            for trial in range(self.config.max_backtracks):
                with torch.no_grad():
                    for parameter, original, gradient in zip(parameters, originals, gradients):
                        parameter.copy_(original - step * gradient)
                    after = float(self._loss(batch, gains))
                result.trials = trial + 1
                if (math.isfinite(after) and (not self.config.rollback_on_worse or
                        (after < before and after <= before - self.config.armijo_c * step * norm * norm))):
                    result.accepted = True
                    if not self.config.rollback_on_worse:
                        result.reason += ":acceptance_test_disabled"
                    result.loss_after, result.step_size = after, step
                    result.update_norm = self._norm([p.detach() - old for p, old in zip(parameters, originals)])
                    result.changed_parameter_count = sum(int(torch.count_nonzero(p.detach() != old))
                                                         for p, old in zip(parameters, originals))
                    return result
                step *= self.config.backtrack_factor
        finally:
            if not result.accepted:
                with torch.no_grad():
                    for parameter, original in zip(parameters, originals):
                        parameter.copy_(original)
        result.reason += ":backtracking_rejected"
        return result

    def _attribute(self, x):
        with torch.enable_grad():
            attribution = self.reference.attribute(x)
        maps = attribution.maps.detach()
        expected = (len(x), self.n_targets, self.n_experts,
                    int(self.model.config.n_variables), int(self.model.config.lookback))
        if maps.shape != expected:
            raise ValueError(f"Reference maps have shape {tuple(maps.shape)}, expected {expected}")
        channels = {}
        for name, value in attribution.channels.items():
            if name not in self.weights:
                continue
            if value.shape != (len(x), self.n_targets, self.weights[name].shape[0]):
                raise ValueError(f"Invalid output-channel relevance shape for {name}")
            channels[name] = value.detach()
        return maps, channels

    def _initialize_reference_windows(self):
        if (self._needs_reference()
                and self.reference_windows is None
                and len(self.history) >= self.config.reference_windows):
            self._recompute_reference_windows()

    def _recompute_reference_windows(self):
        self.reference_windows = torch.cat(list(self.history)[-self.config.reference_windows:]).to(
            device=self.device, dtype=self.dtype)
        self.reference_maps, self.reference_channels = self._attribute(self.reference_windows)

    def _refresh_reference(self) -> bool:
        if (self.config.reference_mode != "ema" or not self._needs_reference()
                or len(self.history) < self.config.reference_windows):
            return False
        with torch.no_grad():
            reference_weights = dict(self.reference.expert_weights())
            for name, parameter in self.weights.items():
                reference_weights[name].lerp_(parameter.detach(), self.config.reference_ema)
        self.reference_version += 1
        self._recompute_reference_windows()
        return True

    def _needs_reference(self):
        return (self.config.stage_ii and self.config.stage_ii_scope == "sparse_experts"
                and self.config.selector in {"relevance_error", "relevance_only"})

    def _distance(self, a: Tensor, b: Tensor) -> Tensor:
        if self._discrepancy_fn is None:
            from .discrepancy import normalized_discrepancy
            self._discrepancy_fn = normalized_discrepancy
        return self._discrepancy_fn(a, b)

    def _relevance_scores(self, current_x):
        fixed = self.config.selection_protocol == "fixed_budget"
        if self.config.reference_mode == "current" and len(self.history) >= self.config.reference_windows:
            self.reference.load_state_dict(self.model.state_dict())
            self._recompute_reference_windows()
            self.reference_version += 1
        stability: list[Optional[float]] = [None] * self.n_experts
        scores: dict[str, Tensor] = {}
        if self.reference_maps is None or current_x is None:
            reason = "reference_warmup" if self.reference_maps is None else "no_current_window"
            if fixed:
                raise SelectionUnavailableError(f"fixed_budget:relevance_only:{reason}")
            return scores, tuple(stability), reason
        maps, channels = self._attribute(self._input(current_x))
        floor = self.config.relevance_floor
        for m in range(self.n_experts):
            current, preceding = maps[0, :, m], self.reference_maps[:, :, m]
            if fixed:
                # Removing the stability threshold does not waive the
                # validity of the attribution that supplies its diagnostic.
                for source, value in (("current", current), ("reference", preceding)):
                    totals = value.double().sum(dim=(-2, -1))
                    if not bool(torch.isfinite(value).all() and (value >= 0).all()
                                and torch.isfinite(totals).all() and (totals > floor).all()):
                        raise SelectionUnavailableError(
                            f"fixed_budget:relevance_only:invalid_input_profile:{source}:expert={m}")
            reliable = all(bool(torch.isfinite(v).all() and (v >= 0).all()
                                and (v.double().sum(dim=(-2, -1)) > floor).all())
                           for v in (current, preceding))
            if reliable:
                distance = self._distance(current.unsqueeze(0).expand_as(preceding), preceding)
                if torch.isfinite(distance).all():
                    stability[m] = 1.0 - float(distance.mean())
                elif fixed:
                    raise SelectionUnavailableError(
                        f"fixed_budget:relevance_only:invalid_input_discrepancy:expert={m}")
        for name, parameter in self.weights.items():
            m, _ = self.layout[name]
            if not fixed and (stability[m] is None or stability[m] >= self.config.stability_threshold):
                continue
            if name not in channels or name not in self.reference_channels:
                if fixed:
                    raise SelectionUnavailableError(f"fixed_budget:relevance_only:missing_channel_profile:{name}")
                continue
            current, preceding = channels[name][0].double(), self.reference_channels[name].double()
            if not all(bool(torch.isfinite(v).all() and (v >= 0).all()
                            and (v.sum(-1) > floor).all()
                            and (not fixed or torch.isfinite(v.sum(-1)).all()))
                       for v in (current, preceding)):
                if fixed:
                    raise SelectionUnavailableError(f"fixed_budget:relevance_only:invalid_channel_profile:{name}")
                continue
            current = current / current.sum(-1, keepdim=True)
            preceding = preceding / preceding.sum(-1, keepdim=True)
            scores[name] = (current - preceding.mean(0)).abs().mean(0)
        return scores, tuple(stability), "relevance_drift"

    def _layer_budget(self, layer):
        return (self.config.channel_budget[layer] if isinstance(self.config.channel_budget, Mapping)
                else self.config.channel_budget)

    def _random_scores(self, time):
        # A release-local CPU generator makes retries independent of global RNG
        # and of the number of previously processed/rejected release events.
        digest = hashlib.sha256(f"{self.config.selection_seed}:{time}".encode("ascii")).digest()
        event_seed = int.from_bytes(digest[:4], "little")
        generator = torch.Generator(device="cpu").manual_seed(event_seed)
        return {name: torch.rand(self.weights[name].shape[0], generator=generator, dtype=torch.float64)
                for name in sorted(self.weights)}, event_seed

    def _layer_budget_record(self, selected, *, dense=False):
        records = {}
        for layer, names in self.layers.items():
            available = sum(self.weights[name].shape[0] for name in names)
            count = available if dense else sum(len(selected.get(name, ())) for name in names)
            width = int(self.weights[names[0]].shape[1])
            records[layer] = {
                "budget_channels": available if dense else self._layer_budget(layer),
                "eligible_channels": available, "selected_channels": count,
                "input_width": width, "selected_parameter_count": count * width,
                "eligible_parameter_count": available * width,
            }
        return records

    def _idle_expert_update(self, reason):
        return StageUpdate("II", reason, selection_protocol=self.config.selection_protocol,
                           selector=self.config.selector,
                           eligible_parameter_count=sum(p.numel() for p in self.update_parameters.values()),
                           layer_budgets=self._layer_budget_record({}))

    def _select(self, scores):
        fixed = self.config.selection_protocol == "fixed_budget"
        selected = {name: [] for name in self.weights}
        for layer, names in self.layers.items():
            budget = self._layer_budget(layer)
            candidates = []
            for name in names:
                if fixed:
                    value = scores.get(name)
                    if (value is None or value.shape != (self.weights[name].shape[0],)
                            or not torch.isfinite(value).all() or (value < 0).any()):
                        raise SelectionUnavailableError(f"fixed_budget:{self.config.selector}:invalid_ranking_scores:{name}")
                if name in scores:
                    for channel, score in enumerate(scores[name].detach().cpu().tolist()):
                        if math.isfinite(score) and (score >= 0 if fixed else score > 0):
                            candidates.append((score, self.layout[name][0], channel, name))
            candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
            if fixed and len(candidates) < budget:
                raise SelectionUnavailableError(f"fixed_budget:{self.config.selector}:insufficient_candidates:{layer}")
            for _, _, channel, name in candidates[:budget]:
                selected[name].append(channel)
        masks = {}
        for name, parameter in self.weights.items():
            mask = torch.zeros_like(parameter)
            if selected[name]:
                mask[selected[name], :] = 1
            masks[name] = mask
        return masks, {name: tuple(channels) for name, channels in selected.items() if channels}

    def release(self, time: int, targets: Mapping[int, Tensor], *,
                current_x: Tensor | None = None) -> UpdateRecord:
        """Apply one release atomically; failed events retain pending forecasts."""
        batch = self.prepare_release(time, targets)
        if current_x is not None:
            current_x = self._input(current_x)
        if batch is None:
            return self._release_event(time, targets, current_x=current_x)
        mutable = dict(self.update_parameters) if self.config.stage_ii else {}
        if self.config.stage_i:
            mutable["router.bias"] = self.router_bias
        originals = {name: parameter.detach().clone() for name, parameter in mutable.items()}
        may_refresh = (self.config.stage_ii
                       and (self.update_occasions + 1) % self.config.reference_interval == 0
                       and len(self.history) >= self.config.reference_windows)
        reference_originals = copy.deepcopy(self.reference.state_dict()) if (may_refresh or self.config.reference_mode == "current") else None
        bookkeeping = (self._clock, self.update_occasions, self.reference_version,
                       self.reference_windows, self.reference_maps, self.reference_channels,
                       self._squared_error, self._absolute_error, self._evaluated_values)
        pending_before, records_before = self.pending.copy(), list(self.records)
        try:
            return self._release_event(time, targets, current_x=current_x)
        except Exception:
            with torch.no_grad():
                for name, original in originals.items():
                    mutable[name].copy_(original)
                if reference_originals is not None:
                    self.reference.load_state_dict(reference_originals)
            (self._clock, self.update_occasions, self.reference_version,
             self.reference_windows, self.reference_maps, self.reference_channels,
             self._squared_error, self._absolute_error, self._evaluated_values) = bookkeeping
            self.pending.clear()
            self.pending.update(pending_before)
            self.records.clear()
            self.records.extend(records_before)
            raise

    def _release_event(self, time: int, targets: Mapping[int, Tensor], *,
                       current_x: Tensor | None = None) -> UpdateRecord:
        batch = self.prepare_release(time, targets)
        self._clock = time
        stability: tuple[Optional[float], ...] = (None,) * self.n_experts
        if batch is None:
            record = UpdateRecord(time, (), None, StageUpdate("I", "no_released_targets"),
                                  self._idle_expert_update("no_released_targets"), stability,
                                  self.reference_version)
            self.records.append(record)
            return record
        self.model.eval()
        gains = self.modulations(batch.inputs)
        stored_error = float((batch.stored_predictions - batch.targets).double().square().mean())
        routing = (self._step("I", "released_feedback", batch, gains, [self.router_bias],
                              self.config.learning_rate_i, self.config.radius_i)
                   if self.config.stage_i else StageUpdate("I", "disabled"))
        experts = self._idle_expert_update("disabled")
        if self.config.stage_ii:
            loss = self._loss(batch, gains)
            parameters = list(self.update_parameters.values())
            gradients = self._gradients(loss, parameters)
            selection_event_seed = None
            layer_budgets = {}
            if self.config.stage_ii_scope != "sparse_experts":
                reason = "dense_update:" + self.config.stage_ii_scope
                masks, selected = None, {}
                if self.config.stage_ii_scope == "dense_eligible":
                    reason += ":ranking_not_used"
                    layer_budgets = self._layer_budget_record({}, dense=True)
            else:
                if self.config.selection_protocol == "fixed_budget":
                    reason = "fixed_budget:" + self.config.selector
                    if self.config.selector == "random":
                        scores, selection_event_seed = self._random_scores(time)
                    elif self.config.selector == "gradient_only":
                        scores = {name: torch.linalg.vector_norm(gradient.double(), dim=-1)
                                  for name, gradient in zip(self.weights, gradients)}
                    else:
                        scores, stability, _ = self._relevance_scores(current_x)
                elif self.config.selector == "gradient_only" or stored_error > self.config.error_threshold:
                    reason = "gradient_only" if self.config.selector == "gradient_only" else "stored_error_threshold"
                    scores = {name: torch.linalg.vector_norm(gradient.double(), dim=-1)
                              for name, gradient in zip(self.weights, gradients)}
                else:
                    scores, stability, reason = self._relevance_scores(current_x)
                masks, selected = self._select(scores)
                layer_budgets = self._layer_budget_record(selected)
                if self.config.selection_protocol == "fixed_budget" and any(
                        bool((scores[name][list(channels)] == 0).any()) for name, channels in selected.items()):
                    reason += ":zero_score_ties"
            experts = self._step("II", reason, batch, gains, parameters,
                                 self.config.learning_rate_ii, self.config.radius_ii,
                                 gradients=gradients, masks=list(masks.values()) if masks is not None else None,
                                 selected=selected, baseline=loss, layer_budgets=layer_budgets,
                                 selection_event_seed=selection_event_seed)
        # Record metrics against the immutable forecasts, never recomputed ones.
        error = (batch.stored_predictions - batch.targets).detach().double()
        self._squared_error += float(error.square().sum())
        self._absolute_error += float(error.abs().sum())
        self._evaluated_values += error.numel()
        for issue_time in batch.issue_times:
            del self.pending[issue_time]
        self.update_occasions += 1
        refreshed = (self._refresh_reference() if self.update_occasions % self.config.reference_interval == 0
                     else False)
        record = UpdateRecord(time, batch.issue_times, stored_error, routing, experts,
                              stability, self.reference_version, refreshed)
        self.records.append(record)
        return record

    @property
    def metrics(self) -> dict[str, float | int | None]:
        count = self._evaluated_values
        return {"count": count, "mse": self._squared_error / count if count else None,
                "mae": self._absolute_error / count if count else None}
