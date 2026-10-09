"""Observation-by-observation feedback release without access to future targets."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import Tensor

from .adaptation import ForecastRecord, OnlineController, ReleasedBatch, UpdateRecord


class DelayedFeedbackStream:
    """Collect only observed single-step targets and release complete horizons.

    At observation index ``t``, ``step(t, x_t, y_t)`` first ingests the already
    observed ``y_t``, releases eligible *older* horizons, and then issues the
    forecast for ``t+1, ..., t+H``.  ``extra_delay`` postpones release beyond
    ``t+H``; no partial horizon is exposed to the controller.

    For offline generator replay, ``before_update`` receives a validated released
    batch before persistent updates.  It must not use targets outside that batch.
    """

    def __init__(self, controller: OnlineController, *, observation_capacity: Optional[int] = None,
                 before_update: Callable[[ReleasedBatch, int], None] | None = None, allow_missing: bool = False):
        self.controller = controller
        minimum = controller.horizon + controller.config.extra_delay + 1
        self.observation_capacity = observation_capacity or minimum
        if not isinstance(self.observation_capacity, int) or self.observation_capacity < minimum:
            raise ValueError("Observation buffer must cover horizon + delay + 1")
        self.before_update = before_update
        self.allow_missing = allow_missing
        self.skipped_forecasts = []
        self.observations: OrderedDict[int, Tensor] = OrderedDict()
        self.last_observation: Optional[int] = None

    def reset(self, *, reset_controller: bool = True):
        self.observations.clear()
        self.last_observation = None
        self.skipped_forecasts.clear()
        if reset_controller:
            self.controller.reset()

    def observe(self, time: int, current_target: Tensor, *, current_x: Tensor | None = None) -> UpdateRecord:
        self.controller._time(time)
        if self.last_observation is not None and time != self.last_observation + 1:
            raise ValueError("Single-step observations must be consecutive and strictly increasing")
        value = torch.as_tensor(current_target).detach().cpu()
        if value.ndim == 0 and self.controller.n_targets == 1:
            value = value.reshape(1)
        if value.shape != (self.controller.n_targets,) or torch.isinf(value).any() or (
                not self.allow_missing and not torch.isfinite(value).all()):
            raise ValueError("current_target must contain the currently observed target vector")
        self.observations[time] = value.clone()
        try:
            targets = {}
            skipped = []
            for record in self.controller.due_forecasts(time):
                indices = range(record.issue_time + 1, record.issue_time + self.controller.horizon + 1)
                missing = [index for index in indices if index not in self.observations]
                if missing:
                    raise ValueError(f"Missing already observed targets for forecast {record.issue_time}: {missing}")
                label = torch.stack([self.observations[index] for index in indices])
                if torch.isfinite(label).all():
                    targets[record.issue_time] = label
                else:
                    skipped.append(record.issue_time)
            if self.before_update is not None and targets:
                batch = self.controller.prepare_release(time, targets)
                self.before_update(batch, time)
            record = self.controller.release(time, targets, current_x=current_x)
            self.controller.discard_unobserved(time, skipped)
            self.skipped_forecasts.extend({"issue_time": issue, "time": time, "reason": "incomplete_observed_horizon"} for issue in skipped)
        except Exception:
            # Do not consume an observation if the release event failed.
            del self.observations[time]
            raise
        self.last_observation = time
        while len(self.observations) > self.observation_capacity:
            self.observations.popitem(last=False)
        return record

    def step(self, time: int, x: Tensor, current_target: Tensor) -> tuple[ForecastRecord, UpdateRecord]:
        update = self.observe(time, current_target, current_x=x)
        forecast = self.controller.predict(x, time)
        return forecast, update
