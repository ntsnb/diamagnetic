from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - old torch fallback
    from torch.nn.utils.stateless import functional_call  # type: ignore


def functional_call_with_param_delta(
    model: nn.Module,
    delta: Mapping[str, torch.Tensor],
    *args: Any,
    **kwargs: Any,
):
    """Run ``model`` with temporary differentiable parameter deltas.

    Unlike the legacy Stage III ``temporary_params`` helper, this function does
    not mutate ``.data`` and does not run under ``no_grad``. That is required
    for training a generator whose outputs are parameter deltas.
    """

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    adapted_params = {}
    for name, param in params.items():
        adapted_params[name] = param + delta[name] if name in delta else param
    try:
        return functional_call(model, (adapted_params, buffers), args, kwargs)
    except TypeError:
        merged = {**adapted_params, **buffers}
        return functional_call(model, merged, args, kwargs)
