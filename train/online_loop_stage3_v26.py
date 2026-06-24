from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

import train.online_loop_stage3_v16 as v16_online


def _extract_v26_representations(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(output, "pooled_servable_change"):
        observable = output.pooled_servable_change
        return torch.zeros_like(observable), observable
    return v16_online._extract_v16_representations(output)


@contextmanager
def _patched_v16_extract():
    original_extract = v16_online._extract_v16_representations
    v16_online._extract_v16_representations = _extract_v26_representations
    try:
        yield
    finally:
        v16_online._extract_v16_representations = original_extract


def run_online_adaptation_v26(*args, **kwargs):
    with _patched_v16_extract():
        return v16_online.run_online_adaptation_v16(*args, **kwargs)


