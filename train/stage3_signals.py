from __future__ import annotations

from dataclasses import dataclass

import torch


SCALAR_STAGE3_MODES = {"level_scalar", "delta_scalar"}
VECTOR_STAGE3_MODES = {"level_vector", "delta_vector"}
V13_COMPAT_VECTOR_MODES = {"level_offset", "fluctuation_delta"}
STAGE3_SIGNAL_MODES = SCALAR_STAGE3_MODES | VECTOR_STAGE3_MODES | V13_COMPAT_VECTOR_MODES


@dataclass
class Stage3SignalState:
    previous_raw_tail: torch.Tensor | None = None


@dataclass
class Stage3SignalResult:
    drift_vec: torch.Tensor
    signal: torch.Tensor
    raw: torch.Tensor
    mode: str
    drift_dim: int
    raw_mean: float
    raw_abs_mean: float
    signal_mean: float
    signal_abs_mean: float
    signal_norm: float
    scalar_ratio: float


def _as_sample_matrix(drift_raw: torch.Tensor) -> torch.Tensor:
    if drift_raw.ndim == 0:
        raise ValueError("stage3 drift signal must have at least one dimension")
    if drift_raw.ndim == 1:
        return drift_raw.unsqueeze(0)
    if drift_raw.ndim == 2:
        return drift_raw
    return drift_raw.flatten(start_dim=1)


def _temporal_difference(
    drift_raw: torch.Tensor,
    state: Stage3SignalState,
    update_state: bool,
) -> torch.Tensor:
    if drift_raw.shape[0] <= 0:
        raise ValueError("stage3 drift signal must contain at least one sample")

    prev_tail = state.previous_raw_tail
    if prev_tail is None:
        prev_rows = torch.cat([drift_raw[:1], drift_raw[:-1]], dim=0)
    else:
        prev_first = prev_tail.to(device=drift_raw.device, dtype=drift_raw.dtype)
        prev_first = prev_first.reshape_as(drift_raw[:1])
        prev_rows = torch.cat([prev_first, drift_raw[:-1]], dim=0)

    if update_state:
        state.previous_raw_tail = drift_raw[-1:].detach().clone()
    return drift_raw - prev_rows


def build_stage3_drift_vector(
    drift_raw: torch.Tensor,
    mode: str,
    bottleneck_dim: int,
    state: Stage3SignalState | None = None,
    update_state: bool = True,
    eps: float = 1e-6,
    normalize_vector: bool = True,
    polarity: float = 1.0,
) -> Stage3SignalResult:
    """Build the Stage III generator input while making scalar/vector semantics explicit.

    Scalar modes intentionally preserve the original implementation contract:
    a normalized global scalar repeated to ``bottleneck_dim``.

    Vector modes preserve embedding-direction information by averaging over the
    batch dimension only. The optional absolute-mean normalization keeps the
    vector scale close to the original scalar gate without collapsing direction.
    """

    if mode not in STAGE3_SIGNAL_MODES:
        raise ValueError(f"unknown stage3 signal mode: {mode}")
    if bottleneck_dim <= 0:
        raise ValueError("bottleneck_dim must be positive")

    if state is None:
        state = Stage3SignalState()

    raw = _as_sample_matrix(drift_raw) * float(polarity)
    if mode.startswith("delta_") or mode == "fluctuation_delta":
        signal = _temporal_difference(raw, state, update_state)
    else:
        if update_state:
            state.previous_raw_tail = raw[-1:].detach().clone()
        signal = raw

    raw_abs_mean_t = raw.abs().mean()
    signal_abs_mean_t = signal.abs().mean()
    signal_mean_t = signal.mean()
    scalar_ratio_t = signal_mean_t / (signal_abs_mean_t + eps)

    if mode in SCALAR_STAGE3_MODES:
        drift_vec = scalar_ratio_t.repeat(bottleneck_dim)
    else:
        drift_vec = signal.mean(dim=0)
        if normalize_vector:
            drift_vec = drift_vec / (signal_abs_mean_t + eps)

    drift_vec = drift_vec.to(device=raw.device, dtype=raw.dtype)
    return Stage3SignalResult(
        drift_vec=drift_vec,
        signal=signal,
        raw=raw,
        mode=mode,
        drift_dim=int(drift_vec.numel()),
        raw_mean=float(raw.mean().detach().cpu().item()),
        raw_abs_mean=float(raw_abs_mean_t.detach().cpu().item()),
        signal_mean=float(signal_mean_t.detach().cpu().item()),
        signal_abs_mean=float(signal_abs_mean_t.detach().cpu().item()),
        signal_norm=float(torch.linalg.vector_norm(signal).detach().cpu().item()),
        scalar_ratio=float(scalar_ratio_t.detach().cpu().item()),
    )
