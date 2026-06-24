from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import torch
import torch.nn as nn

import train.stage3_generator_training_v17 as v17_training
from train.stage3_generator_training_v17 import (
    build_stage3_v17_generator,
    collect_stage3_v17_risk_gate_examples,
    train_stage3_generator_v17,
)


@dataclass(frozen=True)
class Stage3GeneratorCandidateV25:
    epoch: int
    checkpoint_path: Path
    last_train_loss: float | None


def _extract_v25_change_quality(output: Any) -> torch.Tensor:
    if hasattr(output, "pooled_servable_change"):
        return output.pooled_servable_change.detach()
    if hasattr(output, "pooled_phi_i") and hasattr(output, "pooled_phi_v"):
        return (output.pooled_phi_v - output.pooled_phi_i).detach()
    raise ValueError("V25 generator path requires model outputs to expose pooled_servable_change or pooled_phi_i/pooled_phi_v")


def iter_v25_representation_stream(
    dataset,
    batch_size: int,
    device: torch.device,
    model: nn.Module,
    *,
    max_batches: int | None = None,
):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model.eval()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x)
        observable = _extract_v25_change_quality(out)
        phi_zero = torch.zeros_like(observable)
        yield x, y, phi_zero, observable


def _infer_v25_drift_dim(*, model: nn.Module, dataset, device: torch.device) -> int:
    sample_x, _ = dataset[0]
    x = sample_x.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return int(_extract_v25_change_quality(out).shape[-1])


@contextmanager
def _patched_v17_iter():
    original_iter = v17_training.iter_v17_representation_stream
    v17_training.iter_v17_representation_stream = iter_v25_representation_stream
    try:
        yield
    finally:
        v17_training.iter_v17_representation_stream = original_iter


def build_stage3_v25_generator(*, cfg: dict[str, Any], model: nn.Module, drift_dim: int, output_shape: tuple[int, ...]) -> nn.Module:
    gen_cfg = cfg.setdefault("stage3_v2", {}).setdefault("generator", {})
    generator_type = str(gen_cfg.get("type", "layer_aware_hyperadapter_v25")).lower()
    alias_map = {
        "layer_aware_hyperadapter_v25": "layer_aware_hyperadapter_v17",
        "hyperadapter_v25": "hyperadapter_v17",
        "layer_aware_hyperlora_v25": "layer_aware_hyperlora_v17",
        "hyperlora_v25": "hyperlora_v17",
        "identity_signed_separable_v25": "identity_signed_separable_v17",
        "residual_basis_hypergenerator_v25": "residual_basis_hypergenerator_v17",
    }
    gen_cfg["type"] = alias_map.get(generator_type, generator_type)
    return build_stage3_v17_generator(cfg=cfg, model=model, drift_dim=drift_dim, output_shape=output_shape)


def train_stage3_generator_v25(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
    candidate_checkpoint_dir: Path | None = None,
    candidate_metadata: dict[str, Any] | None = None,
):
    alias_cfg = deepcopy(cfg)
    inferred_drift_dim = _infer_v25_drift_dim(model=model, dataset=train_ds, device=device)
    alias_cfg.setdefault("invariant", {})["repr_dim"] = inferred_drift_dim
    build_stage3_v25_generator(
        cfg=alias_cfg,
        model=model,
        drift_dim=inferred_drift_dim,
        output_shape=tuple(train_ds[0][1].shape),
    )
    with _patched_v17_iter():
        generator, rows, candidates_v17 = train_stage3_generator_v17(
            cfg=alias_cfg,
            model=model,
            train_ds=train_ds,
            device=device,
            max_batches=max_batches,
            candidate_checkpoint_dir=candidate_checkpoint_dir,
            candidate_metadata=candidate_metadata,
        )
    candidates_v25 = [
        Stage3GeneratorCandidateV25(
            epoch=candidate.epoch,
            checkpoint_path=candidate.checkpoint_path,
            last_train_loss=candidate.last_train_loss,
        )
        for candidate in candidates_v17
    ]
    return generator, rows, candidates_v25


def collect_stage3_v25_risk_gate_examples(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    generator: nn.Module,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
):
    alias_cfg = deepcopy(cfg)
    alias_cfg.setdefault("invariant", {})["repr_dim"] = _infer_v25_drift_dim(model=model, dataset=train_ds, device=device)
    with _patched_v17_iter():
        return collect_stage3_v17_risk_gate_examples(
            cfg=alias_cfg,
            model=model,
            generator=generator,
            train_ds=train_ds,
            device=device,
            max_batches=max_batches,
        )


