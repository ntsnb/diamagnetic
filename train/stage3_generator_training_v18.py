from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from train.stage3_generator_training_v17 import (
    build_stage3_v17_generator,
    collect_stage3_v17_risk_gate_examples,
    iter_v17_representation_stream,
    train_stage3_generator_v17,
)


@dataclass(frozen=True)
class Stage3GeneratorCandidateV18:
    epoch: int
    checkpoint_path: Path
    last_train_loss: float | None


def iter_v18_representation_stream(
    dataset,
    batch_size: int,
    device: torch.device,
    model: nn.Module,
    *,
    max_batches: int | None = None,
):
    yield from iter_v17_representation_stream(
        dataset,
        batch_size=batch_size,
        device=device,
        model=model,
        max_batches=max_batches,
    )


def build_stage3_v18_generator(*, cfg: dict[str, Any], model: nn.Module, drift_dim: int, output_shape: tuple[int, ...]) -> nn.Module:
    gen_cfg = cfg.setdefault("stage3_v2", {}).setdefault("generator", {})
    generator_type = str(gen_cfg.get("type", "layer_aware_hyperadapter_v18")).lower()
    alias_map = {
        "layer_aware_hyperadapter_v18": "layer_aware_hyperadapter_v17",
        "hyperadapter_v18": "hyperadapter_v17",
        "layer_aware_hyperlora_v18": "layer_aware_hyperlora_v17",
        "hyperlora_v18": "hyperlora_v17",
        "identity_signed_separable_v18": "identity_signed_separable_v17",
        "residual_basis_hypergenerator_v18": "residual_basis_hypergenerator_v17",
    }
    gen_cfg["type"] = alias_map.get(generator_type, generator_type)
    return build_stage3_v17_generator(cfg=cfg, model=model, drift_dim=drift_dim, output_shape=output_shape)


def train_stage3_generator_v18(
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
    build_stage3_v18_generator(
        cfg=alias_cfg,
        model=model,
        drift_dim=int(alias_cfg.get("invariant", {}).get("repr_dim", 8)),
        output_shape=tuple(train_ds[0][1].shape),
    )
    generator, rows, candidates_v17 = train_stage3_generator_v17(
        cfg=alias_cfg,
        model=model,
        train_ds=train_ds,
        device=device,
        max_batches=max_batches,
        candidate_checkpoint_dir=candidate_checkpoint_dir,
        candidate_metadata=candidate_metadata,
    )
    candidates_v18 = [
        Stage3GeneratorCandidateV18(
            epoch=candidate.epoch,
            checkpoint_path=candidate.checkpoint_path,
            last_train_loss=candidate.last_train_loss,
        )
        for candidate in candidates_v17
    ]
    return generator, rows, candidates_v18


def collect_stage3_v18_risk_gate_examples(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    generator: nn.Module,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
):
    return collect_stage3_v17_risk_gate_examples(
        cfg=cfg,
        model=model,
        generator=generator,
        train_ds=train_ds,
        device=device,
        max_batches=max_batches,
    )
