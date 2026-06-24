from __future__ import annotations

import copy
from typing import Any

from train.stage3_generator_training_v26 import (
    Stage3GeneratorCandidateV26,
    build_stage3_v26_generator,
    collect_stage3_v26_risk_gate_examples,
    iter_v26_representation_stream,
    train_stage3_generator_v26,
)

Stage3GeneratorCandidateV27 = Stage3GeneratorCandidateV26
iter_v27_representation_stream = iter_v26_representation_stream


def _normalize_v27_generator_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(cfg)
    gen_cfg = normalized.setdefault("stage3_v2", {}).setdefault("generator", {})
    generator_type = str(gen_cfg.get("type", "")).lower()
    if generator_type == "layer_aware_hyperadapter_v27":
        gen_cfg["type"] = "layer_aware_hyperadapter_v26"
    elif generator_type == "hyperadapter_v27":
        gen_cfg["type"] = "hyperadapter_v16"
    elif generator_type == "identity_signed_separable_v27":
        gen_cfg["type"] = "identity_signed_separable_v17"
    elif generator_type == "residual_basis_hypergenerator_v27":
        gen_cfg["type"] = "residual_basis_hypergenerator_v17"
    elif generator_type == "layer_aware_hyperlora_v27":
        gen_cfg["type"] = "layer_aware_hyperlora_v17"
    return normalized


def build_stage3_v27_generator(*, cfg: dict[str, Any], model, drift_dim: int, output_shape: tuple[int, ...]):
    return build_stage3_v26_generator(
        cfg=_normalize_v27_generator_cfg(cfg),
        model=model,
        drift_dim=drift_dim,
        output_shape=output_shape,
    )


def train_stage3_generator_v27(**kwargs):
    if "cfg" in kwargs:
        kwargs = dict(kwargs)
        kwargs["cfg"] = _normalize_v27_generator_cfg(kwargs["cfg"])
    return train_stage3_generator_v26(**kwargs)


def collect_stage3_v27_risk_gate_examples(**kwargs):
    if "cfg" in kwargs:
        kwargs = dict(kwargs)
        kwargs["cfg"] = _normalize_v27_generator_cfg(kwargs["cfg"])
    return collect_stage3_v26_risk_gate_examples(**kwargs)
