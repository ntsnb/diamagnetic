from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

from train.stage3_generator_training import load_stage3_generator_checkpoint
from train.stage3_risk_gate_v2 import Stage3RiskGateBundle, load_stage3_risk_gate_checkpoint


@dataclass(frozen=True)
class Stage3V19ArtifactBundle:
    generator: nn.Module
    generator_metadata: dict
    risk_gate: Stage3RiskGateBundle | None = None
    risk_gate_metadata: dict | None = None


def _require_existing(path: str | Path, *, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing Stage III V19 {label}: {resolved}")
    return resolved


def load_stage3_v19_artifact_bundle(
    *,
    generator_checkpoint: str | Path,
    model: nn.Module,
    risk_gate_checkpoint: str | Path | None = None,
) -> Stage3V19ArtifactBundle:
    generator_path = _require_existing(generator_checkpoint, label="generator checkpoint")
    generator, generator_metadata = load_stage3_generator_checkpoint(generator_path, model=model)

    risk_gate = None
    risk_gate_metadata = None
    if risk_gate_checkpoint:
        risk_gate_path = _require_existing(risk_gate_checkpoint, label="risk gate checkpoint")
        risk_gate, risk_gate_metadata = load_stage3_risk_gate_checkpoint(risk_gate_path)

    return Stage3V19ArtifactBundle(
        generator=generator,
        generator_metadata=generator_metadata,
        risk_gate=risk_gate,
        risk_gate_metadata=risk_gate_metadata,
    )
