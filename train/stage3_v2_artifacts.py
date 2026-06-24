from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from models.aligned_representation_v2 import (
    TrainSetDeltaStandardizer,
    TrainSetResidualDriftCanonicalizer,
    TrainSetTemporalResidualCCAAligner,
    TrainSetWhitenedProcrustesTemporalResidualCCAAligner,
)
from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner
from models.stage3_alignment_v9 import TrainSetTeacherResidualizedReducedRankCCAAligner
from models.stage3_residual_teacher_v9 import TrainOnlyResidualTeacherV9, load_stage3_v9_teacher_checkpoint
from train.stage3_generator_training import load_stage3_generator_checkpoint
from train.stage3_risk_gate_v2 import Stage3RiskGateBundle, load_stage3_risk_gate_checkpoint


@dataclass(frozen=True)
class Stage3V2ArtifactBundle:
    generator: nn.Module
    generator_metadata: dict
    risk_gate: Stage3RiskGateBundle | None = None
    risk_gate_metadata: dict | None = None
    delta_standardizer: TrainSetDeltaStandardizer | None = None
    residual_canonicalizer: (
        TrainSetResidualDriftCanonicalizer
        | TrainSetTemporalResidualCCAAligner
        | TrainSetWhitenedProcrustesTemporalResidualCCAAligner
        | TrainSetChronologicalResidualizedReducedRankCCAAligner
        | TrainSetTeacherResidualizedReducedRankCCAAligner
        | None
    ) = None
    residual_teacher: TrainOnlyResidualTeacherV9 | None = None
    residual_teacher_metadata: dict | None = None


def _require_existing(path: str | Path, *, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing Stage III V2 {label}: {resolved}")
    return resolved


def _load_delta_standardizer(path: str | Path) -> TrainSetDeltaStandardizer:
    checkpoint_path = _require_existing(path, label="delta standardizer checkpoint")
    state = torch.load(checkpoint_path, map_location="cpu")
    return TrainSetDeltaStandardizer().load_state_dict(state)


def _load_residual_canonicalizer(
    path: str | Path,
) -> (
    TrainSetResidualDriftCanonicalizer
    | TrainSetTemporalResidualCCAAligner
    | TrainSetWhitenedProcrustesTemporalResidualCCAAligner
    | TrainSetChronologicalResidualizedReducedRankCCAAligner
    | TrainSetTeacherResidualizedReducedRankCCAAligner
):
    checkpoint_path = _require_existing(path, label="residual drift canonicalizer checkpoint")
    state = torch.load(checkpoint_path, map_location="cpu")
    if state.get("class_name") == "TrainSetTeacherResidualizedReducedRankCCAAligner":
        return TrainSetTeacherResidualizedReducedRankCCAAligner().load_state_dict(state)
    if state.get("class_name") == "TrainSetChronologicalResidualizedReducedRankCCAAligner":
        return TrainSetChronologicalResidualizedReducedRankCCAAligner().load_state_dict(state)
    if state.get("class_name") == "TrainSetWhitenedProcrustesTemporalResidualCCAAligner":
        return TrainSetWhitenedProcrustesTemporalResidualCCAAligner().load_state_dict(state)
    if state.get("class_name") == "TrainSetTemporalResidualCCAAligner":
        return TrainSetTemporalResidualCCAAligner().load_state_dict(state)
    return TrainSetResidualDriftCanonicalizer().load_state_dict(state)


def load_stage3_v2_artifact_bundle(
    *,
    generator_checkpoint: str | Path,
    model: nn.Module,
    risk_gate_checkpoint: str | Path | None = None,
    delta_standardizer_checkpoint: str | Path | None = None,
    residual_canonicalizer_checkpoint: str | Path | None = None,
    residual_teacher_checkpoint: str | Path | None = None,
) -> Stage3V2ArtifactBundle:
    generator_path = _require_existing(generator_checkpoint, label="generator checkpoint")
    generator, generator_metadata = load_stage3_generator_checkpoint(generator_path, model=model)

    risk_gate = None
    risk_gate_metadata = None
    if risk_gate_checkpoint:
        risk_gate_path = _require_existing(risk_gate_checkpoint, label="risk gate checkpoint")
        risk_gate, risk_gate_metadata = load_stage3_risk_gate_checkpoint(risk_gate_path)

    delta_standardizer = None
    if delta_standardizer_checkpoint:
        delta_standardizer = _load_delta_standardizer(delta_standardizer_checkpoint)

    residual_canonicalizer = None
    if residual_canonicalizer_checkpoint:
        residual_canonicalizer = _load_residual_canonicalizer(residual_canonicalizer_checkpoint)

    residual_teacher = None
    residual_teacher_metadata = None
    if residual_teacher_checkpoint:
        residual_teacher_path = _require_existing(residual_teacher_checkpoint, label="V9 residual teacher checkpoint")
        residual_teacher, residual_teacher_metadata = load_stage3_v9_teacher_checkpoint(residual_teacher_path)

    return Stage3V2ArtifactBundle(
        generator=generator,
        generator_metadata=generator_metadata,
        risk_gate=risk_gate,
        risk_gate_metadata=risk_gate_metadata,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
        residual_teacher=residual_teacher,
        residual_teacher_metadata=residual_teacher_metadata,
    )
