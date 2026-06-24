from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.adaptation_v10 import ResidualBasisHyperGeneratorV10
from models.stage3_alignment_v12 import TrainOnlyOutcomeAlignedTransportAlignerV12
from models.stage3_residual_teacher_v12 import (
    TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    save_stage3_v12_teacher_checkpoint,
)
from train.stage3_generator_training_v10 import save_stage3_generator_checkpoint
from train.stage3_v12_artifacts import load_stage3_v12_artifact_bundle


def test_v12_artifact_bundle_loads_v12_alignment_and_teacher(tmp_path: Path):
    model = nn.Linear(3, 2, bias=True)
    generator = ResidualBasisHyperGeneratorV10(
        named_param_shapes={"weight": tuple(model.weight.shape), "bias": tuple(model.bias.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        output_shape=(2,),
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    save_stage3_generator_checkpoint(path=generator_path, generator=generator, metadata={"run_id": "v12_bundle"})

    sample_count = 96
    phi_i = torch.randn(sample_count, 3)
    t = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    phi_v = phi_i + torch.cat([t, -0.5 * t, torch.zeros_like(t)], dim=1)
    y_base = torch.zeros(sample_count, 2)
    y = y_base + torch.cat([0.5 * t, -0.25 * t], dim=1)
    aligner = TrainOnlyOutcomeAlignedTransportAlignerV12(
        anchor_rank=3,
        max_rank=2,
        max_leakage_rank=1,
        min_calibration_r2=0.0,
        min_block_transfer_r2=0.0,
        min_block_sign_agreement=0.0,
        bootstrap_samples=0,
        reliability_gate_enabled=False,
    ).fit(phi_i_train=phi_i, phi_v_train=phi_v, y_base_train=y_base, y_train=y, split="train")
    aligner_path = tmp_path / "aligner_v12.pt"
    torch.save(aligner.state_dict(), aligner_path)

    teacher = TrainOnlyOutcomeAlignedBlockResidualTeacherV12(
        drift_dim=3,
        output_shape=(2,),
        ridge=1e-5,
        reliability_gate_enabled=False,
    ).fit(drift_train=aligner.transform_phi_pair(phi_i, phi_v), y_base_train=y_base, y_train=y, split="train")
    teacher_path = tmp_path / "teacher_v12.pt"
    save_stage3_v12_teacher_checkpoint(path=teacher_path, teacher=teacher, metadata={"run_id": "v12_bundle"})

    bundle = load_stage3_v12_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
        residual_teacher_checkpoint=teacher_path,
    )

    assert isinstance(bundle.generator, ResidualBasisHyperGeneratorV10)
    assert isinstance(bundle.residual_canonicalizer, TrainOnlyOutcomeAlignedTransportAlignerV12)
    assert isinstance(bundle.residual_teacher, TrainOnlyOutcomeAlignedBlockResidualTeacherV12)
