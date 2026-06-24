from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.adaptation_v2 import LayerAwareHyperAdapterGenerator, LayerAwareHyperLoRAGenerator
from models.aligned_representation_v2 import (
    TrainSetDeltaStandardizer,
    TrainSetResidualDriftCanonicalizer,
    TrainSetTemporalResidualCCAAligner,
    TrainSetWhitenedProcrustesTemporalResidualCCAAligner,
)
from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner
from train.stage3_generator_training import save_stage3_generator_checkpoint
from train.stage3_risk_gate_v2 import Stage3RiskFeatureStandardizer, Stage3RiskGateLogistic, save_stage3_risk_gate_checkpoint
from train.stage3_v2_artifacts import load_stage3_v2_artifact_bundle


def test_stage3_v2_artifact_bundle_loads_generator_and_delta_standardizer(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    delta_path = tmp_path / "delta_standardizer.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test"},
    )
    standardizer = TrainSetDeltaStandardizer().fit(torch.randn(8, 4))
    torch.save(standardizer.state_dict(), delta_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        delta_standardizer_checkpoint=delta_path,
    )

    assert bundle.generator_metadata["run_id"] == "artifact_test"
    assert isinstance(bundle.generator, LayerAwareHyperLoRAGenerator)
    assert bundle.delta_standardizer is not None
    assert bundle.delta_standardizer.fit_sample_count == 8


def test_stage3_v2_artifact_bundle_loads_residual_canonicalizer(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    canonicalizer_path = tmp_path / "residual_canonicalizer.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test"},
    )
    delta = torch.randn(8, 4)
    y_inv = torch.randn(8, 2)
    y = y_inv + torch.randn(8, 2)
    canonicalizer = TrainSetResidualDriftCanonicalizer().fit(
        delta_phi_v_train=delta,
        y_inv_train=y_inv,
        y_train=y,
    )
    torch.save(canonicalizer.state_dict(), canonicalizer_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=canonicalizer_path,
    )

    assert bundle.residual_canonicalizer is not None
    assert bundle.residual_canonicalizer.fit_sample_count == 8


def test_stage3_v2_artifact_bundle_loads_temporal_residual_cca_aligner(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    aligner_path = tmp_path / "temporal_residual_cca.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test"},
    )
    delta = torch.randn(12, 4)
    y_inv = torch.randn(12, 2)
    y = y_inv + torch.stack([delta[:, 0] - delta[:, 1], delta[:, 2]], dim=1)
    aligner = TrainSetTemporalResidualCCAAligner().fit(
        delta_phi_v_train=delta,
        y_inv_train=y_inv,
        y_train=y,
    )
    torch.save(aligner.state_dict(), aligner_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
    )

    assert isinstance(bundle.residual_canonicalizer, TrainSetTemporalResidualCCAAligner)
    assert bundle.residual_canonicalizer.transition_sample_count == 11


def test_stage3_v2_artifact_bundle_loads_whitened_procrustes_temporal_cca_aligner(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    aligner_path = tmp_path / "v6_aligner.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test"},
    )
    phi_i = torch.randn(12, 4)
    phi_v = phi_i + 0.2 * torch.randn(12, 4)
    y_inv = torch.randn(12, 2)
    y = y_inv + torch.stack([phi_v[:, 0] - phi_i[:, 0], phi_v[:, 1] - phi_i[:, 1]], dim=1)
    aligner = TrainSetWhitenedProcrustesTemporalResidualCCAAligner().fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )
    torch.save(aligner.state_dict(), aligner_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
    )

    assert isinstance(bundle.residual_canonicalizer, TrainSetWhitenedProcrustesTemporalResidualCCAAligner)
    assert bundle.residual_canonicalizer.transition_sample_count == 11


def test_stage3_v2_artifact_bundle_loads_residualized_rrcca_v7_aligner(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator.pt"
    aligner_path = tmp_path / "v7_aligner.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test"},
    )
    torch.manual_seed(707)
    phi_i = torch.randn(24, 4)
    latent = torch.cumsum(torch.randn(24, 2), dim=0)
    signal = torch.zeros(24, 4)
    signal[:, 1] = latent[:, 0]
    signal[:, 3] = -latent[:, 1]
    phi_v = phi_i + signal
    y_inv = torch.zeros(24, 2)
    y = torch.stack([latent[:, 0], latent[:, 1]], dim=1)
    aligner = TrainSetChronologicalResidualizedReducedRankCCAAligner(
        max_rank=2,
        min_fit_correlation=0.01,
        min_calibration_correlation=0.01,
        min_calibration_r2=0.0001,
        bootstrap_samples=8,
        min_bootstrap_sign_stability=0.5,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_inv_train=y_inv,
        y_train=y,
    )
    torch.save(aligner.state_dict(), aligner_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
    )

    assert isinstance(bundle.residual_canonicalizer, TrainSetChronologicalResidualizedReducedRankCCAAligner)
    assert bundle.residual_canonicalizer.active_rank >= 1


def test_stage3_v2_artifact_bundle_loads_v8_hyperadapter_generator(tmp_path):
    model = nn.Linear(3, 2, bias=True)
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes={
            "weight": tuple(model.weight.shape),
            "bias": tuple(model.bias.shape),
        },
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator_v8.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=gen,
        metadata={"run_id": "artifact_test_v8"},
    )

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
    )

    assert bundle.generator_metadata["run_id"] == "artifact_test_v8"
    assert isinstance(bundle.generator, LayerAwareHyperAdapterGenerator)
    assert bundle.generator.registered_param_names == gen.registered_param_names


def test_stage3_v2_artifact_bundle_fails_closed_when_generator_missing(tmp_path):
    model = nn.Linear(3, 2, bias=False)

    with pytest.raises(FileNotFoundError):
        load_stage3_v2_artifact_bundle(
            generator_checkpoint=tmp_path / "missing_generator.pt",
            model=model,
        )


def test_stage3_generator_checkpoint_saver_refuses_existing_file_by_default(tmp_path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    path = tmp_path / "generator.pt"
    save_stage3_generator_checkpoint(path=path, generator=gen, metadata={"run_id": "first"})
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        save_stage3_generator_checkpoint(path=path, generator=gen, metadata={"run_id": "second"})

    assert path.read_bytes() == before


def test_stage3_risk_gate_checkpoint_saver_refuses_existing_file_by_default(tmp_path):
    gate = Stage3RiskGateLogistic(input_dim=2)
    standardizer = Stage3RiskFeatureStandardizer().fit(torch.randn(4, 2))
    path = tmp_path / "risk_gate.pt"
    save_stage3_risk_gate_checkpoint(
        path=path,
        gate=gate,
        standardizer=standardizer,
        metadata={"run_id": "first"},
    )
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        save_stage3_risk_gate_checkpoint(
            path=path,
            gate=gate,
            standardizer=standardizer,
            metadata={"run_id": "second"},
        )

    assert path.read_bytes() == before
