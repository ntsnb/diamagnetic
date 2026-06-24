from __future__ import annotations

import torch

from train.stage3_risk_gate_v2 import (
    STAGE3_RISK_FEATURE_NAMES,
    Stage3RiskGateLogistic,
    Stage3RiskGateMLP,
    Stage3RiskFeatureStandardizer,
    load_stage3_risk_gate_checkpoint,
    save_stage3_risk_gate_checkpoint,
    train_stage3_risk_gate,
)


def test_stage3_risk_feature_standardizer_is_train_only_and_serializable():
    torch.manual_seed(71)
    train_features = torch.randn(32, len(STAGE3_RISK_FEATURE_NAMES)) * 2.0 + 5.0
    test_features = torch.randn(4, len(STAGE3_RISK_FEATURE_NAMES)) - 3.0

    standardizer = Stage3RiskFeatureStandardizer().fit(train_features)
    restored = Stage3RiskFeatureStandardizer().load_state_dict(standardizer.state_dict())

    standardized_train = restored.transform(train_features)
    standardized_test = restored.transform(test_features)

    assert restored.fit_sample_count == 32
    assert torch.allclose(standardized_train.mean(dim=0), torch.zeros_like(standardized_train.mean(dim=0)), atol=1e-5)
    assert standardized_test.shape == test_features.shape


def test_stage3_risk_gate_learns_separable_acceptance_labels():
    torch.manual_seed(72)
    n_features = len(STAGE3_RISK_FEATURE_NAMES)
    positive = torch.randn(32, n_features) * 0.1
    positive[:, 4] = 0.05
    negative = torch.randn(32, n_features) * 0.1
    negative[:, 4] = 2.0
    features = torch.cat([positive, negative], dim=0)
    labels = torch.cat([torch.ones(32), torch.zeros(32)], dim=0)

    gate, standardizer, metrics = train_stage3_risk_gate(
        features,
        labels,
        hidden_dim=16,
        epochs=120,
        lr=0.03,
    )
    scores = gate.predict_proba(standardizer.transform(features)).detach()

    assert isinstance(gate, Stage3RiskGateMLP)
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["positive_recall"] > 0.9
    assert metrics["negative_recall"] > 0.9
    assert scores[:32].mean().item() > 0.8
    assert scores[32:].mean().item() < 0.2


def test_stage3_logistic_risk_gate_learns_and_round_trips(tmp_path):
    torch.manual_seed(73)
    n_features = len(STAGE3_RISK_FEATURE_NAMES)
    positive = torch.randn(32, n_features) * 0.1
    positive[:, 4] = -1.0
    negative = torch.randn(32, n_features) * 0.1
    negative[:, 4] = 1.0
    features = torch.cat([positive, negative], dim=0)
    labels = torch.cat([torch.ones(32), torch.zeros(32)], dim=0)

    gate, standardizer, metrics = train_stage3_risk_gate(
        features,
        labels,
        model_type="logistic",
        epochs=180,
        lr=0.08,
    )
    path = tmp_path / "risk_gate.pt"
    save_stage3_risk_gate_checkpoint(
        path=path,
        gate=gate,
        standardizer=standardizer,
        metadata={"run_id": "logistic_roundtrip"},
    )
    loaded, metadata = load_stage3_risk_gate_checkpoint(path)
    scores = loaded(features).detach()

    assert isinstance(gate, Stage3RiskGateLogistic)
    assert metadata["run_id"] == "logistic_roundtrip"
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert scores[:32].mean().item() > 0.75
    assert scores[32:].mean().item() < 0.25
