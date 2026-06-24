from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v20 import FactorMechanismV20Config, build_factor_mechanism_v20
from train.stage3_generator_training import save_stage3_generator_checkpoint
from train.stage3_generator_training_v20 import (
    build_stage3_v20_generator,
    collect_stage3_v20_risk_gate_examples,
    train_stage3_generator_v20,
)
from train.stage3_risk_gate_v2 import save_stage3_risk_gate_checkpoint, train_stage3_risk_gate
from train.stage3_v20_artifacts import load_stage3_v20_artifact_bundle


def _tiny_cfg() -> dict:
    return {
        "invariant": {"repr_dim": 8},
        "stage3_v2": {
            "online_batch_size": 1,
            "generator": {
                "enabled": True,
                "type": "layer_aware_hyperadapter_v20",
                "hidden_dim": 16,
                "rank": 2,
                "max_scale": 0.1,
                "epochs": 2,
                "lr": 1e-2,
                "lambda_delta": 0.0,
                "lambda_zero": 0.0,
                "lambda_shift": 0.0,
                "lambda_improvement": 0.0,
                "lambda_intervention_compare": 0.0,
                "target_names": ["weight", "bias"],
                "include_random_matched_norm_negative": True,
                "amplified_negative_factors": [2.0],
            },
            "risk_gate": {
                "model_type": "logistic",
                "epochs": 20,
                "lr": 1e-2,
                "include_counterfactual_examples": True,
                "counterfactual_modes": ["zero", "sign_flipped"],
                "counterfactual_label": 0.0,
            },
        },
        "online": {
            "stage3": {
                "signal_mode": "fluctuation_delta",
                "normalize_vector": True,
                "drift_polarity": 1.0,
                "drift_bottleneck": 8,
                "target_modules": ["history_output_heads", "variant_output_heads"],
                "target_names": ["weight", "bias"],
                "max_delta": 0.05,
                "blend_alpha": 0.35,
                "max_pred_shift": 0.25,
                "risk_gate_enable": True,
                "risk_gate_threshold": 0.5,
            }
        },
    }


def _tiny_model() -> torch.nn.Module:
    return build_factor_mechanism_v20(
        FactorMechanismV20Config(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            history_hidden_dim=12,
            release_hidden_dim=12,
            env_count=3,
            router_env_hidden_dim=8,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_revin=False,
            use_vae=True,
            vae_eval_mode="mean",
            zero_init_variant=True,
            use_projected_residual=True,
        )
    )


def _tiny_dataset(sample_count: int = 6) -> TensorDataset:
    x = torch.randn(sample_count, 4, 3)
    y = torch.randn(sample_count, 1)
    return TensorDataset(x, y)


def test_build_stage3_v20_generator_targets_history_and_variant_heads():
    cfg = _tiny_cfg()
    model = _tiny_model()
    gen = build_stage3_v20_generator(cfg=cfg, model=model, drift_dim=8, output_shape=(1,))
    assert gen.registered_param_names
    assert all(name.startswith(("history_output_heads", "variant_output_heads")) for name in gen.registered_param_names)


def test_train_stage3_generator_v20_writes_candidates(tmp_path: Path):
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    model = _tiny_model()
    train_ds = _tiny_dataset()
    gen, rows, candidates = train_stage3_generator_v20(
        cfg=cfg,
        model=model,
        train_ds=train_ds,
        device=torch.device("cpu"),
        max_batches=3,
        candidate_checkpoint_dir=tmp_path / "candidates",
        candidate_metadata={"test": "v20"},
    )
    assert rows
    assert len(candidates) == cfg["stage3_v2"]["generator"]["epochs"]
    assert candidates[0].checkpoint_path.exists()
    assert gen.training is False


def test_collect_stage3_v20_risk_gate_examples_returns_features_and_labels():
    torch.manual_seed(8)
    cfg = _tiny_cfg()
    model = _tiny_model()
    train_ds = _tiny_dataset()
    generator = build_stage3_v20_generator(cfg=cfg, model=model, drift_dim=8, output_shape=(1,))
    features, labels, rows = collect_stage3_v20_risk_gate_examples(
        cfg=cfg,
        model=model,
        generator=generator,
        train_ds=train_ds,
        device=torch.device("cpu"),
        max_batches=2,
    )
    assert features.ndim == 2
    assert labels.ndim == 1
    assert features.shape[0] == labels.shape[0]
    assert rows


def test_load_stage3_v20_artifact_bundle_round_trip(tmp_path: Path):
    torch.manual_seed(9)
    cfg = _tiny_cfg()
    model = _tiny_model()
    generator = build_stage3_v20_generator(cfg=cfg, model=model, drift_dim=8, output_shape=(1,))
    generator_ckpt = tmp_path / "generator.pt"
    save_stage3_generator_checkpoint(
        path=generator_ckpt,
        generator=generator,
        metadata={"protocol_version": "v20_corrective_budgeted_probe"},
        overwrite=True,
    )

    features = torch.randn(6, 9)
    labels = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    gate, standardizer, _ = train_stage3_risk_gate(features, labels, model_type="logistic", epochs=10, lr=1e-2)
    risk_gate_ckpt = tmp_path / "risk_gate.pt"
    save_stage3_risk_gate_checkpoint(
        path=risk_gate_ckpt,
        gate=gate,
        standardizer=standardizer,
        metadata={"protocol_version": "v20_corrective_budgeted_probe"},
        overwrite=True,
    )

    bundle = load_stage3_v20_artifact_bundle(
        generator_checkpoint=generator_ckpt,
        model=model,
        risk_gate_checkpoint=risk_gate_ckpt,
    )
    assert bundle.generator is not None
    assert bundle.risk_gate is not None
    assert bundle.generator_metadata["protocol_version"] == "v20_corrective_budgeted_probe"
