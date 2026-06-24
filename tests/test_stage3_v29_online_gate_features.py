from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v29 import FactorMechanismV29Config, build_factor_mechanism_v29
from train.stage3_v29_online_gate_features import build_online_gate_dynamic_feature_row


def _small_cfg() -> FactorMechanismV29Config:
    return FactorMechanismV29Config(
        input_dim=4,
        seq_len=6,
        out_dim=2,
        d_model=24,
        n_experts=3,
        latent_dim=10,
        repr_dim=12,
        expert_hidden_dim=18,
        history_hidden_dim=14,
        release_hidden_dim=14,
        change_memory_hidden_dim=14,
        env_count=3,
        router_env_hidden_dim=12,
        invariant_release_hidden_dim=16,
        router_invariant_hidden_dim=12,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode="mean",
    )


def test_v29_online_gate_dynamic_features_use_only_forward_outputs() -> None:
    torch.manual_seed(41)
    model = build_factor_mechanism_v29(_small_cfg()).eval()
    x = torch.randn(5, 6, 4)

    with torch.no_grad():
        out = model(x)
    row = build_online_gate_dynamic_feature_row(
        out,
        split="validation",
        batch_index=2,
        sample_start=24,
        sample_end=29,
        validation_window_id="2",
        target_name="raw_residual_vector",
    )

    assert row["target_name"] == "raw_residual_vector"
    assert row["validation_window_id"] == "2"
    assert row["feature_dynamic_status"] == "available_from_model_forward"
    assert row["feature_dynamic_sample_count"] == 5
    assert row["feature_dynamic_alpha_entropy_mean"] >= 0.0
    assert row["feature_dynamic_alpha_top1_mean"] <= 1.0
    assert row["feature_dynamic_release_confidence_mean"] >= 0.0
    assert row["feature_dynamic_release_confidence_mean"] <= 1.0
    assert row["feature_dynamic_servable_change_norm_mean"] >= 0.0
    banned_tokens = ("gain", "counterfactual", "label", "partial", "teacher_real", "outcome", "best", "target_y")
    leaking_feature_columns = [
        key
        for key in row
        if key.startswith("feature_") and any(token in key for token in banned_tokens)
    ]
    assert leaking_feature_columns == []
