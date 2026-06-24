from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v17 import FactorMechanismV17Config, build_factor_mechanism_v17
from train.online_loop_stage3_v17 import run_online_adaptation_v17
from train.run_stage3_v17_causal_env_dual_stream_probe import (
    _collect_split_tensors_v17,
    validate_v17_config_contract,
)


def test_v17_config_locks_versioned_non_overwriting_roots() -> None:
    text = Path("configs/stage3_v17_causal_env_dual_stream_probe.yaml").read_text(encoding="utf-8")

    assert "protocol_version: v17_causal_env_dual_stream_probe" in text
    assert "version: v17_causal_env_dual_stream_probe" in text
    assert "drift_source: delta_phi_v" in text
    assert "signal_mode: fluctuation_delta" in text
    assert "logs/s3v17expert" in text
    assert "ckpt/s3v17expert" in text
    assert "results/s3v17expert" in text


def test_v17_config_validator_rejects_wrong_roots() -> None:
    cfg = {
        "stage3_v2": {"protocol_version": "v17_causal_env_dual_stream_probe", "log_root": "logs/s3v16expert"},
        "online": {"stage3": {"version": "v17_causal_env_dual_stream_probe", "drift_source": "delta_phi_v", "signal_mode": "fluctuation_delta"}},
    }
    with pytest.raises(ValueError, match="s3v17expert"):
        validate_v17_config_contract(cfg)


def test_collect_split_tensors_v17_uses_model_native_delta() -> None:
    torch.manual_seed(17)
    model = build_factor_mechanism_v17(
        FactorMechanismV17Config(
            input_dim=4,
            seq_len=6,
            out_dim=2,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            env_count=2,
            temporal_encoder="transformer",
            nhead=4,
            num_layers=1,
            dropout=0.0,
            use_vae=False,
        )
    ).eval()
    x = torch.randn(6, 6, 4)
    y = torch.randn(6, 2)
    raw, y_base, y_true = _collect_split_tensors_v17(
        dataset=TensorDataset(x, y),
        batch_size=1,
        device=torch.device("cpu"),
        model=model,
        max_batches=None,
    )

    assert raw.shape == (6, 8)
    assert y_base.shape == (6, 2)
    assert y_true.shape == (6, 2)


def test_online_loop_v17_rejects_non_causal_batching() -> None:
    torch.manual_seed(19)
    model = build_factor_mechanism_v17(
        FactorMechanismV17Config(
            input_dim=4,
            seq_len=6,
            out_dim=2,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            env_count=2,
            temporal_encoder="transformer",
            nhead=4,
            num_layers=1,
            dropout=0.0,
            use_vae=False,
        )
    ).eval()
    stream = [(torch.randn(2, 6, 4), torch.randn(2, 2))]
    cfg = {
        "online": {
            "stage3": {
                "causal_online": True,
                "allow_untrained_generator": True,
                "signal_mode": "delta_vector",
                "normalize_vector": True,
                "drift_polarity": 1.0,
                "max_delta": 0.01,
                "max_pred_shift": 0.1,
                "blend_alpha": 1.0,
                "target_names": ["weight"],
                "target_modules": ["variant_output_heads"],
            }
        }
    }

    with pytest.raises(ValueError, match="batch_size=1"):
        run_online_adaptation_v17(
            stream=stream,
            model=model,
            generator=None,
            risk_gate=None,
            cfg=cfg,
            device=torch.device("cpu"),
        )
