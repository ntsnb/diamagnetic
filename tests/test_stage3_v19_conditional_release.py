from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v19 import FactorMechanismV19Config, build_factor_mechanism_v19
from train.online_loop_stage3_v19 import run_online_adaptation_v19
from train.run_stage3_v19_conditional_release_probe import _collect_split_tensors_v19


def test_v19_config_locks_versioned_non_overwriting_roots() -> None:
    text = Path("configs/stage3_v19_conditional_release_probe.yaml").read_text(encoding="utf-8")

    assert "protocol_version: v19_conditional_release_probe" in text
    assert "version: v19_conditional_release_probe" in text
    assert "logs/s3v19expert" in text
    assert "ckpt/s3v19expert" in text
    assert "results/s3v19expert" in text
    assert "history_output_heads" in text
    assert "variant_output_heads" in text


def test_online_loop_v19_rejects_non_causal_batching() -> None:
    torch.manual_seed(19)
    model = build_factor_mechanism_v19(
        FactorMechanismV19Config(
            input_dim=4,
            seq_len=6,
            out_dim=2,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            release_hidden_dim=12,
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
        run_online_adaptation_v19(
            stream=stream,
            model=model,
            generator=None,
            risk_gate=None,
            cfg=cfg,
            device=torch.device("cpu"),
        )


def test_collect_split_tensors_v19_uses_model_native_release_state() -> None:
    torch.manual_seed(23)
    model = build_factor_mechanism_v19(
        FactorMechanismV19Config(
            input_dim=4,
            seq_len=6,
            out_dim=2,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            release_hidden_dim=12,
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
    raw, y_base, y_true = _collect_split_tensors_v19(
        dataset=TensorDataset(x, y),
        batch_size=1,
        device=torch.device("cpu"),
        model=model,
        max_batches=None,
    )

    assert raw.shape == (6, 8)
    assert y_base.shape == (6, 2)
    assert y_true.shape == (6, 2)
