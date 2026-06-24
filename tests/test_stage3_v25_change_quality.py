from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v25 import FactorMechanismV25Config, build_factor_mechanism_v25
from train.run_stage3_v25_change_quality_probe import (
    _collect_split_tensors_v25,
    validate_v25_config_contract,
)
from train.online_loop_stage3_v25 import run_online_adaptation_v25


def _tiny_cfg() -> dict:
    return {
        "data": {"lookback": 4, "horizon": 1},
        "model_v25": {"repr_dim": 8},
        "stage3_v2": {
            "protocol_version": "v25_change_quality_probe",
            "log_root": "logs/s3v25expert",
            "checkpoint_root": "ckpt/s3v25expert",
            "results_root": "results/s3v25expert",
        },
        "online": {
            "stage3": {
                "version": "v25_change_quality_probe",
                "signal_mode": "fluctuation_delta",
                "causal_online": True,
                "online_batch_size": 1,
            }
        },
    }


def _tiny_model():
    return build_factor_mechanism_v25(
        FactorMechanismV25Config(
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


def test_validate_v25_config_contract_requires_v25_roots() -> None:
    cfg = _tiny_cfg()
    validate_v25_config_contract(cfg)
    cfg["stage3_v2"]["log_root"] = "logs/other"
    with pytest.raises(ValueError):
        validate_v25_config_contract(cfg)


def test_collect_split_tensors_v25_uses_change_quality_source() -> None:
    torch.manual_seed(13)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(6, 4, 3), torch.randn(6, 1))
    raw, y_base, y = _collect_split_tensors_v25(
        dataset=ds,
        batch_size=2,
        device=torch.device("cpu"),
        model=model,
        max_batches=2,
    )
    assert raw.shape[0] == y_base.shape[0] == y.shape[0] == 4
    assert raw.shape[1] == 36


def test_run_online_adaptation_v25_rejects_non_causal_batch() -> None:
    with pytest.raises(Exception):
        run_online_adaptation_v25(
            base_model=None,
            generator=None,
            stream=[None, None],
            stage3_cfg={"causal_online": True, "online_batch_size": 2},
            device=torch.device("cpu"),
        )


