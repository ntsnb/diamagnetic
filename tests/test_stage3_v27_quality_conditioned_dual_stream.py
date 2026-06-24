from __future__ import annotations

import os
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v27 import FactorMechanismV27Config, build_factor_mechanism_v27
from train.online_loop_stage3_v27 import run_online_adaptation_v27
from train.run_stage3_v27_quality_conditioned_dual_stream_probe import (
    _collect_split_tensors_v27,
    validate_v27_config_contract,
)


def _tiny_cfg() -> dict:
    return {
        "data": {"lookback": 4, "horizon": 1},
        "model_v27": {"repr_dim": 8},
        "stage3_v2": {
            "protocol_version": "v27_quality_conditioned_dual_stream_probe",
            "log_root": "logs/s3v27expert",
            "checkpoint_root": "ckpt/s3v27expert",
            "results_root": "results/s3v27expert",
        },
        "online": {
            "stage3": {
                "version": "v27_quality_conditioned_dual_stream_probe",
                "signal_mode": "fluctuation_delta",
                "causal_online": True,
                "online_batch_size": 1,
            }
        },
    }


def _tiny_model():
    return build_factor_mechanism_v27(
        FactorMechanismV27Config(
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
            change_memory_hidden_dim=12,
            env_count=3,
            router_env_hidden_dim=8,
            quality_hidden_dim=12,
            router_quality_hidden_dim=8,
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


def test_validate_v27_config_contract_requires_v27_roots() -> None:
    cfg = _tiny_cfg()
    validate_v27_config_contract(cfg)
    cfg["stage3_v2"]["log_root"] = "logs/other"
    with pytest.raises(ValueError):
        validate_v27_config_contract(cfg)


def test_collect_split_tensors_v27_uses_quality_conditioned_drift_source() -> None:
    torch.manual_seed(13)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(6, 4, 3), torch.randn(6, 1))
    raw, y_base, y = _collect_split_tensors_v27(
        dataset=ds,
        batch_size=2,
        device=torch.device("cpu"),
        model=model,
        max_batches=2,
    )
    assert raw.shape[0] == y_base.shape[0] == y.shape[0] == 4
    assert raw.shape[1] == 38


def test_run_online_adaptation_v27_rejects_non_causal_batch() -> None:
    with pytest.raises(Exception):
        run_online_adaptation_v27(
            base_model=None,
            generator=None,
            stream=[None, None],
            stage3_cfg={"causal_online": True, "online_batch_size": 2},
            device=torch.device("cpu"),
        )
