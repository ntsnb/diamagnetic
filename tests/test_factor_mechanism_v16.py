from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v16 import FactorMechanismV16Config, build_factor_mechanism_v16


def _small_cfg(vae_eval_mode: str = "mean") -> FactorMechanismV16Config:
    return FactorMechanismV16Config(
        input_dim=5,
        seq_len=8,
        out_dim=2,
        d_model=32,
        n_experts=3,
        latent_dim=12,
        repr_dim=16,
        expert_hidden_dim=24,
        env_count=3,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode=vae_eval_mode,
    )


def test_v16_forward_exposes_pooled_representations_and_decomposition() -> None:
    torch.manual_seed(5)
    model = build_factor_mechanism_v16(_small_cfg()).eval()
    x = torch.randn(4, 8, 5)

    out = model(x)

    assert out.y_hat.shape == (4, 2)
    assert out.y_inv.shape == (4, 2)
    assert out.y_var.shape == (4, 2)
    assert out.pooled_phi_i.shape == (4, 16)
    assert out.pooled_delta_phi_v.shape == (4, 16)
    assert out.pooled_phi_v.shape == (4, 16)
    assert out.expert_phi_i.shape == (4, 3, 16)
    assert out.expert_delta_phi_v.shape == (4, 3, 16)
    assert out.env_posteriors.shape == (4, 3, 3)
    assert torch.allclose(out.pooled_phi_v, out.pooled_phi_i + out.pooled_delta_phi_v, atol=1e-6)
    assert torch.allclose(out.y_hat, out.y_inv + out.y_var, atol=1e-6)


def test_v16_eval_is_deterministic_in_mean_mode() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v16(_small_cfg("mean")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1.z, out2.z)
    assert torch.allclose(out1.y_hat, out2.y_hat)
    assert torch.allclose(out1.pooled_delta_phi_v, out2.pooled_delta_phi_v)


def test_v16_eval_sample_mode_remains_stochastic() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v16(_small_cfg("sample")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)
