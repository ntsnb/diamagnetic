from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v27 import FactorMechanismV27Config, build_factor_mechanism_v27


def _small_cfg(vae_eval_mode: str = "mean") -> FactorMechanismV27Config:
    return FactorMechanismV27Config(
        input_dim=5,
        seq_len=8,
        out_dim=2,
        d_model=32,
        n_experts=3,
        latent_dim=12,
        repr_dim=16,
        expert_hidden_dim=24,
        history_hidden_dim=18,
        release_hidden_dim=20,
        change_memory_hidden_dim=18,
        env_count=3,
        quality_hidden_dim=24,
        router_quality_hidden_dim=16,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode=vae_eval_mode,
    )


def test_v27_forward_exposes_quality_conditioned_change_state() -> None:
    torch.manual_seed(5)
    model = build_factor_mechanism_v27(_small_cfg()).eval()
    x = torch.randn(4, 8, 5)

    out = model(x)

    assert out.y_hat.shape == (4, 2)
    assert out.y_inv.shape == (4, 2)
    assert out.y_hist.shape == (4, 2)
    assert out.y_var.shape == (4, 2)
    assert out.expert_y_inv.shape == (4, 3, 2)
    assert out.expert_y_hist.shape == (4, 3, 2)
    assert out.expert_y_var.shape == (4, 3, 2)
    assert out.expert_y_hat.shape == (4, 3, 2)
    assert out.expert_quality_state.shape == (4, 3, 16)
    assert out.expert_quality_score.shape == (4, 3, 1)
    assert out.quality_router_bias.shape == (4, 3)
    assert out.pooled_quality_state.shape == (4, 16)
    assert out.pooled_quality_score.shape == (4, 1)
    assert out.pooled_servable_change.shape[-1] == 73
    assert torch.allclose(
        out.expert_y_var,
        out.expert_y_change_raw + out.expert_quality_score * out.expert_change_delta,
        atol=1e-6,
    )
    assert torch.allclose(out.expert_y_hat, out.expert_y_inv + out.expert_y_hist + out.expert_y_var, atol=1e-6)
    assert torch.allclose(out.expert_multiscale_change_score, out.expert_quality_score, atol=1e-6)
    assert torch.all((out.expert_quality_score >= 0.0) & (out.expert_quality_score <= 1.0))
    assert torch.allclose(out.alpha.sum(dim=-1), torch.ones(out.alpha.shape[0]), atol=1e-6)


def test_v27_eval_is_deterministic_in_mean_mode() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v27(_small_cfg("mean")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1.z, out2.z)
    assert torch.allclose(out1.y_hat, out2.y_hat)
    assert torch.allclose(out1.pooled_quality_state, out2.pooled_quality_state)
    assert torch.allclose(out1.pooled_quality_score, out2.pooled_quality_score)
    assert torch.allclose(out1.pooled_servable_change, out2.pooled_servable_change)


def test_v27_eval_sample_mode_remains_stochastic() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v27(_small_cfg("sample")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)
