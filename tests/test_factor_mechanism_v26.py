from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v26 import FactorMechanismV26Config, build_factor_mechanism_v26


def _small_cfg(vae_eval_mode: str = "mean") -> FactorMechanismV26Config:
    return FactorMechanismV26Config(
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
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode=vae_eval_mode,
    )


def test_v26_forward_exposes_multiscale_change_state() -> None:
    torch.manual_seed(5)
    model = build_factor_mechanism_v26(_small_cfg()).eval()
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
    assert torch.allclose(
        out.pooled_phi_i,
        out.pooled_base + out.pooled_invariant_adapter - out.pooled_env_correction,
        atol=1e-6,
    )
    assert out.pooled_release_state.shape == out.pooled_budgeted_release.shape
    assert torch.allclose(out.pooled_delta_phi_v, out.pooled_release_state, atol=1e-6)
    assert out.pooled_observable_change.shape == out.pooled_delta_phi_v.shape
    assert out.pooled_change_memory.shape == out.pooled_delta_phi_v.shape
    assert out.pooled_multiscale_change_score.shape == (4, 1)
    assert out.pooled_servable_change.shape[-1] == out.pooled_delta_phi_v.shape[-1] * 5 + 2 + out.y_hat.shape[-1] * 2
    assert torch.allclose(out.pooled_raw_release, out.pooled_servable_change, atol=1e-6)
    assert torch.allclose(out.pooled_phi_v, out.pooled_phi_i + out.pooled_release_state, atol=1e-6)
    assert torch.allclose(out.expert_y_hat, out.expert_y_inv + out.expert_y_var + out.expert_y_hist, atol=1e-6)
    assert torch.allclose(
        out.expert_y_var,
        out.expert_y_change_raw + out.expert_multiscale_change_score * out.expert_change_delta,
        atol=1e-6,
    )
    assert torch.allclose(out.expert_history_budget + out.expert_release_budget, torch.ones_like(out.expert_history_budget), atol=1e-5)
    assert torch.all((out.expert_release_confidence >= 0.0) & (out.expert_release_confidence <= 1.0))
    assert torch.all((out.expert_multiscale_change_score >= 0.0) & (out.expert_multiscale_change_score <= 1.0))
    assert out.expert_release_state.shape == out.expert_raw_release.shape


def test_v26_eval_is_deterministic_in_mean_mode() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v26(_small_cfg("mean")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1.z, out2.z)
    assert torch.allclose(out1.y_hat, out2.y_hat)
    assert torch.allclose(out1.pooled_budgeted_release, out2.pooled_budgeted_release)
    assert torch.allclose(out1.pooled_release_state, out2.pooled_release_state)
    assert torch.allclose(out1.pooled_observable_change, out2.pooled_observable_change)
    assert torch.allclose(out1.pooled_change_memory, out2.pooled_change_memory)
    assert torch.allclose(out1.pooled_multiscale_change_score, out2.pooled_multiscale_change_score)
    assert torch.allclose(out1.pooled_servable_change, out2.pooled_servable_change)


def test_v26_eval_sample_mode_remains_stochastic() -> None:
    torch.manual_seed(7)
    model = build_factor_mechanism_v26(_small_cfg("sample")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)



