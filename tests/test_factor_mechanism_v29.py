from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v29 import FactorMechanismV29Config, build_factor_mechanism_v29


def _small_cfg(vae_eval_mode: str = "mean") -> FactorMechanismV29Config:
    return FactorMechanismV29Config(
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
        invariant_release_hidden_dim=28,
        router_invariant_hidden_dim=16,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode=vae_eval_mode,
    )


def test_v29_forward_exposes_expert_local_invariant_release_states() -> None:
    torch.manual_seed(17)
    cfg = _small_cfg()
    model = build_factor_mechanism_v29(cfg).eval()
    x = torch.randn(4, 8, 5)

    out = model(x)

    assert out.y_hat.shape == (4, 2)
    assert out.y_inv.shape == (4, 2)
    assert out.y_hist.shape == (4, 2)
    assert out.y_var.shape == (4, 2)
    assert out.expert_shared_anchor.shape == (4, 3, 16)
    assert out.expert_local_env_logits.shape == (4, 3, 3)
    assert out.expert_local_env_posteriors.shape == (4, 3, 3)
    assert out.expert_invariant_correction.shape == (4, 3, 16)
    assert out.expert_invariant_correction_gate.shape == (4, 3, 16)
    assert out.expert_phi_i.shape == (4, 3, 16)
    assert out.expert_residual_release_state.shape == (4, 3, 16)
    assert out.expert_release_support_score.shape == (4, 3, 1)
    assert out.expert_release_suppression_score.shape == (4, 3, 1)
    assert out.expert_release_delta_y.shape == (4, 3, 2)
    assert out.expert_release_suppression_y.shape == (4, 3, 2)
    assert out.support_router_bias.shape == out.invariant_router_bias.shape if hasattr(out, "support_router_bias") else True
    assert out.pooled_servable_change.shape[-1] == 7 * cfg.repr_dim + 2 * cfg.env_count + 4 + 5 * cfg.out_dim
    assert torch.allclose(
        out.expert_y_var,
        out.expert_y_change_raw
        + out.expert_release_support_score * out.expert_release_delta_y
        - out.expert_release_suppression_score * out.expert_release_suppression_y,
        atol=1e-6,
    )
    assert torch.allclose(out.expert_delta_phi_v, out.expert_residual_release_state, atol=1e-6)
    assert torch.allclose(out.pooled_delta_phi_v, out.pooled_residual_release_state, atol=1e-6)
    assert torch.allclose(out.expert_phi_v, out.expert_phi_i + out.expert_residual_release_state, atol=1e-6)
    assert torch.allclose(out.expert_y_hat, out.expert_y_inv + out.expert_y_hist + out.expert_y_var, atol=1e-6)
    assert torch.all((out.expert_release_support_score >= 0.0) & (out.expert_release_support_score <= 1.0))
    assert torch.all((out.expert_release_suppression_score >= 0.0) & (out.expert_release_suppression_score <= 1.0))
    assert torch.allclose(out.expert_local_env_posteriors.sum(dim=-1), torch.ones(4, 3), atol=1e-6)
    assert torch.allclose(out.alpha.sum(dim=-1), torch.ones(out.alpha.shape[0]), atol=1e-6)


def test_v29_eval_is_deterministic_in_mean_mode() -> None:
    torch.manual_seed(19)
    model = build_factor_mechanism_v29(_small_cfg("mean")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1.z, out2.z)
    assert torch.allclose(out1.y_hat, out2.y_hat)
    assert torch.allclose(out1.pooled_phi_i, out2.pooled_phi_i)
    assert torch.allclose(out1.pooled_residual_release_state, out2.pooled_residual_release_state)
    assert torch.allclose(out1.pooled_servable_change, out2.pooled_servable_change)


def test_v29_eval_sample_mode_remains_stochastic() -> None:
    torch.manual_seed(23)
    model = build_factor_mechanism_v29(_small_cfg("sample")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)
