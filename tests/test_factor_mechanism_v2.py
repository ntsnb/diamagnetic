from __future__ import annotations

import torch

from models.factor_mechanism_v2 import FactorMechanismV2Config, build_factor_mechanism_v2


def _small_cfg(vae_eval_mode: str = "mean") -> FactorMechanismV2Config:
    return FactorMechanismV2Config(
        input_dim=5,
        seq_len=8,
        out_dim=2,
        d_model=32,
        n_experts=2,
        latent_dim=12,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=True,
        vae_eval_mode=vae_eval_mode,
    )


def test_v2_vae_eval_uses_mu_by_default_for_deterministic_predictions():
    torch.manual_seed(7)
    model = build_factor_mechanism_v2(_small_cfg("mean")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1.z, out2.z)
    assert torch.allclose(out1.y_hat, out2.y_hat)


def test_v2_vae_eval_sample_mode_remains_explicitly_stochastic():
    torch.manual_seed(7)
    model = build_factor_mechanism_v2(_small_cfg("sample")).eval()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)


def test_v2_vae_train_mode_keeps_reparameterization_sampling():
    torch.manual_seed(7)
    model = build_factor_mechanism_v2(_small_cfg("mean")).train()
    x = torch.randn(4, 8, 5)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1.z, out2.z)
