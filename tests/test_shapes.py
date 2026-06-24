from __future__ import annotations

import torch

from models.factor_mechanism import FactorMechanismConfig, build_factor_mechanism


def test_factor_outputs_shapes():
    cfg = FactorMechanismConfig(input_dim=5, seq_len=12, out_dim=3, d_model=32, n_experts=2, latent_dim=16)
    model = build_factor_mechanism(cfg)
    x = torch.randn(4, 12, 5)
    out = model(x)
    assert out.y_hat.shape == (4, 3)
    assert out.z.shape[0] == 4
    assert out.x_recon.shape == x.shape
