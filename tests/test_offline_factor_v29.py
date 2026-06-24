from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v29 import FactorMechanismV29Config, build_factor_mechanism_v29
from train.offline_factor_v29 import FactorMechanismV29LossConfig, train_offline_factor_v29


def test_train_offline_factor_v29_reports_invariant_release_losses() -> None:
    torch.manual_seed(29)
    model = build_factor_mechanism_v29(
        FactorMechanismV29Config(
            input_dim=4,
            seq_len=6,
            out_dim=2,
            d_model=24,
            n_experts=2,
            latent_dim=10,
            repr_dim=12,
            expert_hidden_dim=20,
            history_hidden_dim=16,
            release_hidden_dim=16,
            change_memory_hidden_dim=16,
            env_count=3,
            invariant_release_hidden_dim=20,
            router_invariant_hidden_dim=12,
            temporal_encoder="transformer",
            nhead=4,
            num_layers=1,
            dropout=0.0,
            use_revin=False,
            use_vae=True,
            vae_eval_mode="mean",
        )
    )
    ds = TensorDataset(torch.randn(8, 6, 4), torch.randn(8, 2))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = train_offline_factor_v29(loader, model, opt, FactorMechanismV29LossConfig(), torch.device("cpu"))

    assert rows
    row = rows[0]
    assert "loss_support_alignment" in row
    assert "loss_suppression_alignment" in row
    assert "loss_local_env_consistency" in row
    assert "loss_invariant_correction_budget" in row
    assert "loss_residual_release_alignment" in row
    assert "loss_residual_release_advantage" in row
    assert "loss_expert_release_orthogonal" in row
    assert "loss_residual_state_noncollapse" in row
