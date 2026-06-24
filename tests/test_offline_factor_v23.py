from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v23 import FactorMechanismV23Config, build_factor_mechanism_v23
from train.offline_factor_v23 import FactorMechanismV23LossConfig, train_offline_factor_v23


def test_train_offline_factor_v23_reports_observable_drift_losses() -> None:
    torch.manual_seed(11)
    model = build_factor_mechanism_v23(
        FactorMechanismV23Config(
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
            env_count=3,
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
    rows = train_offline_factor_v23(loader, model, opt, FactorMechanismV23LossConfig(), torch.device("cpu"))

    assert rows
    row = rows[0]
    assert "loss_history_after_release" in row
    assert "loss_release_first_increment" in row
    assert "loss_release_priority_advantage" in row
    assert "loss_release_state_advantage" in row
    assert "loss_completion_after_release" in row
    assert "loss_budget_alignment" in row

