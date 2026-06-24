from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factor_mechanism_v19 import FactorMechanismV19Config, build_factor_mechanism_v19
from train.offline_factor_v19 import FactorMechanismV19LossConfig, evaluate_factor_v19, train_offline_factor_v19


def _model() -> torch.nn.Module:
    cfg = FactorMechanismV19Config(
        input_dim=4,
        seq_len=6,
        out_dim=2,
        d_model=16,
        n_experts=2,
        latent_dim=8,
        repr_dim=8,
        expert_hidden_dim=16,
        release_hidden_dim=12,
        env_count=2,
        temporal_encoder="transformer",
        nhead=4,
        num_layers=1,
        dropout=0.0,
        use_vae=False,
    )
    return build_factor_mechanism_v19(cfg)


def test_offline_factor_v19_train_step_runs_and_reports_new_loss_terms() -> None:
    torch.manual_seed(11)
    model = _model()
    x = torch.randn(10, 6, 4)
    y = torch.randn(10, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rows = train_offline_factor_v19(loader, model, opt, FactorMechanismV19LossConfig(), torch.device("cpu"))

    assert rows
    required = {
        "loss_local_history_proxy",
        "loss_local_release_increment",
        "loss_conditional_advantage",
        "loss_release_null",
        "loss_counterfactual_shuffle",
    }
    assert required.issubset(rows[-1].keys())
    for key, value in rows[-1].items():
        assert torch.isfinite(torch.tensor(value)), key


def test_offline_factor_v19_evaluate_returns_scalar_metrics() -> None:
    torch.manual_seed(13)
    model = _model().eval()
    x = torch.randn(8, 6, 4)
    y = torch.randn(8, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    mse, mae = evaluate_factor_v19(loader, model, torch.device("cpu"))

    assert isinstance(mse, float)
    assert isinstance(mae, float)
