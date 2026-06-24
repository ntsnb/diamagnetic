from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.loaders import build_dataloaders_from_series
from models.factor_mechanism import FactorMechanismConfig, build_factor_mechanism
from train.offline_factor import train_offline_factor
from utils.seed import set_seed
from utils.config import load_yaml


def main():
    base = load_yaml("configs/base.yaml")
    cfg = base
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    if cfg["data"]["source"] == "synthetic":
        series = torch.randn(500, 8)
    else:
        raise NotImplementedError("file data source not implemented")

    train_dl, _, _ = build_dataloaders_from_series(
        series=series,
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        target_dim=cfg["data"]["target_dim"],
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    mcfg = FactorMechanismConfig(
        input_dim=series.shape[1],
        seq_len=cfg["data"]["lookback"],
        out_dim=cfg["data"]["horizon"],
        d_model=cfg["model"]["d_model"],
        n_experts=cfg["model"]["n_experts"],
        latent_dim=cfg["model"]["latent_dim"],
        temporal_encoder=cfg["model"]["temporal_encoder"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )
    model = build_factor_mechanism(mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(cfg["train"]["epochs"]):
        train_offline_factor(
            dataloader=train_dl,
            model=model,
            opt=opt,
            beta_vae=cfg["train"]["beta_vae"],
            lambda_disc=cfg["train"]["lambda_disc"],
            device=device,
            use_lrp=cfg["train"]["use_lrp"],
            lrp_method=cfg["train"]["lrp_method"],
        )


if __name__ == "__main__":
    main()
