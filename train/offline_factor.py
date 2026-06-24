from __future__ import annotations

from typing import Iterable, Tuple

import torch

from losses.discrepancy import expert_discrepancy
from losses.vae_loss import vae_loss
from models.lrp import LRPEngine
from utils.progress import get_progress


def evaluate_factor(
    dataloader: Iterable,
    model,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_mse += torch.mean((out.y_hat - y) ** 2).item() * x.size(0)
            total_mae += torch.mean(torch.abs(out.y_hat - y)).item() * x.size(0)
            n += x.size(0)
    return total_mse / max(n, 1), total_mae / max(n, 1)


def train_offline_factor(
    dataloader: Iterable,
    model,
    opt: torch.optim.Optimizer,
    beta_vae: float,
    lambda_disc: float,
    device: torch.device,
    use_lrp: bool = False,
    lrp_method: str = "rrp",
    save_lrp: bool = False,
    pred_loss_type: str = "mse",
    pred_mae_weight: float = 0.0,
    pred_huber_delta: float = 1.0,
):
    """Offline training for factor mechanism (Eq. 24-26)."""

    model.train()
    lrp_engine = LRPEngine() if use_lrp else None

    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_factor")
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        # Prediction loss
        mse = torch.mean((outputs.y_hat - y) ** 2)
        mae = torch.mean(torch.abs(outputs.y_hat - y))
        if pred_loss_type == "mse":
            l_pred = mse
        elif pred_loss_type == "mae":
            l_pred = mae
        elif pred_loss_type == "huber":
            l_pred = torch.nn.functional.huber_loss(outputs.y_hat, y, delta=pred_huber_delta)
        elif pred_loss_type == "mixed":
            l_pred = mse + pred_mae_weight * mae
        else:
            raise ValueError(f"unknown pred_loss_type: {pred_loss_type}")

        # VAE loss
        if getattr(model, "use_vae", True):
            l_vae = vae_loss(outputs.x_recon, x, outputs.mu_z, outputs.log_var_z, beta_vae)
        else:
            l_vae = torch.tensor(0.0, device=device)

        # Discrepancy loss (optional LRP)
        l_disc = torch.tensor(0.0, device=device)
        if use_lrp and lrp_engine is not None:
            # placeholder: compute relevance maps per expert, per sample
            # You should replace with proper indexing for targets and batch.
            rel_maps = []
            for _ in outputs.expert_embs:
                rel_maps.append(lrp_engine.compute_relevance_map(model, x, method=lrp_method))
            l_disc = expert_discrepancy(rel_maps, lambda_mass=1.0)
            if save_lrp:
                torch.save(rel_maps, "logs/lrp_maps.pt")

        loss = l_pred + l_vae + lambda_disc * l_disc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if pbar is not None:
            pbar.update(x.size(0))
    if pbar is not None:
        pbar.close()
