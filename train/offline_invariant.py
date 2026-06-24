from __future__ import annotations

from typing import Iterable

import torch

from losses.irn_loss import irn_surrogate_loss
from models.invariant_learning import MTIL, MTILConfig, MTEI
from utils.progress import get_progress


def train_offline_invariant(
    dataloader: Iterable,
    mtil: MTIL,
    mtei: MTEI,
    cfg: MTILConfig,
    opt_mtil: torch.optim.Optimizer,
    opt_mtei: torch.optim.Optimizer,
    device: torch.device,
):
    """Alternating optimization for MTIL + MTEI (FOIL style)."""

    mtil.train()
    mtei.train()

    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    pbar = get_progress(total=total, desc="offline_invariant")
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # Forward to get phi and y_hat
        phi, y_hat = mtil(x)

        # E-step: infer env labels
        labels = mtei.e_step(phi.detach(), y.detach(), irn_surrogate_loss)

        # M-step for MTEI regressors
        per_env_loss = mtei.m_step(phi.detach(), y.detach(), labels, irn_surrogate_loss)
        opt_mtei.zero_grad(set_to_none=True)
        if per_env_loss:
            mtei_loss = torch.stack(list(per_env_loss.values())).mean()
            mtei_loss.backward()
            opt_mtei.step()

        # MTIL update (invariant risk + ERM)
        per_env_loss = mtei.m_step(phi, y, labels, irn_surrogate_loss)
        mtil_loss = MTIL.loss(per_env_loss, y_hat, y, cfg, irn_surrogate_loss)
        opt_mtil.zero_grad(set_to_none=True)
        mtil_loss.backward()
        opt_mtil.step()
        if pbar is not None:
            pbar.update(x.size(0))
    if pbar is not None:
        pbar.close()
