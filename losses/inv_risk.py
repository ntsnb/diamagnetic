from __future__ import annotations

from typing import Dict

import torch


def env_invariant_risk(per_env_loss: Dict[int, torch.Tensor], lambda_var: float) -> torch.Tensor:
    """Environment-invariant objective (Eq. 8).

    R_mean + lambda_var * Var_e(R_suf^e)
    """

    if not per_env_loss:
        raise ValueError("per_env_loss is empty")
    losses = torch.stack(list(per_env_loss.values()))
    mean = losses.mean()
    var = losses.var(unbiased=False)
    return mean + lambda_var * var
