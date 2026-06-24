from __future__ import annotations

from typing import Any


def assert_deterministic_eval_model(
    model: Any,
    *,
    context: str,
    allow_mc_eval: bool = False,
) -> None:
    """Reject stochastic VAE evaluation in Stage III V2 boundaries.

    Legacy ``FactorMechanism`` keeps sampling in ``eval()`` when ``use_vae`` is
    true. Stage III V2 metrics and generator training require deterministic
    point predictions unless an explicit Monte Carlo protocol is implemented.
    """

    if not bool(getattr(model, "use_vae", False)):
        return
    vae_eval_mode = getattr(model, "vae_eval_mode", None)
    if vae_eval_mode == "mean":
        return
    if allow_mc_eval and vae_eval_mode == "sample":
        return
    raise ValueError(
        f"{context} requires deterministic VAE eval; use FactorMechanismV2 "
        "with vae_eval_mode='mean' or disable VAE sampling for this boundary"
    )
