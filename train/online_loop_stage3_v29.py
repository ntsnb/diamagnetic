from __future__ import annotations

from typing import Any

from train.online_loop_stage3_v26 import run_online_adaptation_v26
from train.stage3_v29_online_mode_gate import OnlineModeGatePolicy, select_online_mode_for_row

run_online_adaptation_v29 = run_online_adaptation_v26


def select_stage3_v29_serving_mode(
    online_feature_row: dict[str, Any],
    *,
    online_mode_gate_policy: OnlineModeGatePolicy | None = None,
    allowed_modes: list[str] | tuple[str, ...] | None = None,
    default_mode: str = "history_only",
) -> dict[str, Any]:
    """Select a V29 serving feature mode using online-available features only.

    This helper is intentionally separate from the inherited V26 adaptation loop:
    V29 currently has no generator-unlock evidence, so the safest serving
    integration point is a fail-closed mode decision before any active Stage III
    correction is applied.
    """

    return select_online_mode_for_row(
        online_feature_row,
        policy=online_mode_gate_policy,
        allowed_modes=allowed_modes,
        default_mode=default_mode,
    )
