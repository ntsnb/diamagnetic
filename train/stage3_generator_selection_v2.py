from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Stage3GeneratorCandidate:
    epoch: int
    checkpoint_path: Path
    last_train_loss: float | None = None


def _float_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def score_generator_candidate(aggregate: Mapping[str, Any]) -> float:
    """Score a generator checkpoint using validation-only rolling evidence."""

    return score_served_policy_candidate(aggregate)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def score_served_policy_candidate(aggregate: Mapping[str, Any]) -> float:
    """Score validation evidence using the served Stage III policy.

    The score intentionally gives strong weight to the worst rolling
    real-vs-counterfactual margin. A checkpoint with high raw gain but weak or
    negative attribution should not outrank a lower-gain checkpoint whose real
    drift separates from amplified/off-manifold counterfactuals.
    """

    pass_bonus = 100.0 if bool(aggregate.get("rolling_claim_gate_pass", False)) else 0.0
    pass_rate = _float_value(aggregate, "validation_window_pass_rate")
    worst_mse_gain = _float_value(aggregate, "worst_window_real_mse_gain")
    worst_mae_gain = _float_value(aggregate, "worst_window_real_mae_gain")
    worst_margin = _float_value(
        aggregate,
        "worst_window_real_vs_best_counterfactual_mse_gain_margin",
        worst_mse_gain - max(0.0, _float_value(aggregate, "best_window_counterfactual_mse_gain")),
    )
    max_reject_rate = _float_value(aggregate, "max_window_real_reject_rate")
    return pass_bonus + pass_rate + worst_mse_gain + worst_mae_gain + (20.0 * worst_margin) - (5.0 * max_reject_rate)


def build_generator_selection_row(
    candidate: Stage3GeneratorCandidate,
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(aggregate)
    row.update(
        {
            "candidate_epoch": int(candidate.epoch),
            "candidate_checkpoint": str(candidate.checkpoint_path),
            "candidate_last_train_loss": candidate.last_train_loss,
            "generator_selection_score": score_generator_candidate(aggregate),
            "generator_selection_abstained": False,
            "generator_selection_abstain_reason": "",
        }
    )
    return row


def select_best_generator_candidate(
    rows: list[Mapping[str, Any]],
    *,
    abstain_if_no_pass: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select Stage III generator checkpoint from an empty candidate list")
    selected = dict(
        max(
            rows,
            key=lambda row: (
                _float_value(row, "generator_selection_score", float("-inf")),
                -int(row.get("candidate_epoch", 0)),
            ),
        )
    )
    if abstain_if_no_pass and not any(_as_bool(row.get("rolling_claim_gate_pass", False)) for row in rows):
        selected["generator_selection_abstained"] = True
        selected["generator_selection_abstain_reason"] = "no_candidate_passed_claim_gate"
    else:
        selected.setdefault("generator_selection_abstained", False)
        selected.setdefault("generator_selection_abstain_reason", "")
    return selected
