from __future__ import annotations

from pathlib import Path

from train.stage3_generator_selection_v2 import (
    Stage3GeneratorCandidate,
    build_generator_selection_row,
    select_best_generator_candidate,
    score_generator_candidate,
    score_served_policy_candidate,
)


def test_generator_selection_score_prioritizes_rolling_gate_pass():
    failing_but_high_gain = {
        "rolling_claim_gate_pass": False,
        "validation_window_pass_rate": 0.9,
        "worst_window_real_mse_gain": 0.5,
        "worst_window_real_mae_gain": 0.5,
        "best_window_counterfactual_mse_gain": 0.0,
    }
    passing_but_small_gain = {
        "rolling_claim_gate_pass": True,
        "validation_window_pass_rate": 1.0,
        "worst_window_real_mse_gain": 0.01,
        "worst_window_real_mae_gain": 0.01,
        "best_window_counterfactual_mse_gain": 0.0,
    }

    assert score_generator_candidate(passing_but_small_gain) > score_generator_candidate(failing_but_high_gain)


def test_build_generator_selection_row_records_candidate_and_score():
    candidate = Stage3GeneratorCandidate(
        epoch=2,
        checkpoint_path=Path("checkpoints/stage3_v2/run/generator_candidates/generator_epoch_002.pt"),
        last_train_loss=0.123,
    )
    aggregate = {
        "rolling_claim_gate_pass": False,
        "rolling_failure_reasons": "window_pass_rate",
        "validation_window_count": 3,
        "validation_window_pass_count": 2,
        "validation_window_pass_rate": 2 / 3,
        "failed_validation_windows": "0",
        "worst_window_real_mse_gain": -0.001,
        "worst_window_real_mae_gain": 0.002,
        "best_window_counterfactual_mse_gain": 0.003,
    }

    row = build_generator_selection_row(candidate, aggregate)

    assert row["candidate_epoch"] == 2
    assert row["candidate_checkpoint"].endswith("generator_epoch_002.pt")
    assert row["candidate_last_train_loss"] == 0.123
    assert row["generator_selection_score"] == score_generator_candidate(aggregate)
    assert row["rolling_failure_reasons"] == "window_pass_rate"


def test_select_best_generator_candidate_uses_score_then_earliest_epoch():
    rows = [
        {"candidate_epoch": 3, "candidate_checkpoint": "epoch3.pt", "generator_selection_score": 1.0},
        {"candidate_epoch": 1, "candidate_checkpoint": "epoch1.pt", "generator_selection_score": 2.0},
        {"candidate_epoch": 0, "candidate_checkpoint": "epoch0.pt", "generator_selection_score": 2.0},
    ]

    selected = select_best_generator_candidate(rows)

    assert selected["candidate_epoch"] == 0
    assert selected["candidate_checkpoint"] == "epoch0.pt"


def test_served_policy_score_prioritizes_real_counterfactual_margin_over_raw_gain():
    high_gain_bad_margin = {
        "rolling_claim_gate_pass": False,
        "validation_window_pass_rate": 0.0,
        "worst_window_real_mse_gain": 0.08,
        "worst_window_real_mae_gain": 0.06,
        "best_window_counterfactual_mse_gain": 0.081,
        "worst_window_real_vs_best_counterfactual_mse_gain_margin": -0.001,
        "max_window_real_reject_rate": 0.01,
    }
    lower_gain_better_margin = {
        "rolling_claim_gate_pass": False,
        "validation_window_pass_rate": 0.0,
        "worst_window_real_mse_gain": 0.04,
        "worst_window_real_mae_gain": 0.03,
        "best_window_counterfactual_mse_gain": 0.036,
        "worst_window_real_vs_best_counterfactual_mse_gain_margin": 0.004,
        "max_window_real_reject_rate": 0.01,
    }

    assert score_served_policy_candidate(lower_gain_better_margin) > score_served_policy_candidate(high_gain_bad_margin)


def test_select_best_generator_candidate_marks_abstention_when_all_candidates_fail():
    rows = [
        {
            "candidate_epoch": 2,
            "candidate_checkpoint": "epoch2.pt",
            "generator_selection_score": 0.2,
            "rolling_claim_gate_pass": False,
            "rolling_failure_reasons": "counterfactual",
        },
        {
            "candidate_epoch": 1,
            "candidate_checkpoint": "epoch1.pt",
            "generator_selection_score": 0.3,
            "rolling_claim_gate_pass": False,
            "rolling_failure_reasons": "window_pass_rate;counterfactual",
        },
    ]

    selected = select_best_generator_candidate(rows, abstain_if_no_pass=True)

    assert selected["candidate_epoch"] == 1
    assert selected["generator_selection_abstained"] is True
    assert selected["generator_selection_abstain_reason"] == "no_candidate_passed_claim_gate"
