import torch

from train.stage3_v12e_conditional_permutation import build_history_norm_strata
from train.stage3_v12e_conditional_permutation import conditional_permute_drift
from train.stage3_v12e_conditional_permutation import conditional_random_matched_norm


def test_v12e_conditional_permutation_stays_within_strata() -> None:
    drift = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    strata = torch.tensor([0, 0, 1, 1])

    shuffled = conditional_permute_drift(drift_signal=drift, strata=strata, seed=7)

    assert shuffled.shape == drift.shape
    assert {tuple(row.tolist()) for row in shuffled[:2]} <= {tuple(row.tolist()) for row in drift[:2]}
    assert {tuple(row.tolist()) for row in shuffled[2:]} <= {tuple(row.tolist()) for row in drift[2:]}
    assert not torch.allclose(shuffled, drift)


def test_v12e_history_norm_strata_are_deterministic_and_shape_safe() -> None:
    history = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])

    first = build_history_norm_strata(history_features=history, bin_count=3)
    second = build_history_norm_strata(history_features=history, bin_count=3)

    assert torch.equal(first, second)
    assert first.shape == (6,)
    assert first.min().item() >= 0
    assert first.max().item() <= 2


def test_v12e_conditional_random_matched_norm_preserves_row_norms() -> None:
    drift = torch.tensor([[3.0, 4.0], [0.0, 0.0], [5.0, 12.0]])
    strata = torch.tensor([0, 0, 1])

    random = conditional_random_matched_norm(drift_signal=drift, strata=strata, seed=11)

    assert random.shape == drift.shape
    assert torch.allclose(torch.linalg.vector_norm(random, dim=1), torch.linalg.vector_norm(drift, dim=1), atol=1e-6)
    assert torch.allclose(random[1], torch.zeros(2))
