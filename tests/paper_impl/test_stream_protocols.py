import json

import numpy as np
import pytest
import torch

from data.loaders import _build_shifted_splits, _shift_score
from paper_impl.stream_protocols import (chronological_issue_indices, legacy_shift_score,
                                         select_stream_indices, write_selection_manifest)


def test_legacy_reproduces_real_old_function_formula_candidates_and_indices():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(100, 4, 3, generator=generator)
    inputs[79:99, :, 1] += 4
    old = _build_shifted_splits(inputs, 55, 10, 20, 35)
    selected = select_stream_indices(inputs, protocol="legacy_shifted")
    assert selected.test_indices.tolist() == list(old[2])
    assert selected.manifest["candidate_starts"] == list(range(65, 81))
    assert selected.manifest["shift_score"] == pytest.approx(old[3]["shift_score"], rel=2e-6)
    assert legacy_shift_score(inputs[:55], inputs[79:99]) == pytest.approx(
        _shift_score(inputs[:55], inputs[79:99]), rel=2e-6)
    assert selected.manifest["future_inputs_for_selection"] is True
    assert selected.manifest["uses_future_targets_y_for_selection"] is False
    assert "not online drift detection" in selected.manifest["selection_bias"]


def test_chronological_uses_entire_suffix_independently_of_values():
    first = select_stream_indices(100)
    values = np.random.default_rng(0).normal(size=(100, 4, 2))
    second = select_stream_indices(values, online_test_fraction=.01)
    values[80:] += 1e8
    third = select_stream_indices(values)
    assert first.test_indices.tolist() == list(range(65, 100))
    np.testing.assert_equal(first.test_indices, second.test_indices)
    np.testing.assert_equal(second.test_indices, third.test_indices)
    assert first.manifest["future_inputs_for_selection"] is False


def test_legacy_first_tie_and_short_pool_match_old():
    inputs = torch.ones(101, 2, 1)
    selection = select_stream_indices(inputs, protocol="legacy_shifted", shift_pool_fraction=.05)
    n_train, n_val, n_test = int(101*.55), int(101*.1), int(101*.2)
    old = _build_shifted_splits(inputs, n_train, n_val, n_test, max(n_test, int(101*.05)))
    assert selection.test_indices.tolist() == list(old[2])
    assert selection.manifest["shift_score"] == 0


def test_strict_target_boundaries_and_complete_horizon_release():
    result = chronological_issue_indices(100, lookback=4, horizon=3, extra_delay=7)
    previous_end = -1
    for split in result["splits"].values():
        issues = np.asarray(split["issue_indices"])
        assert (issues + 1 >= split["target_start"]).all()
        assert (issues + 3 < split["target_end"]).all()
        assert (issues - 3 >= 0).all()
        np.testing.assert_array_equal(split["release_indices"], issues + 3 + 7)
        assert issues[0] + 1 > previous_end
        previous_end = issues[-1] + 3
    assert result["splits"]["test"]["issue_indices"][-1] == 96
    assert result["splits"]["test"]["release_indices"][-1] == 106


def test_frozen_manifest_idempotent_but_cannot_change(tmp_path):
    path = tmp_path / "frozen.json"
    selection = select_stream_indices(100)
    one = write_selection_manifest(path, selection)
    two = write_selection_manifest(path, selection)
    assert one == two == json.loads(path.read_text(encoding="utf8"))
    with pytest.raises(FileExistsError, match="frozen selection"):
        write_selection_manifest(path, select_stream_indices(101))


@pytest.mark.parametrize("kwargs", [{"lookback": 0, "horizon": 1}, {"lookback": 2, "horizon": 0},
                                    {"lookback": 2, "horizon": 1, "extra_delay": -1},
                                    {"lookback": 2, "horizon": 99}])
def test_invalid_boundary_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        chronological_issue_indices(100, **kwargs)


def test_legacy_requires_finite_actual_inputs():
    with pytest.raises(ValueError, match="actual input"):
        select_stream_indices(100, protocol="legacy_shifted")
    with pytest.raises(ValueError, match="finite"):
        select_stream_indices(np.full((100, 2, 1), np.nan), protocol="legacy_shifted")
