"""Regression tests for independently reviewed full-experiment boundary cases."""
from dataclasses import asdict
import json

import pytest
import torch

from paper_impl.model import ModelConfig, PaperForecaster
from paper_impl.adaptation import OnlineConfig, OnlineController, ReleasedBatch


def test_official_subset_forecasting_requires_explicit_target_columns():
    with pytest.raises(ValueError, match="explicit target_indices"):
        ModelConfig(3, 16, 2, targets=1, expert_backbone="transformer")
    valid = ModelConfig(3, 16, 2, targets=1, expert_backbone="transformer", target_indices=(2,))
    assert valid.target_indices == (2,)
    # With every variable predicted, all columns are unambiguous.
    ModelConfig(3, 16, 2, targets=3, expert_backbone="transformer")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_update_is_skipped_and_remains_strict_json_serializable(nonfinite, monkeypatch):
    model = PaperForecaster(ModelConfig(2, 3, 2, hidden=4, heads=1, n_experts=2))
    controller = OnlineController(model, OnlineConfig(stage_ii=False, stage_iii=False))
    x = torch.ones(1, 3, 2)
    batch = ReleasedBatch((2,), x, torch.zeros(1, 2, 1), torch.zeros(1, 2, 1))
    monkeypatch.setattr(controller, "_loss", lambda *args: torch.tensor(nonfinite))
    before = model.router.bias.detach().clone()
    record = controller._step("I", "test", batch, None, [model.router.bias], .01, .1)
    assert not record.accepted and record.reason.endswith(":nonfinite_loss")
    assert record.loss_before is None and record.loss_after is None
    assert torch.equal(before, model.router.bias)
    assert json.loads(json.dumps(asdict(record), allow_nan=False))["loss_before"] is None
