"""Matched Stage II controls: exact masks, candidate scope and audited failures."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from paper_impl.adaptation import OnlineConfig, OnlineController, SelectionUnavailableError


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 3)
        self.norm = nn.LayerNorm(3)
        self.second = nn.Linear(3, 2)
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        return self.head(self.second(self.norm(self.first(x))).tanh())


class InspectableForecaster(nn.Module):
    """Controller test double; production LRP formulas are tested separately."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(horizon=2, targets=1, n_experts=2, lookback=3, n_variables=2)
        self.experts = nn.ModuleList([Expert(), Expert()])
        self.router = nn.Linear(2, 2)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(.3 * torch.sin(values + index) + .1)

    def expert_weights(self):
        return {f"experts.{m}.{layer}.weight": getattr(expert, layer).weight
                for m, expert in enumerate(self.experts) for layer in ("first", "second")}

    def forward(self, x, stochastic=False, modulations=None):
        h = x.mean(1)
        predictions = torch.stack([expert(h) for expert in self.experts], 1)
        gates = self.router(h).softmax(-1)
        return SimpleNamespace(prediction=(predictions * gates[:, :, None]).sum(1).unsqueeze(-1))

    def attribute(self, x):
        maps = x.abs().transpose(1, 2)[:, None, None].expand(-1, 1, 2, -1, -1)
        channels = {}
        for m, expert in enumerate(self.experts):
            first = expert.first(x.mean(1)).abs() + .1
            second = expert.second(first).abs() + .1
            channels[f"experts.{m}.first.weight"] = first[:, None]
            channels[f"experts.{m}.second.weight"] = second[:, None]
        return SimpleNamespace(maps=maps, channels=channels)


def window():
    return torch.tensor([[1., .2], [.8, .4], [1.1, .1]], dtype=torch.double)


def controller(selector="gradient_only", *, reference=True, **overrides):
    settings = dict(stage_i=False, stage_iii=False, selector=selector,
                    selection_protocol="fixed_budget", selection_seed=93,
                    channel_budget={"first.weight": 2, "second.weight": 1},
                    reference_windows=2, reference_interval=2, history_capacity=4,
                    learning_rate_ii=.02, radius_ii=.1)
    settings.update(overrides)
    initial = window().unsqueeze(0).repeat(2, 1, 1) if reference else None
    return OnlineController(InspectableForecaster().double(), OnlineConfig(**settings), reference_inputs=initial)


def release_once(instance):
    forecast = instance.predict(window(), 0)
    before = {name: p.detach().clone() for name, p in instance.model.named_parameters()}
    record = instance.release(2, {0: forecast.prediction + 1.}, current_x=window())
    return before, record.experts


@pytest.mark.parametrize("selector", ["relevance_only", "gradient_only", "random"])
def test_sparse_selectors_have_exact_equal_scalar_budgets_and_only_touch_selected_rows(selector):
    instance = controller(selector, stability_threshold=0., error_threshold=0.)
    before, update = release_once(instance)
    assert update.accepted
    assert update.selected_parameter_count == 2 * 2 + 1 * 3 == 7
    assert update.eligible_parameter_count == 2 * (3 * 2 + 2 * 3) == 24
    assert update.selection_protocol == "fixed_budget" and update.selector == selector
    assert update.reason.startswith("fixed_budget:" + selector)
    assert "stored_error_threshold" not in update.reason
    assert update.layer_budgets["first.weight"]["selected_channels"] == 2
    assert update.layer_budgets["second.weight"]["selected_channels"] == 1
    assert sum(row["selected_parameter_count"] for row in update.layer_budgets.values()) == 7
    changed = 0
    for name, parameter in instance.model.named_parameters():
        difference = parameter.detach() != before[name]
        if name in update.selected_channels:
            mask = torch.zeros_like(parameter, dtype=torch.bool)
            mask[list(update.selected_channels[name])] = True
            assert not difference[~mask].any()
        else:
            assert not difference.any(), name
        changed += int(difference.sum())
    assert update.changed_parameter_count == changed > 0
    assert changed <= update.selected_parameter_count
    json.dumps(asdict(update), allow_nan=False)
    if selector == "relevance_only":
        # Identical valid reference/current profiles have zero drift; neither
        # the stability gate nor error fallback may alter this matched control.
        assert "zero_score_ties" in update.reason
        assert update.selected_channels == {"experts.0.first.weight": (0, 1), "experts.0.second.weight": (0,)}


@pytest.mark.parametrize("positive", [False, True])
def test_zero_and_partly_zero_scores_still_fill_the_exact_budget(positive):
    instance = controller()
    scores = {name: torch.zeros(p.shape[0]) for name, p in instance.weights.items()}
    if positive:
        scores["experts.1.first.weight"][2] = 1.
    masks, selected = instance._select(scores)
    assert sum(int(mask.sum()) for mask in masks.values()) == 7
    if positive:
        assert selected["experts.1.first.weight"] == (2,)
        assert selected["experts.0.first.weight"] == (0,)
    else:
        assert selected["experts.0.first.weight"] == (0, 1)
    assert selected["experts.0.second.weight"] == (0,)


def test_dense_control_updates_only_all_eligible_weights_and_logs_actual_changes():
    instance = controller(stage_ii_scope="dense_eligible")
    assert set(instance.update_parameters) == set(instance.weights)
    before, update = release_once(instance)
    assert update.accepted and update.selected_parameter_count == update.eligible_parameter_count == 24
    assert update.reason == "dense_update:dense_eligible:ranking_not_used"
    assert sum(row["selected_parameter_count"] for row in update.layer_budgets.values()) == 24
    assert all(row["selected_channels"] == row["eligible_channels"] for row in update.layer_budgets.values())
    changed = 0
    for name, parameter in instance.model.named_parameters():
        difference = parameter.detach() != before[name]
        if name not in instance.weights:
            assert not difference.any(), name
        changed += int(difference.sum())
    assert update.changed_parameter_count == changed > 0
    legacy = controller(selection_protocol="adaptive", stage_ii_scope="dense_experts")
    assert any(name.endswith("norm.weight") for name in legacy.update_parameters)
    assert any(name.endswith("head.weight") for name in legacy.update_parameters)
    assert len(legacy.update_parameters) > len(instance.update_parameters)


def test_random_selection_is_release_seeded_and_does_not_touch_global_rng():
    instance = controller("random")
    rng = torch.random.get_rng_state().clone()
    first, seed = instance._random_scores(2)
    repeated, repeated_seed = instance._random_scores(2)
    other, other_seed = instance._random_scores(3)
    assert seed == repeated_seed and seed != other_seed
    assert all(torch.equal(first[name], repeated[name]) for name in first)
    assert any(not torch.equal(first[name], other[name]) for name in first)
    assert torch.equal(rng, torch.random.get_rng_state())
    _, update = release_once(instance)
    assert update.selection_event_seed == seed
    assert torch.equal(rng, torch.random.get_rng_state())


def test_random_retry_after_release_failure_reuses_mask_and_rolls_back_state(monkeypatch):
    instance = controller("random")
    forecast = instance.predict(window(), 0)
    before = deepcopy(instance.model.state_dict())
    original_step, attempted = instance._step, []

    def fail_after_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        attempted.append(result)
        raise RuntimeError("after selected update")

    monkeypatch.setattr(instance, "_step", fail_after_step)
    with pytest.raises(RuntimeError, match="after selected update"):
        instance.release(2, {0: forecast.prediction + 1.}, current_x=window())
    assert all(torch.equal(value, before[name]) for name, value in instance.model.state_dict().items())
    assert len(instance.pending_records) == 1 and instance.update_occasions == 0
    monkeypatch.setattr(instance, "_step", original_step)
    update = instance.release(2, {0: forecast.prediction + 1.}, current_x=window()).experts
    assert update.selected_channels == attempted[0].selected_channels
    assert update.selection_event_seed == attempted[0].selection_event_seed


@pytest.mark.parametrize("failure", ["warmup", "missing", "zero", "nonfinite", "no_current"])
def test_unavailable_lrp_is_explicit_and_rolls_back_the_complete_release(failure):
    instance = controller("relevance_only", reference=failure != "warmup", stage_i=True)
    name = next(iter(instance.weights))
    if failure == "missing":
        instance.reference_channels.pop(name)
    elif failure == "zero":
        instance.reference_channels[name].zero_()
    elif failure == "nonfinite":
        instance.reference_channels[name].fill_(float("nan"))
    forecast = instance.predict(window(), 0)
    before = deepcopy(instance.model.state_dict())
    with pytest.raises(SelectionUnavailableError, match="fixed_budget:relevance_only"):
        instance.release(2, {0: forecast.prediction + 1.}, current_x=None if failure == "no_current" else window())
    assert all(torch.equal(value, before[key]) for key, value in instance.model.state_dict().items())
    assert instance.metrics["count"] == 0 and instance.update_occasions == 0
    assert len(instance.pending_records) == 1 and not instance.records


@pytest.mark.parametrize("source", ["current", "reference"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), torch.finfo(torch.double).max])
def test_invalid_input_relevance_is_unavailable_even_with_valid_channel_scores(source, value, monkeypatch):
    instance = controller("relevance_only", stage_i=True)
    if source == "reference":
        instance.reference_maps = torch.full_like(instance.reference_maps, value)
    else:
        original_attribute = instance._attribute

        def invalid_input_maps(x):
            maps, channels = original_attribute(x)
            return torch.full_like(maps, value), channels

        monkeypatch.setattr(instance, "_attribute", invalid_input_maps)
    forecast = instance.predict(window(), 0)
    before = deepcopy(instance.model.state_dict())
    with pytest.raises(SelectionUnavailableError, match=f"invalid_input_profile:{source}:expert=0"):
        instance.release(2, {0: forecast.prediction + 1.}, current_x=window())
    assert all(torch.equal(parameter, before[name]) for name, parameter in instance.model.state_dict().items())
    assert instance.metrics["count"] == 0 and instance.update_occasions == 0
    assert len(instance.pending_records) == 1 and not instance.records


def test_nonfinite_input_discrepancy_is_unavailable(monkeypatch):
    instance = controller("relevance_only")
    monkeypatch.setattr(instance, "_distance", lambda a, b: torch.full((), float("nan")))
    with pytest.raises(SelectionUnavailableError, match="invalid_input_discrepancy:expert=0"):
        instance._relevance_scores(window())


@pytest.mark.parametrize("source", ["current", "reference"])
def test_overflowed_channel_normalization_is_unavailable(source, monkeypatch):
    instance = controller("relevance_only")
    name = next(iter(instance.weights))
    maximum = torch.finfo(torch.double).max
    if source == "reference":
        instance.reference_channels[name] = torch.full_like(instance.reference_channels[name], maximum)
    else:
        original_attribute = instance._attribute

        def overflowing_channels(x):
            maps, channels = original_attribute(x)
            channels[name] = torch.full_like(channels[name], maximum)
            return maps, channels

        monkeypatch.setattr(instance, "_attribute", overflowing_channels)
    with pytest.raises(SelectionUnavailableError, match="invalid_channel_profile"):
        instance._relevance_scores(window())


def test_zero_gradient_records_selected_capacity_separately_from_no_parameter_changes(monkeypatch):
    instance = controller()
    monkeypatch.setattr(instance, "_gradients", lambda loss, parameters: [torch.zeros_like(p) for p in parameters])
    before, update = release_once(instance)
    assert not update.accepted and update.selected_parameter_count == 7
    assert update.changed_parameter_count == 0 and "zero_gradient" in update.reason
    assert all(torch.equal(parameter, before[name]) for name, parameter in instance.model.named_parameters())


def test_rejected_step_keeps_selected_budget_but_reports_zero_changed_parameters(monkeypatch):
    instance = controller(max_backtracks=2)
    initial = {name: p.detach().clone() for name, p in instance.weights.items()}
    # Supply a deliberately wrong descent direction at the minimum of a known
    # objective, so every candidate step must fail the acceptance test.
    monkeypatch.setattr(instance, "_loss", lambda batch, gains:
                        1 + sum((parameter - initial[name]).square().sum()
                                for name, parameter in instance.weights.items()))
    monkeypatch.setattr(instance, "_gradients", lambda loss, parameters: [torch.ones_like(p) for p in parameters])
    before, update = release_once(instance)
    assert not update.accepted and update.trials == 2
    assert update.selected_parameter_count == 7 and update.changed_parameter_count == 0
    assert "backtracking_rejected" in update.reason
    assert all(torch.equal(parameter, before[name]) for name, parameter in instance.model.named_parameters())


def test_original_adaptive_selection_retains_positive_only_and_error_fallback():
    instance = controller("relevance_error", selection_protocol="adaptive", error_threshold=0.)
    scores = {name: torch.zeros(p.shape[0]) for name, p in instance.weights.items()}
    masks, selected = instance._select(scores)
    assert not selected and not any(mask.any() for mask in masks.values())
    _, update = release_once(instance)
    assert update.reason.startswith("stored_error_threshold")
    assert update.selection_protocol == "adaptive"


def test_fixed_budget_larger_than_the_candidate_set_is_rejected():
    with pytest.raises(ValueError, match="exceeds.*eligible channels"):
        controller(channel_budget={"first.weight": 7, "second.weight": 1})


@pytest.mark.parametrize("settings", [
    {"selection_protocol": "fixed_budget", "selector": "relevance_error"},
    {"selection_protocol": "adaptive", "selector": "random"},
    {"selection_protocol": "fixed_budget", "selector": "gradient_only", "stage_ii_scope": "dense_experts"},
    {"selection_seed": True}, {"selection_seed": -1}, {"selection_seed": 2**32},
])
def test_invalid_matched_protocol_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        OnlineConfig(**settings)
