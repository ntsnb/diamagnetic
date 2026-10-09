"""Mathematical/state tests for the actual online operators, using an inspectable MoE."""

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from paper_impl.adaptation import CenteredBoundedGenerator, OnlineConfig, OnlineController


class ToyExpert(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1., .2], [.1, .7]]) * (index + 1))
            self.linear.bias.copy_(torch.tensor([.1, .2]))


class ToyForecaster(nn.Module):
    def __init__(self, horizon=2, targets=1):
        super().__init__()
        self.config = SimpleNamespace(horizon=horizon, targets=targets, n_experts=2,
                                      lookback=3, n_variables=2)
        self.experts = nn.ModuleList([ToyExpert(i) for i in range(2)])
        self.router = nn.Linear(2, 2)
        self.output_head = nn.Linear(2, horizon * targets)
        with torch.no_grad():
            self.router.weight.copy_(torch.tensor([[.1, .2], [.2, -.1]]))
            self.router.bias.zero_()
            self.output_head.weight.fill_(.5)
            self.output_head.bias.zero_()

    def expert_weights(self):
        return {f"experts.{m}.linear.weight": expert.linear.weight
                for m, expert in enumerate(self.experts)}

    def forward(self, x, stochastic=False, modulations=None):
        h = x.mean(1)
        branches = []
        for m, expert in enumerate(self.experts):
            name = f"experts.{m}.linear.weight"
            weight = expert.linear.weight
            if modulations:
                embedding = torch.einsum("bi,boi->bo", h, weight * modulations[name])
            else:
                embedding = h @ weight.T
            branches.append(embedding + expert.linear.bias)
        gates = self.router(h).softmax(-1)
        mixed = (torch.stack(branches, 1) * gates.unsqueeze(-1)).sum(1)
        prediction = self.output_head(mixed).reshape(len(x), self.config.horizon, self.config.targets)
        return SimpleNamespace(prediction=prediction, gates=gates)

    def attribute(self, x):
        # This toy's explicit linear relevance is only for auditing the controller.
        # Production Transformer LRP is tested separately in test_lrp.py.
        maps, channels = [], {}
        for m, expert in enumerate(self.experts):
            relevance = (x * expert.linear.weight.abs().sum(0)).abs().transpose(1, 2)
            maps.append(relevance[:, None].expand(-1, self.config.targets, -1, -1))
            node_relevance = torch.einsum("bli,oi->blo", x, expert.linear.weight).abs().sum(1)
            channels[f"experts.{m}.linear.weight"] = node_relevance[:, None].expand(
                -1, self.config.targets, -1)
        return SimpleNamespace(maps=torch.stack(maps, 2), channels=channels)


def make_controller(config=None, **kwargs):
    torch.manual_seed(11)
    model = ToyForecaster().double()
    generator = CenteredBoundedGenerator({n: tuple(p.shape) for n, p in model.expert_weights().items()},
                                         2, hidden_dim=3, epsilon=.2).double()
    with torch.no_grad():
        for block in generator.blocks.values():
            block.w2.weight.fill_(1.)
    contrast = lambda x: x.mean(1)
    controller = OnlineController(model, config or OnlineConfig(), generator=generator,
                                  contrast_fn=contrast, **kwargs)
    return controller


def window():
    return torch.tensor([[1., .2], [.8, .4], [1.1, .1]], dtype=torch.double)


def states(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_identical(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[name], b[name]) for name in a)


@pytest.mark.parametrize("switches", itertools.product((False, True), repeat=3))
def test_all_eight_stage_combinations_run_and_preserve_frozen_parameters(switches):
    cfg = OnlineConfig(stage_i=switches[0], stage_ii=switches[1], stage_iii=switches[2],
                       error_threshold=0., reference_windows=2)
    controller = make_controller(cfg)
    before = states(controller.model)
    forecast = controller.predict(window(), 0)
    record = controller.release(2, {0: forecast.prediction + 1.}, current_x=window())
    after = states(controller.model)
    allowed = set(controller.weights) if switches[1] else set()
    if switches[0]:
        allowed.add("router.bias")
    for name in before:
        if name not in allowed:
            assert torch.equal(before[name], after[name]), name
    assert record.routing.accepted == switches[0]
    assert record.experts.accepted == switches[1]
    assert len(controller.pending_records) == 0
    assert controller.metrics["mse"] == pytest.approx(1.)


def test_generator_identity_bound_orientation_and_learning_signal():
    torch.manual_seed(5)
    generator = CenteredBoundedGenerator({"experts.0.fc.weight": (3, 2)}, 4, epsilon=.3).double()
    contrast = torch.randn(7, 4, dtype=torch.double)
    initial = generator(contrast)["experts.0.fc.weight"]
    assert initial.shape == (7, 3, 2)
    assert torch.equal(initial, torch.ones_like(initial))
    initial.sum().backward()
    block = generator.blocks["layer_0"]
    assert block.w2.weight.grad.abs().sum() > 0
    assert block.w1.weight.detach().abs().sum() > 0
    with torch.no_grad():
        block.w2.weight.normal_(0, 10)
    gains = generator(contrast)["experts.0.fc.weight"]
    zero = generator(torch.zeros_like(contrast))["experts.0.fc.weight"]
    assert torch.equal(zero, torch.ones_like(zero))
    assert bool(((1 - .3) ** 2 <= gains).all() and (gains <= (1 + .3) ** 2).all())
    weight = torch.randn(3, 2, dtype=torch.double)
    difference = torch.linalg.vector_norm((weight * gains - weight).flatten(1), dim=1)
    assert (difference <= (2 * .3 + .3 ** 2) * weight.norm()).all()
    alpha, beta = generator.gain_vectors(contrast)["experts.0.fc.weight"]
    assert torch.equal(gains, beta.unsqueeze(-1) * alpha.unsqueeze(-2))


def test_generator_identity_ablation_and_matching_shape_sharing():
    shapes = {"experts.0.fc.weight": (3, 2), "experts.1.fc.weight": (3, 2)}
    generator = CenteredBoundedGenerator(shapes, 2, mode="identity", share_matching_shapes=True)
    assert len(generator.blocks) == 0
    assert not list(generator.parameters())
    shared = CenteredBoundedGenerator(shapes, 2, share_matching_shapes=True)
    assert len(shared.blocks) == 1
    for gains in generator(torch.randn(4, 2)).values():
        assert torch.equal(gains, torch.ones(4, 3, 2))


def test_no_release_and_early_release_never_update_or_accept_future_labels():
    controller = make_controller(OnlineConfig(extra_delay=3))
    snapshot = states(controller.model)
    forecast = controller.predict(window(), 10)
    record = controller.release(11, {})
    assert record.routing.reason == "no_released_targets"
    with pytest.raises(ValueError, match="unavailable until 15"):
        controller.release(14, {10: torch.zeros_like(forecast.prediction)})
    assert_identical(snapshot, states(controller.model))
    controller.release(15, {10: forecast.prediction})
    with pytest.raises(KeyError):
        controller.release(16, {10: forecast.prediction})


def test_router_and_expert_steps_are_clipped_armijo_descent_with_exact_support():
    config = OnlineConfig(error_threshold=0., learning_rate_i=100., learning_rate_ii=100.,
                          radius_i=.01, radius_ii=.02, channel_budget=1)
    controller = make_controller(config)
    before = states(controller.model)
    forecast = controller.predict(window(), 0)
    record = controller.release(2, {0: forecast.prediction + 2}, current_x=window())
    assert record.routing.update_norm <= .01 + 1e-12
    assert record.experts.update_norm <= .02 + 1e-12
    for step in (record.routing, record.experts):
        assert step.accepted
        assert step.loss_after <= step.loss_before - config.armijo_c * step.step_size * step.gradient_norm ** 2
    assert record.experts.loss_before == pytest.approx(record.routing.loss_after)
    assert sum(len(v) for v in record.experts.selected_channels.values()) == 1
    for name, parameter in controller.weights.items():
        rows = record.experts.selected_channels.get(name, ())
        for channel in range(parameter.shape[0]):
            if channel not in rows:
                assert torch.equal(parameter[channel].detach(), before[name][channel])
    for name, value in controller.model.state_dict().items():
        if name.endswith("bias") and name != "router.bias":
            assert torch.equal(value, before[name])


def test_loss_gradient_passes_through_frozen_gains_to_persistent_weights():
    config = OnlineConfig(stage_i=False, selector="gradient_only", learning_rate_ii=1e-3,
                          radius_ii=1., channel_budget=4)
    controller = make_controller(config)
    forecast = controller.predict(window(), 0)
    targets = {0: forecast.prediction + .5}
    batch = controller.prepare_release(2, targets)
    gains = controller.modulations(batch.inputs)
    loss = (controller.model(batch.inputs, modulations=gains).prediction - batch.targets).square().mean()
    parameters = list(controller.weights.values())
    gradients = torch.autograd.grad(loss, parameters)
    expected = [p.detach() - config.learning_rate_ii * g for p, g in zip(parameters, gradients)]
    record = controller.release(2, targets, current_x=window())
    assert record.experts.accepted
    for parameter, target in zip(parameters, expected):
        torch.testing.assert_close(parameter.detach(), target, rtol=0, atol=1e-14)
    assert all(parameter.grad is None for parameter in controller.generator.parameters())
    assert all(parameter.requires_grad for parameter in controller.generator.parameters())


def test_stored_forecasts_are_immutable_and_error_trigger_uses_them():
    controller = make_controller(OnlineConfig(stage_i=False, stage_iii=False, error_threshold=.01))
    forecast = controller.predict(window(), 0)
    original = forecast.prediction.clone()
    forecast.prediction.add_(100)
    assert torch.equal(controller.pending_records[0].prediction, original)
    # Make the current predictor fit the released target; the issued forecast still erred.
    with torch.no_grad():
        controller.model.output_head.bias.add_(1)
    record = controller.release(2, {0: original + 1}, current_x=window())
    assert record.stored_error == pytest.approx(1.)
    assert record.experts.reason.startswith("stored_error_threshold")
    assert record.experts.loss_before < 1e-28
    assert controller.metrics["mse"] == pytest.approx(1.)


def test_temporary_modulation_never_accumulates_into_model_parameters():
    controller = make_controller(OnlineConfig(stage_i=False, stage_ii=False, stage_iii=True))
    before = states(controller.model)
    first = controller.predict(window(), 0)
    controller.release(2, {0: first.prediction})
    second = controller.predict(window(), 2)
    assert_identical(before, states(controller.model))
    assert torch.equal(first.prediction, second.prediction)
    unmodulated = controller.model(window().unsqueeze(0)).prediction
    assert not torch.equal(first.prediction, unmodulated)


def test_reference_is_fixed_then_ema_uses_only_persistent_expert_weights():
    initial_windows = window().unsqueeze(0).repeat(2, 1, 1)
    controller = make_controller(OnlineConfig(stage_i=False, error_threshold=0.,
                                             reference_windows=2, reference_interval=2,
                                             reference_ema=.25), reference_inputs=initial_windows)
    reference_before = states(controller.reference)
    windows_before = controller.reference_windows.clone()
    first = controller.predict(window(), 0)
    controller.release(2, {0: first.prediction + 2}, current_x=window())
    assert_identical(reference_before, states(controller.reference))
    assert torch.equal(controller.reference_windows, windows_before)
    second = controller.predict(window() * 2, 2)
    update = controller.release(4, {2: second.prediction + 2}, current_x=window() * 3)
    assert update.reference_refreshed and controller.reference_version == 1
    for name, parameter in controller.reference.named_parameters():
        expected = (.75 * reference_before[name] + .25 * dict(controller.model.named_parameters())[name]
                    if name in controller.weights else reference_before[name])
        torch.testing.assert_close(parameter, expected, rtol=0, atol=2e-16)
    assert not torch.equal(controller.reference_windows, windows_before)


def test_relevance_selection_preserves_expert_identity_and_channel_budget():
    reference = torch.tensor([[[2., .1]] * 3] * 2, dtype=torch.double)
    controller = make_controller(OnlineConfig(stage_i=False, stage_iii=False, reference_windows=2,
                                             stability_threshold=1., error_threshold=float("inf")),
                                 reference_inputs=reference)
    x = torch.tensor([[.1, 2.]] * 3, dtype=torch.double)
    record = controller.predict(x, 0)
    update = controller.release(2, {0: record.prediction + 1}, current_x=x)
    assert update.experts.reason == "relevance_drift"
    assert all(s is not None and s < 1 for s in update.stability)
    assert sum(len(v) for v in update.experts.selected_channels.values()) == 1


def test_zero_reference_map_makes_full_comparison_unavailable_without_false_distance():
    reference = torch.stack([window(), torch.zeros_like(window())])
    controller = make_controller(OnlineConfig(stage_i=False, stage_iii=False, reference_windows=2),
                                 reference_inputs=reference)
    before = states(controller.model)
    forecast = controller.predict(window(), 0)
    update = controller.release(2, {0: forecast.prediction + 1}, current_x=window())
    assert update.stability == (None, None)
    assert not update.experts.selected_channels
    assert not update.experts.accepted
    assert_identical(before, states(controller.model))


def test_fixed_tie_order_budget_across_all_experts_and_pytorch_row_mask():
    controller = make_controller(OnlineConfig(stage_iii=False, channel_budget=2))
    scores = {name: torch.ones(2) for name in controller.weights}
    masks, selected = controller._select(scores)
    assert selected == {"experts.0.linear.weight": (0, 1)}
    assert torch.equal(masks["experts.0.linear.weight"], torch.ones(2, 2, dtype=torch.double))
    assert not masks["experts.1.linear.weight"].any()


def test_reset_restores_predictor_reference_and_bounded_state_not_generator_training():
    controller = make_controller(OnlineConfig(error_threshold=0., reference_windows=1,
                                             reference_interval=1))
    initial = states(controller.model)
    forecast = controller.predict(window(), 0)
    controller.release(2, {0: forecast.prediction + 1}, current_x=window())
    with torch.no_grad():
        next(controller.generator.parameters()).add_(.1)
    learned_generator = states(controller.generator)
    controller.reset()
    assert_identical(initial, states(controller.model))
    assert_identical(initial, states(controller.reference))
    assert_identical(learned_generator, states(controller.generator))
    assert not controller.pending_records and not controller.history and not controller.records
    assert controller.metrics["count"] == 0


def test_real_forecaster_controller_interface_runs():
    from paper_impl.model import ModelConfig, PaperForecaster
    model = PaperForecaster(ModelConfig(n_variables=2, lookback=3, horizon=1, targets=1,
                                        n_experts=2, hidden=4, heads=1))
    controller = OnlineController(model, OnlineConfig(stage_iii=False, error_threshold=0.,
                                                       reference_windows=1))
    x = torch.tensor([[.1, .4], [.3, .2], [.5, .6]])
    record = controller.predict(x, 0)
    update = controller.release(1, {0: record.prediction + 1}, current_x=x)
    assert update.routing.accepted
    assert update.experts.accepted
    assert update.experts.loss_after <= update.experts.loss_before


def test_backtracking_exhaustion_returns_exact_original_parameters():
    config = OnlineConfig(stage_i=False, stage_iii=False, selector="gradient_only",
                          learning_rate_ii=1e6, radius_ii=1e6, max_backtracks=1)
    controller = make_controller(config)
    before = states(controller.model)
    record = controller.predict(window(), 0)
    update = controller.release(2, {0: record.prediction + 1}, current_x=window())
    assert not update.experts.accepted
    assert update.experts.reason.endswith("backtracking_rejected")
    assert update.experts.loss_after == update.experts.loss_before
    assert_identical(before, states(controller.model))


def test_exception_after_router_step_rolls_back_entire_release(monkeypatch):
    controller = make_controller(OnlineConfig(reference_windows=1))
    forecast = controller.predict(window(), 0)
    before = states(controller.model)
    def fail(_):
        raise RuntimeError("Injected attribution failure")
    monkeypatch.setattr(controller, "_relevance_scores", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        controller.release(2, {0: forecast.prediction + 1}, current_x=window())
    assert_identical(before, states(controller.model))
    assert len(controller.pending_records) == 1
    assert controller.metrics["count"] == 0
    assert not controller.records and controller.update_occasions == 0


def test_multiple_fully_released_horizons_form_one_chronological_batch():
    controller = make_controller(OnlineConfig(stage_i=False, stage_ii=False, stage_iii=False))
    first = controller.predict(window(), 0)
    second = controller.predict(window() * 2, 1)
    targets = {1: second.prediction + 2, 0: first.prediction + 1}
    batch = controller.prepare_release(3, targets)
    assert batch.issue_times == (0, 1)
    assert torch.equal(batch.inputs[0], window())
    update = controller.release(3, targets)
    assert update.stored_error == pytest.approx(2.5)
    assert controller.metrics["count"] == 4


def test_real_forecaster_per_sample_temporary_gains_with_multiple_release_records():
    from paper_impl.model import ModelConfig, PaperForecaster, ForecastBranches
    config = ModelConfig(n_variables=2, lookback=3, horizon=1, n_experts=2, hidden=4, heads=1)
    model = PaperForecaster(config)
    branches = ForecastBranches(config).freeze()
    generator = CenteredBoundedGenerator({name: tuple(value.shape)
                                         for name, value in model.expert_weights().items()}, 1)
    with torch.no_grad():
        for block in generator.blocks.values():
            block.w2.weight.normal_(0, .2)
    controller = OnlineController(model, OnlineConfig(error_threshold=0., reference_windows=2),
                                  generator=generator, contrast_fn=branches)
    x = window().float()
    first = controller.predict(x, 0)
    second = controller.predict(x * 2, 1)
    update = controller.release(2, {0: first.prediction + 1, 1: second.prediction + 1}, current_x=x * 3)
    assert update.routing.accepted and update.experts.accepted
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in branches.parameters())
