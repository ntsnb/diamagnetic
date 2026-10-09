import pytest
import torch

from paper_impl.adaptation import OnlineConfig
from paper_impl.stream import DelayedFeedbackStream
from test_adaptation import make_controller, window, states, assert_identical


def test_complete_horizon_and_extra_delay_are_enforced_with_real_observation_timeline():
    controller = make_controller(OnlineConfig(stage_iii=False, stage_ii=False, extra_delay=2))
    events = []
    stream = DelayedFeedbackStream(controller, before_update=lambda batch, time: events.append(
        (time, batch.issue_times, batch.targets.detach().clone())))
    snapshot = states(controller.model)
    for time in range(4):
        _, update = stream.step(time, window(), torch.tensor([float(time)]))
        assert not update.issue_times
        assert_identical(snapshot, states(controller.model))
    _, update = stream.step(4, window(), torch.tensor([4.]))
    assert update.issue_times == (0,)
    assert events[0][0:2] == (4, (0,))
    assert torch.equal(events[0][2], torch.tensor([[[1.], [2.]]], dtype=torch.double))
    assert update.routing.accepted


def test_replay_callback_precedes_persistent_updates_and_receives_historical_inputs():
    controller = make_controller(OnlineConfig(stage_iii=False, stage_ii=False))
    callbacks = []
    def before(batch, time):
        callbacks.append((states(controller.model), batch.inputs.clone(), time))
    stream = DelayedFeedbackStream(controller, before_update=before)
    initial = states(controller.model)
    stream.step(0, window(), torch.tensor([0.]))
    stream.step(1, window() * 2, torch.tensor([1.]))
    stream.step(2, window() * 3, torch.tensor([2.]))
    assert_identical(callbacks[0][0], initial)
    assert torch.equal(callbacks[0][1], window().unsqueeze(0))
    assert callbacks[0][2] == 2


def test_stream_buffers_are_bounded_and_metrics_use_issued_forecasts():
    controller = make_controller(OnlineConfig(stage_i=False, stage_ii=False, stage_iii=False,
                                             extra_delay=2, pending_capacity=5,
                                             history_capacity=4, record_capacity=3))
    stream = DelayedFeedbackStream(controller)
    issued, actual = {}, {}
    for time in range(30):
        target = torch.tensor([float(time % 7)])
        actual[time] = target.double()
        forecast, _ = stream.step(time, window() * (time % 3 + 1), target)
        issued[time] = forecast.prediction
        assert len(stream.observations) <= 5
        assert len(controller.pending_records) <= 4
        assert len(controller.history) <= 4
        assert len(controller.records) <= 3
    errors = [issued[t] - torch.stack([actual[t + 1], actual[t + 2]]).unsqueeze(0)
              for t in range(26)]
    expected = torch.cat(errors).square().mean()
    assert controller.metrics["mse"] == pytest.approx(float(expected))
    assert controller.metrics["count"] == 26 * 2


def test_observation_validation_prevents_skipped_or_replayed_time():
    stream = DelayedFeedbackStream(make_controller(OnlineConfig(stage_iii=False)))
    stream.step(0, window(), torch.tensor([0.]))
    with pytest.raises(ValueError, match="consecutive"):
        stream.observe(2, torch.tensor([2.]))
    with pytest.raises(ValueError, match="consecutive"):
        stream.observe(0, torch.tensor([0.]))
    with pytest.raises(ValueError, match="currently observed"):
        stream.observe(1, torch.tensor([[1.], [2.]]))


def test_stream_reset_clears_observations_for_new_training_episode():
    stream = DelayedFeedbackStream(make_controller(OnlineConfig(stage_iii=False)))
    stream.step(5, window(), torch.tensor([5.]))
    stream.reset()
    assert not stream.observations and stream.last_observation is None
    stream.step(0, window(), torch.tensor([0.]))


def test_replay_resets_persistent_episodes_detaches_base_and_ignores_future_suffix(monkeypatch):
    import copy
    from torch import nn
    import paper_impl.replay as replay
    from paper_impl.adaptation import OnlineController

    class FrozenBranches(nn.Module):
        def forward(self, x):
            return x.mean(1)

        def freeze(self):
            return self.eval().requires_grad_(False)

    config = OnlineConfig(extra_delay=1, error_threshold=0., reference_windows=2)
    initial = make_controller(config)
    model, generator = initial.model, initial.generator
    original = states(model)
    second_generator = copy.deepcopy(generator)
    values = torch.stack((torch.linspace(.1, 2., 20), torch.linspace(2., .1, 20)), dim=-1).double()
    targets = values[:, :1] * 1.3
    captured = []
    def controller_factory(*args, **kwargs):
        controller = OnlineController(*args, **kwargs)
        captured.append((controller, states(controller.model)))
        return controller
    monkeypatch.setattr(replay, "OnlineController", controller_factory)
    replay_config = replay.ReplayConfig(episodes=2, max_windows=3)
    report = replay.train_generator(model, FrozenBranches(), generator, values, targets, 12,
                                    config, replay_config)
    altered_values, altered_targets = values.clone(), targets.clone()
    altered_values[12:] += 10000
    altered_targets[12:] -= 10000
    other_report = replay.train_generator(model, FrozenBranches(), second_generator,
                                          altered_values, altered_targets, 12, config, replay_config)
    assert_identical(original, states(model))
    assert_identical(states(generator), states(second_generator))
    assert report == other_report
    for controller, initial_state in captured:
        assert_identical(initial_state, original)
        assert all(parameter.grad is None for parameter in controller.model.parameters())
    assert report["episodes"][0]["max_gradient_norm"] > 0
    assert report["last_release"] < 12
    for episode in report["episodes"]:
        for event in episode["events"]:
            assert all(event["time"] >= t + 2 + 1 for t in event["issue_times"])


def test_evaluation_returns_issue_time_forecasts_after_later_parameter_updates(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import paper_impl.evaluation as evaluation
    from paper_impl.adaptation import OnlineController

    config = OnlineConfig(stage_ii=False, stage_iii=False, learning_rate_i=.1, radius_i=.1)
    initial = make_controller(config)
    values = torch.stack((torch.linspace(.1, 2., 12), torch.linspace(2., .1, 12)), dim=-1).double()
    data = SimpleNamespace(values=values, targets=values[:, :1],
                           metadata={"std": [1., 1.], "target_indices": [0]})
    split = SimpleNamespace(issue_times=np.array([3, 4, 5]))
    captured, issued = [], []
    def controller_factory(*args, **kwargs):
        controller = OnlineController(*args, **kwargs)
        captured.append(controller)
        original_predict = controller.predict
        def capture(x, time):
            record = original_predict(x, time)
            issued.append(record.clone())
            return record
        controller.predict = capture
        return controller
    monkeypatch.setattr(evaluation, "OnlineController", controller_factory)
    result = evaluation.evaluate_stream(initial.model, None, None, data, split, config)
    np.testing.assert_array_equal(result["predictions"], torch.cat([r.prediction for r in issued]).numpy())
    assert result["metrics"]["accepted_router_updates"] > 0
    final_prediction = captured[0].model(issued[0].inputs).prediction.detach().cpu()
    assert not torch.equal(final_prediction, issued[0].prediction)
