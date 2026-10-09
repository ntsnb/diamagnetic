"""Bounded protocol, algorithm-state and upstream-numerics tests (no datasets)."""
import ast
import copy
import math
from pathlib import Path
import typing

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from paper_impl.online_baselines import (
    OnlineBaselineConfig, ReleasedBatch, make_online_baseline,
    ProceedAdapter, _ProceedBottleneck, _ProceedLinear, _ProceedConv1d,
)


class TinyForecaster(nn.Module):
    def __init__(self, seed=4):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.Linear(8, 7)
            self.head = nn.Linear(7, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.encoder(x.flatten(1)))).unsqueeze(-1)

    def head_parameters(self):
        return self.head.parameters()


class ModeSensitiveForecaster(TinyForecaster):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(7)
        self.dropout = nn.Dropout(.5)
        self.forward_records = []

    def forward(self, x):
        record = {"training": self.training, "dropout": self.dropout.training,
                  "batch_norm": self.batch_norm.training,
                  "before": copy.deepcopy(self.state_dict())}
        representation = torch.tanh(self.batch_norm(self.encoder(x.flatten(1))))
        dropped = self.dropout(representation)
        record.update(representation=representation.detach().clone(), dropped=dropped.detach().clone())
        self.forward_records.append(record)
        return self.head(dropped).unsqueeze(-1)


def config(**kwargs):
    return OnlineBaselineConfig(lookback=4, horizon=2, target_indices=(0,),
                                learning_rate=0.015, concept_dim=5, bottleneck_dim=3,
                                **kwargs)


def data(n=8):
    rng = torch.Generator().manual_seed(28)
    x = torch.randn(n, 4, 2, generator=rng)
    y = x[:, -1:, :1].repeat(1, 2, 1) + 0.7 * x[:, :2, :1]
    return x, y


def method(name, **kwargs):
    instance = make_online_baseline(name, TinyForecaster(), config(**kwargs), secondary_model=TinyForecaster(19))
    if name == "PROCEED":
        x, y = data()
        instance.fit_offline(x, y, epochs=2, batch_size=2)
    return instance


def batch(issue=0, n=1):
    x, y = data(n)
    issues = tuple(range(issue, issue + n))
    return ReleasedBatch(x, y, issues, tuple(t + 2 for t in issues))


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def changed(before, model):
    return {name for name, tensor in model.state_dict().items()
            if isinstance(tensor, torch.Tensor) and not torch.equal(before[name], tensor.cpu())}


def test_gd_changes_backbone_and_head_without_mutating_supplied_model():
    source = TinyForecaster()
    initial = copy.deepcopy(source.state_dict())
    adapter = make_online_baseline("GD", source, config())
    report = adapter.update(batch(), time=2)
    assert report.steps == 1 and report.issue_times == (0,)
    assert any(name.startswith("encoder") for name in changed(initial, adapter.model))
    assert any(name.startswith("head") for name in changed(initial, adapter.model))
    equal_state(initial, source.state_dict())


def test_head_gd_keeps_representation_parameters_frozen():
    adapter = method("Head-GD")
    initial = copy.deepcopy(adapter.model.state_dict())
    adapter.update(batch(), time=2)
    assert changed(initial, adapter.model) == {"head.weight", "head.bias"}


@pytest.mark.parametrize("name", ["GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED"])
def test_unreleased_batch_is_rejected_atomically(name):
    adapter = method(name)
    before = adapter.state_dict()
    released = batch(issue=0, n=2)
    # First target is available, second target still contains the future.
    with pytest.raises(ValueError, match="future/unreleased"):
        adapter.update(released, time=2)
    equal_state(before, adapter.state_dict())
    # Forged available_at cannot bypass the inherent H-step delay.
    forged = ReleasedBatch(released.x, released.y, (0, 1), (1, 1))
    with pytest.raises(ValueError, match="future/unreleased"):
        adapter.update(forged, time=2)
    equal_state(before, adapter.state_dict())


def test_feedback_delay_and_duplicate_guards():
    adapter = method("GD", extra_delay=3)
    released = batch()
    with pytest.raises(ValueError, match="future/unreleased"):
        adapter.update(released, time=5)
    delayed = ReleasedBatch(released.x, released.y, (0,), (5,))
    adapter.update(delayed, time=5)
    with pytest.raises(ValueError, match="duplicate"):
        adapter.update(delayed, time=6)


@pytest.mark.parametrize("name", ["GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED"])
def test_checkpoint_restores_exact_next_prediction_and_update(name, tmp_path):
    adapter = method(name)
    x, _ = data(1)
    adapter.update(batch(), time=2, current_context=x)
    adapter.forecast(x, time=2)
    path = tmp_path / "online.pt"
    adapter.save_checkpoint(path)
    resumed = method(name)
    resumed.load_checkpoint(path)
    equal_state(adapter.state_dict(), resumed.state_dict())
    next_batch = batch(issue=1)
    a = adapter.update(next_batch, time=3, current_context=x)
    b = resumed.update(next_batch, time=3, current_context=x)
    assert a == b
    torch.testing.assert_close(adapter.forecast(x, 3), resumed.forecast(x, 3), rtol=0, atol=0)
    equal_state(adapter.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("name", ["GD", "OneNet", "SOLID", "SOLID++", "PROCEED"])
def test_reset_restores_pretrained_algorithm_state(name):
    adapter = method(name)
    initial = adapter.state_dict()
    x, _ = data(1)
    adapter.update(batch(), time=2, current_context=x)
    adapter.forecast(x, 2)
    adapter.reset()
    equal_state(initial, adapter.state_dict())


def test_solid_is_episodic_and_solid_plus_plus_is_continual():
    x, y = data(3)
    history = ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3))
    solid, continual = method("SOLID"), method("SOLID++")
    for controller in (solid, continual):
        controller.seed_history(history, time=3)
    frozen = copy.deepcopy(solid.model.state_dict())
    continual_before = copy.deepcopy(continual.model.state_dict())
    p1, p2 = solid.forecast(x[2:], 3), solid.forecast(x[2:], 4)
    torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    assert changed(frozen, solid.model) == {"head.weight", "head.bias"}
    p1, p2 = continual.forecast(x[2:], 3), continual.forecast(x[2:], 4)
    assert not torch.equal(p1, p2)
    assert any(name.startswith("encoder") for name in changed(continual_before, continual.model))


def test_solid_periodic_context_selection_and_no_double_update():
    adapter = method("SOLID", period=4, lambda_period=0.1, selected_data_num=2)
    x, y = data(4)
    history = ReleasedBatch(x[:3], y[:3], (0, 1, 3), (2, 3, 5))
    adapter.update(history, time=7, current_context=x[3:])
    assert adapter.last_update.selected_issue_times == (3,)
    steps = adapter.total_update_steps
    adapter.forecast(x[3:], 7)
    assert adapter.total_update_steps == steps
    with pytest.raises(ValueError, match="different current inputs"):
        adapter.forecast(x[3:] + 0.1, 7)


def test_solid_original_summed_gradient_matches_independent_sgd_oracle():
    adapter = method("SOLID", selected_data_num=2)
    x, y = data(3)
    adapter.seed_history(ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3)), time=3)
    reference = copy.deepcopy(adapter.model)
    reference.eval()
    parameters = list(reference.head_parameters())
    for parameter in parameters:
        parameter.grad = None
    for i in range(2):
        F.mse_loss(reference(x[i:i + 1]), y[i:i + 1]).backward()
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(parameter.grad, alpha=-adapter.config.learning_rate)
    adapter.forecast(x[2:], 3)
    for expected, actual in zip(reference.head_parameters(), adapter.model.head_parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mode,reduction", [("eval", "sum"), ("train", "mean")])
def test_solid_explicit_mode_controls_dropout_bn_and_only_head_gradients(mode, reduction):
    source = ModeSensitiveForecaster()
    adapter = make_online_baseline("SOLID", source, config(solid_adaptation_mode=mode,
                                   solid_gradient_reduction=reduction, selected_data_num=2))
    x, y = data(3)
    adapter.seed_history(ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3)), time=3)
    parameters_before = {key: p.detach().clone() for key, p in adapter.model.named_parameters()}
    adapter.forecast(x[2:], 3)
    adaptation, prediction = adapter.model.forward_records
    assert adaptation["training"] == adaptation["dropout"] == adaptation["batch_norm"] == (mode == "train")
    assert not prediction["training"] and not any(module.training for module in adapter.model.modules())
    assert int(adapter.model.batch_norm.num_batches_tracked) == int(mode == "train")
    altered = {key for key, p in adapter.model.named_parameters() if not torch.equal(parameters_before[key], p)}
    assert altered == {"head.weight", "head.bias"}
    assert {key for key, p in adapter.model.named_parameters() if p.requires_grad} == altered
    if mode == "eval":
        assert torch.equal(adaptation["representation"], adaptation["dropped"])
        assert adapter.source_variant == "solid_original_eval_sum"
    else:
        assert (adaptation["dropped"] == 0).any()
        kept = adaptation["dropped"] != 0
        assert kept.any()
        torch.testing.assert_close(adaptation["dropped"][kept], 2 * adaptation["representation"][kept])
        assert adapter.source_variant == "onlinetsf_train_mean_full_state_reset"


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_solid_resets_complete_pretrained_state_before_every_new_input(mode):
    adapter = make_online_baseline("SOLID", ModeSensitiveForecaster(),
                                   config(solid_adaptation_mode=mode, selected_data_num=2))
    x, y = data(3)
    adapter.seed_history(ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3)), time=3)
    pretrained = copy.deepcopy(adapter._pretrained)
    adapter.forecast(x[2:], 3)
    with torch.no_grad():
        for p in adapter.model.parameters():
            p.add_(17)
        adapter.model.batch_norm.running_mean.fill_(37)
        adapter.model.batch_norm.running_var.fill_(91)
        adapter.model.batch_norm.num_batches_tracked.fill_(200)
    adapter.forecast(x[2:], 4)
    equal_state(adapter.model.forward_records[-2]["before"], pretrained)
    assert int(adapter.model.batch_norm.num_batches_tracked) == int(mode == "train")


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_solid_mode_dropout_and_bn_state_resume_exactly(mode):
    source = ModeSensitiveForecaster()
    cfg = config(solid_adaptation_mode=mode, selected_data_num=2)
    first = make_online_baseline("SOLID", source, cfg)
    x, y = data(3)
    first.seed_history(ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3)), time=3)
    first.forecast(x[2:], 3)
    saved = first.state_dict()
    assert saved["config"]["solid_adaptation_mode"] == mode
    restored = make_online_baseline("SOLID", source, cfg)
    restored.load_state_dict(saved)
    assert torch.equal(first.forecast(x[2:], 4), restored.forecast(x[2:], 4))
    equal_state(first.state_dict(), restored.state_dict())


@pytest.mark.parametrize("requested_mode", ["eval", "train"])
def test_solid_plus_plus_always_trains_and_preserves_bn_history(requested_mode):
    adapter = make_online_baseline("SOLID++", ModeSensitiveForecaster(),
                                   config(solid_adaptation_mode=requested_mode, selected_data_num=2))
    x, y = data(3)
    adapter.seed_history(ReleasedBatch(x[:2], y[:2], (0, 1), (2, 3)), time=3)
    assert adapter.adaptation_mode == "train" and adapter.gradient_reduction == "mean"
    assert adapter.source_variant == "onlinetsf_continual_train_mean"
    for time in (3, 4):
        adapter.forecast(x[2:], time)
        assert adapter.model.forward_records[-2]["dropout"]
    assert int(adapter.model.batch_norm.num_batches_tracked) == 2


def test_solid_mode_validation_and_memory_age_boundary():
    assert config().solid_adaptation_mode == "eval"
    with pytest.raises(ValueError, match="solid_adaptation_mode"):
        config(solid_adaptation_mode="automatic")
    adapter = method("SOLID", extra_delay=1, memory_size=3, selected_data_num=3)
    x, y = data(4)
    adapter.seed_history(ReleasedBatch(x[:3], y[:3], (0, 1, 2), (3, 4, 5)), time=5)
    adapter.forecast(x[3:], 6)
    # H + delay + memory_size - 1 = 5: age 5 is included, age 6 is not.
    assert set(adapter.last_update.selected_issue_times) == {1, 2}
    adapter.forecast(x[3:], 8)
    assert not adapter.last_update.selected_issue_times and adapter.last_update.steps == 0


def test_onenet_updates_both_branches_global_weight_and_decision():
    adapter = method("OneNet")
    a, b = copy.deepcopy(adapter.model.state_dict()), copy.deepcopy(adapter.secondary_model.state_dict())
    decision = copy.deepcopy(adapter.decision.state_dict())
    adapter.update(batch(), time=2)
    assert changed(a, adapter.model) and changed(b, adapter.secondary_model)
    assert changed(decision, adapter.decision)
    assert torch.count_nonzero(adapter.weight) and torch.count_nonzero(adapter.bias)
    x, _ = data(1)
    prediction = adapter.forecast(x, 2)
    with torch.no_grad():
        first, second = adapter._branches(x)
        assert not torch.allclose(prediction, (first + second) / 2)


def test_onenet_requires_explicit_second_branch_and_has_offline_training():
    model = TinyForecaster()
    with pytest.raises(ValueError, match="secondary_model"):
        make_online_baseline("OneNet", model, config())
    with pytest.raises(ValueError, match="secondary_model"):
        make_online_baseline("OneNet", model, config(), secondary_model=model)
    adapter = method("OneNet")
    x, y = data()
    report = adapter.fit_offline(x, y, epochs=1, batch_size=2, issue_times=tuple(range(8)), cutoff=9)
    assert report["epochs"] == 1 and adapter.offline_epochs == 1
    assert torch.count_nonzero(adapter.weight)


def test_proceed_rejects_untrained_adapter_and_trains_every_component():
    adapter = make_online_baseline("PROCEED", TinyForecaster(), config())
    x, y = data()
    with pytest.raises(RuntimeError, match="fit_offline"):
        adapter.forecast(x[:1], 0)
    before = copy.deepcopy(adapter.adapter.state_dict())
    backbone_before = copy.deepcopy(adapter.model.state_dict())
    report = adapter.fit_offline(x, y, epochs=2, batch_size=2, issue_times=tuple(range(8)), cutoff=9)
    assert report["training_steps"] == 8
    names = changed(before, adapter.adapter)
    assert any(n.startswith("current_encoder") for n in names)
    assert any(n.startswith("history_encoder") for n in names)
    assert any(n.startswith("generators") for n in names)
    assert changed(backbone_before, adapter.model)


def test_proceed_online_freezes_concept_and_generator_weights_but_updates_biases():
    adapter = method("PROCEED")
    before = copy.deepcopy(adapter.adapter.state_dict())
    adapter.update(batch(), time=2)
    updated = changed(before, adapter.adapter)
    assert updated and all(name.endswith("output_bias") for name in updated)
    x, _ = data(1)
    backbone = copy.deepcopy(adapter.model.state_dict())
    p1, p2 = adapter.forecast(x, 2), adapter.forecast(x + 1.5, 3)
    assert not torch.equal(p1, p2)
    equal_state(backbone, adapter.model.state_dict())
    assert all(adapter.model.get_submodule(name).adaptation is None
               for names in adapter.adapter.groups.values() for name in names)


def test_proceed_initial_scaling_is_identity_even_with_exogenous_target_subset():
    source = TinyForecaster()
    model = copy.deepcopy(source)
    adapter = ProceedAdapter(model, config())
    x, _ = data(2)
    adapter.assign(model, torch.randn(2, 5))
    torch.testing.assert_close(source(x), model(x), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("convolution", [False, True])
def test_proceed_activation_scaling_matches_explicit_per_sample_parameter_map(convolution):
    torch.manual_seed(21)
    layer = _ProceedConv1d(2, 3, 3, padding=1) if convolution else _ProceedLinear(2, 3)
    x = torch.randn(2, 2, 5) if convolution else torch.randn(2, 5, 2)
    a, b, shift = torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 3)
    layer.adaptation = (a, b, shift)
    expected = []
    for i in range(2):
        matrix = a[i, :, None] * b[i, None, :]
        weight = layer.weight * (matrix[:, :, None] if convolution else matrix)
        bias = layer.bias + shift[i]
        expected.append(F.conv1d(x[i:i + 1], weight, bias, padding=1) if convolution
                        else F.linear(x[i:i + 1], weight, bias))
    torch.testing.assert_close(layer(x), torch.cat(expected), rtol=1e-5, atol=1e-6)


def test_proceed_generator_numerical_parity_with_unmodified_author_class():
    # Load only the unmodified Bottleneck class AST from the pinned official
    # snapshot: unrelated transformers imports are not needed for this oracle.
    path = Path(__file__).resolve().parents[2] / "third_party/official_online/OnlineTSF/adapter/module/generator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Bottleneck")
    namespace = {"torch": torch, "nn": nn, "math": math, "typing": typing}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    upstream = namespace["Bottleneck"](5, 9, 2, 3, activation=nn.Sigmoid, shared=True, need_bias=True)
    port = _ProceedBottleneck(5, 9, 2, 3)
    with torch.no_grad():
        port.up.weight.normal_()
        port.output_bias.normal_()
        upstream.weights[0].weight.copy_(port.down.weight)
        upstream.weights[1].copy_(port.up.weight[None, None])
        upstream.biases[0].copy_(port.layer_bias)
        upstream.biases[1].copy_(port.output_bias)
    drift = torch.randn(4, 5)
    torch.testing.assert_close(port(drift), upstream(drift), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["OneNet", "PROCEED"])
def test_offline_cutoff_rejects_future_training_targets(name):
    adapter = make_online_baseline(name, TinyForecaster(), config(), secondary_model=TinyForecaster(5))
    x, y = data()
    before = adapter.state_dict()
    with pytest.raises(ValueError, match="cutoff"):
        adapter.fit_offline(x, y, issue_times=tuple(range(8)), cutoff=8)
    equal_state(before, adapter.state_dict())


@pytest.mark.parametrize("name", ["OneNet", "PROCEED"])
def test_offline_cutoff_includes_extra_feedback_delay(name):
    adapter = method(name, extra_delay=3)
    x, y = data(8)
    before = adapter.state_dict()
    with pytest.raises(ValueError, match="release crosses training cutoff"):
        adapter.fit_offline(x, y, issue_times=tuple(range(8)), cutoff=9)
    equal_state(before, adapter.state_dict())


def test_predictions_are_fixed_clones_and_protocol_mismatch_cannot_resume():
    adapter = method("GD")
    x, _ = data(1)
    issued = adapter.forecast(x, 0)
    fixed = issued.clone()
    adapter.update(batch(), time=2)
    torch.testing.assert_close(issued, fixed, rtol=0, atol=0)
    incompatible = method("GD", extra_delay=1)
    with pytest.raises(ValueError, match="protocol"):
        incompatible.load_state_dict(adapter.state_dict())


@pytest.mark.parametrize("backbone", ["itransformer", "crossformer", "autoformer", "transformer",
                                      "patchtst", "informer", "fedformer", "pyraformer"])
@pytest.mark.parametrize("name", ["GD", "Head-GD", "OneNet", "SOLID", "SOLID++", "PROCEED"])
def test_all_methods_integrate_with_complete_official_backbones(backbone, name):
    from paper_impl.official_backbones import OfficialForecastConfig, OfficialForecaster

    def official(model_name):
        return OfficialForecaster(OfficialForecastConfig(
            model_name=model_name, lookback=24, horizon=8, n_variables=3,
            target_indices=(0,), hidden=16, heads=8, depth=2, dropout=0.,
            patch_len=4, patch_stride=2, moving_avg=3, fedformer_modes=4, crossformer_seg_len=4))

    torch.manual_seed(2026)
    cfg = OnlineBaselineConfig(lookback=24, horizon=8, target_indices=(0,), learning_rate=1e-4,
                               concept_dim=5, bottleneck_dim=3, selected_data_num=2)
    primary = official(backbone)
    secondary = official("patchtst" if backbone != "patchtst" else "itransformer") if name == "OneNet" else None
    controller = make_online_baseline(name, primary, cfg, secondary_model=secondary)
    x, y = torch.randn(4, 24, 3), torch.randn(4, 8, 1)
    if name in ("PROCEED", "OneNet"):
        controller.fit_offline(x, y, epochs=1, batch_size=2)
    initial = copy.deepcopy(controller.model.state_dict())
    report = controller.update(ReleasedBatch(x[:1], y[:1], (0,), (8,)), time=8, current_context=x[1:2])
    prediction = controller.forecast(x[1:2], time=8)
    assert prediction.shape == (1, 8, 1) and torch.isfinite(prediction).all()
    assert report.steps == 1
    assert changed(initial, controller.model)
