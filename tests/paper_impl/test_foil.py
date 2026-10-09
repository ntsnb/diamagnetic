"""FOIL 数学、全局样本索引、参数所有权及离线训练的有界验证。"""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from paper_impl.foil import (
    FoilConfig,
    assign_environments,
    infer_environments,
    initialize_environments,
    irn_loss,
    mei_step,
    mil_objective,
    mil_step,
    repair_empty_environments,
    set_foil_phase,
    smooth_environment_labels,
    train_foil_branches,
)


@pytest.fixture(autouse=True, scope="module")
def _small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyBranch(nn.Module):
    def __init__(self, horizon=3, targets=2):
        super().__init__()
        self.config = SimpleNamespace(horizon=horizon, targets=targets)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(4, 6))
        self.head = nn.Linear(6, horizon * targets)

    def forward(self, x):
        return self.head(self.encoder(x)).reshape(-1, self.config.horizon, self.config.targets)


class TinyBranches(nn.Module):
    def __init__(self, horizon=3, targets=2):
        super().__init__()
        self.invariant = TinyBranch(horizon, targets)
        self.variant = TinyBranch(horizon, targets)


def _data(n=16, horizon=3, targets=2, seed=19):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, 2, dtype=torch.float64, generator=generator)
    weights = torch.arange(4 * horizon * targets, dtype=torch.float64).reshape(4, -1)
    weights = torch.sin(weights * 1.7) / 2
    y = (x.flatten(1) @ weights + torch.arange(horizon * targets) * .17).reshape(n, horizon, targets)
    return x, y


def _config(**kwargs):
    fields = dict(n_environments=2, rounds=2, mil_epochs=2, mei_epochs=2, batch_size=3,
                  mil_learning_rate=.01, mei_learning_rate=.02, smoothing_radius=1, seed=91)
    fields.update(kwargs)
    return FoilConfig(**fields)


def _heads(branch, environments=2):
    return nn.ModuleList([nn.Linear(6, branch.config.horizon * branch.config.targets).double()
                          for _ in range(environments)])


def test_irn_normalizes_each_sample_and_target_separately_with_population_std():
    prediction = torch.tensor([[[0., 100.], [1., 110.], [2., 120.]]], dtype=torch.float64)
    target = prediction.flip(-2)
    torch.testing.assert_close(irn_loss(prediction, target), torch.tensor(4., dtype=torch.float64))
    scale = torch.tensor([[[2., .4]]], dtype=torch.float64)
    shift = torch.tensor([[[15., -600.]]], dtype=torch.float64)
    assert irn_loss(prediction, prediction * scale + shift).item() < 1e-25


def test_irn_is_positive_scale_and_shift_invariant_above_floor():
    generator = torch.Generator().manual_seed(13)
    prediction = torch.randn(5, 7, 3, dtype=torch.float64, generator=generator)
    target = torch.randn(5, 7, 3, dtype=torch.float64, generator=generator)
    pscale = torch.rand(5, 1, 3, dtype=torch.float64, generator=generator) + .2
    yscale = torch.rand(5, 1, 3, dtype=torch.float64, generator=generator) + .2
    pshift = torch.randn(5, 1, 3, dtype=torch.float64, generator=generator)
    yshift = torch.randn(5, 1, 3, dtype=torch.float64, generator=generator)
    torch.testing.assert_close(
        irn_loss(prediction, target, reduction="none"),
        irn_loss(prediction * pscale + pshift, target * yscale + yshift, reduction="none"),
        rtol=1e-12, atol=1e-12,
    )


def test_irn_keeps_normalization_gradients_and_constant_predictions_finite():
    _, target = _data(n=4)
    prediction = (target.flip(1) + .2 * torch.randn_like(target)).requires_grad_()
    assert torch.autograd.gradcheck(lambda p: irn_loss(p, target), (prediction,))
    gradient, = torch.autograd.grad(irn_loss(prediction, target), prediction)
    torch.testing.assert_close(gradient.sum(1), torch.zeros(4, 2, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close((gradient * prediction).sum(1), torch.zeros(4, 2, dtype=torch.float64),
                               atol=1e-12, rtol=0)
    constant = torch.ones_like(target, requires_grad=True)
    irn_loss(constant, target, std_floor=.01).backward()
    assert torch.isfinite(constant.grad).all()
    assert constant.grad.abs().sum() > 0


def test_horizon_one_requires_explicit_degenerate_policy():
    prediction = torch.randn(4, 1, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    with pytest.raises(ValueError, match="H=1"):
        irn_loss(prediction, target)
    loss = irn_loss(prediction, target, horizon_one="zero_irn")
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and torch.count_nonzero(prediction.grad) == 0


def test_initialization_is_random_reproducible_and_nonempty():
    labels = initialize_environments(17, 4, seed=21)
    assert torch.equal(labels, initialize_environments(17, 4, seed=21))
    assert not torch.equal(labels, initialize_environments(17, 4, seed=22))
    assert torch.bincount(labels).tolist() == [5, 4, 4, 4]
    with pytest.raises(ValueError, match="at least"):
        initialize_environments(2, 3)


def test_argmin_ties_keep_previous_then_use_fixed_environment_order():
    scores = torch.tensor([[.1, .1, .3], [.5, .1, .1], [.1, .1, .5]])
    actual = assign_environments(scores, torch.tensor([1, 0, 2]))
    assert actual.tolist() == [1, 1, 0]


def test_temporal_smoothing_is_synchronous_and_independent_of_shuffle():
    labels = torch.tensor([0, 1, 0, 1, 1])
    ids = torch.arange(5)
    expected = torch.tensor([0, 0, 1, 1, 1])
    assert torch.equal(smooth_environment_labels(labels, ids, radius=1), expected)
    permutation = torch.tensor([4, 1, 3, 0, 2])
    shuffled = smooth_environment_labels(labels[permutation], ids[permutation], radius=1)
    assert torch.equal(shuffled, expected[permutation])
    assert torch.equal(smooth_environment_labels(labels, radius=0), labels)
    gap_labels = torch.tensor([0, 1, 0])
    assert torch.equal(smooth_environment_labels(gap_labels, torch.tensor([0, 1, 100]), radius=2), gap_labels)


def test_empty_environment_repair_moves_hardest_without_emptying_donors():
    labels = torch.zeros(5, dtype=torch.long)
    scores = torch.tensor([[1., 3., 4.], [7., 2., 1.], [7., 5., 9.], [3., 1., 1.], [2., 1., 1.]])
    ids = torch.tensor([4, 8, 2, 1, 5])
    result = repair_empty_environments(labels, scores, ids)
    assert result.tolist() == [0, 2, 1, 0, 0]
    assert (torch.bincount(result, minlength=3) > 0).all()
    order = torch.tensor([2, 3, 1, 4, 0])
    shuffled = repair_empty_environments(labels[order], scores[order], ids[order])
    assert torch.equal(shuffled, result[order])
    singleton = repair_empty_environments(torch.tensor([0, 0, 1]), torch.ones(3, 3))
    assert torch.bincount(singleton, minlength=3).tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="at least"):
        repair_empty_environments(torch.zeros(2, dtype=torch.long), torch.ones(2, 3))


def test_mil_has_equal_environment_weight_global_erm_and_population_variance():
    x, target = _data(5)
    branch = TinyBranch().double()
    prediction = branch(x)
    labels = torch.tensor([0, 0, 0, 0, 1])
    terms = mil_objective(prediction, target, labels, n_environments=2, lambda_erm=1.3, lambda_env=.7)
    row_irn = irn_loss(prediction, target, reduction="none")
    risks = torch.stack((row_irn[:4].mean(), row_irn[4]))
    torch.testing.assert_close(terms.irn_mean, risks.mean())
    torch.testing.assert_close(terms.erm, (prediction - target).square().mean())
    torch.testing.assert_close(terms.environment_variance, ((risks - risks.mean()) ** 2).mean())
    torch.testing.assert_close(terms.total, risks.mean() + 1.3 * terms.erm + .7 * terms.environment_variance)
    with pytest.raises(ValueError, match="every environment"):
        mil_objective(prediction, target, torch.zeros(5, dtype=torch.long),
                      n_environments=2, lambda_erm=1, lambda_env=1)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_streamed_mil_gradient_matches_dense_equation_10(batch_size):
    torch.manual_seed(11)
    x, target = _data(9)
    streamed, dense = TinyBranch().double(), None
    dense = deepcopy(streamed)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2])
    config = _config(n_environments=3, batch_size=batch_size, lambda_erm=1.3)
    streamed_optimizer = torch.optim.SGD(streamed.parameters(), lr=.003)
    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=.003)
    objective = mil_objective(dense(x), target, labels, n_environments=3, lambda_erm=1.3, lambda_env=.7)
    objective.total.backward()
    dense_optimizer.step()
    metrics = mil_step(streamed, streamed_optimizer, x, target, labels, config,
                       lambda_env=.7, generator=torch.Generator().manual_seed(8))
    assert metrics["total"] == pytest.approx(objective.total.item(), abs=1e-12)
    for actual, expected in zip(streamed.parameters(), dense.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_mil_learns_raw_target_scale_and_offset():
    torch.manual_seed(17)
    x, target = _data(24)
    branch = TinyBranch().double()
    config = _config(batch_size=5, lambda_erm=10)
    labels = initialize_environments(len(x), 2)
    optimizer = torch.optim.Adam(branch.parameters(), lr=.035)
    initial = (branch(x) - target).square().mean().item()
    for _ in range(45):
        mil_step(branch, optimizer, x, target, labels, config, lambda_env=.1)
    final = (branch(x) - target).square().mean().item()
    assert final < initial * .08
    assert all(p.requires_grad for p in branch.parameters())


def test_mei_only_changes_environment_heads_and_phase_switch_is_explicit():
    torch.manual_seed(8)
    branch = TinyBranch().double()
    heads = _heads(branch)
    x, target = _data(9)
    labels = initialize_environments(9, 2)
    original_branch = deepcopy(branch.state_dict())
    original_heads = deepcopy(heads.state_dict())
    for parameter in branch.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.Adam(heads.parameters(), lr=.03)
    metrics = mei_step(branch, heads, optimizer, x, target, labels, _config())
    assert metrics["irn_mean"] >= 0
    assert all(not p.requires_grad and p.grad is None for p in branch.parameters())
    assert all(p.requires_grad for p in heads.parameters())
    assert any(not torch.equal(value, original_heads[key]) for key, value in heads.state_dict().items())
    assert all(torch.equal(value, original_branch[key]) for key, value in branch.state_dict().items())
    set_foil_phase(branch, heads, "mil")
    assert all(p.requires_grad for p in branch.parameters())
    assert all(not p.requires_grad and p.grad is None for p in heads.parameters())
    set_foil_phase(branch, heads, "frozen")
    assert all(not p.requires_grad for p in branch.parameters())
    assert all(not p.requires_grad for p in heads.parameters())


def test_streamed_mei_matches_environment_equal_weight_dense_loss():
    torch.manual_seed(6)
    x, target = _data(8)
    branch = TinyBranch().double()
    heads = _heads(branch)
    dense_heads = deepcopy(heads)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    optimizer = torch.optim.SGD(heads.parameters(), lr=.02)
    dense_optimizer = torch.optim.SGD(dense_heads.parameters(), lr=.02)
    features = branch.encoder(x).detach()
    risks = torch.stack([irn_loss(head(features[labels == e]).reshape(-1, 3, 2), target[labels == e])
                         for e, head in enumerate(dense_heads)])
    risks.mean().backward()
    dense_optimizer.step()
    actual = mei_step(branch, heads, optimizer, x, target, labels, _config(batch_size=1))
    assert actual["irn_mean"] == pytest.approx(risks.mean().item(), abs=1e-12)
    for a, b in zip(heads.parameters(), dense_heads.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_mei_rejects_optimizer_with_branch_parameters():
    branch = TinyBranch().double()
    heads = _heads(branch)
    optimizer = torch.optim.SGD(list(heads.parameters()) + list(branch.parameters()), lr=.01)
    x, target = _data(6)
    with pytest.raises(ValueError, match="current phase"):
        mei_step(branch, heads, optimizer, x, target, initialize_environments(6, 2), _config())


def test_full_sequence_environment_inference_retains_global_indices():
    torch.manual_seed(7)
    branch = TinyBranch().double()
    heads = _heads(branch)
    x, target = _data(14)
    labels = initialize_environments(14, 2)
    shuffled = infer_environments(branch, heads, x, target, labels, _config(batch_size=2),
                                   generator=torch.Generator().manual_seed(6))
    ordered = infer_environments(branch, heads, x, target, labels,
                                 _config(batch_size=14, shuffle=False))
    torch.testing.assert_close(shuffled.losses, ordered.losses, atol=1e-12, rtol=1e-12)
    assert torch.equal(shuffled.labels, ordered.labels)
    assert all(count > 0 for count in shuffled.counts) and sum(shuffled.counts) == 14


def test_complete_training_is_reproducible_json_serializable_and_restores_best():
    torch.manual_seed(54)
    branches = TinyBranches().double()
    duplicate = deepcopy(branches)
    train_x, train_y = _data(15)
    val_x, val_y = _data(7, seed=20)
    config = _config()
    before_rng = torch.get_rng_state().clone()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    history = train_foil_branches(branches, train_x, train_y, val_x, val_y, config)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_deterministic == torch.are_deterministic_algorithms_enabled()
    repeated = train_foil_branches(duplicate, train_x, train_y, val_x, val_y, config)
    assert json.loads(json.dumps(history, allow_nan=False)) == history == repeated
    assert not branches.training and all(not p.requires_grad for p in branches.parameters())
    assert history["invariant"]["seed"] != history["variant"]["seed"]
    for name in ("invariant", "variant"):
        branch_history = history[name]
        observed = [branch_history["initial_validation_mse"]]
        observed.extend(row["validation_mse"] for row in branch_history["mil"])
        assert branch_history["best_validation_mse"] == min(observed)
        assert branch_history["restored_validation_mse"] == branch_history["best_validation_mse"]
        assert len(branch_history["mil"]) == config.rounds * config.mil_epochs
        assert len(branch_history["final_environment_labels"]) == len(train_x)
        assert all(c > 0 for c in branch_history["final_environment_counts"])
    for a, b in zip(branches.parameters(), duplicate.parameters()):
        assert torch.equal(a, b)
    contrast = branches.variant(val_x) - branches.invariant(val_x)
    assert history["contrast"]["rms"] == pytest.approx(contrast.square().mean().sqrt().item())
    assert history["contrast"]["zero_fraction"] == (contrast == 0).double().mean().item()
    assert history["contrast"]["separation_guaranteed"] is False


def test_horizon_one_training_records_erm_only_and_can_restore_initial_checkpoint():
    torch.manual_seed(31)
    branches = TinyBranches(horizon=1, targets=1).double()
    initial = deepcopy(branches.state_dict())
    train_x, train_y = _data(10, horizon=1, targets=1)
    val_x, val_y = _data(6, horizon=1, targets=1, seed=21)
    config = _config(rounds=1, mil_epochs=1, mei_epochs=1, horizon_one="zero_irn", mil_learning_rate=5.)
    history = train_foil_branches(branches, train_x, train_y, val_x, val_y, config)
    assert history["horizon_one_degenerate"] is True
    for name in ("invariant", "variant"):
        assert history[name]["best_step"] == 0
        assert history[name]["mil"][0]["irn_mean"] == 0
        assert history[name]["mil"][0]["environment_variance"] == 0
    assert all(torch.equal(value, initial[key]) for key, value in branches.state_dict().items())


def test_production_forecast_branches_train_on_small_batches():
    from paper_impl.model import ForecastBranches, ModelConfig

    torch.manual_seed(9)
    model_config = ModelConfig(n_variables=2, lookback=2, horizon=3, targets=2,
                              hidden=4, heads=1, depth=1)
    branches = ForecastBranches(model_config).double()
    train_x, train_y = _data(6)
    val_x, val_y = _data(3, seed=21)
    history = train_foil_branches(branches, train_x, train_y, val_x, val_y,
                                  _config(rounds=1, mil_epochs=1, mei_epochs=1, batch_size=2))
    assert len(history["invariant"]["rounds"]) == 1
    assert branches(val_x).shape == (3, 6)
    assert all(not p.requires_grad for p in branches.parameters())


def test_training_rejects_shared_branches_too_few_samples_and_horizon_one():
    x, target = _data(3)
    branches = TinyBranches().double()
    branches.variant = branches.invariant
    with pytest.raises(ValueError, match="share parameters"):
        train_foil_branches(branches, x, target, x, target, _config())
    branches = TinyBranches().double()
    with pytest.raises(ValueError, match="at least"):
        train_foil_branches(branches, x[:1], target[:1], x, target, _config())
    one_x, one_target = _data(4, horizon=1, targets=1)
    with pytest.raises(ValueError, match="H=1"):
        train_foil_branches(TinyBranches(1, 1).double(), one_x, one_target, one_x, one_target, _config())


@pytest.mark.parametrize("kwargs", [
    {"n_environments": 0}, {"batch_size": 0}, {"lambda_variant": -1}, {"lambda_erm": float("nan")},
    {"mil_learning_rate": 0}, {"std_floor": 0}, {"smoothing_radius": -1},
    {"horizon_one": "silently_mse"}, {"max_grad_norm": -3}, {"seed": -1},
])
def test_config_rejects_invalid_training_settings(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_streams_cpu_dataset_and_freezes_result():
    torch.manual_seed(18)
    branches = TinyBranches().float().cuda()
    x, y = _data(6)
    vx, vy = _data(3, seed=22)
    history = train_foil_branches(branches, x, y, vx, vy,
                                  _config(rounds=1, mil_epochs=1, mei_epochs=1, batch_size=2,
                                          deterministic=False))
    json.dumps(history, allow_nan=False)
    assert all(p.is_cuda and not p.requires_grad for p in branches.parameters())
