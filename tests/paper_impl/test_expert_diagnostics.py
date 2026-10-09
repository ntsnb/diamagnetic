"""专家证据在发行时记录，不能改变在线状态或随机序列。"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl.adaptation import OnlineConfig
from paper_impl.data import DataConfig, prepare_data
from paper_impl.diagnostics import ExpertDiagnostics, ExpertDiagnosticsConfig
from paper_impl.evaluation import evaluate_stream
from paper_impl.model import ModelConfig, PaperForecaster


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('backbone', ['native', 'itransformer', 'informer'])
def test_diagnostics_preserve_predictions_updates_and_rng(tmp_path, backbone):
    path = tmp_path / 'load.csv'
    count = 90
    pd.DataFrame({'time': pd.date_range('2020-01-01', periods=count, freq='h'),
                  'load': 2 + np.sin(np.arange(count) / 3), 'weather': np.cos(np.arange(count) / 7)}).to_csv(path, index=False)
    data = prepare_data(DataConfig(str(path), 'time', ['load'], lookback=24, horizon=2))
    torch.manual_seed(43)
    model = PaperForecaster(ModelConfig(2, 24, 2, hidden=8, heads=8, depth=1, n_experts=2,
                                        expert_backbone=backbone, official_sampling_seed=43, target_indices=(0,)))
    config = OnlineConfig(stage_i=True, stage_ii=True, stage_iii=False, selector='gradient_only',
                          reference_windows=2, reference_interval=2, history_capacity=8, extra_delay=1)
    before = copy.deepcopy(model.state_dict())
    state = torch.random.get_rng_state()
    first = evaluate_stream(model, None, None, data, data.test, config, max_windows=5)
    torch.random.set_rng_state(state)
    second = evaluate_stream(model, None, None, data, data.test, config, max_windows=5,
                             diagnostics_config=ExpertDiagnosticsConfig(True, 2))
    assert torch.equal(state, torch.random.get_rng_state())
    np.testing.assert_array_equal(first['predictions'], second['predictions'])
    assert first['updates'] == second['updates']
    assert any(event['time'] < second['issue_times'][-1] and event['experts']['accepted'] for event in second['updates'])
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key])
    artifact = second['expert_diagnostics']; summary, arrays = artifact['summary'], artifact['arrays']
    np.testing.assert_array_equal(arrays['issue_times'], second['issue_times'])
    np.testing.assert_array_equal(arrays['pair_issue_times'], second['issue_times'][::2])
    assert arrays['routing_weights'].shape == (5, 2)
    assert arrays['expert_predictions'].shape == (5, 2, 2, 1)
    assert arrays['pair_distances'].shape == (3, 1, 1)
    assert np.isclose(sum(summary['mean_gate_weight']), 1)
    assert sum(summary['top1_count']) == 5
    assert np.isclose(sum(summary['top1_share']), 1)
    assert summary['diagnostic_seconds'] > 0
    assert second['metrics']['total_seconds_including_diagnostics'] >= second['metrics']['seconds']
    for m in range(2):
        error = arrays['expert_predictions'][:, m] - second['targets']
        np.testing.assert_allclose(summary['per_expert_error'][m]['normalized_mse'], np.mean(error ** 2))
    json.dumps(summary, allow_nan=False)


class DiagnosticFixture(torch.nn.Module):
    def __init__(self, experts=2, zero=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.))
        self.config = SimpleNamespace(n_experts=experts, targets=1, horizon=2, n_variables=2, lookback=2)
        self.zero = zero

    def forward(self, x, stochastic=False):
        torch.rand(1)  # Capture must restore even an incidental stochastic operation.
        return SimpleNamespace(gates=torch.ones(1, self.config.n_experts) / self.config.n_experts)

    def expert_prediction(self, x, m):
        return torch.full((1, 2, 1), float(m + 1))

    def attribute(self, x, freeze_head=True):
        maps = torch.ones(1, 1, self.config.n_experts, 2, 2)
        if self.zero:
            maps.zero_()
        return SimpleNamespace(maps=maps)


@pytest.mark.parametrize('experts,zero', [(1, False), (2, True)])
def test_single_expert_zero_maps_and_missing_truth_are_explicit(experts, zero):
    model = DiagnosticFixture(experts, zero)
    collector = ExpertDiagnostics(ExpertDiagnosticsConfig(True), model)
    state = torch.random.get_rng_state()
    collector.capture(model, torch.ones(1, 2, 2), 5)
    assert torch.equal(state, torch.random.get_rng_state())
    truth = np.array([[[np.nan], [2.]]])
    result = collector.finish(np.array([5]), truth, np.ones_like(truth), np.array([10.]))
    summary, arrays = result['summary'], result['arrays']
    assert summary['mean_specialization_distance'] is None
    assert summary['per_expert_error'][0]['observed_target_values'] == 1
    assert summary['per_expert_error'][0]['normalized_mse'] == 1
    assert summary['per_expert_error'][0]['native_mse'] == 100
    assert arrays['observed_target_mask'].tolist() == [[[False], [True]]]
    assert summary['invalid_pair_target_windows'] == (1 if zero else 0)
    assert summary['single_expert'] == (experts == 1)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize('values', [{'enabled': 'true'}, {'distance_interval': 0}, {'distance_interval': True}])
def test_invalid_diagnostic_settings(values):
    with pytest.raises(ValueError):
        ExpertDiagnosticsConfig(**values)
