"""同离线checkpoint完成四个定位对照，并保存可核验的专家证据。"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl import run as cli
from paper_impl.evaluation import selection_variants, selection_comparison_summary


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('controlled_experiment')
    path = root / 'load.csv'
    rows = np.arange(60)
    pd.DataFrame({'time': pd.date_range('2024-01-01', periods=60, freq='h'),
                  'load': 2 + np.sin(rows / 3) + rows / 20,
                  'weather': np.cos(rows / 5)}).to_csv(path, index=False)
    configuration = {'seed': 43, 'device': 'cpu', 'cpu_threads': 1,
        'data': {'path': str(path), 'timestamp_column': 'time', 'target_columns': ['load'],
                 'feature_columns': ['load', 'weather'], 'lookback': 4, 'horizon': 2,
                 'train_fraction': .6, 'validation_fraction': .2},
        'model': {'hidden': 4, 'heads': 1, 'n_experts': 2, 'latent_block': 2, 'use_revin': True},
        'foil': {'n_environments': 2, 'rounds': 1, 'mil_epochs': 1, 'mei_epochs': 1, 'batch_size': 8},
        'training': {'epochs': 1, 'batch_size': 8, 'specialization_weight': 0.},
        'generator': {'mode': 'identity'},
        'online': {'reference_windows': 2, 'reference_interval': 2, 'history_capacity': 4,
                   'extra_delay': 1, 'channel_budget': 1},
        'variants': ['stages_000'], 'test_windows': 5,
        'selection_comparisons': list(selection_variants()),
        'expert_diagnostics': {'enabled': True, 'distance_interval': 1}}
    result = cli.run(configuration, root / 'trained')
    return root / 'trained', configuration, result


def test_full_run_writes_all_matched_and_expert_evidence(completed):
    output, _, result = completed
    assert len(result['results']) == 5
    protocols = load(output / 'evaluation_protocols.json')
    assert load(output / 'seed_plan.json')['selection'] == 43
    validation = load(output / 'selection_comparison.json')
    assert validation['equal_selected_parameters_per_sparse_event']
    assert validation['all_three_sparse_controls_present']
    counts = []
    for name in selection_variants():
        protocol = protocols[name]
        assert (protocol['stage_i'], protocol['stage_ii'], protocol['stage_iii']) == (False, True, False)
        assert protocol['selection_seed'] == 43
        events = load(output / f'{name}_updates.json')
        assert events
        assert any(event['time'] < load(output / 'data_metadata.json')['issue_times']['test'][4] and event['experts']['accepted'] for event in events)
        assert all(event['time'] >= max(event['issue_times']) + 3 for event in events)
        assert all(not event['routing']['accepted'] for event in events)
        assert all(event['experts']['changed_parameter_count'] <= event['experts']['selected_parameter_count'] for event in events)
        counts.append([event['experts']['selected_parameter_count'] for event in events])
        diagnostic = load(output / f'{name}_expert_diagnostics.json')
        assert diagnostic['issued_windows'] == diagnostic['distance_windows'] == 5
        with np.load(output / f'{name}_expert_diagnostics.npz') as evidence, np.load(output / f'{name}_forecasts.npz') as forecasts:
            np.testing.assert_array_equal(evidence['issue_times'], forecasts['issue_times'])
            assert evidence['expert_errors_normalized'].shape == (5, 2, 2, 1)
    assert counts[0] == counts[1] == counts[2]
    assert all(dense > sparse for dense, sparse in zip(counts[3], counts[0]))


def test_checkpoint_reevaluation_preserves_matched_predictions_and_diagnostics(completed, tmp_path):
    original, _, _ = completed
    restored = tmp_path / 'restored'
    cli.run({}, restored, original / 'model.pt')
    assert load(original / 'selection_comparison.json') == load(restored / 'selection_comparison.json')
    for name in ['stages_000', *selection_variants()]:
        with np.load(original / f'{name}_expert_diagnostics.npz') as before, np.load(restored / f'{name}_expert_diagnostics.npz') as after:
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key])


def test_old_v2_checkpoint_without_selection_fields_remains_readable(completed, tmp_path):
    original, _, _ = completed
    saved = torch.load(original / 'model.pt', map_location='cpu', weights_only=False)
    for key in ('selection_protocol', 'selection_seed'):
        saved['configuration']['online'].pop(key)
        saved['online_config'].pop(key, None)  # Pickle may preserve the shared dictionary identity.
    saved['configuration'].pop('selection_comparisons')
    saved['configuration'].pop('expert_diagnostics')
    path = tmp_path / 'old.pt'
    torch.save(saved, path)
    _, normalized, _ = cli._load_checkpoint(path)
    assert normalized['online']['selection_protocol'] == 'adaptive'
    assert normalized['online']['selection_seed'] == 42
    assert normalized['selection_comparisons'] == []
    assert not normalized['expert_diagnostics']['enabled']


def test_comparison_summary_rejects_per_event_budget_mismatch(completed):
    output, _, _ = completed
    records = {name: load(output / f'{name}_updates.json') for name in selection_variants()}
    records['matched_gradient'][0]['experts']['selected_parameter_count'] += 1
    with pytest.raises(ValueError, match='matched parameter budgets'):
        selection_comparison_summary(records)


@pytest.mark.parametrize('key,value', [('selection_comparisons', ['unknown']),
                                      ('selection_comparisons', ['matched_lrp', 'matched_lrp']),
                                      ('expert_diagnostics', {'distance_interval': 0})])
def test_invalid_controls_rejected_before_training(completed, key, value):
    configuration = copy.deepcopy(completed[1])
    configuration[key] = value
    with pytest.raises(ValueError):
        cli._normalize(configuration)
