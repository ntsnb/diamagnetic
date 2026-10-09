"""Seed requests must produce complete, independently configured experiment tasks."""
import hashlib
import json
from pathlib import Path

import pytest

from paper_impl.experiment import plan_matrix, baseline_configuration
from paper_impl.run import _normalize
from paper_impl.baseline_run import normalize_baseline_config
from tools.build_full_experiment_matrix import generate


ROOT = Path(__file__).resolve().parents[2]
OLD_MATRIX = ROOT / 'configs/full_experiment/structural/matrix.json'


@pytest.fixture
def matrix(tmp_path):
    source = tmp_path / 'observations.csv'
    source.write_text('time,load\n2020-01-01,1\n', encoding='utf-8')
    inventory = [{
        'id': name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_data': {'path': str(source), 'timestamp_column': 'time',
                       'target_columns': ['load'], 'feature_columns': ['load'],
                       'expected_interval_seconds': 3600},
    } for name in ('task_a', 'task_b')]
    output = tmp_path / 'two_seeds'
    generate(output, inventory, seeds=[42, 43], backbones=['native', 'itransformer'])
    return output / 'matrix.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def test_partially_missing_seed_is_not_silently_ignored():
    with pytest.raises(ValueError, match='种子 43 缺少配置'):
        plan_matrix(OLD_MATRIX, datasets=['panama'], backbones=['itransformer'],
                    ablations=['full'], families=['framework'], seeds=[42, 43])


def test_two_seeds_have_real_framework_and_baseline_configs(matrix):
    manifest = read(matrix)
    for job in manifest['jobs']:
        config_path = matrix.parent / job['config']
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == job['config_sha256']
        effective = _normalize(read(config_path))
        assert effective['seed'] == effective['foil']['seed'] == effective['training']['seed'] == job['seed']
        assert effective['model']['official_sampling_seed'] == job['seed']
    for job in manifest['baseline_jobs']:
        config_path = matrix.parent / job['config']
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == job['config_sha256']
        effective = normalize_baseline_config(read(config_path))
        assert effective['seed'] == job['seed']
        assert effective['method'] == job['method']
        assert effective['model']['model_name'] == job['backbone']
    plan = plan_matrix(matrix, datasets=['task_a'], backbones=['itransformer'],
                       ablations=['full'], seeds=[42, 43], device='cpu')
    assert plan['seeds'] == [42, 43] and plan['seed_coverage_validated']
    assert plan['job_count'] == 16
    for seed in (42, 43):
        jobs = [j for j in plan['jobs'] if j['seed'] == seed]
        assert len(jobs) == 8 and len({j['id'] for j in jobs}) == 8
        framework = _normalize(next(j['configuration'] for j in jobs if j['kind'] == 'framework'))
        for job in jobs:
            if job['kind'] == 'baseline':
                assert normalize_baseline_config(job['configuration'])['data'] == framework['data']


@pytest.mark.parametrize('family', ['framework', 'baseline'])
@pytest.mark.parametrize('explicit', [True, False])
def test_seed_coverage_is_per_task_even_when_seed_exists_elsewhere(matrix, family, explicit):
    manifest = read(matrix)
    field, variant = ('jobs', 'ablation') if family == 'framework' else ('baseline_jobs', 'method')
    selected = 'full' if family == 'framework' else 'GD'
    manifest[field] = [j for j in manifest[field] if not (
        j['dataset'] == 'task_a' and j['backbone'] == 'itransformer'
        and j['seed'] == 43 and j[variant] == selected)]
    write(matrix, manifest)
    kwargs = {'ablations': ['full']} if family == 'framework' else {'methods': ['GD']}
    with pytest.raises(ValueError, match=r'种子 43 缺少配置.*task_a'):
        plan_matrix(matrix, backbones=['itransformer'], families=[family],
                    seeds=[42, 43] if explicit else None, **kwargs)


@pytest.mark.parametrize('family', ['framework', 'baseline'])
@pytest.mark.parametrize('explicit', [True, False])
def test_entire_task_missing_all_seeds_cannot_disappear_from_plan(matrix, family, explicit):
    manifest = read(matrix)
    field, variant = ('jobs', 'ablation') if family == 'framework' else ('baseline_jobs', 'method')
    selected = 'full' if family == 'framework' else 'GD'
    manifest[field] = [j for j in manifest[field] if not (
        j['dataset'] == 'task_a' and j['backbone'] == 'itransformer' and j[variant] == selected)]
    write(matrix, manifest)
    kwargs = {'ablations': ['full']} if family == 'framework' else {'methods': ['GD']}
    with pytest.raises(ValueError, match=r'种子 42 缺少配置.*task_a'):
        plan_matrix(matrix, datasets=['task_a', 'task_b'], backbones=['itransformer'],
                    families=[family], seeds=[42, 43] if explicit else None, **kwargs)


@pytest.mark.parametrize('missing_dimension', ['backbone', 'ablation'])
def test_missing_declared_dimension_is_reported_as_missing_configuration(matrix, missing_dimension):
    manifest = read(matrix)
    missing_value = 'itransformer' if missing_dimension == 'backbone' else 'full'
    manifest['jobs'] = [j for j in manifest['jobs'] if j[missing_dimension] != missing_value]
    write(matrix, manifest)
    with pytest.raises(ValueError, match='种子 42 缺少配置'):
        plan_matrix(matrix, datasets=['task_a'], backbones=['itransformer'],
                    ablations=['full'], families=['framework'], seeds=[42, 43])


def test_scope_validation_respects_backbone_applicability():
    plan = plan_matrix(OLD_MATRIX, datasets=['panama'], backbones=['itransformer', 'transformer', 'native'],
                       ablations=['woRevIN'], methods=['GD'], seeds=[42])
    assert {(j['kind'], j['backbone']) for j in plan['jobs']} == {
        ('framework', 'itransformer'), ('framework', 'native'),
        ('baseline', 'itransformer'), ('baseline', 'transformer'),
    }


@pytest.mark.parametrize('family', ['framework', 'baseline'])
def test_manifest_entry_without_real_seed_file_is_rejected(matrix, family):
    manifest = read(matrix)
    field = 'jobs' if family == 'framework' else 'baseline_jobs'
    job = next(j for j in manifest[field] if j['seed'] == 43 and j['backbone'] == 'itransformer')
    (matrix.parent / job['config']).unlink()
    with pytest.raises(ValueError, match='种子 43 缺少配置文件'):
        plan_matrix(matrix, seeds=[42, 43], families=[family])


@pytest.mark.parametrize('field', ['seed', 'foil', 'training', 'model', 'online'])
def test_seed43_task_cannot_reuse_nested_seed42_configuration(matrix, field):
    manifest = read(matrix)
    job = next(j for j in manifest['jobs'] if j['seed'] == 43 and j['ablation'] == 'full')
    path = matrix.parent / job['config']; config = read(path)
    if field == 'seed':
        config['seed'] = 42
    elif field == 'model':
        config['model']['official_sampling_seed'] = 42
    elif field == 'online':
        config['online']['selection_seed'] = 42
    else:
        config[field]['seed'] = 42
    write(path, config)
    with pytest.raises(ValueError, match='seed mismatch'):
        plan_matrix(matrix, families=['framework'], ablations=['full'], seeds=[42, 43])


def test_baseline_file_seed_must_match_manifest(matrix):
    manifest = read(matrix)
    job = next(j for j in manifest['baseline_jobs'] if j['seed'] == 43)
    path = matrix.parent / job['config']; config = read(path); config['seed'] = 42; write(path, config)
    with pytest.raises(ValueError, match='seed mismatch for baseline'):
        plan_matrix(matrix, families=['baseline'], seeds=[42, 43])


def test_new_explicit_baseline_files_preserve_legacy_derivation(matrix):
    manifest = read(matrix)
    jobs = {(j['dataset'], j['backbone'], j['seed']): j for j in manifest['jobs'] if j['ablation'] == 'full'}
    for job in manifest['baseline_jobs']:
        full = jobs[(job['dataset'], job['backbone'], job['seed'])]
        derived = baseline_configuration(read(matrix.parent / full['config']),
                                         backbone=job['backbone'], method=job['method'])
        assert derived == read(matrix.parent / job['config'])


@pytest.mark.parametrize('seeds', [[], [True], [42, 42], [-1], [2**32], [42.0]])
def test_invalid_seed_requests_fail_before_planning(seeds):
    with pytest.raises(ValueError, match='seeds must'):
        plan_matrix(OLD_MATRIX, seeds=seeds)


def test_sampling_seed_defaults_to_root_but_explicit_override_is_preserved():
    old = read(OLD_MATRIX)
    config = read(OLD_MATRIX.parent / old['jobs'][0]['config'])
    config['seed'] = config['foil']['seed'] = config['training']['seed'] = 43
    config['model'].pop('official_sampling_seed', None)
    assert _normalize(config)['model']['official_sampling_seed'] == 43
    config['model']['official_sampling_seed'] = 47
    assert _normalize(config)['model']['official_sampling_seed'] == 47
