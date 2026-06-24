from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.online_adapt import LabelBuffer
from train.online_loop import run_online_adaptation
from train.run_experiment import run_single_experiment
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from utils.config import load_yaml


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_online_loader(test_ds, batch_size: int):
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def evaluate_frozen(model, online_dl, device) -> dict[str, float]:
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for x, y in online_dl:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            losses.append(torch.mean((out.y_hat - y) ** 2).item())
            maes.append(torch.mean(torch.abs(out.y_hat - y)).item())
    return {"mse": _mean(losses), "mae": _mean(maes)}


def evaluate_online_gd(model, online_dl, device, lr: float, prefixes: list[str] | None = None, weight_decay: float = 0.0, grad_clip: float | None = 1.0, delay_steps: int = 24) -> dict[str, float]:
    model = copy.deepcopy(model).to(device)
    named_params = list(model.named_parameters())
    if prefixes:
        params = [p for n, p in named_params if any(n.startswith(pref) for pref in prefixes)]
    else:
        params = [p for _, p in named_params]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    buffer = LabelBuffer(delay_steps)
    losses = []
    maes = []
    for x, y in online_dl:
        x = x.to(device)
        y = y.to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
            losses.append(torch.mean((out.y_hat - y) ** 2).item())
            maes.append(torch.mean(torch.abs(out.y_hat - y)).item())
        buffer.push(out.z, x, out.alpha)
        delayed = buffer.pop_with_label(y)
        if delayed is None:
            continue
        _, x_delayed, _, y_delayed = delayed
        model.train()
        out_delayed = model(x_delayed)
        loss = torch.mean((out_delayed.y_hat - y_delayed) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
    return {"mse": _mean(losses), "mae": _mean(maes)}


def make_ours_stream(online_dl, invariant_bundle, device):
    def stream():
        for x, y in online_dl:
            x = x.to(device)
            y = y.to(device)
            if invariant_bundle is None:
                yield x, y
                continue
            with torch.no_grad():
                phi_i = invariant_bundle['encoder_i'](x)
                phi_v = invariant_bundle['encoder_v'](x)
            yield x, y, phi_i, phi_v
    return stream()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tuning/load_candidate_i_shifted.yaml')
    parser.add_argument('--results', default='logs/load_benchmarks/shifted_online_baselines.csv')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--invariant-epochs', type=int, default=1)
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    rows = []
    summary = {}
    for ds in DATASETS:
        cfg = copy.deepcopy(base_cfg)
        cfg['train']['epochs'] = args.epochs
        cfg['invariant']['epochs'] = args.invariant_epochs
        cfg['data']['path'] = ds['path']
        cfg['data']['timestamp_col'] = ds['timestamp_col']
        cfg['data']['csv_encoding'] = ds.get('encoding')
        cfg['data']['target_dim'] = target_dim_for_dataset(ds)
        cfg['data']['target_col'] = ds['target_col']
        cfg['data']['missing_sentinels'] = ds.get('missing_sentinels', [])
        cfg['data']['impute_method'] = ds.get('impute_method', 'drop')
        cfg['online_run']['enable'] = False

        tag = f"{ds['name']}_shifted_offline"
        result = run_single_experiment(cfg, save_ckpt=False, variant_name=tag, return_artifacts=True)
        model = result['model']
        device = result['device']
        test_ds = result['test_ds']
        invariant_bundle = result['invariant_bundle']
        split_info = result['split_info']
        online_dl = build_online_loader(test_ds, cfg['train']['batch_size'])

        frozen = evaluate_frozen(copy.deepcopy(model).to(device), online_dl, device)
        gd_full = evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=None, weight_decay=1e-5, delay_steps=cfg['data']['lookback'])
        gd_head = evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=['output_head'], weight_decay=1e-5, delay_steps=cfg['data']['lookback'])

        ours_stage3_cfg = copy.deepcopy(cfg)
        ours_stage3_cfg['online_run']['enable'] = True
        ours_stage3_cfg['online']['stage1']['enable'] = False
        ours_stage3_cfg['online']['stage2']['enable'] = False
        ours_stage3_cfg['online']['stage3']['enable'] = True
        ours_stage3 = run_online_adaptation(make_ours_stream(online_dl, invariant_bundle, device), copy.deepcopy(model).to(device), None, ours_stage3_cfg, device)

        ours_full_cfg = copy.deepcopy(cfg)
        ours_full_cfg['online_run']['enable'] = True
        ours_full_cfg['online']['stage1']['enable'] = True
        ours_full_cfg['online']['stage2']['enable'] = True
        ours_full_cfg['online']['stage3']['enable'] = True
        ours_full = run_online_adaptation(make_ours_stream(online_dl, invariant_bundle, device), copy.deepcopy(model).to(device), None, ours_full_cfg, device)

        methods = {
            'frozen': frozen,
            'gd_full': gd_full,
            'gd_last_layer': gd_head,
            'ours_stage3': {'mse': ours_stage3['online_stage3_mse'], 'mae': ours_stage3['online_stage3_mae']},
            'ours_full': {'mse': ours_full['online_stage3_mse'], 'mae': ours_full['online_stage3_mae']},
        }
        summary[ds['name']] = methods
        for method, metrics in methods.items():
            rows.append({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'dataset': ds['name'],
                'backbone': cfg['model']['temporal_encoder'],
                'method': method,
                'offline_test_mse': f"{result['offline_test_mse']:.6f}",
                'offline_test_mae': f"{result['offline_test_mae']:.6f}",
                'online_mse': f"{metrics['mse']:.6f}",
                'online_mae': f"{metrics['mae']:.6f}",
                'shift_score': f"{float(split_info.get('shift_score', 0.0)):.6f}",
                'test_start': split_info.get('test_start', ''),
                'test_end': split_info.get('test_end', ''),
            })
        print(json.dumps({ds['name']: methods, 'shift': split_info}, ensure_ascii=False, indent=2))

    append_rows(args.results, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
