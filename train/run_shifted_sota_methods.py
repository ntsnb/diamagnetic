from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.online_adapt import LabelBuffer
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


def mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - y) ** 2)


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
            losses.append(mse(out.y_hat, y).item())
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
            losses.append(mse(out.y_hat, y).item())
            maes.append(torch.mean(torch.abs(out.y_hat - y)).item())
        buffer.push(out.z, x, out.alpha)
        delayed = buffer.pop_with_label(y)
        if delayed is None:
            continue
        _, x_delayed, _, y_delayed = delayed
        model.train()
        out_delayed = model(x_delayed)
        loss = mse(out_delayed.y_hat, y_delayed)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
    return {"mse": _mean(losses), "mae": _mean(maes)}


def evaluate_onenet(model, online_dl, device, lr: float = 1e-4, eta: float = 4.0, delay_steps: int = 24) -> dict[str, float]:
    frozen = copy.deepcopy(model).to(device).eval()
    adapted = copy.deepcopy(model).to(device)
    params = [p for n, p in adapted.named_parameters() if n.startswith('output_head')]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    buffer = LabelBuffer(delay_steps)
    w_frozen = 0.5
    losses = []
    maes = []
    for x, y in online_dl:
        x = x.to(device)
        y = y.to(device)
        frozen.eval(); adapted.eval()
        with torch.no_grad():
            out_f = frozen(x).y_hat
            out_a = adapted(x).y_hat
            pred = w_frozen * out_f + (1.0 - w_frozen) * out_a
            losses.append(mse(pred, y).item())
            maes.append(torch.mean(torch.abs(pred - y)).item())
            out_adapt = adapted(x)
        buffer.push(out_adapt.z, x, out_adapt.alpha)
        delayed = buffer.pop_with_label(y)
        if delayed is None:
            continue
        _, x_delayed, _, y_delayed = delayed
        with torch.no_grad():
            lf = mse(frozen(x_delayed).y_hat, y_delayed).item()
            la = mse(adapted(x_delayed).y_hat, y_delayed).item()
            score_f = torch.exp(torch.tensor(-eta * lf))
            score_a = torch.exp(torch.tensor(-eta * la))
            w_frozen = (score_f / (score_f + score_a + 1e-8)).item()
        adapted.train()
        out = adapted(x_delayed)
        loss = mse(out.y_hat, y_delayed)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return {"mse": _mean(losses), "mae": _mean(maes)}


def _prepare_memory(ds, max_items: int, device: torch.device):
    items = []
    n = len(ds)
    step = max(1, n // max_items)
    for idx in range(0, n, step):
        x, y = ds[idx]
        items.append((x.to(device), y.to(device)))
        if len(items) >= max_items:
            break
    return items


def _retrieve_topk(memory, x_t: torch.Tensor, k: int):
    xq = x_t.mean(dim=0).reshape(1, -1)
    dists = []
    for i, (mx, my) in enumerate(memory):
        dist = torch.norm(mx.reshape(1, -1) - xq, p=2).item()
        dists.append((dist, i))
    dists.sort(key=lambda z: z[0])
    idxs = [i for _, i in dists[: min(k, len(dists))]]
    xs = torch.stack([memory[i][0] for i in idxs], dim=0)
    ys = torch.stack([memory[i][1] for i in idxs], dim=0)
    return xs, ys


def _adapt_head_once(model, x_batch: torch.Tensor, y_batch: torch.Tensor, lr: float = 1e-4, steps: int = 1):
    params = [p for n, p in model.named_parameters() if n.startswith('output_head')]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(steps):
        out = model(x_batch)
        loss = mse(out.y_hat, y_batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


def evaluate_solid(model, online_dl, train_memory, device, k: int = 32, lr: float = 1e-4, delay_steps: int = 24, continual: bool = False, max_online_memory: int = 2048) -> dict[str, float]:
    model = copy.deepcopy(model).to(device)
    memory = list(train_memory)
    buffer = LabelBuffer(delay_steps)
    losses = []
    maes = []
    for x, y in online_dl:
        x = x.to(device)
        y = y.to(device)
        if memory:
            x_mem, y_mem = _retrieve_topk(memory, x, k)
            _adapt_head_once(model, x_mem, y_mem, lr=lr, steps=1 if not continual else 2)
        model.eval()
        with torch.no_grad():
            out = model(x)
            losses.append(mse(out.y_hat, y).item())
            maes.append(torch.mean(torch.abs(out.y_hat - y)).item())
        buffer.push(out.z, x, out.alpha)
        delayed = buffer.pop_with_label(y)
        if delayed is not None and continual:
            _, x_delayed, _, y_delayed = delayed
            for xb, yb in zip(x_delayed, y_delayed):
                memory.append((xb.detach(), yb.detach()))
            if len(memory) > max_online_memory:
                memory = memory[-max_online_memory:]
    return {"mse": _mean(losses), "mae": _mean(maes)}


class ProceedAdapter(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_prev: torch.Tensor, z_cur: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_prev, z_cur - z_prev], dim=-1))


def train_proceed_adapter(model, train_ds, device, max_pairs: int = 4096, epochs: int = 3, batch_size: int = 64):
    model = copy.deepcopy(model).to(device).eval()
    n = len(train_ds)
    rng = random.Random(42)
    idx_pairs = []
    for _ in range(min(max_pairs, max(1, n - 32))):
        i = rng.randint(0, n - 2)
        j = min(n - 1, i + rng.randint(1, 24))
        idx_pairs.append((i, j))
    z_prev_list = []
    z_cur_list = []
    resid_list = []
    with torch.no_grad():
        for i, j in idx_pairs:
            x_prev, _ = train_ds[i]
            x_cur, y_cur = train_ds[j]
            x_prev = x_prev.unsqueeze(0).to(device)
            x_cur = x_cur.unsqueeze(0).to(device)
            y_cur = y_cur.unsqueeze(0).to(device)
            out_prev = model(x_prev)
            out_cur = model(x_cur)
            z_prev_list.append(out_prev.z.squeeze(0).cpu())
            z_cur_list.append(out_cur.z.squeeze(0).cpu())
            resid_list.append((y_cur - out_cur.y_hat).squeeze(0).cpu())
    z_prev = torch.stack(z_prev_list)
    z_cur = torch.stack(z_cur_list)
    resid = torch.stack(resid_list)
    ds = TensorDataset(z_prev, z_cur, resid)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    adapter = ProceedAdapter(z_prev.shape[1], resid.shape[1]).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        for zp, zc, rr in dl:
            zp = zp.to(device); zc = zc.to(device); rr = rr.to(device)
            pred = adapter(zp, zc)
            loss = mse(pred, rr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
    return adapter.eval(), model


def evaluate_proceed_style(base_model, adapter, online_dl, device, delay_steps: int = 24) -> dict[str, float]:
    model = copy.deepcopy(base_model).to(device).eval()
    buffer = LabelBuffer(delay_steps)
    last_labeled_x = None
    losses = []
    maes = []
    for x, y in online_dl:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out_cur = model(x)
            pred = out_cur.y_hat
            if last_labeled_x is not None:
                out_prev = model(last_labeled_x)
                resid = adapter(out_prev.z, out_cur.z)
                pred = pred + resid
            losses.append(mse(pred, y).item())
            maes.append(torch.mean(torch.abs(pred - y)).item())
        buffer.push(out_cur.z, x, out_cur.alpha)
        delayed = buffer.pop_with_label(y)
        if delayed is not None:
            _, x_delayed, _, _ = delayed
            last_labeled_x = x_delayed
    return {"mse": _mean(losses), "mae": _mean(maes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tuning/load_candidate_i_shifted.yaml')
    parser.add_argument('--results', default='logs/load_benchmarks/shifted_sota_methods.csv')
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

        tag = f"{ds['name']}_shifted_sota_offline"
        result = run_single_experiment(cfg, save_ckpt=False, variant_name=tag, return_artifacts=True)
        model = result['model']
        device = result['device']
        train_ds = result['train_ds']
        test_ds = result['test_ds']
        split_info = result['split_info']
        online_dl = build_online_loader(test_ds, cfg['train']['batch_size'])
        train_memory = _prepare_memory(train_ds, max_items=1024, device=device)
        proceed_adapter, proceed_backbone = train_proceed_adapter(model, train_ds, device)

        methods = {
            'frozen': evaluate_frozen(copy.deepcopy(model).to(device), online_dl, device),
            'gd_full': evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=None, weight_decay=1e-5, delay_steps=cfg['data']['lookback']),
            'gd_last_layer': evaluate_online_gd(copy.deepcopy(model), online_dl, device, lr=1e-4, prefixes=['output_head'], weight_decay=1e-5, delay_steps=cfg['data']['lookback']),
            'onenet': evaluate_onenet(copy.deepcopy(model), online_dl, device, lr=1e-4, delay_steps=cfg['data']['lookback']),
            'solid': evaluate_solid(copy.deepcopy(model), online_dl, train_memory, device, k=32, lr=1e-4, delay_steps=cfg['data']['lookback'], continual=False),
            'solidpp': evaluate_solid(copy.deepcopy(model), online_dl, train_memory, device, k=32, lr=1e-4, delay_steps=cfg['data']['lookback'], continual=True),
            'proceed': evaluate_proceed_style(proceed_backbone, proceed_adapter, online_dl, device, delay_steps=cfg['data']['lookback']),
        }

        # ours
        from train.online_loop import run_online_adaptation
        invariant_bundle = result['invariant_bundle']
        def ours_stream():
            for x, y in online_dl:
                x = x.to(device); y = y.to(device)
                if invariant_bundle is None:
                    yield x, y
                else:
                    with torch.no_grad():
                        phi_i = invariant_bundle['encoder_i'](x)
                        phi_v = invariant_bundle['encoder_v'](x)
                    yield x, y, phi_i, phi_v
        ours_stage3_cfg = copy.deepcopy(cfg)
        ours_stage3_cfg['online_run']['enable'] = True
        ours_stage3_cfg['online']['stage1']['enable'] = False
        ours_stage3_cfg['online']['stage2']['enable'] = False
        ours_stage3_cfg['online']['stage3']['enable'] = True
        ours_stage3 = run_online_adaptation(ours_stream(), copy.deepcopy(model).to(device), None, ours_stage3_cfg, device)
        ours_full_cfg = copy.deepcopy(cfg)
        ours_full_cfg['online_run']['enable'] = True
        ours_full_cfg['online']['stage1']['enable'] = True
        ours_full_cfg['online']['stage2']['enable'] = True
        ours_full_cfg['online']['stage3']['enable'] = True
        ours_full = run_online_adaptation(ours_stream(), copy.deepcopy(model).to(device), None, ours_full_cfg, device)
        methods['ours_stage3'] = {'mse': ours_stage3['online_stage3_mse'], 'mae': ours_stage3['online_stage3_mae']}
        methods['ours_full'] = {'mse': ours_full['online_stage3_mse'], 'mae': ours_full['online_stage3_mae']}

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
    from utils.config import load_yaml
    main()
