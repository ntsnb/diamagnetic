from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data.datasets import WindowedDataset
from data.windowing import build_windows


def _shift_score(train_x: torch.Tensor, cand_x: torch.Tensor) -> float:
    train_flat = train_x.reshape(train_x.shape[0], -1)
    cand_flat = cand_x.reshape(cand_x.shape[0], -1)
    train_mean = train_flat.mean(dim=0)
    train_std = train_flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    cand_mean = cand_flat.mean(dim=0)
    mean_shift = ((cand_mean - train_mean).abs() / train_std).mean().item()
    train_var = train_flat.var(dim=0, unbiased=False).clamp_min(1e-6)
    cand_var = cand_flat.var(dim=0, unbiased=False).clamp_min(1e-6)
    var_shift = ((cand_var.log() - train_var.log()).abs()).mean().item()
    return float(mean_shift + 0.1 * var_shift)


def _build_shifted_splits(
    x: torch.Tensor,
    n_train: int,
    n_val: int,
    eval_test_size: int,
    pool_size: int,
):
    n = x.shape[0]
    train_range = range(0, n_train)
    val_range = range(n_train, n_train + n_val)
    pool_start = max(n_train + n_val, n - pool_size)
    pool_end = n
    if pool_end - pool_start < eval_test_size:
        pool_start = max(n_train + n_val, n - eval_test_size)
    best_start = pool_start
    best_score = float('-inf')
    train_x = x[list(train_range)]
    for start in range(pool_start, pool_end - eval_test_size + 1):
        stop = start + eval_test_size
        score = _shift_score(train_x, x[start:stop])
        if score > best_score:
            best_score = score
            best_start = start
    test_range = range(best_start, best_start + eval_test_size)
    return train_range, val_range, test_range, {
        'pool_start': pool_start,
        'pool_end': pool_end,
        'test_start': best_start,
        'test_end': best_start + eval_test_size,
        'shift_score': best_score,
    }


def build_dataloaders_from_series(
    series: torch.Tensor,
    lookback: int,
    horizon: int,
    target_dim: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    drop_last: bool = False,
    return_datasets: bool = False,
    split_mode: str = 'chronological',
    test_ratio: float | None = None,
    shift_pool_ratio: float | None = None,
    return_split_info: bool = False,
):
    x, y = build_windows(series, lookback, horizon, target_dim)
    ds = WindowedDataset(x, y)
    n = len(ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    remaining = n - n_train - n_val
    if test_ratio is None:
        n_test = remaining
    else:
        n_test = min(remaining, max(1, int(n * test_ratio)))
    split_info = {'split_mode': split_mode}
    if split_mode == 'chronological':
        train_ds = Subset(ds, range(0, n_train))
        val_ds = Subset(ds, range(n_train, n_train + n_val))
        test_ds = Subset(ds, range(n_train + n_val, n_train + n_val + n_test))
        split_info.update({
            'train_start': 0,
            'train_end': n_train,
            'val_start': n_train,
            'val_end': n_train + n_val,
            'test_start': n_train + n_val,
            'test_end': n_train + n_val + n_test,
            'shift_score': 0.0,
        })
    elif split_mode == 'chronological_shifted':
        pool_ratio = shift_pool_ratio if shift_pool_ratio is not None else (1.0 - train_ratio - val_ratio)
        pool_size = max(n_test, int(n * pool_ratio))
        train_range, val_range, test_range, shifted = _build_shifted_splits(x, n_train, n_val, n_test, pool_size)
        train_ds = Subset(ds, train_range)
        val_ds = Subset(ds, val_range)
        test_ds = Subset(ds, test_range)
        split_info.update({
            'train_start': train_range.start,
            'train_end': train_range.stop,
            'val_start': val_range.start,
            'val_end': val_range.stop,
            'test_start': test_range.start,
            'test_end': test_range.stop,
            **shifted,
        })
    elif split_mode == 'random':
        from torch.utils.data import random_split

        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    else:
        raise ValueError(f'unknown split_mode: {split_mode}')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if return_datasets and return_split_info:
        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, split_info
    if return_datasets:
        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
    if return_split_info:
        return train_dl, val_dl, test_dl, split_info
    return train_dl, val_dl, test_dl
