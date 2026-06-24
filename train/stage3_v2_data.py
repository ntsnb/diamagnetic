from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from data.datasets import WindowedDataset
from data.windowing import build_windows


@dataclass(frozen=True)
class RawSegment:
    name: str
    start: int
    end: int

    def validate(self, *, lookback: int, horizon: int) -> None:
        if self.end - self.start < lookback + horizon:
            raise ValueError(
                f"{self.name} segment [{self.start}, {self.end}) is too short "
                f"for lookback={lookback}, horizon={horizon}"
            )


def _window_segment(
    series: torch.Tensor,
    segment: RawSegment,
    *,
    lookback: int,
    horizon: int,
    target_dim: int,
) -> tuple[Subset, dict[str, int]]:
    segment.validate(lookback=lookback, horizon=horizon)
    x, y = build_windows(series[segment.start : segment.end], lookback, horizon, target_dim)
    ds = WindowedDataset(x, y)
    wrapped = Subset(ds, range(len(ds)))
    info = {
        f"{segment.name}_raw_start": segment.start,
        f"{segment.name}_raw_end": segment.end,
        f"{segment.name}_target_start": segment.start + lookback,
        f"{segment.name}_target_end": segment.end,
        f"{segment.name}_window_count": len(ds),
    }
    return wrapped, info


def _chronological_segments(
    series_len: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None,
) -> tuple[RawSegment, RawSegment, RawSegment]:
    train_end = int(series_len * train_ratio)
    val_end = train_end + int(series_len * val_ratio)
    if test_ratio is None:
        test_end = series_len
    else:
        test_end = min(series_len, val_end + max(1, int(series_len * test_ratio)))
    return (
        RawSegment("train", 0, train_end),
        RawSegment("val", train_end, val_end),
        RawSegment("test", val_end, test_end),
    )


def build_causal_dataloaders_from_series(
    *,
    series: torch.Tensor,
    lookback: int,
    horizon: int,
    target_dim: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    drop_last: bool = False,
    return_datasets: bool = False,
    split_mode: str = "chronological",
    test_ratio: float | None = None,
    shift_pool_ratio: float | None = None,
    return_split_info: bool = False,
    train_shuffle: bool = True,
):
    """Build split-safe dataloaders by splitting raw time before windowing.

    This V2 helper avoids the legacy leakage mode where full-series sliding
    windows are built first and then sliced into splits.
    """

    if split_mode != "chronological":
        raise ValueError("Stage III V2 causal loader currently supports split_mode='chronological' only")
    if shift_pool_ratio is not None:
        raise ValueError("chronological_shifted V2 causal split is not implemented yet")

    train_seg, val_seg, test_seg = _chronological_segments(
        int(series.shape[0]),
        train_ratio,
        val_ratio,
        test_ratio,
    )
    train_ds, train_info = _window_segment(
        series,
        train_seg,
        lookback=lookback,
        horizon=horizon,
        target_dim=target_dim,
    )
    val_ds, val_info = _window_segment(
        series,
        val_seg,
        lookback=lookback,
        horizon=horizon,
        target_dim=target_dim,
    )
    test_ds, test_info = _window_segment(
        series,
        test_seg,
        lookback=lookback,
        horizon=horizon,
        target_dim=target_dim,
    )
    split_info = {
        "split_mode": "chronological_raw_first",
        "leakage_guard": "raw_split_before_windowing",
        "shift_score": 0.0,
        **train_info,
        **val_info,
        **test_info,
    }

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if return_datasets and return_split_info:
        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, split_info
    if return_datasets:
        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
    if return_split_info:
        return train_dl, val_dl, test_dl, split_info
    return train_dl, val_dl, test_dl
