from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd
import torch


def _read_table(path: str, encoding: Optional[str]) -> pd.DataFrame:
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    if path.endswith(".xml"):
        return pd.read_xml(path)
    if path.endswith(".csv"):
        if encoding is not None:
            return pd.read_csv(path, encoding=encoding)
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.read_csv(path, encoding="latin1")
    raise ValueError("Unsupported file type. Use .xlsx/.xls, .xml, or .csv")


def _prepare_raw_frame(
    *,
    path: str,
    timestamp_col: str,
    feature_cols: Optional[list[str]],
    encoding: Optional[str],
    missing_sentinels: Optional[Sequence[float]],
) -> tuple[pd.DataFrame, list[str]]:
    df = _read_table(path, encoding)
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not found in file")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    if missing_sentinels:
        for sentinel in missing_sentinels:
            df = df.replace(sentinel, pd.NA)

    selected_cols = feature_cols or [c for c in df.columns if c != timestamp_col]
    if not selected_cols:
        raise ValueError("no feature columns available for split-safe loading")
    for col in selected_cols:
        if col not in df.columns:
            raise ValueError(f"feature_col '{col}' not found in file")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[[timestamp_col, *selected_cols]].copy(), selected_cols


def _train_fill_values(train_df: pd.DataFrame, value_cols: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in value_cols:
        if train_df[col].notna().sum() == 0:
            raise ValueError(f"split-safe imputation cannot fill feature '{col}' because train split is all missing")
        median = train_df[col].median(skipna=True)
        values[col] = float(median)
    return values


def _causal_ffill_segment(
    segment: pd.DataFrame,
    *,
    value_cols: list[str],
    fill_values: dict[str, float],
    previous_values: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series]:
    imputed = segment.copy()
    if previous_values is not None and len(imputed) > 0:
        seed = pd.DataFrame([previous_values], columns=value_cols)
        values = pd.concat([seed, imputed[value_cols]], ignore_index=True).ffill().iloc[1:].reset_index(drop=True)
        imputed.loc[:, value_cols] = values.to_numpy()
    else:
        imputed.loc[:, value_cols] = imputed[value_cols].ffill()
    imputed.loc[:, value_cols] = imputed[value_cols].fillna(fill_values)
    if len(imputed) == 0:
        last_values = pd.Series(fill_values, dtype="float64")
    else:
        last_values = imputed[value_cols].iloc[-1].astype("float64")
    return imputed, last_values


def _split_bounds(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end <= 0:
        raise ValueError("train split is empty after raw chronological split")
    if val_end <= train_end:
        raise ValueError("validation split is empty after raw chronological split")
    if val_end >= n:
        raise ValueError("test split is empty after raw chronological split")
    return train_end, val_end


def load_split_safe_tabular_timeseries(
    *,
    path: str,
    timestamp_col: str,
    feature_cols: Optional[list[str]] = None,
    encoding: Optional[str] = None,
    missing_sentinels: Optional[Sequence[float]] = None,
    impute_method: str = "causal_ffill_train_median",
    train_ratio: float,
    val_ratio: float,
) -> tuple[torch.Tensor, pd.DataFrame, dict[str, Any]]:
    """Load tabular time series with raw-split-safe causal imputation.

    This Stage III V2 reader intentionally avoids full-series interpolation.
    Missing values are imputed after the chronological train/val/test raw split:
    each segment is forward-filled from already visible values, and remaining
    leading gaps are filled with train-only medians.
    """

    if impute_method != "causal_ffill_train_median":
        raise ValueError("Stage III V2 split-safe reader supports only causal_ffill_train_median")

    raw_df, selected_cols = _prepare_raw_frame(
        path=path,
        timestamp_col=timestamp_col,
        feature_cols=feature_cols,
        encoding=encoding,
        missing_sentinels=missing_sentinels,
    )
    train_end, val_end = _split_bounds(len(raw_df), train_ratio, val_ratio)
    segments = {
        "train": raw_df.iloc[:train_end].copy(),
        "val": raw_df.iloc[train_end:val_end].copy(),
        "test": raw_df.iloc[val_end:].copy(),
    }
    fill_values = _train_fill_values(segments["train"], selected_cols)
    info: dict[str, Any] = {
        "split_safe_imputation": True,
        "preprocessing_mode": "split_safe_v2",
        "preprocessing_module": "data.readers_stage3_v2.load_split_safe_tabular_timeseries",
        "preprocessing_contract_version": "stage3_v2_split_safe_imputation_v1",
        "imputation_method": impute_method,
        "preprocessing_impute_method_requested": impute_method,
        "preprocessing_impute_method_effective": impute_method,
        "imputation_uses_future": False,
        "imputation_uses_bfill": False,
        "imputation_uses_cross_split_interpolation": False,
        "preprocessing_split_safe": True,
        "preprocessing_uses_future_values": False,
        "preprocessing_fail_on_remaining_nan": True,
        "raw_row_count": len(raw_df),
        "preprocessing_row_count_before": len(raw_df),
        "preprocessing_missing_before_total": int(raw_df[selected_cols].isna().sum().sum()),
        "train_raw_end": train_end,
        "val_raw_end": val_end,
        "train_fill_values": fill_values,
        "preprocessing_unresolved_columns": [],
    }

    imputed_segments = []
    previous_values: pd.Series | None = None
    for name, segment in segments.items():
        before = int(segment[selected_cols].isna().sum().sum())
        imputed, previous_values = _causal_ffill_segment(
            segment,
            value_cols=selected_cols,
            fill_values=fill_values,
            previous_values=previous_values,
        )
        after = int(imputed[selected_cols].isna().sum().sum())
        info[f"{name}_missing_before"] = before
        info[f"{name}_missing_after"] = after
        info[f"{name}_raw_start"] = int(segment.index[0]) if len(segment) else 0
        info[f"{name}_raw_end"] = int(segment.index[-1]) + 1 if len(segment) else 0
        imputed_segments.append(imputed)

    df = pd.concat(imputed_segments, ignore_index=True)
    unresolved = [col for col in selected_cols if df[col].isna().any()]
    info["preprocessing_unresolved_columns"] = unresolved
    info["preprocessing_row_count_after"] = len(df)
    info["preprocessing_missing_after_total"] = int(df[selected_cols].isna().sum().sum())
    if unresolved:
        raise ValueError(f"split-safe imputation left unresolved NaNs in columns: {unresolved}")
    series = torch.tensor(df[selected_cols].values, dtype=torch.float32)
    return series, df, info
