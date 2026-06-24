from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch

from utils.paths import resolve_project_path


def _apply_missing_value_policy(
    df: pd.DataFrame,
    timestamp_col: str,
    missing_sentinels: Optional[Sequence[float]],
    impute_method: str,
) -> pd.DataFrame:
    if missing_sentinels:
        for sentinel in missing_sentinels:
            df = df.replace(sentinel, pd.NA)

    value_cols = [c for c in df.columns if c != timestamp_col]
    if not value_cols:
        return df

    # Ensure numeric columns stay numeric after sentinel replacement.
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if impute_method == "drop":
        return df.dropna()
    if impute_method == "ffill_bfill":
        df[value_cols] = df[value_cols].ffill().bfill()
        return df.dropna()
    if impute_method == "time_interpolate":
        df = df.set_index(timestamp_col)
        df[value_cols] = df[value_cols].interpolate(method="time").ffill().bfill()
        df = df.reset_index()
        return df.dropna()
    raise ValueError(f"unknown impute_method: {impute_method}")


def load_tabular_timeseries(
    path: str,
    timestamp_col: str,
    feature_cols: Optional[List[str]] = None,
    dropna: bool = True,
    encoding: Optional[str] = None,
    missing_sentinels: Optional[Sequence[float]] = None,
    impute_method: str = "drop",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """Load time series from Excel or XML.

    Returns:
        series: [T, N] torch tensor
        df: raw DataFrame (sorted by timestamp)
    """

    resolved_path = resolve_project_path(path, must_exist=True)
    resolved_path_text = str(resolved_path)

    if resolved_path_text.endswith(".xlsx") or resolved_path_text.endswith(".xls"):
        df = pd.read_excel(resolved_path)
    elif resolved_path_text.endswith(".xml"):
        df = pd.read_xml(resolved_path)
    elif resolved_path_text.endswith(".csv"):
        if encoding is not None:
            df = pd.read_csv(resolved_path, encoding=encoding)
        else:
            # try utf-8 first, then latin1
            try:
                df = pd.read_csv(resolved_path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(resolved_path, encoding="latin1")
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls, .xml, or .csv")

    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not found in file")

    # parse timestamp for robust sorting
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = _apply_missing_value_policy(
        df,
        timestamp_col=timestamp_col,
        missing_sentinels=missing_sentinels,
        impute_method=impute_method,
    )
    if dropna:
        df = df.dropna()

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != timestamp_col]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"feature_col '{c}' not found in file")

    series = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    return series, df
