from __future__ import annotations

import pandas as pd
import pytest

from data.readers_stage3_v2 import load_split_safe_tabular_timeseries


def test_split_safe_imputation_does_not_use_validation_value_for_train_gap(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="h"),
            "load": [0.0, 1.0, 2.0, 3.0, -9999.0, 100.0, 101.0, 102.0],
        }
    ).to_csv(path, index=False)

    series, df, info = load_split_safe_tabular_timeseries(
        path=str(path),
        timestamp_col="date",
        feature_cols=["load"],
        missing_sentinels=[-9999.0],
        train_ratio=0.625,
        val_ratio=0.125,
        impute_method="causal_ffill_train_median",
    )

    assert series[4, 0].item() == 3.0
    assert df.loc[4, "load"] == 3.0
    assert info["split_safe_imputation"] is True
    assert info["imputation_uses_future"] is False
    assert info["train_missing_before"] == 1
    assert info["train_missing_after"] == 0


def test_split_safe_imputation_uses_train_stat_for_leading_train_gap(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="h"),
            "load": [-9999.0, 2.0, 4.0, 100.0, 101.0, 102.0],
        }
    ).to_csv(path, index=False)

    series, _, info = load_split_safe_tabular_timeseries(
        path=str(path),
        timestamp_col="date",
        feature_cols=["load"],
        missing_sentinels=[-9999.0],
        train_ratio=0.5,
        val_ratio=0.1667,
        impute_method="causal_ffill_train_median",
    )

    assert series[0, 0].item() == 3.0
    assert info["train_fill_values"]["load"] == 3.0


def test_split_safe_imputation_val_and_test_leading_gaps_use_history_only(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="h"),
            "load": [1.0, 2.0, 3.0, 4.0, -9999.0, 100.0, -9999.0, -9999.0, 200.0, 201.0],
        }
    ).to_csv(path, index=False)

    series, _, info = load_split_safe_tabular_timeseries(
        path=str(path),
        timestamp_col="date",
        feature_cols=["load"],
        missing_sentinels=[-9999.0],
        train_ratio=0.4,
        val_ratio=0.2,
        impute_method="causal_ffill_train_median",
    )

    assert series[4, 0].item() == 4.0
    assert series[6, 0].item() == 100.0
    assert series[7, 0].item() == 100.0
    assert info["val_missing_before"] == 1
    assert info["test_missing_before"] == 2
    assert info["preprocessing_row_count_before"] == 10
    assert info["preprocessing_row_count_after"] == 10


def test_split_safe_imputation_rejects_feature_all_missing_in_train(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="h"),
            "load": [-9999.0, -9999.0, -9999.0, 10.0, 11.0, 12.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="train split is all missing"):
        load_split_safe_tabular_timeseries(
            path=str(path),
            timestamp_col="date",
            feature_cols=["load"],
            missing_sentinels=[-9999.0],
            train_ratio=0.5,
            val_ratio=0.1667,
            impute_method="causal_ffill_train_median",
        )
