import json

import numpy as np
import pandas as pd
import pytest

from tools.prepare_lishui_online import (LoadSource, add_causal_inputs, aggregate_hourly,
                                        build_lishui_online, file_sha256, read_load_source)


def source_file(path, count=400, values=None, **source_kwargs):
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 00:15", periods=count, freq="15min"),
                          "col2": np.arange(count, dtype=float) + 100 if values is None else values})
    frame.to_csv(path, index=False)
    return LoadSource(path.stem, path.stem, str(path), **source_kwargs), frame


def test_hourly_right_ending_bins_do_not_read_future(tmp_path):
    source, original = source_file(tmp_path / "county.csv", count=12)
    native, _ = read_load_source(source)
    hourly = add_causal_inputs(aggregate_hourly(native, 900))
    assert hourly.index[0].hour == 1
    assert hourly.iloc[0].load == pytest.approx(101.5)
    original.loc[4:, "col2"] += 1e9
    original.to_csv(source.path, index=False)
    changed_native, _ = read_load_source(source)
    changed = add_causal_inputs(aggregate_hourly(changed_native, 900))
    pd.testing.assert_frame_equal(hourly.iloc[:1], changed.iloc[:1])


def test_missing_positions_and_zero_targets_are_not_fabricated(tmp_path):
    source, frame = source_file(tmp_path / "county.csv", count=20)
    frame.loc[:3, "col2"] = 0
    frame = frame.drop(index=5)
    frame.loc[10, "col2"] = np.nan
    frame.to_csv(source.path, index=False)
    native, metadata = read_load_source(source)
    hourly = add_causal_inputs(aggregate_hourly(native, 900))
    assert metadata["missing_source_time_positions"] == 1
    assert metadata["invalid_value_count"] == 1
    assert metadata["zero_value_count"] == 4
    assert hourly.iloc[0].load == 0 and hourly.iloc[0].load_observed == 1
    for index in (1, 2):
        assert np.isnan(hourly.iloc[index].load)
        assert hourly.iloc[index].load_observed == 0
        assert hourly.iloc[index].coverage_fraction == .75
        assert hourly.iloc[index].load_input == 0


def test_leading_missing_stays_missing_and_negative_source_value_retained(tmp_path):
    source, frame = source_file(tmp_path / "county.csv", count=12)
    frame.loc[0, "col2"] = -1
    frame.to_csv(source.path, index=False)
    native, metadata = read_load_source(source)
    hourly = add_causal_inputs(aggregate_hourly(native, 900))
    assert native.iloc[0].source_load == -1
    assert np.isnan(native.iloc[0].load)
    assert metadata["negative_value_count"] == 1
    assert np.isnan(hourly.iloc[0].load_input)
    assert np.isfinite(hourly.iloc[1].load_input)


def test_units_unknown_never_silently_become_mw(tmp_path):
    source, _ = source_file(tmp_path / "county.csv", count=8)
    _, metadata = read_load_source(source)
    assert metadata["unit"] is None
    assert metadata["unit_multiplier"] == 1
    with pytest.raises(ValueError, match="unknown physical units"):
        read_load_source(source, output_unit="MW")
    known = LoadSource(**{**source.__dict__, "unit": "kW"})
    native, metadata = read_load_source(known, output_unit="MW")
    assert native.iloc[0].load == pytest.approx(.1)
    assert metadata["unit"] == "MW" and metadata["unit_multiplier"] == .001


@pytest.mark.parametrize("operation", ["duplicate", "reverse", "off_grid"])
def test_ambiguous_timestamps_rejected(tmp_path, operation):
    source, frame = source_file(tmp_path / "county.csv", count=8)
    if operation == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif operation == "reverse":
        frame = frame.iloc[::-1]
    else:
        frame.loc[1, "timestamp"] += pd.Timedelta(minutes=1)
    frame.to_csv(source.path, index=False)
    with pytest.raises(ValueError, match="timestamps"):
        read_load_source(source)


def test_builder_freezes_sources_masks_windows_and_missing_city_status(tmp_path):
    source, frame = source_file(tmp_path / "county.csv")
    frame.loc[225, "col2"] = np.nan
    frame.to_csv(source.path, index=False)
    before = file_sha256(source.path)
    output = tmp_path / "out"
    result = build_lishui_online([source], output, lookback=4, horizon=2, extra_delay=3)
    assert file_sha256(source.path) == before
    assert result["missing_city_source"] is True
    assert result["county_aggregation_used_to_create_city"] is False
    assert result["independent_task_count"] == 1
    meta = json.loads((output / "metadata/county.json").read_text(encoding="utf8"))
    assert meta["source_sha256"] == before
    assert meta["target_input_columns"] == {"load": "load_input"}
    assert meta["target_mask_columns"] == ["load_observed"]
    assert meta["window_validity"]["validation"]["incomplete_target_window_count"] > 0
    prepared = pd.read_csv(output / "hourly/county.csv")
    assert prepared.load.isna().sum() == 1
    assert prepared.load_input.isna().sum() == 0
    assert (output / "native/county.csv.gz").exists()
    indices = json.loads((output / "splits/county.json").read_text(encoding="utf8"))
    assert indices["splits"]["test"]["release_indices"][0] == indices["splits"]["test"]["issue_indices"][0] + 5
    with pytest.raises(FileExistsError):
        build_lishui_online([source], output, lookback=4, horizon=2)


def test_duplicate_series_cannot_be_counted_as_independent_city(tmp_path):
    county, frame = source_file(tmp_path / "county.csv")
    frame.to_csv(tmp_path / "city.csv", index=False)
    city = LoadSource("city", "city", str(tmp_path / "city.csv"), scope="city_total")
    with pytest.raises(ValueError, match="duplicate observation series"):
        build_lishui_online([county, city], tmp_path / "out", lookback=4, horizon=2)


def test_independent_city_and_alternate_5min_import(tmp_path):
    county, _ = source_file(tmp_path / "county.csv")
    city, _ = source_file(tmp_path / "city.csv", values=np.arange(400, dtype=float) + 1000, scope="city_total")
    path = tmp_path / "alternate.csv"
    pd.DataFrame({"t": pd.date_range("2025-01-01", periods=100, freq="5min"),
                  "v": np.arange(100) + 100}).to_csv(path, index=False, header=False)
    alternate = LoadSource("alternate", "alternate", str(path), source_interval_seconds=300,
                           header=False, timestamp_column=0, value_column=1, scope="city_alternate")
    result = build_lishui_online([county, city], tmp_path / "out", alternate_sources=[alternate],
                                 lookback=4, horizon=2)
    assert result["independent_task_count"] == 2
    assert result["missing_city_source"] is False
    assert len(result["alternate_sources"]) == 1
    assert result["alternate_sources"][0]["primary_benchmark_entity"] is False


def test_confirmed_5min_city_has_independent_time_axis_and_unit_null(tmp_path):
    county, _ = source_file(tmp_path / "county.csv")
    city_path = tmp_path / "city.csv"
    pd.DataFrame({"t": pd.date_range("2026-01-01", periods=1200, freq="5min"),
                  "v": np.arange(1200) + 1000}).to_csv(city_path, index=False, header=False)
    city = LoadSource("city", "city", str(city_path), source_interval_seconds=300,
                      header=False, timestamp_column=0, value_column=1,
                      scope="city_societal_total_user_confirmed_20261007")
    result = build_lishui_online([county, city], tmp_path / "out", lookback=4, horizon=2)
    assert result["time_alignment"] == "per_entity"
    assert result["unit"] is None
    assert result["tasks"][0]["first_timestamp"].startswith("2024-")
    assert result["tasks"][1]["first_timestamp"].startswith("2026-")
    assert "user_confirmed" in result["city_identity_status"]
    assert result["tasks"][0]["split_manifest"] != result["tasks"][1]["split_manifest"]
    output = pd.read_csv(tmp_path / "out/hourly/city.csv")
    assert output.loc[0, "load_observed"] == 0
    assert np.isnan(output.loc[0, "load_input"])
    assert output.loc[1, "load"] == pytest.approx(np.arange(1001, 1013).mean())
    with pytest.raises(ValueError, match="no overlapping"):
        build_lishui_online([county, city], tmp_path / "common", lookback=4, horizon=2, alignment="common")
