from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOAD_TO_STATION = {
    "丽水莲都区全社会负荷总有功.csv": "莲都",
    "丽水龙泉市全社会负荷总有功.csv": "龙泉",
    "丽水遂昌县全社会负荷总有功.csv": "遂昌",
    "丽水庆元县全社会负荷总有功.csv": "庆元",
    "丽水云和县全社会负荷总有功.csv": "云和",
    "丽水青田县全社会负荷总有功.csv": "青田",
    "丽水缙云县全社会负荷总有功.csv": "缙云",
    "丽水景宁县全社会负荷总有功.csv": "景宁",
    "丽水松阳县全社会负荷总有功.csv": "松阳",
}

WEATHER_RENAMES = {
    "风向": "wind_dir",
    "WindDirect": "wind_dir",
    "风速": "wind_speed",
    "WindVelocity": "wind_speed",
    "雨量": "precip",
    "降水": "precip",
    "Precipitation": "precip",
    "气温": "temp",
    "DryBulTemp": "temp",
    "相对湿度": "rel_humidity",
    "RelHumidity": "rel_humidity",
    "总辐射辐照度(W/m²)": "global_rad",
    "直接辐射辐照度(W/m²)": "direct_rad",
}


@dataclass
class PreparedDataset:
    county: str
    station: str
    path: Path
    summary: dict


def _station_key(name: str) -> str:
    name = name.replace("丽水", "").replace("全社会负荷总有功", "")
    for suffix in ("县", "区", "市"):
        name = name.replace(suffix, "")
    return name.replace(".csv", "")


def _parse_weather_timestamp(series: pd.Series) -> pd.Series:
    values = []
    excel_origin = pd.Timestamp("1899-12-30")
    for value in series:
        if pd.isna(value):
            values.append(pd.NaT)
        elif isinstance(value, (pd.Timestamp, datetime, date)):
            values.append(pd.Timestamp(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values.append(excel_origin + pd.to_timedelta(float(value), unit="D"))
        else:
            values.append(pd.to_datetime(value, errors="coerce"))
    return pd.Series(values, index=series.index, dtype="datetime64[ns]")


def _clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"}).copy()
    df["timestamp"] = _parse_weather_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.rename(columns={c: WEATHER_RENAMES.get(c, c) for c in df.columns})

    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "wind_dir" in df.columns:
        df.loc[(df["wind_dir"] < 0) | (df["wind_dir"] > 360), "wind_dir"] = np.nan
    if "wind_speed" in df.columns:
        df.loc[(df["wind_speed"] < 0) | (df["wind_speed"] > 60), "wind_speed"] = np.nan
    if "precip" in df.columns:
        df.loc[(df["precip"] < 0) | (df["precip"] > 500), "precip"] = np.nan
    if "temp" in df.columns:
        df.loc[(df["temp"] < -40) | (df["temp"] > 55), "temp"] = np.nan
    if "rel_humidity" in df.columns:
        df.loc[(df["rel_humidity"] < 0) | (df["rel_humidity"] > 100), "rel_humidity"] = np.nan
    for col in ("global_rad", "direct_rad"):
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    df = df.set_index("timestamp").resample("1h").mean()
    df = df.interpolate(method="time").ffill().bfill()
    return df.reset_index()


def _clean_load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["timestamp", "load"], na_values=["<NULL>"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df.loc[df["load"] <= 0, "load"] = np.nan

    valid = df["load"].dropna()
    if not valid.empty:
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - 3.0 * iqr)
        upper = q3 + 3.0 * iqr
        df.loc[(df["load"] < lower) | (df["load"] > upper), "load"] = np.nan

    df = df.set_index("timestamp").resample("1h").mean()
    df["load"] = df["load"].interpolate(method="time").ffill().bfill()
    return df.reset_index()


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    month = ts.dt.month.astype(float)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1.0) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1.0) / 12.0)
    return df


def prepare_lishui_family(raw_root: Path | str, output_dir: Path | str) -> list[PreparedDataset]:
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_path = next(raw_root.glob("气象要素与辐照度监测数据*.xlsx"))
    xls = pd.ExcelFile(weather_path)
    weather_map = {sheet: _clean_weather(pd.read_excel(weather_path, sheet_name=sheet)) for sheet in xls.sheet_names}

    prepared: list[PreparedDataset] = []
    summary_rows: list[dict] = []
    for load_name, station in LOAD_TO_STATION.items():
        load_path = raw_root / load_name
        load_df = _clean_load(load_path)
        weather_df = weather_map[station]

        overlap_start = max(load_df["timestamp"].min(), weather_df["timestamp"].min())
        overlap_end = min(load_df["timestamp"].max(), weather_df["timestamp"].max())
        overlap_start = pd.Timestamp(overlap_start).ceil("1h")
        overlap_end = pd.Timestamp(overlap_end).floor("1h")
        full_index = pd.date_range(overlap_start, overlap_end, freq="1h")

        load_hourly = load_df.set_index("timestamp").reindex(full_index).rename_axis("timestamp")
        weather_hourly = weather_df.set_index("timestamp").reindex(full_index).rename_axis("timestamp")

        load_hourly["load"] = load_hourly["load"].interpolate(method="time").ffill().bfill()
        weather_hourly = weather_hourly.interpolate(method="time").ffill().bfill()

        merged = load_hourly.reset_index().merge(weather_hourly.reset_index(), on="timestamp", how="inner")
        merged = _add_calendar_features(merged)

        county = _station_key(load_name)
        out_path = output_dir / f"lishui_{county}.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")

        summary = {
            "county": county,
            "station": station,
            "start": merged["timestamp"].min(),
            "end": merged["timestamp"].max(),
            "rows": int(len(merged)),
            "load_mean": float(merged["load"].mean()),
            "load_std": float(merged["load"].std()),
            "feature_count": int(len(merged.columns) - 1),
        }
        summary_rows.append(summary)
        prepared.append(PreparedDataset(county=county, station=station, path=out_path, summary=summary))

    pd.DataFrame(summary_rows).sort_values("county").to_csv(output_dir / "lishui_summary.csv", index=False, encoding="utf-8-sig")
    return prepared


def load_lishui_registry(prepared_dir: Path | str) -> list[dict]:
    prepared_dir = Path(prepared_dir)
    rows = []
    for csv_path in sorted(prepared_dir.glob("lishui_*.csv")):
        if csv_path.name == "lishui_summary.csv":
            continue
        rows.append(
            {
                "name": csv_path.stem,
                "path": str(csv_path.as_posix()),
                "timestamp_col": "timestamp",
                "target_col": "load",
                "encoding": "utf-8-sig",
                "impute_method": "time_interpolate",
            }
        )
    return rows


def iter_lishui_counties(prepared_dir: Path | str) -> Iterable[str]:
    for item in load_lishui_registry(prepared_dir):
        yield item["name"]
