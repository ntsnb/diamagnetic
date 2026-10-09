"""从独立负荷观测文件构建带真值掩码的丽水在线数据；不修改原文件。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_impl.stream_protocols import chronological_issue_indices, write_selection_manifest


COUNTY_FILES = {
    "云和": "丽水云和县全社会负荷总有功.csv",
    "庆元": "丽水庆元县全社会负荷总有功.csv",
    "景宁": "丽水景宁县全社会负荷总有功.csv",
    "松阳": "丽水松阳县全社会负荷总有功.csv",
    "缙云": "丽水缙云县全社会负荷总有功.csv",
    "莲都": "丽水莲都区全社会负荷总有功(含南城).csv",
    "遂昌": "丽水遂昌县全社会负荷总有功.csv",
    "青田": "丽水青田县全社会负荷总有功.csv",
    "龙泉": "丽水龙泉市全社会负荷总有功.csv",
}
UNIT_TO_MW = {"W": 1e-6, "kW": 1e-3, "MW": 1.0, "GW": 1e3}


@dataclass(frozen=True)
class LoadSource:
    entity_id: str
    name: str
    path: str
    source_interval_seconds: int = 900
    unit: str = "unknown"
    scope: str = "county_societal_active_power"
    header: bool = True
    timezone: str = "Asia/Shanghai"
    timestamp_column: str | int = "timestamp"
    value_column: str | int = "col2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_conversion(unit: str, output_unit: str) -> tuple[str, float]:
    if unit not in (*UNIT_TO_MW, "unknown"):
        raise ValueError("unit must be W, kW, MW, GW, or unknown")
    if output_unit == "source":
        return unit, 1.0
    if output_unit not in UNIT_TO_MW:
        raise ValueError("output_unit must be source, W, kW, MW, or GW")
    if unit == "unknown":
        raise ValueError("cannot convert unknown physical units")
    return output_unit, UNIT_TO_MW[unit] / UNIT_TO_MW[output_unit]


def read_load_source(source: LoadSource, *, output_unit: str = "source") -> tuple[pd.DataFrame, dict]:
    """恢复原生规则网格，保留缺测位置，拒绝重复时间和暗中排序。"""
    if not re.fullmatch(r"[\w-]+", source.entity_id, flags=re.UNICODE):
        raise ValueError("entity_id must not contain path separators or punctuation")
    if (isinstance(source.source_interval_seconds, bool) or
            not isinstance(source.source_interval_seconds, int) or source.source_interval_seconds <= 0):
        raise ValueError("source_interval_seconds must be a positive integer")
    path = Path(source.path).resolve()
    frame = pd.read_csv(path, header=0 if source.header else None, encoding="utf-8-sig",
                        na_values=["<NULL>"], keep_default_na=True)
    if len(frame) < 2:
        raise ValueError("source must contain at least two observations")
    if source.timestamp_column not in frame or source.value_column not in frame:
        raise ValueError("declared timestamp/value columns are missing")
    times = pd.DatetimeIndex(pd.to_datetime(frame[source.timestamp_column], errors="raise", format="mixed"))
    if times.hasnans or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("source timestamps must be strictly increasing and unique")
    naive = times.tz is None
    times = times.tz_localize(source.timezone) if naive else times.tz_convert(source.timezone)
    tick = source.source_interval_seconds * 1_000_000_000
    if np.any(np.diff(times.asi8) % tick) or np.any(times.asi8 % tick):
        raise ValueError("timestamps do not lie on the declared source sampling grid")
    raw = pd.to_numeric(frame[source.value_column], errors="coerce").to_numpy(dtype=float)
    output_unit, multiplier = _unit_conversion(source.unit, output_unit)
    raw *= multiplier
    valid = np.isfinite(raw) & (raw >= 0)
    native = pd.DataFrame({"source_load": raw, "load": np.where(valid, raw, np.nan),
                           "load_observed": valid.astype(np.int8), "source_row_present": 1}, index=times)
    index = pd.date_range(times[0], times[-1], freq=pd.Timedelta(seconds=source.source_interval_seconds))
    native = native.reindex(index)
    native[["load_observed", "source_row_present"]] = native[["load_observed", "source_row_present"]].fillna(0).astype(np.int8)
    native.index.name = "timestamp"
    # Separate hashes keep identity exact even when a float cannot hold timestamp ns.
    observation_hash = hashlib.sha256(times.asi8.tobytes() + raw.tobytes()).hexdigest()
    metadata = {
        "source": asdict(source), "source_path": path.as_posix(), "source_sha256": file_sha256(path),
        "observation_sha256": observation_hash, "original_columns": [str(c) for c in frame.columns],
        "source_rows": len(frame), "native_grid_rows": len(native),
        "first_timestamp": times[0].isoformat(), "last_timestamp": times[-1].isoformat(),
        "source_interval_seconds": source.source_interval_seconds,
        "source_timezone_was_undeclared": naive,
        "timezone_assumption": source.timezone if naive else None,
        "source_unit": None if source.unit == "unknown" else source.unit,
        "unit": None if output_unit == "unknown" else output_unit, "unit_multiplier": multiplier,
        "unit_status": "unverified_source_unit" if source.unit == "unknown" else "explicit_caller_declaration",
        "physical_comparison_across_entities_verified": False,
        "prior_source_processing": "unknown; observed file values, not a certified acquisition export",
        "missing_source_time_positions": len(native) - len(frame),
        "invalid_value_count": int((~np.isfinite(raw)).sum()),
        "negative_value_count": int((np.isfinite(raw) & (raw < 0)).sum()),
        "zero_value_count": int((raw == 0).sum()),
        "invalid_value_policy": "nonfinite and negative values masked; zero retained as observed",
    }
    return native, metadata


def aggregate_hourly(native: pd.DataFrame, source_interval_seconds: int) -> pd.DataFrame:
    """小时结束时刻 t 只使用 (t-1h,t] 的原始观测，覆盖率不足不生成真值。"""
    if source_interval_seconds <= 0 or 3600 % source_interval_seconds:
        raise ValueError("source interval must divide one hour exactly")
    expected = 3600 // source_interval_seconds
    grouped = native.resample("1h", closed="right", label="right", origin="epoch")
    observed = grouped.load_observed.sum().astype(int)
    present = grouped.source_row_present.sum().astype(int)
    hourly = pd.DataFrame({"load": grouped.load.mean(), "source_observed_count": observed,
                           "source_row_count": present, "source_expected_count": expected})
    hourly["load_observed"] = (observed == expected).astype(np.int8)
    hourly.loc[hourly.load_observed == 0, "load"] = np.nan
    hourly["coverage_fraction"] = observed / expected
    hourly.index.name = "timestamp"
    return hourly


def add_causal_inputs(hourly: pd.DataFrame) -> pd.DataFrame:
    """真值保持缺失；输入只前向填充，首个有效观测前仍为 NaN。"""
    frame = hourly.copy()
    frame["load_input"] = frame.load.ffill()
    frame["load_input_observed"] = frame.load_observed.astype(np.int8)
    for label, values, period in (("hour", frame.index.hour, 24),
                                  ("dow", frame.index.dayofweek, 7),
                                  ("month", frame.index.month - 1, 12)):
        frame[f"{label}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"{label}_cos"] = np.cos(2 * np.pi * values / period)
    return frame


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # Stable compression header and stable float/text formatting aid reproducibility.
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, encoding="utf-8", float_format="%.17g", compression=compression,
                 index=True, date_format="%Y-%m-%dT%H:%M:%S%z", lineterminator="\n")


def _window_validity(frame: pd.DataFrame, manifest: dict, feature_columns: list[str]) -> dict:
    result = {}
    valid_target = frame.load_observed.to_numpy(dtype=bool)
    finite_inputs = np.isfinite(frame[feature_columns].to_numpy()).all(axis=1)
    lookback, horizon = manifest["lookback"], manifest["horizon"]
    for name, split in manifest["splits"].items():
        issues = np.asarray(split["issue_indices"], dtype=np.int64)
        input_ok = np.array([finite_inputs[t-lookback+1:t+1].all() for t in issues])
        target_ok = np.array([valid_target[t+1:t+horizon+1].all() for t in issues])
        result[name] = {"issued_window_count": len(issues),
                        "finite_input_window_count": int(input_ok.sum()),
                        "complete_supervised_window_count": int((input_ok & target_ok).sum()),
                        "incomplete_target_window_count": int((~target_ok).sum()),
                        "unavailable_input_window_count": int((~input_ok).sum())}
    return result


def build_lishui_online(sources: list[LoadSource], output: str | Path, *,
                        alternate_sources: list[LoadSource] | None = None,
                        output_unit: str = "source", lookback: int = 24, horizon: int = 12,
                        extra_delay: int = 0, train_fraction: float = .55,
                        validation_fraction: float = .10, alignment: str = "per_entity") -> dict:
    """构建独立任务及各自冻结时间索引；不同年份的县、市不强求交集。"""
    if not sources:
        raise ValueError("at least one independent source is required")
    alternates = alternate_sources or []
    if alignment not in ("per_entity", "common"):
        raise ValueError("alignment must be per_entity or common")
    ids = [s.entity_id for s in sources + alternates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate entity identity")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be a new or empty directory; original datasets are immutable")
    loaded = [(*read_load_source(source, output_unit=output_unit), source) for source in sources]
    # Even a differently named or formatted duplicate must not become another county/city.
    hashes = [metadata["observation_sha256"] for _, metadata, _ in loaded]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate observation series assigned to different entities")
    units = {metadata["unit"] for _, metadata, _ in loaded}
    if len(units) != 1:
        raise ValueError("task output units differ; explicitly convert known units before comparing")
    hourly_frames = [aggregate_hourly(native, source.source_interval_seconds) for native, _, source in loaded]
    bounds = [(frame.index.min(), native.index.max().floor("1h"))
              for frame, (native, _, _) in zip(hourly_frames, loaded)]
    if alignment == "common":
        start, end = max(b[0] for b in bounds), min(b[1] for b in bounds)
        if end < start:
            raise ValueError("independent primary sources have no overlapping time range")
        bounds = [(start, end)] * len(sources)
    # Validate every source, candidate and split before publishing output files.
    alternate_loaded = [(*read_load_source(source, output_unit=output_unit), source) for source in alternates]
    indexed = []
    for (start, end), (_, metadata, source) in zip(bounds, loaded):
        index = pd.date_range(start, end, freq="1h", name="timestamp")
        manifest = chronological_issue_indices(len(index), lookback=lookback, horizon=horizon,
                                                train_fraction=train_fraction,
                                                validation_fraction=validation_fraction, extra_delay=extra_delay)
        manifest.update(entity_id=source.entity_id, source_sha256=metadata["source_sha256"],
                        first_timestamp=index[0].isoformat(), last_timestamp=index[-1].isoformat(),
                        interval_seconds=3600, alignment=alignment)
        for split in manifest["splits"].values():
            split["first_target_timestamp"] = index[split["target_start"]].isoformat()
            split["last_target_timestamp"] = index[split["target_end"] - 1].isoformat()
            split["first_issue_timestamp"] = index[split["issue_indices"][0]].isoformat()
            split["last_issue_timestamp"] = index[split["issue_indices"][-1]].isoformat()
        indexed.append((index, manifest))
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("hourly", "native", "metadata", "splits"):
        (output / directory).mkdir()
    feature_columns = ["load_input", "load_input_observed", "hour_sin", "hour_cos", "dow_sin",
                       "dow_cos", "month_sin", "month_cos"]
    registry = []
    summaries = []
    for (native, metadata, source), hourly, (index, manifest) in zip(loaded, hourly_frames, indexed):
        split_path = output / "splits" / (source.entity_id + ".json")
        frozen = write_selection_manifest(split_path, manifest)
        native_path = output / "native" / (source.entity_id + ".csv.gz")
        _write_csv(native, native_path)
        # Fill on the native source's hourly history, then align; never backward-fill.
        frame = add_causal_inputs(hourly).reindex(index)
        path = output / "hourly" / (source.entity_id + ".csv")
        _write_csv(frame, path)
        metadata.update(
            entity_id=source.entity_id, name=source.name, scope=source.scope,
            native_path=native_path.relative_to(output).as_posix(), native_sha256=file_sha256(native_path),
            hourly_path=path.relative_to(output).as_posix(), hourly_sha256=file_sha256(path),
            hourly_rows=len(frame), hourly_valid_targets=int(frame.load_observed.sum()),
            hourly_missing_targets=int((frame.load_observed == 0).sum()),
            hourly_interval_seconds=3600, aggregation="mean of all expected samples in (t-1h,t]; right ending label",
            partial_hour_policy="target=NaN unless every expected native position has an observed valid value",
            target_imputation="none", input_imputation="forward_fill_only; leading_missing_remains_NaN",
            zero_policy="zero is an observed value and is not imputed or dropped",
            feature_columns=feature_columns, target_columns=["load"],
            target_input_columns={"load": "load_input"}, target_mask_columns=["load_observed"],
            hourly_first_timestamp=index[0].isoformat(), hourly_last_timestamp=index[-1].isoformat(),
            split_manifest=split_path.relative_to(output).as_posix(), split_manifest_sha256=frozen["manifest_sha256"],
            window_validity=_window_validity(frame, manifest, feature_columns),
            issue_release_rule="issue t forecasts t+1..t+H; complete labels release at t+H+extra_delay",
        )
        meta_path = output / "metadata" / (source.entity_id + ".json")
        _json(meta_path, metadata)
        registry.append({"entity_id": source.entity_id, "name": source.name, "scope": source.scope,
                         "path": path.relative_to(output).as_posix(), "metadata": meta_path.relative_to(output).as_posix(),
                         "unit": metadata["unit"], "target_columns": ["load"], "feature_columns": feature_columns,
                         "target_input_columns": {"load": "load_input"}, "target_mask_columns": ["load_observed"],
                         "timestamp_column": "timestamp", "expected_interval_seconds": 3600,
                         "source_sha256": metadata["source_sha256"], "split_manifest": metadata["split_manifest"],
                         "split_manifest_sha256": frozen["manifest_sha256"],
                         "train_fraction": train_fraction, "validation_fraction": validation_fraction,
                         "hourly_rows": len(frame), "first_timestamp": index[0].isoformat(),
                         "last_timestamp": index[-1].isoformat()})
        summaries.append({"entity_id": source.entity_id, "name": source.name, "unit": metadata["unit"],
                          "source_interval_seconds": source.source_interval_seconds,
                          "source_rows": metadata["source_rows"], "hourly_rows": len(frame),
                          "first_timestamp": index[0].isoformat(), "last_timestamp": index[-1].isoformat(),
                          "missing_native_positions": metadata["missing_source_time_positions"],
                          "missing_hourly_targets": metadata["hourly_missing_targets"],
                          "window_validity": metadata["window_validity"],
                          "split_target_time_ranges": {name: {k: part[k] for k in
                              ("first_target_timestamp", "last_target_timestamp", "target_start", "target_end")}
                              for name, part in manifest["splits"].items()}})
    alternate_registry = []
    for native, metadata, source in alternate_loaded:
        native_path = output / "native" / (source.entity_id + ".csv.gz")
        _write_csv(native, native_path)
        metadata.update(entity_id=source.entity_id, name=source.name, scope=source.scope,
                        native_path=native_path.relative_to(output).as_posix(),
                        native_sha256=file_sha256(native_path), primary_benchmark_entity=False,
                        role="candidate_source",
                        reason="candidate city source/frequency; never counted as an eleventh independent target")
        _json(output / "metadata" / (source.entity_id + ".json"), metadata)
        alternate_registry.append({"entity_id": source.entity_id, "native_path": metadata["native_path"],
                                   "metadata": f"metadata/{source.entity_id}.json", "primary_benchmark_entity": False,
                                   "role": "candidate_source"})
    city_status = ("user_confirmed_designated_city_file_20261007; physical_unit_pending" if
                   any("user_confirmed" in s.scope for s in sources) else
                   "filename says societal total; municipal-only scope and units require source confirmation")
    result = {
        "schema_version": 2, "build_kind": "independent_lishui_online_observation_tasks",
        "tasks": registry, "alternate_sources": alternate_registry,
        "candidate_sources": alternate_registry, "independent_task_count": len(registry),
        "time_alignment": alignment, "interval_seconds": 3600,
        "timezone": str(indexed[0][0].tz), "unit": next(iter(units)),
        "county_aggregation_used_to_create_city": False,
        "city_identity_status": city_status,
        "missing_city_source": not any(s.scope.startswith("city_") for s in sources),
        "missing_city_requirements": [] if any(s.scope.startswith("city_") for s in sources) else
            ["independent timestamped city observation file", "unit and population/measurement scope declaration"],
        "weather_source": "not found in the audited allowed paths; no weather was synthesized",
        "legacy_cleaned_counties_used_as_truth": False,
        "legacy_cleaned_counties_warning": "2023-2024 cleaned files have full-series interpolation and different provenance",
        "physical_comparison_across_entities_verified": False,
        "split_manifests": {task["entity_id"]: task["split_manifest"] for task in registry},
        "build_parameters": {"lookback": lookback, "horizon": horizon, "extra_delay": extra_delay,
                              "train_fraction": train_fraction, "validation_fraction": validation_fraction,
                              "output_unit": output_unit, "alignment": alignment},
        "builder_sha256": file_sha256(__file__),
    }
    _json(output / "registry.json", result)
    _json(output / "summary.json", {"schema_version": 2, "independent_task_count": len(registry),
                                    "time_alignment": alignment, "unit": next(iter(units)),
                                    "city_identity_status": city_status, "tasks": summaries,
                                    "candidate_sources": alternate_registry})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county-root", type=Path, required=True)
    parser.add_argument("--city-file", type=Path)
    parser.add_argument("--city-no-header", action="store_true")
    parser.add_argument("--city-interval-seconds", type=int, default=900)
    parser.add_argument("--city-value-column", default="col2")
    parser.add_argument("--city-timestamp-column", default="timestamp")
    parser.add_argument("--city-unit", default="unknown", choices=["unknown", *UNIT_TO_MW])
    parser.add_argument("--city-confirmed", action="store_true", help="record explicit user confirmation of the designated city source")
    parser.add_argument("--county-unit", default="unknown", choices=["unknown", *UNIT_TO_MW])
    parser.add_argument("--alternate-city-file", type=Path, help="headerless 5-minute file, kept outside benchmark task count")
    parser.add_argument("--candidate-city-file", type=Path, help="15-minute timestamp,col2 candidate, kept outside benchmark task count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-unit", default="source", choices=["source", *UNIT_TO_MW])
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--extra-delay", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=.55)
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--alignment", default="per_entity", choices=["per_entity", "common"])
    args = parser.parse_args()
    sources = [LoadSource(f"lishui_{name}", name, str(args.county_root / file), unit=args.county_unit,
                          scope="county_societal_active_power_including_nancheng" if name == "莲都" else
                                "county_societal_active_power") for name, file in COUNTY_FILES.items()]
    if args.city_file is not None:
        sources.append(LoadSource("lishui_city", "丽水全社会负荷", str(args.city_file),
                                  source_interval_seconds=args.city_interval_seconds,
                                  unit=args.city_unit, scope="city_societal_total_user_confirmed_20261007" if args.city_confirmed else
                                                           "city_societal_total_scope_unconfirmed",
                                  header=not args.city_no_header,
                                  timestamp_column=0 if args.city_no_header else args.city_timestamp_column,
                                  value_column=1 if args.city_no_header else args.city_value_column))
    alternates = []
    if args.alternate_city_file is not None:
        alternates.append(LoadSource("lishui_city_alternate_5min", "丽水全社会负荷_另一5分钟来源",
                                      str(args.alternate_city_file), source_interval_seconds=300,
                                      unit="unknown", scope="city_alternate_scope_unconfirmed", header=False,
                                      timestamp_column=0, value_column=1))
    if args.candidate_city_file is not None:
        alternates.append(LoadSource("lishui_city_candidate_15min", "丽水全社会负荷_候选15分钟来源",
                                      str(args.candidate_city_file), source_interval_seconds=900,
                                      unit="unknown", scope="city_candidate_scope_unconfirmed"))
    result = build_lishui_online(sources, args.output, alternate_sources=alternates,
                                 output_unit=args.output_unit, lookback=args.lookback, horizon=args.horizon,
                                 extra_delay=args.extra_delay, train_fraction=args.train_fraction,
                                 validation_fraction=args.validation_fraction, alignment=args.alignment)
    print(json.dumps({"output": args.output.resolve().as_posix(),
                      "independent_tasks": result["independent_task_count"],
                      "hourly_rows": {task["entity_id"]: task["hourly_rows"] for task in result["tasks"]},
                      "unit": result["unit"], "city_source_missing": result["missing_city_source"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
