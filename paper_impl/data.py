"""时间顺序划分、只用训练统计量的归一化与可追踪窗口。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch


@dataclass
class DataConfig:
    path: str
    timestamp_column: str
    target_columns: list[str]
    lookback: int = 24
    horizon: int = 12
    feature_columns: list[str] | None = None
    train_fraction: float = .6
    validation_fraction: float = .2
    max_rows: int | None = None
    encoding: str = "utf-8"
    expected_interval_seconds: float | None = None
    target_input_columns: dict[str, str] | None = None
    target_mask_columns: list[str] | None = None
    protocol: str = "chronological"
    online_test_fraction: float = .20
    shift_pool_fraction: float = .35
    extra_delay: int = 0


@dataclass
class WindowSplit:
    inputs: torch.Tensor
    targets: torch.Tensor
    issue_times: np.ndarray

    def __len__(self):
        return len(self.issue_times)


@dataclass
class PreparedData:
    values: torch.Tensor
    targets: torch.Tensor
    train: WindowSplit
    validation: WindowSplit
    test: WindowSplit
    metadata: dict


def _windows(values, target_indices, begin, end, lookback, horizon, *, labels=None, issues=None, complete_labels=False):
    # [begin,end) 指目标所在行；允许前缀历史成为下一分段输入。
    issue = (np.arange(max(lookback - 1, begin - 1), end - horizon, dtype=np.int64)
             if issues is None else np.asarray(issues, dtype=np.int64))
    if not len(issue):
        raise ValueError("split too short for lookback and horizon")
    xs = torch.stack([values[t - lookback + 1:t + 1] for t in issue])
    target_values = values[:, target_indices] if labels is None else labels
    ys = torch.stack([target_values[t + 1:t + horizon + 1] for t in issue])
    if complete_labels:
        eligible = torch.isfinite(ys).all(dim=(1, 2))
        xs, ys, issue = xs[eligible], ys[eligible], issue[eligible.numpy()]
        if not len(issue):
            raise ValueError("split has no fully observed training/validation horizons")
    return WindowSplit(xs, ys, issue)


def prepare_data(config: DataConfig) -> PreparedData:
    if config.lookback < 1 or config.horizon < 1:
        raise ValueError("lookback/horizon must be positive")
    if not (0 < config.train_fraction < 1 and 0 < config.validation_fraction < 1 - config.train_fraction):
        raise ValueError("invalid chronological split fractions")
    if config.protocol not in ("chronological", "legacy_shifted"):
        raise ValueError("protocol must be chronological or legacy_shifted")
    if not isinstance(config.extra_delay, int) or isinstance(config.extra_delay, bool) or config.extra_delay < 0:
        raise ValueError("extra_delay must be a nonnegative integer")
    path = Path(config.path).resolve()
    frame = pd.read_csv(path, encoding=config.encoding)
    if config.timestamp_column not in frame:
        raise ValueError("timestamp column is required for chronology verification")
    times = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("timestamps must be strictly increasing; do not silently sort or deduplicate")
    intervals = times.diff().dropna().dt.total_seconds()
    if len(intervals) == 0 or intervals.nunique() != 1:
        raise ValueError("regular sampling is required; missing time positions need explicit preprocessing")
    interval_seconds = float(intervals.iloc[0])
    if config.expected_interval_seconds is not None and interval_seconds != config.expected_interval_seconds:
        raise ValueError("sampling interval differs from the declared interval")
    if config.max_rows is not None:
        if config.max_rows < 1:
            raise ValueError("max_rows must be positive")
        frame = frame.iloc[:config.max_rows].copy()
        times = times.iloc[:config.max_rows]
    target_masks = config.target_mask_columns or []
    mapping = config.target_input_columns or {name: name for name in config.target_columns}
    if set(mapping) != set(config.target_columns) or any(not isinstance(c, str) for c in mapping.values()):
        raise ValueError("target_input_columns must map every target to an input column")
    if target_masks and (len(target_masks) != len(config.target_columns) or len(set(target_masks)) != len(target_masks)):
        raise ValueError("target_mask_columns must specify one distinct observation mask per target")
    excluded = set(target_masks) | {name for name, source in mapping.items() if source != name}
    columns = config.feature_columns or [c for c in frame.select_dtypes(include="number").columns
                                         if c != config.timestamp_column and c not in excluded]
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError("features must be nonempty and unique")
    if not config.target_columns or len(set(config.target_columns)) != len(config.target_columns):
        raise ValueError("targets must be nonempty and unique")
    if not set(mapping.values()).issubset(columns):
        raise ValueError("forecast targets require explicit input columns or a target_input_columns mapping")
    if set(target_masks) & set(columns):
        raise ValueError("target evaluation masks cannot be used as input features")
    raw = frame[columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    true_targets = frame[config.target_columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if target_masks:
        masks = frame[target_masks].apply(pd.to_numeric, errors="raise").to_numpy()
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("target masks must contain only zero and one")
        if (masks.astype(bool) & ~np.isfinite(true_targets.to_numpy())).any():
            raise ValueError("a target marked observed must be finite")
        true_targets = true_targets.mask(~masks.astype(bool))
    elif true_targets.isna().any().any():
        raise ValueError("missing forecast targets require an explicit evaluation mask; they are never imputed as labels")
    count = len(raw)
    train_end = int(count * config.train_fraction)
    validation_end = int(count * (config.train_fraction + config.validation_fraction))
    if config.protocol == "legacy_shifted":
        n_windows = count - config.lookback - config.horizon + 1
        n_train = int(n_windows * config.train_fraction)
        n_val = int(n_windows * config.validation_fraction)
        train_end = n_train + config.lookback
        validation_end = n_train + n_val + config.lookback
    # 只能forward fill，缺失初值用训练前缀中位数；绝不跨未来插值。
    medians = raw.iloc[:train_end].median()
    if medians.isna().any():
        raise ValueError("all-missing training feature cannot be imputed")
    filled = raw.ffill().fillna(medians)
    mean = filled.iloc[:train_end].mean()
    original_std = filled.iloc[:train_end].std(ddof=0)
    unit_std_columns = original_std.index[original_std <= 1e-8].tolist()
    std = original_std.where(original_std > 1e-8, 1.0)
    values = torch.tensor(((filled - mean) / std).to_numpy(), dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("standardized observations are outside the finite float32 range")
    target_indices = [columns.index(mapping[c]) for c in config.target_columns]
    target_mean = mean.iloc[target_indices].to_numpy()
    target_std = std.iloc[target_indices].to_numpy()
    labels = torch.tensor((true_targets.to_numpy() - target_mean) / target_std, dtype=torch.float32)
    if torch.isinf(labels).any():
        raise ValueError("standardized targets are outside the float32 range")
    from .stream_protocols import chronological_issue_indices, select_stream_indices
    if config.protocol == "chronological":
        selection = chronological_issue_indices(count, lookback=config.lookback, horizon=config.horizon,
                         train_fraction=config.train_fraction, validation_fraction=config.validation_fraction, extra_delay=config.extra_delay)
        issues = {name: np.array(part["issue_indices"], dtype=np.int64) for name, part in selection["splits"].items()}
    else:
        all_issues = np.arange(config.lookback - 1, count - config.horizon, dtype=np.int64)
        all_x = torch.stack([values[t - config.lookback + 1:t + 1] for t in all_issues])
        selected = select_stream_indices(all_x, protocol=config.protocol, train_fraction=config.train_fraction,
                    validation_fraction=config.validation_fraction, online_test_fraction=config.online_test_fraction,
                    shift_pool_fraction=config.shift_pool_fraction)
        selection = selected.manifest
        issues = {"train": all_issues[selected.train_indices], "validation": all_issues[selected.validation_indices],
                  "test": all_issues[selected.test_indices]}
        selection["normalization_scope"] = "observations through first validation issue; causal preprocessing, legacy selection formula"
    cutoffs = {"train": int(issues["validation"][0]), "validation": int(issues["test"][0])}
    admissible = {name: (indices[indices + config.horizon + config.extra_delay <= cutoffs[name]] if name != "test" else indices)
                  for name, indices in issues.items()}
    splits = {name: _windows(values, target_indices, 0, count, config.lookback, config.horizon,
                  labels=labels, issues=indices, complete_labels=name != "test") for name, indices in admissible.items()}
    metadata = dict(config=asdict(config), path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    rows=count, columns=columns, target_indices=target_indices, train_end=train_end, validation_end=validation_end,
                    mean=mean.tolist(), std=std.tolist(), imputation_medians=medians.tolist(),
                    original_training_std=original_std.tolist(), unit_std_columns=unit_std_columns,
                    first_timestamp=str(times.iloc[0]), last_timestamp=str(times.iloc[-1]),
                    interval_seconds=interval_seconds,
                    observed_missing_count=int(raw.isna().sum().sum()), imputation="forward_fill_then_training_median",
                    target_mean=target_mean.tolist(), target_std=target_std.tolist(),
                    target_missing_count=int(true_targets.isna().sum().sum()), selection=selection,
                    excluded_incomplete_supervised_horizons={name: len(admissible[name]) - len(splits[name]) for name in ("train", "validation")},
                    excluded_unreleased_supervised_horizons={name: len(issues[name]) - len(admissible[name]) for name in ("train", "validation")},
                    offline_training_cutoff=cutoffs["train"], offline_validation_cutoff=cutoffs["validation"],
                    missing_label_policy="predict every test window; score observed points; adapt only after complete observed horizons",
                    issue_times={name: part.issue_times.tolist() for name, part in splits.items()})
    return PreparedData(values, labels, splits["train"], splits["validation"], splits["test"], metadata)
