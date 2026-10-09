"""冻结在线评估索引；将因果时间后缀与事后最大偏移压力测试明确区分。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class StreamSelection:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    manifest: dict


def _fractions(train_fraction: float, validation_fraction: float) -> None:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("invalid chronological split fractions")


def _numpy(inputs) -> np.ndarray:
    if hasattr(inputs, "detach"):
        inputs = inputs.detach().cpu().numpy()
    values = np.asarray(inputs)
    if values.ndim < 2 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("inputs must be finite windows with a sample and feature dimension")
    return values


def _digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def legacy_shift_score(train_inputs, candidate_inputs) -> float:
    """旧 data/loaders.py 的输入均值/方差公式，以 float64 计算。"""
    train = _numpy(train_inputs).reshape(len(train_inputs), -1).astype(np.float64)
    candidate = _numpy(candidate_inputs).reshape(len(candidate_inputs), -1).astype(np.float64)
    if train.shape[1] != candidate.shape[1]:
        raise ValueError("training and candidate input shapes differ")
    train_var = train.var(axis=0)
    mean_shift = (np.abs(candidate.mean(axis=0) - train.mean(axis=0)) /
                  np.maximum(np.sqrt(train_var), 1e-6)).mean()
    variance_shift = np.abs(np.log(np.maximum(candidate.var(axis=0), 1e-6)) -
                            np.log(np.maximum(train_var, 1e-6))).mean()
    return float(mean_shift + .1 * variance_shift)


def select_stream_indices(inputs, *, protocol: str = "chronological",
                          train_fraction: float = .55, validation_fraction: float = .10,
                          online_test_fraction: float = .20,
                          shift_pool_fraction: float = .35) -> StreamSelection:
    """在已构造的窗口数组上选样本索引。

    chronological 始终使用完整剩余后缀，不查看输入数值。legacy_shifted
    复现旧窗口计数、每个候选起点和最大分数规则；它读取候选期间的未来输入，
    只适合作为事后压力测试。输入可能包含历史负荷，但函数不接收预测目标 y。
    后者沿用旧的窗口边界（可能有跨分段目标重叠），不是严格目标隔离切分。
    严格切分请先调用 chronological_issue_indices。
    """
    _fractions(train_fraction, validation_fraction)
    if protocol not in ("chronological", "legacy_shifted"):
        raise ValueError("protocol must be chronological or legacy_shifted")
    # A count suffices for chronological selection, proving value independence.
    if isinstance(inputs, (int, np.integer)):
        if protocol != "chronological":
            raise ValueError("legacy_shifted requires the actual input windows")
        count, values = int(inputs), None
    else:
        values = _numpy(inputs)
        count = len(values)
    n_train = int(count * train_fraction)
    n_val = int(count * validation_fraction)
    remaining = count - n_train - n_val
    if min(n_train, n_val, remaining) < 1:
        raise ValueError("all three splits must contain samples")
    start, stop = n_train + n_val, count
    metadata = {
        "schema_version": 1, "protocol": protocol, "sample_count": count,
        "index_semantics": "zero_based_input_window_indices; half_open_ranges",
        "train_fraction": train_fraction, "validation_fraction": validation_fraction,
        "train_start": 0, "train_end": n_train,
        "val_start": n_train, "val_end": n_train + n_val,
        "future_inputs_for_selection": protocol == "legacy_shifted",
        "uses_future_targets_y_for_selection": False,
        "target_boundary_safety": "not_guaranteed_by_window_index_splitting",
    }
    if protocol == "legacy_shifted":
        if not (0 < online_test_fraction <= 1 and 0 < shift_pool_fraction <= 1):
            raise ValueError("test and pool fractions must be in (0, 1]")
        n_test = min(remaining, max(1, int(count * online_test_fraction)))
        pool_size = max(n_test, int(count * shift_pool_fraction))
        pool_start = max(n_train + n_val, count - pool_size)
        if count - pool_start < n_test:
            pool_start = max(n_train + n_val, count - n_test)
        candidates = np.arange(pool_start, count - n_test + 1, dtype=np.int64)
        flat = values.reshape(count, -1).astype(np.float64)
        train_mean = flat[:n_train].mean(axis=0)
        train_var = flat[:n_train].var(axis=0)
        train_std = np.maximum(np.sqrt(train_var), 1e-6)
        # Centering before cumulative moments prevents cancellation for high offsets.
        centered = flat - train_mean
        sums = np.concatenate([np.zeros((1, flat.shape[1])), centered.cumsum(axis=0)])
        squares = np.concatenate([np.zeros((1, flat.shape[1])), (centered ** 2).cumsum(axis=0)])
        scores = np.empty(len(candidates), dtype=np.float64)
        for i, candidate in enumerate(candidates):
            mean = (sums[candidate + n_test] - sums[candidate]) / n_test
            var = (squares[candidate + n_test] - squares[candidate]) / n_test - mean ** 2
            scores[i] = (np.abs(mean) / train_std).mean() + .1 * np.abs(
                np.log(np.maximum(var, 1e-6)) - np.log(np.maximum(train_var, 1e-6))).mean()
        # np.argmax, like the original strict '>', keeps the first exact tie.
        winner = int(np.argmax(scores))
        start, stop = int(candidates[winner]), int(candidates[winner]) + n_test
        metadata.update(
            evaluation_kind="retrospective_maximum_input_shift_stress_test",
            selection_bias="selected using all candidate future input blocks; not online drift detection",
            numeric_backend="float64_centered_cumulative_moments; formula_equivalent_not_bitwise_float32",
            source_reference="data/loaders.py::_shift_score,_build_shifted_splits",
            online_test_fraction=online_test_fraction, shift_pool_fraction=shift_pool_fraction,
            pool_start=pool_start, pool_end=count, candidate_starts=candidates.tolist(),
            candidate_scores=scores.tolist(), shift_score=float(scores[winner]),
            input_sha256=_digest(values),
            score_formula="mean(abs(mu_c-mu_tr)/max(sd_tr,1e-6)) + 0.1*mean(abs(log(max(var_c,1e-6))-log(max(var_tr,1e-6))))",
            tie_break="first candidate attaining maximum computed score",
        )
    else:
        metadata.update(evaluation_kind="complete_chronological_suffix", shift_score=None)
    metadata.update(test_start=start, test_end=stop)
    return StreamSelection(np.arange(n_train, dtype=np.int64),
                           np.arange(n_train, n_train + n_val, dtype=np.int64),
                           np.arange(start, stop, dtype=np.int64), metadata)


def chronological_issue_indices(n_rows: int, *, lookback: int, horizon: int,
                                train_fraction: float = .55,
                                validation_fraction: float = .10,
                                extra_delay: int = 0) -> dict:
    """按目标行严格分段；发行 t 预测 t+1..t+H，释放 t+H+delay。"""
    _fractions(train_fraction, validation_fraction)
    for name, value, minimum in (("n_rows", n_rows, 1), ("lookback", lookback, 1),
                                 ("horizon", horizon, 1), ("extra_delay", extra_delay, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    result = {"schema_version": 1, "protocol": "chronological", "rows": int(n_rows),
              "lookback": int(lookback), "horizon": int(horizon), "extra_delay": int(extra_delay),
              "train_end": train_end, "validation_end": validation_end,
              "index_semantics": "zero_based_observation_rows; target ranges are half open",
              "forecast_rule": "issue=t; target=t+1..t+H; release=t+H+extra_delay",
              "split_boundary_rule": "each complete target horizon belongs to exactly one split",
              "future_inputs_for_selection": False, "splits": {}}
    for name, begin, end in (("train", 0, train_end), ("validation", train_end, validation_end),
                             ("test", validation_end, n_rows)):
        issues = np.arange(max(lookback - 1, begin - 1), end - horizon, dtype=np.int64)
        if not len(issues):
            raise ValueError(f"{name} split too short for lookback/horizon")
        result["splits"][name] = {"target_start": begin, "target_end": end,
                                   "issue_indices": issues.tolist(),
                                   "release_indices": (issues + horizon + extra_delay).tolist()}
    return result


def write_selection_manifest(path: str | Path, selection: StreamSelection | dict) -> dict:
    """保存冻结索引及内容摘要；不同内容不能覆盖已经冻结的文件。"""
    if isinstance(selection, StreamSelection):
        payload = dict(selection.manifest, train_indices=selection.train_indices.tolist(),
                       validation_indices=selection.validation_indices.tolist(),
                       test_indices=selection.test_indices.tolist())
    else:
        payload = dict(selection)
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf8")).hexdigest()
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf8") != text:
            raise FileExistsError(f"frozen selection manifest already exists: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    return payload
