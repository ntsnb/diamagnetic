# Stage III V13 Fluctuation-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-overwriting Stage III V13 path that selectively inherits audited V2-V12 mechanisms and tests whether `delta_phi_v(t)-delta_phi_v(t-1)` provides real residual-correction value beyond causal history controls.

**Architecture:** V13 is added as a new versioned family. It introduces a V13 signal contract, a V13 teacher wrapper, a V13 teacher-only probe runner, a V13 public-3 matrix runner, V13 configs, V13 tests, and V13 audit records. It reuses real V12 data/model/training utilities and fail-closed gates; it does not create toy algorithm or toy experiment paths.

**Tech Stack:** Python, PyTorch, pytest, YAML configs, existing Stage III V12 runner utilities, train-only signal scaling, chronological OOF teacher, per-horizon label availability history controls.

---

## File Structure

Create these files:

- `train/stage3_v13_signal_contract.py`
  - Owns V13 signal semantics: `level_offset`, `fluctuation_delta`, train-only scaling, base counterfactual transforms, diagnostics, stop-condition summaries, and a separate optional helper for `variant_temporal_delta` when the runner has actual `phi_v` sequences.
- `models/stage3_residual_teacher_v13.py`
  - Wraps the V12d chronological OOF residual teacher with V13 feature modes, metadata, checkpoint save/load, and V13 diagnostic names.
- `train/run_stage3_v13_fluctuation_teacher_probe.py`
  - Real-data Panama/public single-run teacher-only V13 probe. Reuses V12d/V12c runner utilities, writes manifests, diagnostics, abort/complete files, and no-overwrite outputs.
- `train/run_stage3_v13_fluctuation_teacher_matrix.py`
  - Public-3 multi-seed matrix runner. Calls the V13 probe with non-overwriting roots, summarizes gates, and stops if a directionality stop condition appears.
- `configs/stage3_v13_fluctuation_teacher_probe.yaml`
  - V13 real-data config inherited from V12d/V12e settings, with V13 protocol/version, V13 logs/checkpoint roots, fluctuation signal modes, teacher-only gates, and no generator by default.
- `tests/test_stage3_v13_signal_contract.py`
  - Unit tests for V13 signal contract, train-only scale, previous-tail behavior, base counterfactuals, optional `phi_v` temporal-delta diagnostics, and stop-condition summaries.
- `tests/test_stage3_v13_teacher_protocol.py`
  - Unit tests for V13 feature modes, teacher wrapper metadata, checkpoint no-overwrite behavior, and abort/complete gate semantics.
- `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`
  - Evidence record for V13 design, implementation checks, pilot command, pilot output, gate status, stop conditions, and remaining risks.

Modify these existing files:

- `docs/superpowers/specs/2026-06-02-stage3-v13-fluctuation-first-design.md`
  - Already updated. Keep it as the authoritative design spec.
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
  - Add V13 implementation and pilot evidence only after commands produce actual output.
- `docs/audit/RISK_REGISTER.md`
  - Add V13 risks and stop-condition status after implementation checks.

Do not modify V2-V12 runner behavior in place.

No commits should be made unless the user explicitly asks for commits. This overrides the generic superpowers frequent-commit recommendation because `AGENTS.md` says not to commit unless explicitly asked.

---

## Task 1: V13 Signal Contract Tests

**Files:**
- Create: `tests/test_stage3_v13_signal_contract.py`
- Create later in Task 2: `train/stage3_v13_signal_contract.py`

- [ ] **Step 1: Write failing tests for level offset and fluctuation semantics**

Add this test file:

```python
from __future__ import annotations

import pytest
import torch

from train.stage3_v13_signal_contract import Stage3SignalBuilderV13, compute_v13_sequence_delta


def test_v13_fluctuation_delta_uses_level_offset_transition() -> None:
    level = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [3.0, 5.0, 9.0],
            [4.0, 8.0, 15.0],
        ]
    )
    builder = Stage3SignalBuilderV13(drift_dim=3, mode="fluctuation_delta", normalize=False)
    builder.fit(level, split="train")

    result = builder.transform(level, counterfactual_mode="real")

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 5.0],
            [1.0, 3.0, 6.0],
        ]
    )
    assert torch.allclose(result.level_offset, level)
    assert torch.allclose(result.fluctuation, expected)
    assert torch.allclose(result.signal, expected)
    assert result.mode == "fluctuation_delta"
    assert result.protocol_version == builder.protocol_version
```

- [ ] **Step 2: Write failing tests for previous-tail continuity**

Append:

```python
def test_v13_fluctuation_delta_uses_previous_tail() -> None:
    train = torch.tensor([[10.0, 20.0], [13.0, 25.0]])
    validation = torch.tensor([[18.0, 31.0], [20.0, 40.0]])
    builder = Stage3SignalBuilderV13(drift_dim=2, mode="fluctuation_delta", normalize=False)
    builder.fit(train, split="train")

    result = builder.transform(validation, previous_level_tail=train[-1:], counterfactual_mode="real")

    assert torch.allclose(result.fluctuation[0], torch.tensor([5.0, 6.0]))
    assert torch.allclose(result.fluctuation[1], torch.tensor([2.0, 9.0]))
```

- [ ] **Step 3: Write failing tests for optional `phi_v` temporal-delta diagnostics**

Append:

```python
def test_v13_variant_temporal_delta_requires_actual_variant_sequence() -> None:
    phi_v = torch.tensor([[2.0, 4.0], [5.0, 8.0], [6.0, 9.0]])
    previous_phi_v = torch.tensor([[1.0, 3.0]])

    delta = compute_v13_sequence_delta(phi_v, previous_tail=previous_phi_v)

    assert torch.allclose(delta[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(delta[1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(delta[2], torch.tensor([1.0, 1.0]))
```

- [ ] **Step 4: Write failing tests for train-only scaling and zero preservation**

Append:

```python
def test_v13_train_only_scaling_and_zero_preservation() -> None:
    train = torch.cat([torch.ones(16, 3), 3.0 * torch.ones(16, 3)], dim=0)
    builder = Stage3SignalBuilderV13(
        drift_dim=3,
        mode="fluctuation_delta",
        normalize=True,
        robust_scale_quantile=0.75,
    )

    with pytest.raises(ValueError, match="train split"):
        builder.fit(train, split="validation")

    builder.fit(train, split="train")
    zero = builder.transform(torch.zeros(4, 3), counterfactual_mode="zero")

    assert torch.allclose(zero.signal, torch.zeros_like(zero.signal), atol=1e-8)
    assert builder.diagnostic_summary()["signal_contract_version"] == builder.protocol_version
```

- [ ] **Step 5: Write failing tests for base counterfactual modes and stop diagnostics**

Append:

```python
def test_v13_counterfactual_modes_and_stop_diagnostics() -> None:
    level = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    builder = Stage3SignalBuilderV13(drift_dim=3, mode="fluctuation_delta", normalize=False)
    builder.fit(level, split="train")

    real = builder.transform(level, counterfactual_mode="real")
    flipped = builder.transform(level, counterfactual_mode="sign_flipped")
    zero = builder.transform(level, counterfactual_mode="zero")
    amplified = builder.transform(level, counterfactual_mode="amplified_2x")
    random_matched = builder.transform(level, counterfactual_mode="random_matched_norm")

    assert torch.allclose(flipped.signal, -real.signal)
    assert torch.allclose(zero.signal, torch.zeros_like(real.signal))
    assert torch.allclose(amplified.signal, 2.0 * real.signal)
    assert random_matched.signal.shape == real.signal.shape

    stop = builder.stop_diagnostic(real.signal, zero_fraction_threshold=0.99)
    assert stop["stop_condition_triggered"] is False
    assert stop["stop_condition_name"] == ""
```

- [ ] **Step 6: Run tests and verify they fail**

Run:

```powershell
pytest tests\test_stage3_v13_signal_contract.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'train.stage3_v13_signal_contract'`.

---

## Task 2: V13 Signal Contract Implementation

**Files:**
- Create: `train/stage3_v13_signal_contract.py`
- Test: `tests/test_stage3_v13_signal_contract.py`

- [ ] **Step 1: Implement V13 dataclasses and builder skeleton**

Create `train/stage3_v13_signal_contract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


V13_SIGNAL_MODES = {"level_offset", "fluctuation_delta"}
V13_COUNTERFACTUAL_MODES = {
    "real",
    "zero",
    "shuffled",
    "sign_flipped",
    "amplified_2x",
    "amplified_5x",
    "random_matched_norm",
}


@dataclass
class Stage3SignalResultV13:
    signal: torch.Tensor
    level_offset: torch.Tensor
    fluctuation: torch.Tensor
    raw: torch.Tensor
    mode: str
    counterfactual_mode: str
    drift_dim: int
    protocol_version: str
    first_step_zero_fraction: float
    zero_mask_fraction: float
    level_norm_mean: float
    fluctuation_norm_mean: float
    signal_norm_mean: float
    scaled_abs_mean: float


def compute_v13_sequence_delta(sequence: torch.Tensor, previous_tail: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a temporal sequence delta for diagnostics when the real sequence is available."""
    if sequence.ndim == 1:
        matrix = sequence.unsqueeze(0)
    elif sequence.ndim > 2:
        matrix = sequence.flatten(start_dim=1)
    else:
        matrix = sequence
    matrix = matrix.detach().float()
    if matrix.shape[0] == 0:
        raise ValueError("sequence must contain at least one sample")
    if previous_tail is None:
        previous = torch.cat([matrix[:1], matrix[:-1]], dim=0)
    else:
        previous_first = previous_tail.detach().float().to(device=matrix.device, dtype=matrix.dtype).reshape_as(matrix[:1])
        previous = torch.cat([previous_first, matrix[:-1]], dim=0)
    return matrix - previous
```

- [ ] **Step 2: Implement matrix conversion and sequence transitions**

Add below the dataclass:

```python
class Stage3SignalBuilderV13:
    protocol_version = "v13_fluctuation_first_signal_contract"

    def __init__(
        self,
        *,
        drift_dim: int,
        mode: Literal["level_offset", "fluctuation_delta"] = "fluctuation_delta",
        normalize: bool = True,
        robust_scale_quantile: float = 0.75,
        eps: float = 1e-6,
        polarity: float = 1.0,
        random_seed: int = 1729,
    ) -> None:
        if drift_dim <= 0:
            raise ValueError("drift_dim must be positive")
        if mode not in V13_SIGNAL_MODES:
            raise ValueError(f"unsupported V13 signal mode: {mode}")
        if not 0.0 < robust_scale_quantile <= 1.0:
            raise ValueError("robust_scale_quantile must be in (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.drift_dim = int(drift_dim)
        self.mode = mode
        self.normalize = bool(normalize)
        self.robust_scale_quantile = float(robust_scale_quantile)
        self.eps = float(eps)
        self.polarity = float(polarity)
        self.random_seed = int(random_seed)
        self.train_scale: torch.Tensor | None = None
        self.fit_sample_count = 0
        self.fit_split = ""

    def _as_matrix(self, level_offset: torch.Tensor) -> torch.Tensor:
        if level_offset.ndim == 1:
            level_offset = level_offset.unsqueeze(0)
        elif level_offset.ndim > 2:
            level_offset = level_offset.flatten(start_dim=1)
        if level_offset.ndim != 2 or level_offset.shape[1] != self.drift_dim:
            raise ValueError(f"level_offset must have shape [N, {self.drift_dim}]")
        return level_offset.detach().float() * self.polarity

    def _transition(self, raw: torch.Tensor, previous_level_tail: torch.Tensor | None = None) -> torch.Tensor:
        return compute_v13_sequence_delta(raw, previous_tail=previous_level_tail)

    def _mode_signal(self, raw: torch.Tensor, previous_level_tail: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        fluctuation = self._transition(raw, previous_level_tail=previous_level_tail)
        if self.mode == "level_offset":
            return raw, fluctuation
        if self.mode == "fluctuation_delta":
            return fluctuation, fluctuation
        raise ValueError(f"unsupported V13 signal mode: {self.mode}")
```

- [ ] **Step 3: Implement fit, transform, counterfactuals, diagnostics**

Add:

```python
    def fit(self, level_offset_train: torch.Tensor, *, split: str = "train") -> "Stage3SignalBuilderV13":
        if split != "train":
            raise ValueError("Stage3SignalBuilderV13 can only be fit on the train split")
        raw = self._as_matrix(level_offset_train)
        signal, _ = self._mode_signal(raw)
        scale = torch.quantile(signal.abs(), self.robust_scale_quantile, dim=0)
        fallback = signal.abs().mean(dim=0)
        scale = torch.where(scale > self.eps, scale, fallback)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        self.train_scale = scale.detach().float()
        self.fit_sample_count = int(raw.shape[0])
        self.fit_split = "train"
        return self

    def transform(
        self,
        level_offset: torch.Tensor,
        *,
        previous_level_tail: torch.Tensor | None = None,
        counterfactual_mode: str = "real",
    ) -> Stage3SignalResultV13:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before transform")
        if counterfactual_mode not in V13_COUNTERFACTUAL_MODES:
            raise ValueError(f"unsupported V13 counterfactual mode: {counterfactual_mode}")
        raw = self._as_matrix(level_offset)
        signal, fluctuation = self._mode_signal(raw, previous_level_tail=previous_level_tail)
        signal = self._apply_counterfactual(signal, mode=counterfactual_mode)
        zero_mask = torch.linalg.vector_norm(signal, dim=1, keepdim=True) <= self.eps
        if self.normalize:
            scale = self.train_scale.to(device=signal.device, dtype=signal.dtype).clamp_min(self.eps)
            scaled = signal / scale.unsqueeze(0)
        else:
            scaled = signal
        scaled = torch.where(zero_mask, torch.zeros_like(scaled), scaled)
        first_zero = float((torch.linalg.vector_norm(fluctuation[:1], dim=1) <= self.eps).float().mean().item())
        return Stage3SignalResultV13(
            signal=scaled.detach().float(),
            level_offset=raw.detach().float(),
            fluctuation=fluctuation.detach().float(),
            raw=raw.detach().float(),
            mode=self.mode,
            counterfactual_mode=counterfactual_mode,
            drift_dim=self.drift_dim,
            protocol_version=self.protocol_version,
            first_step_zero_fraction=first_zero,
            zero_mask_fraction=float(zero_mask.float().mean().item()),
            level_norm_mean=float(torch.linalg.vector_norm(raw, dim=1).mean().item()) if raw.numel() else 0.0,
            fluctuation_norm_mean=float(torch.linalg.vector_norm(fluctuation, dim=1).mean().item()) if fluctuation.numel() else 0.0,
            signal_norm_mean=float(torch.linalg.vector_norm(scaled, dim=1).mean().item()) if scaled.numel() else 0.0,
            scaled_abs_mean=float(scaled.abs().mean().item()) if scaled.numel() else 0.0,
        )

    def _apply_counterfactual(self, signal: torch.Tensor, *, mode: str) -> torch.Tensor:
        if mode == "real":
            return signal
        if mode == "zero":
            return torch.zeros_like(signal)
        if mode == "sign_flipped":
            return -signal
        if mode == "amplified_2x":
            return 2.0 * signal
        if mode == "amplified_5x":
            return 5.0 * signal
        if mode == "shuffled":
            if signal.shape[0] <= 1:
                return signal.clone()
            return signal.flip(0)
        if mode == "random_matched_norm":
            gen = torch.Generator(device=signal.device)
            gen.manual_seed(self.random_seed)
            random = torch.randn(signal.shape, generator=gen, device=signal.device, dtype=signal.dtype)
            random_norm = torch.linalg.vector_norm(random, dim=1, keepdim=True).clamp_min(self.eps)
            signal_norm = torch.linalg.vector_norm(signal, dim=1, keepdim=True)
            return random / random_norm * signal_norm
        raise ValueError(f"unsupported V13 counterfactual mode: {mode}")
```

- [ ] **Step 4: Implement state and stop diagnostics**

Add:

```python
    def diagnostic_summary(self) -> dict[str, int | float | str]:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before diagnostics")
        return {
            "signal_contract_version": self.protocol_version,
            "signal_mode": self.mode,
            "signal_normalize": str(self.normalize),
            "signal_fit_split": self.fit_split,
            "signal_fit_sample_count": self.fit_sample_count,
            "signal_scale_abs_mean": float(self.train_scale.abs().mean().item()),
            "signal_scale_min": float(self.train_scale.min().item()),
            "signal_scale_max": float(self.train_scale.max().item()),
        }

    def stop_diagnostic(self, signal: torch.Tensor, *, zero_fraction_threshold: float = 0.99) -> dict[str, object]:
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        zero_fraction = float((torch.linalg.vector_norm(signal, dim=1) <= self.eps).float().mean().item())
        triggered = zero_fraction >= float(zero_fraction_threshold)
        return {
            "stop_condition_triggered": triggered,
            "stop_condition_name": "fluctuation_collapsed_to_zero" if triggered else "",
            "zero_fraction": zero_fraction,
            "zero_fraction_threshold": float(zero_fraction_threshold),
        }

    def state_dict(self) -> dict[str, object]:
        if self.train_scale is None:
            raise RuntimeError("Stage3SignalBuilderV13 must be fit before serialization")
        return {
            "protocol_version": self.protocol_version,
            "drift_dim": self.drift_dim,
            "mode": self.mode,
            "normalize": self.normalize,
            "robust_scale_quantile": self.robust_scale_quantile,
            "eps": self.eps,
            "polarity": self.polarity,
            "random_seed": self.random_seed,
            "train_scale": self.train_scale.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "fit_split": self.fit_split,
        }

    def load_state_dict(self, state: dict[str, object]) -> "Stage3SignalBuilderV13":
        self.drift_dim = int(state["drift_dim"])
        self.mode = str(state.get("mode", self.mode))
        if self.mode not in V13_SIGNAL_MODES:
            raise ValueError(f"unsupported loaded V13 signal mode: {self.mode}")
        self.normalize = bool(state.get("normalize", self.normalize))
        self.robust_scale_quantile = float(state.get("robust_scale_quantile", self.robust_scale_quantile))
        self.eps = float(state.get("eps", self.eps))
        self.polarity = float(state.get("polarity", self.polarity))
        self.random_seed = int(state.get("random_seed", self.random_seed))
        self.train_scale = torch.as_tensor(state["train_scale"]).detach().float()
        self.fit_sample_count = int(state.get("fit_sample_count", 0))
        self.fit_split = str(state.get("fit_split", "train"))
        return self
```

- [ ] **Step 5: Run V13 signal tests**

Run:

```powershell
pytest tests\test_stage3_v13_signal_contract.py -q
```

Expected: all V13 signal contract tests pass.

---

## Task 3: V13 Teacher Wrapper Tests

**Files:**
- Create: `tests/test_stage3_v13_teacher_protocol.py`
- Create later in Task 4: `models/stage3_residual_teacher_v13.py`

- [ ] **Step 1: Write failing tests for feature modes**

Create `tests/test_stage3_v13_teacher_protocol.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.stage3_residual_teacher_v13 import (
    V13_CONTEXT_FEATURE_MODES,
    TrainOnlyFluctuationContextResidualTeacherV13,
    build_stage3_v13_context_features,
    load_stage3_v13_teacher_checkpoint,
    save_stage3_v13_teacher_checkpoint,
)


def _fit_teacher_for_unit_test(teacher: TrainOnlyFluctuationContextResidualTeacherV13) -> None:
    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.2],
            [1.0, 0.2, 0.4],
            [1.5, 0.3, 0.6],
            [2.0, 0.4, 0.8],
            [2.5, 0.5, 1.0],
            [3.0, 0.6, 1.2],
            [3.5, 0.7, 1.4],
        ],
        dtype=torch.float32,
    )
    y_base = torch.zeros(features.shape[0], 2)
    y = torch.stack([0.1 * features[:, 0], -0.2 * features[:, 1]], dim=1)
    teacher.fit(drift_train=features, y_base_train=y_base, y_train=y, split="train")


def test_v13_context_feature_modes_are_explicit() -> None:
    fluctuation = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    level = 10.0 + fluctuation
    history = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    fluct_only = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="fluctuation_only",
    )
    history_only = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="history_only",
    )
    both = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="fluctuation_plus_history",
    )
    level_diag = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="level_offset_only",
    )

    assert "fluctuation_plus_history" in V13_CONTEXT_FEATURE_MODES
    assert torch.allclose(fluct_only.features, fluctuation)
    assert torch.allclose(history_only.features, history)
    assert both.features.shape == (4, 5)
    assert torch.allclose(level_diag.features, level)
    assert both.fluctuation_dim == 3
    assert both.history_dim == 2
    assert both.level_offset_dim == 3
```

- [ ] **Step 2: Write failing tests for shape safety and metadata**

Append:

```python
def test_v13_context_features_are_shape_safe() -> None:
    fluctuation = torch.zeros(4, 3)
    history = torch.zeros(5, 2)
    level = torch.zeros(4, 3)

    with pytest.raises(ValueError, match="same sample count"):
        build_stage3_v13_context_features(
            fluctuation_signal=fluctuation,
            history_features=history,
            level_offset=level,
            mode="fluctuation_plus_history",
        )


def test_v13_teacher_diagnostic_summary_has_v13_metadata() -> None:
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=2,
        purge_gap=0,
    )
    _fit_teacher_for_unit_test(teacher)

    summary = teacher.diagnostic_summary()

    assert summary["teacher_method"] == teacher.protocol_version
    assert summary["teacher_context_feature_mode"] == "fluctuation_only"
    assert summary["teacher_fluctuation_signal_dim"] == 3
```

- [ ] **Step 3: Write failing tests for no-overwrite checkpoint behavior**

Append:

```python
def test_v13_teacher_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=2,
        purge_gap=0,
    )
    _fit_teacher_for_unit_test(teacher)
    path = tmp_path / "teacher_v13.pt"

    save_stage3_v13_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        save_stage3_v13_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    loaded, metadata = load_stage3_v13_teacher_checkpoint(path)
    assert loaded.protocol_version == teacher.protocol_version
    assert metadata["protocol_version"] == teacher.protocol_version
```

- [ ] **Step 4: Run tests and verify they fail**

Run:

```powershell
pytest tests\test_stage3_v13_teacher_protocol.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'models.stage3_residual_teacher_v13'`.

---

## Task 4: V13 Teacher Wrapper Implementation

**Files:**
- Create: `models/stage3_residual_teacher_v13.py`
- Test: `tests/test_stage3_v13_teacher_protocol.py`

- [ ] **Step 1: Implement V13 feature dataclass and builder**

Create `models/stage3_residual_teacher_v13.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.stage3_residual_teacher_v12d import TrainOnlyContextOOFResidualTeacherV12d


V13_CONTEXT_FEATURE_MODES = {
    "fluctuation_only",
    "history_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
}


@dataclass
class Stage3V13ContextFeatures:
    features: torch.Tensor
    mode: str
    fluctuation_dim: int
    history_dim: int
    level_offset_dim: int
    feature_dim: int


def build_stage3_v13_context_features(
    *,
    fluctuation_signal: torch.Tensor,
    history_features: torch.Tensor,
    level_offset: torch.Tensor,
    mode: str,
) -> Stage3V13ContextFeatures:
    if mode not in V13_CONTEXT_FEATURE_MODES:
        raise ValueError(f"unsupported V13 context feature mode: {mode}")
    fluctuation = fluctuation_signal.detach().float()
    history = history_features.detach().float()
    level = level_offset.detach().float()
    if fluctuation.ndim != 2:
        raise ValueError("fluctuation_signal must be a 2-D tensor")
    if history.ndim != 2:
        raise ValueError("history_features must be a 2-D tensor")
    if level.ndim != 2:
        raise ValueError("level_offset must be a 2-D tensor")
    if fluctuation.shape[0] != history.shape[0] or fluctuation.shape[0] != level.shape[0]:
        raise ValueError("fluctuation_signal, history_features, and level_offset must have the same sample count")
    if mode == "fluctuation_only":
        features = fluctuation
    elif mode == "history_only":
        features = history
    elif mode == "fluctuation_plus_history":
        features = torch.cat([fluctuation, history], dim=1)
    elif mode == "level_offset_only":
        features = level
    elif mode == "level_offset_plus_history":
        features = torch.cat([level, history], dim=1)
    else:
        raise ValueError(f"unsupported V13 context feature mode: {mode}")
    return Stage3V13ContextFeatures(
        features=features.detach().float(),
        mode=mode,
        fluctuation_dim=int(fluctuation.shape[1]),
        history_dim=int(history.shape[1]),
        level_offset_dim=int(level.shape[1]),
        feature_dim=int(features.shape[1]),
    )
```

- [ ] **Step 2: Implement V13 teacher wrapper**

Append:

```python
class TrainOnlyFluctuationContextResidualTeacherV13(TrainOnlyContextOOFResidualTeacherV12d):
    protocol_version = "v13_fluctuation_context_oof_residual_teacher"

    def __init__(
        self,
        *,
        context_feature_mode: str,
        fluctuation_signal_dim: int = 0,
        history_feature_dim: int = 0,
        level_offset_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        if context_feature_mode not in V13_CONTEXT_FEATURE_MODES:
            raise ValueError(f"unsupported V13 context feature mode: {context_feature_mode}")
        inherited_context_mode = _v12d_inherited_context_mode(context_feature_mode)
        super().__init__(
            context_feature_mode=inherited_context_mode,
            history_lags=kwargs.pop("history_lags", ()),
            feedback_delay=int(kwargs.pop("feedback_delay", 0)),
            history_feature_dim=int(history_feature_dim),
            drift_signal_dim=int(fluctuation_signal_dim),
            **kwargs,
        )
        self.context_feature_mode = context_feature_mode
        self.fluctuation_signal_dim = int(fluctuation_signal_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.level_offset_dim = int(level_offset_dim)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        summary = super().diagnostic_summary()
        summary["teacher_method"] = self.protocol_version
        summary["teacher_context_feature_mode"] = self.context_feature_mode
        summary["teacher_fluctuation_signal_dim"] = self.fluctuation_signal_dim
        summary["teacher_history_feature_dim"] = self.history_feature_dim
        summary["teacher_level_offset_dim"] = self.level_offset_dim
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["protocol_version"] = self.protocol_version
        state["context_feature_mode"] = self.context_feature_mode
        state["fluctuation_signal_dim"] = self.fluctuation_signal_dim
        state["history_feature_dim"] = self.history_feature_dim
        state["level_offset_dim"] = self.level_offset_dim
        return state

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyFluctuationContextResidualTeacherV13":
        super().load_state_dict(state)
        self.context_feature_mode = str(state.get("context_feature_mode", self.context_feature_mode))
        if self.context_feature_mode not in V13_CONTEXT_FEATURE_MODES:
            raise ValueError(f"loaded unsupported V13 context feature mode: {self.context_feature_mode}")
        self.fluctuation_signal_dim = int(state.get("fluctuation_signal_dim", self.fluctuation_signal_dim))
        self.history_feature_dim = int(state.get("history_feature_dim", self.history_feature_dim))
        self.level_offset_dim = int(state.get("level_offset_dim", self.level_offset_dim))
        return self
```

Add this helper above the class:

```python
def _v12d_inherited_context_mode(v13_mode: str) -> str:
    if v13_mode == "history_only":
        return "history_only"
    if v13_mode in {"fluctuation_plus_history", "level_offset_plus_history"}:
        return "drift_plus_history"
    if v13_mode in {"fluctuation_only", "level_offset_only"}:
        return "drift_only"
    raise ValueError(f"unsupported V13 context feature mode: {v13_mode}")
```

- [ ] **Step 3: Implement V13 checkpoint save/load**

Append:

```python
def save_stage3_v13_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyFluctuationContextResidualTeacherV13,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V13 residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyFluctuationContextResidualTeacherV13.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyFluctuationContextResidualTeacherV13.__name__)
    signal_builder = getattr(teacher, "stage3_signal_builder_v13", None)
    if signal_builder is not None and hasattr(signal_builder, "state_dict"):
        payload_metadata.setdefault("stage3_signal_builder_v13_state", signal_builder.state_dict())
    history_builder = getattr(teacher, "stage3_history_builder_v13", None)
    if history_builder is not None and hasattr(history_builder, "state_dict"):
        payload_metadata.setdefault("stage3_history_builder_v13_state", history_builder.state_dict())
    torch.save(
        {
            "class_name": TrainOnlyFluctuationContextResidualTeacherV13.__name__,
            "protocol_version": TrainOnlyFluctuationContextResidualTeacherV13.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v13_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyFluctuationContextResidualTeacherV13, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        context_feature_mode=str(state.get("context_feature_mode", "fluctuation_only")),
        fluctuation_signal_dim=int(state.get("fluctuation_signal_dim", 0)),
        history_feature_dim=int(state.get("history_feature_dim", 0)),
        level_offset_dim=int(state.get("level_offset_dim", 0)),
        ridge_grid=torch.as_tensor(state.get("ridge_grid", [1e-3])).flatten().tolist(),
        alpha_grid=torch.as_tensor(state.get("alpha_grid", [0.0])).flatten().tolist(),
        eps=float(state.get("eps", 1e-8)),
        fold_count=int(state.get("fold_count", 5)),
        min_prefix_samples=int(state.get("min_prefix_samples", 32)),
        purge_gap=int(state.get("purge_gap", 0)),
        min_block_mse_gain=float(state.get("min_block_mse_gain", 0.0)),
        min_block_mae_gain=float(state.get("min_block_mae_gain", 0.0)),
        min_block_sign_agreement=float(state.get("min_block_sign_agreement", 0.5)),
        min_block_pass_fraction=float(state.get("min_block_pass_fraction", 0.6)),
        min_output_gain=float(state.get("min_output_gain", 0.0)),
        calibration_mae_weight=float(state.get("calibration_mae_weight", 0.0)),
        allow_negative_alpha=bool(state.get("allow_negative_alpha", True)),
        reliability_gate_enabled=bool(state.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(state.get("reliability_gate_quantile", 0.95)),
        min_reliable_fraction=float(state.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(state.get("disable_if_unreliable", False)),
        selected_policy=str(state.get("selected_policy", "relaxed")),
        claim_policy=str(state.get("claim_policy", "strict")),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V13 residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V13 residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyFluctuationContextResidualTeacherV13.protocol_version)
    metadata.setdefault("class_name", TrainOnlyFluctuationContextResidualTeacherV13.__name__)
    return teacher, metadata
```

- [ ] **Step 4: Run teacher protocol tests**

Run:

```powershell
pytest tests\test_stage3_v13_teacher_protocol.py -q
```

Expected: all V13 teacher protocol tests pass.

---

## Task 5: V13 Probe Runner And Config

**Files:**
- Create: `configs/stage3_v13_fluctuation_teacher_probe.yaml`
- Create: `train/run_stage3_v13_fluctuation_teacher_probe.py`
- Create: `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`

- [ ] **Step 1: Create V13 config by inheriting V12d/V12e settings**

Create `configs/stage3_v13_fluctuation_teacher_probe.yaml` by copying the structure of `configs/stage3_v12d_context_teacher_probe.yaml` with these explicit changes:

```yaml
invariant:
  representation_version: aligned_residual_v13_fluctuation_first
stage3_v2:
  protocol_version: v13_fluctuation_teacher_probe
  log_root: logs/s3v13fluct
  checkpoint_root: ckpt/s3v13fluct
  results_root: results/s3v13fluct
  run_id: manual_stage3_v13_fluctuation_teacher_probe
  v13_feature_modes:
    - fluctuation_only
    - history_only
    - fluctuation_plus_history
    - level_offset_only
    - level_offset_plus_history
  v12d_history:
    protocol: v12e_per_horizon_label_availability
    lags: [1, 2, 3, 6, 12, 24]
    label_delay: 0
    include_availability_mask: true
    standardize: true
    robust_scale_quantile: 0.75
    eps: 0.000001
  residual_teacher:
    enabled: true
    version: v13
    signal_mode: fluctuation_delta
    normalize_vector: true
    robust_scale_quantile: 0.75
    signal_eps: 0.000001
    abort_if_teacher_diagnostic_fails: true
    diagnostic_min_real_mse_gain: 0.005
    diagnostic_min_real_mae_gain: 0.002
    diagnostic_counterfactual_margin: 0.002
    diagnostic_min_sign_agreement: 0.55
    diagnostic_min_partial_mse_gain_vs_history: 0.001
    diagnostic_min_partial_mae_gain_vs_history: 0.0005
online:
  stage3:
    version: v13_fluctuation_teacher_probe
    drift_source: delta_phi_v
    signal_mode: fluctuation_delta
    normalize_vector: true
```

Keep the real data, model, train, invariant, canonicalizer, and teacher grid sections from V12d unless the copied keys conflict with the V13 signal names.

- [ ] **Step 2: Create V13 probe runner by adapting V12d probe**

Create `train/run_stage3_v13_fluctuation_teacher_probe.py` as a V13-specific copy/adaptation of `train/run_stage3_v12d_context_teacher_probe.py`.

Required V13 substitutions:

```python
from models.stage3_residual_teacher_v13 import (
    TrainOnlyFluctuationContextResidualTeacherV13,
    build_stage3_v13_context_features,
    save_stage3_v13_teacher_checkpoint,
)
from train.stage3_v12e_conditional_permutation import (
    build_history_norm_strata,
    conditional_permute_drift,
    conditional_random_matched_norm,
)
from train.stage3_v13_signal_contract import Stage3SignalBuilderV13
```

The runner must:

- force `cfg["stage3_v2"]["protocol_version"] = "v13_fluctuation_teacher_probe"`;
- force `cfg["online"]["stage3"]["version"] = "v13_fluctuation_teacher_probe"`;
- collect raw `level_offset = phi_v - phi_i`;
- collect raw `level_offset` directly from `representation.delta_phi_v`, falling back to exact `representation.phi_v - representation.phi_i` only when `delta_phi_v` is unavailable;
- reject any main-signal path that passes through V12 residual canonicalizers, latent-space transport, or V12 delta standardizers before V13 computes `fluctuation_delta`;
- fit `Stage3SignalBuilderV13` on train `level_offset`;
- transform train/validation/test into V13 fluctuation signals with correct previous split tails;
- build V13 context features with `build_stage3_v13_context_features`;
- use V12e per-horizon history builder by default;
- evaluate `conditional_shuffled` and `conditional_random_matched_norm` at the runner level by first building the real V13 fluctuation signal and then applying the V12e history-norm stratified functions to that signal;
- record `teacher_conditional_permutation_protocol` and `teacher_conditional_permutation_bin_count` for every conditional counterfactual row;
- force `diagnostic_include_offline_counterfactuals: true` for V13 and reject `--no-diagnostic-include-offline-counterfactuals`;
- summarize V13 gate rows using the complete mode set instead of V12's online-causal-only gate filter;
- save `teacher_v13_<feature_slug>.pt`;
- write `stage3_v13_signal_contract_diagnostics.csv`;
- write `stage3_v13_history_contract_diagnostics.csv`;
- write `teacher_only_abort.json` if stop conditions or gates fail;
- write `teacher_only_complete.json` only if V13 gate passes.

- [ ] **Step 3: Add directionality stop checks in the probe**

Add a helper in the V13 probe:

```python
def _v13_directional_stop_reasons(
    *,
    signal_stop: dict[str, object],
    rolling_aggregate_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
) -> list[str]:
    reasons: list[str] = []
    if bool(signal_stop.get("stop_condition_triggered", False)):
        reasons.append(str(signal_stop.get("stop_condition_name", "signal_stop")))
    if not rolling_aggregate_rows:
        reasons.append("rolling_validation_gate_missing")
    by_feature = {str(row.get("feature_mode", "")): row for row in rolling_aggregate_rows}
    main = by_feature.get("fluctuation_plus_history", {})
    if main and not bool(main.get("teacher_diagnostic_gate_pass", False)):
        reasons.append("fluctuation_plus_history_teacher_gate_failed")
    for row in partial_rows:
        try:
            if float(row.get("partial_mse_gain_vs_history", 0.0)) <= 0.0:
                reasons.append("partial_mse_gain_vs_history_nonpositive")
                break
        except (TypeError, ValueError):
            reasons.append("partial_mse_gain_vs_history_invalid")
            break
    for row in partial_rows:
        try:
            if float(row.get("partial_mae_gain_vs_history", 0.0)) <= 0.0:
                reasons.append("partial_mae_gain_vs_history_nonpositive")
                break
        except (TypeError, ValueError):
            reasons.append("partial_mae_gain_vs_history_invalid")
            break
    return sorted(set(reasons))
```

Use these reasons to write `teacher_only_abort.json` and stop the runner from reporting a complete status.

- [ ] **Step 4: Create initial V13 audit record**

Create `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`:

```markdown
# Stage III V13 Fluctuation-First Design And Experiment Record

## Scope

V13 is a new non-overwriting Stage III version. It selectively inherits audited V2-V12 mechanisms and tests whether `delta_phi_v(t)-delta_phi_v(t-1)` provides incremental residual-correction value beyond causal history controls.

## Inherited Mechanisms

- V2 deterministic evaluation and zero-drift identity requirement.
- V2.1 split-safe preprocessing and projected residual alignment.
- V12c train-only zero-preserving signal scaling.
- V12d drift-vs-history partial gain comparison.
- V12e per-horizon label availability.

## Stop Rule

If V13 shows a directional error, experiments stop and an abort/diagnostic record is written. Larger public-3 experiments are not launched from a failed pilot.

## Current Status

Implementation planned. No V13 experiment has been run yet.
```

- [ ] **Step 5: Run import-level protocol check**

Run:

```powershell
python train\run_stage3_v13_fluctuation_teacher_probe.py --help
```

Expected: command exits 0 and shows V13 probe CLI arguments. This is a protocol-connectivity check only, not an experiment.

---

## Task 6: V13 Matrix Runner

**Files:**
- Create: `train/run_stage3_v13_fluctuation_teacher_matrix.py`

- [ ] **Step 1: Create matrix runner based on V12d matrix**

Create `train/run_stage3_v13_fluctuation_teacher_matrix.py` by adapting `train/run_stage3_v12d_context_teacher_matrix.py`.

Required changes:

```python
DEFAULT_DATASETS = ("panama", "tetouan", "spain_energy_weather")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_FEATURE_MODES = (
    "fluctuation_only",
    "history_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
)
```

The command builder must call:

```python
"train\\run_stage3_v13_fluctuation_teacher_probe.py"
```

Default roots:

```python
parser.add_argument("--out-root", default="logs/s3v13fluct_matrix")
parser.add_argument("--checkpoint-root", default="ckpt/s3v13fluct_matrix")
parser.add_argument("--run-prefix", default="v13fluctm")
```

- [ ] **Step 2: Add matrix-level direction stop**

If any single probe returns a `teacher_only_abort.json` reason containing:

- `fluctuation_collapsed_to_zero`
- `partial_mse_gain_vs_history_nonpositive`
- `fluctuation_plus_history_teacher_gate_failed`

the matrix runner should record that status in `matrix_summary.csv`. If `--stop-on-direction-error` is set, it should stop launching remaining runs.

Add CLI:

```python
parser.add_argument("--stop-on-direction-error", action="store_true")
```

- [ ] **Step 3: Run matrix help check**

Run:

```powershell
python train\run_stage3_v13_fluctuation_teacher_matrix.py --help
```

Expected: command exits 0 and prints V13 matrix options.

---

## Task 7: V13 Test Suite And Real Pilot

**Files:**
- Modify only after real command output exists:
  - `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`
  - `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
  - `docs/audit/RISK_REGISTER.md`

- [ ] **Step 1: Run V13 unit tests**

Run:

```powershell
pytest tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run inherited safety tests likely touched by imports**

Run:

```powershell
pytest tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12d_context_features.py tests\test_stage3_v12b_signal_contract.py -q
```

Expected: existing inherited safety tests still pass.

- [ ] **Step 3: Run Panama real-data pilot**

Run:

```powershell
python train\run_stage3_v13_fluctuation_teacher_probe.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --dataset panama --run-id v13fluct_pilot_panama_s42_01 --seed 42 --epochs 3 --invariant-epochs 3 --device auto --max-train-batches 120 --max-generator-batches 120 --max-eval-batches 120 --validation-windows 5 --out-root logs\s3v13fluct_pilot --checkpoint-root ckpt\s3v13fluct_pilot
```

Expected:

- output directory under `logs\s3v13fluct_pilot\v13fluct_pilot_panama_s42_01\panama\seed_42`;
- checkpoint directory under `ckpt\s3v13fluct_pilot\panama\v13fluct_pilot_panama_s42_01\seed_42`;
- `resolved_config.json`;
- `manifest.json`;
- `stage3_v13_signal_contract_diagnostics.csv`;
- `teacher_only_rolling_validation_aggregate.csv`;
- either `teacher_only_abort.json` or `teacher_only_complete.json`.

- [ ] **Step 4: Stop if the pilot exposes directionality error**

If the pilot writes `teacher_only_abort.json` with directional stop reasons, do not run the public-3 matrix. Update `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md` with:

```markdown
## Panama Pilot Result

Status: stopped after V13 pilot.

Reason: [copy exact abort reasons from teacher_only_abort.json]

Interpretation: V13 fluctuation-first should not be scaled until the directional failure is diagnosed.
```

When filling this section, replace the reason line with the exact JSON value from `teacher_only_abort.json`, for example:

```markdown
Reason: `["partial_mse_gain_vs_history_nonpositive"]`
```

- [ ] **Step 5: Run public-3 matrix only if pilot is directionally safe**

Run only if the pilot does not trigger directional stop reasons:

```powershell
python train\run_stage3_v13_fluctuation_teacher_matrix.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --device auto --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v13fluct_matrix --checkpoint-root ckpt\s3v13fluct_matrix --stop-on-direction-error
```

Expected:

- `_manifests/<matrix_id>/matrix_manifest.json`;
- `_manifests/<matrix_id>/planned_runs.json`;
- `_manifests/<matrix_id>/matrix_summary.csv`;
- each run writes V13 probe artifacts or stops with a diagnostic abort.

---

## Task 8: Evidence Updates And Completion Audit

**Files:**
- Modify: `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`
- Modify: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify: `docs/audit/RISK_REGISTER.md`
- Modify only if claims change: `docs/audit/PAPER_CLAIMS_MATRIX.md`

- [ ] **Step 1: Update V13 audit record with actual commands and outputs**

Add sections:

```markdown
## Verification Commands

- `pytest tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py -q`
- `pytest tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12d_context_features.py tests\test_stage3_v12b_signal_contract.py -q`
- `python train\run_stage3_v13_fluctuation_teacher_probe.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --dataset panama --run-id v13fluct_pilot_panama_s42_01 --seed 42 --epochs 3 --invariant-epochs 3 --device auto --max-train-batches 120 --max-generator-batches 120 --max-eval-batches 120 --validation-windows 5 --out-root logs\s3v13fluct_pilot --checkpoint-root ckpt\s3v13fluct_pilot`

## Output Paths

- Pilot output: `logs\s3v13fluct_pilot\v13fluct_pilot_panama_s42_01\panama\seed_42`
- Pilot checkpoints: `ckpt\s3v13fluct_pilot\panama\v13fluct_pilot_panama_s42_01\seed_42`
- Pilot manifest: `logs\s3v13fluct_pilot\v13fluct_pilot_panama_s42_01\panama\seed_42\manifest.json`

## Gate Status

- Teacher-only gate: record `pass` only if `teacher_only_complete.json` exists; otherwise record `fail` with the abort JSON reason.
- Directional stop: record `triggered` only if `teacher_only_abort.json` lists a V13 directionality stop reason; otherwise record `not triggered`.
- Public-3 matrix launched: record `yes` only if the matrix command is actually run after the pilot; otherwise record `no`.
```

- [ ] **Step 2: Update evidence ledger with only verified facts**

Add a V13 entry that distinguishes implementation checks from real-data experiments. Do not include invented results.

- [ ] **Step 3: Update risk register**

Add V13 risks:

- fluctuation signal may be non-informative;
- history-only may dominate fluctuation-plus-history;
- counterfactuals may beat real fluctuation;
- generator training remains blocked until teacher-only gate passes.

- [ ] **Step 4: Completion audit**

Before claiming this stage complete, verify:

- V13 files exist;
- V13 tests pass;
- V13 config uses V13 roots;
- V13 runner refuses overwrites by default;
- V13 pilot either produced abort or complete status;
- no V13 result is described as SOTA or proof of concept drift;
- experiments stopped if a directional failure appeared.

Do not mark the active goal complete unless V13 implementation, experiment restart, and evidence updates have all been verified.
