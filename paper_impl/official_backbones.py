"""Full, commit-pinned author forecasters with an isolated import namespace.

This module is independent of the legacy ``*_style`` summary encoders. The
author model predicts every input channel with its original prediction head;
only the final channel selection is applied here. No future target is input.
See docs/audit/full_experiment_20261007/OFFICIAL_BACKBONES.md for provenance,
supported variants, compatibility patches, and the exact Head-GD boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn


OFFICIAL_MODEL_NAMES = (
    "itransformer", "crossformer", "autoformer", "transformer",
    "patchtst", "informer", "fedformer", "pyraformer",
)
_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "official_forecasters"
_MODULES = {
    "itransformer": ("itransformer.model.iTransformer", "Model"),
    "crossformer": ("crossformer.cross_models.cross_former", "Crossformer"),
    "autoformer": ("autoformer.models.Autoformer", "Model"),
    "transformer": ("transformer.models.Transformer", "Model"),
    "patchtst": ("patchtst.PatchTST_supervised.models.PatchTST", "Model"),
    "informer": ("informer.models.model", "Informer"),
    "fedformer": ("fedformer.models.FEDformer", "Model"),
    "pyraformer": ("pyraformer.pyraformer.Pyraformer_LR", "Model"),
}
_TIME_FEATURES = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}


def _positive(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dropout(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a finite number in [0, 1)")


@dataclass(frozen=True)
class OfficialForecastConfig:
    """Explicit forecast settings; model-specific fields use author mechanisms.

    ``depth`` controls encoder layers; ``decoder_depth`` controls decoder layers
    where present. ``heads=8`` also satisfies FEDformer's fixed Fourier layout.
    Calendar covariates are normalized timeF features, never target values.
    Missing covariates are zeros for models requiring them, or None for the
    iTransformer, which otherwise treats them as additional variate tokens.
    """

    model_name: str
    lookback: int
    horizon: int
    n_variables: int
    target_indices: tuple[int, ...] | None = None
    hidden: int = 64
    heads: int = 8
    depth: int = 2
    dropout: float = 0.1
    d_ff: int | None = None
    decoder_depth: int = 1
    label_len: int | None = None
    activation: str = "gelu"
    freq: str = "h"
    factor: int = 1
    moving_avg: int = 25
    itransformer_use_norm: bool = True
    crossformer_seg_len: int = 6
    crossformer_win_size: int = 2
    crossformer_factor: int = 10
    crossformer_baseline: bool = False
    patch_len: int = 16
    patch_stride: int = 8
    patch_padding: str | None = "end"
    patch_revin: bool = True
    patch_affine: bool = True
    patch_subtract_last: bool = False
    patch_individual: bool = False
    patch_decomposition: bool = False
    patch_kernel_size: int = 25
    patch_head_dropout: float = 0.0
    patch_fc_dropout: float = 0.0
    informer_distil: bool = True
    informer_mix: bool = True
    fedformer_version: str = "Fourier"
    fedformer_modes: int = 32
    fedformer_mode_select: str = "random"
    pyraformer_decoder: str = "FC"
    pyraformer_window_size: tuple[int, int, int] = (2, 2, 2)
    pyraformer_inner_size: int = 3
    pyraformer_d_bottleneck: int | None = None
    pyraformer_d_k: int | None = None
    pyraformer_d_v: int | None = None
    pyraformer_cscm: str = "Bottleneck_Construct"
    pyraformer_truncate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or self.model_name.lower() not in OFFICIAL_MODEL_NAMES:
            raise ValueError(f"model_name must be one of {OFFICIAL_MODEL_NAMES}")
        object.__setattr__(self, "model_name", self.model_name.lower())
        for field in ("lookback", "horizon", "n_variables", "hidden", "heads", "depth", "decoder_depth", "factor"):
            object.__setattr__(self, field, _positive(field, getattr(self, field)))
        if self.hidden % self.heads or self.hidden % 2:
            raise ValueError("hidden must be even and divisible by heads")
        _dropout("dropout", self.dropout)
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 4 * self.hidden)
        _positive("d_ff", self.d_ff)
        if self.label_len is None:
            object.__setattr__(self, "label_len", max(1, self.lookback // 2))
        _positive("label_len", self.label_len)
        if self.label_len > self.lookback:
            raise ValueError("label_len cannot exceed lookback")
        if self.activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        if self.freq not in _TIME_FEATURES:
            raise ValueError(f"freq must be one of {tuple(_TIME_FEATURES)}")
        indices = tuple(range(self.n_variables)) if self.target_indices is None else tuple(self.target_indices)
        if not indices or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.n_variables for i in indices):
            raise ValueError("target_indices must be nonempty integer column indices in [0, n_variables)")
        if len(indices) != len(set(indices)):
            raise ValueError("target_indices must not contain duplicates")
        object.__setattr__(self, "target_indices", tuple(int(i) for i in indices))
        for field in (
            "itransformer_use_norm", "crossformer_baseline", "patch_revin", "patch_affine",
            "patch_subtract_last", "patch_individual", "patch_decomposition", "informer_distil",
            "informer_mix", "pyraformer_truncate",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean")
        if self.model_name in ("transformer", "informer", "pyraformer"):
            if max(self.lookback + 1, self.label_len + self.horizon) > 5000:
                raise ValueError("author positional embeddings support at most 5000 time steps")
        if self.model_name in ("autoformer", "fedformer"):
            _positive("moving_avg", self.moving_avg)
            if self.moving_avg % 2 == 0:
                raise ValueError("moving_avg must be odd to preserve the author's decomposition length")
        if self.model_name == "autoformer":
            for length in (self.lookback, self.label_len + self.horizon):
                if not 1 <= int(self.factor * math.log(length)) <= length:
                    raise ValueError("Autoformer requires 1 <= int(factor * log(length)) <= length for encoder and decoder")
        if self.model_name == "crossformer":
            for field in ("crossformer_seg_len", "crossformer_win_size", "crossformer_factor"):
                _positive(field, getattr(self, field))
            segments = math.ceil(self.lookback / self.crossformer_seg_len)
            for _ in range(self.depth - 1):
                padding = (-segments) % self.crossformer_win_size
                if padding > segments:
                    raise ValueError("Crossformer segment merging needs padding <= available segments; reduce win_size/depth or seg_len")
                segments = math.ceil(segments / self.crossformer_win_size)
        if self.model_name == "patchtst":
            for field in ("patch_len", "patch_stride", "patch_kernel_size"):
                _positive(field, getattr(self, field))
            if self.patch_len > self.lookback:
                raise ValueError("PatchTST requires patch_len <= lookback; choose an explicit smaller patch_len")
            if self.patch_padding not in (None, "end"):
                raise ValueError("patch_padding must be None or 'end'")
            if self.patch_kernel_size % 2 == 0:
                raise ValueError("patch_kernel_size must be odd")
            _dropout("patch_head_dropout", self.patch_head_dropout)
            _dropout("patch_fc_dropout", self.patch_fc_dropout)
        if self.model_name == "informer":
            if self.lookback < 2:
                raise ValueError("Informer ProbSparse attention requires lookback >= 2")
            if self.informer_distil and math.ceil(self.lookback / (2 ** (self.depth - 1))) < 2:
                raise ValueError("Informer ProbSparse attention requires length >= 2 after encoder distillation; reduce depth or disable distil")
        if self.model_name == "fedformer":
            if self.fedformer_version != "Fourier":
                raise ValueError("this audited adapter supports the author's full FEDformer Fourier variant only")
            if self.heads != 8:
                raise ValueError("official FEDformer Fourier weights require heads=8")
            if self.label_len != self.lookback // 2 or self.lookback < 4:
                raise ValueError("official FEDformer requires lookback >= 4 and label_len=lookback//2")
            _positive("fedformer_modes", self.fedformer_modes)
            if self.fedformer_mode_select not in ("random", "low"):
                raise ValueError("fedformer_mode_select must be 'random' or 'low'")
        if self.model_name == "pyraformer":
            if self.freq != "h":
                raise ValueError("Pyraformer DataEmbedding expects four hourly timeF features (freq='h')")
            if self.pyraformer_decoder not in ("FC", "attention"):
                raise ValueError("pyraformer_decoder must be 'FC' or 'attention'")
            windows = tuple(self.pyraformer_window_size)
            if len(windows) != 3:
                raise ValueError("the author's Pyraformer predictor requires exactly three pyramid windows")
            for window in windows:
                _positive("pyraformer_window_size entry", window)
                if window < 2:
                    raise ValueError("pyramid windows must be >= 2")
            object.__setattr__(self, "pyraformer_window_size", windows)
            _positive("pyraformer_inner_size", self.pyraformer_inner_size)
            if self.pyraformer_inner_size % 2 == 0:
                raise ValueError("pyraformer_inner_size must be odd")
            if self.lookback + (self.pyraformer_decoder == "FC") < math.prod(windows):
                raise ValueError("Pyraformer input must leave at least one node after three pyramid reductions")
            if self.pyraformer_cscm not in ("Bottleneck_Construct", "Conv_Construct", "MaxPooling_Construct", "AvgPooling_Construct"):
                raise ValueError("unsupported author Pyraformer CSCM")
            for field, default in (("pyraformer_d_k", self.hidden // self.heads), ("pyraformer_d_v", self.hidden // self.heads), ("pyraformer_d_bottleneck", max(1, self.hidden // 4))):
                if getattr(self, field) is None:
                    object.__setattr__(self, field, default)
                _positive(field, getattr(self, field))


def official_source_info(model_name: str) -> dict[str, Any]:
    """Return repository, exact revision, original/runtime hashes and patch path."""
    name = model_name.lower()
    if name not in OFFICIAL_MODEL_NAMES:
        raise ValueError(f"unknown official model {model_name!r}")
    sources = json.loads((_VENDOR_ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    runtime = json.loads((_VENDOR_ROOT / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))
    source = sources[name]
    return {
        "model_name": name,
        "implementation_kind": "full_time_series_reference" if name == "transformer" else "full_author_forecaster",
        "repository": source["repository"], "commit": source["commit"],
        "commit_url": source["repository"] + "/tree/" + source["commit"],
        "license_path": str(_VENDOR_ROOT / "upstream" / name / "LICENSE"),
        "upstream_files": source["files"], "runtime_files": runtime[name]["files"],
        "patch_path": str(_VENDOR_ROOT / "patches" / (name + ".patch")),
        "patch_sha256": runtime[name]["patch_sha256"],
    }


class OfficialForecaster(nn.Module):
    """A complete official network, returning ``[batch, horizon, targets]``."""

    def __init__(self, config: OfficialForecastConfig):
        super().__init__()
        if not isinstance(config, OfficialForecastConfig):
            raise TypeError("config must be an OfficialForecastConfig")
        self.config = config
        self.model_name = config.model_name
        self.target_indices = config.target_indices
        self.n_targets = len(self.target_indices)
        module_path, class_name = _MODULES[self.model_name]
        model_type = getattr(import_module("third_party.official_forecasters.runtime." + module_path), class_name)
        c = config
        params = SimpleNamespace(
            task_name="long_term_forecast", seq_len=c.lookback, pred_len=c.horizon,
            label_len=c.label_len, enc_in=c.n_variables, dec_in=c.n_variables,
            c_out=c.n_variables, d_model=c.hidden, n_heads=c.heads, e_layers=c.depth,
            d_layers=c.decoder_depth, d_ff=c.d_ff, dropout=c.dropout, activation=c.activation,
            output_attention=False, embed="timeF", freq=c.freq, factor=c.factor,
            moving_avg=c.moving_avg, use_norm=c.itransformer_use_norm, class_strategy="projection",
        )
        if self.model_name == "crossformer":
            self.model = model_type(
                c.n_variables, c.lookback, c.horizon, c.crossformer_seg_len,
                win_size=c.crossformer_win_size, factor=c.crossformer_factor,
                d_model=c.hidden, d_ff=c.d_ff, n_heads=c.heads, e_layers=c.depth,
                dropout=c.dropout, baseline=c.crossformer_baseline, device=torch.device("cpu"),
            )
        elif self.model_name == "informer":
            self.model = model_type(
                c.n_variables, c.n_variables, c.n_variables, c.lookback, c.label_len, c.horizon,
                factor=c.factor, d_model=c.hidden, n_heads=c.heads, e_layers=c.depth,
                d_layers=c.decoder_depth, d_ff=c.d_ff, dropout=c.dropout, attn="prob", embed="timeF",
                freq=c.freq, activation=c.activation, output_attention=False,
                distil=c.informer_distil, mix=c.informer_mix, device=torch.device("cpu"),
            )
        elif self.model_name == "patchtst":
            params.__dict__.update(
                fc_dropout=c.patch_fc_dropout, head_dropout=c.patch_head_dropout,
                individual=c.patch_individual, patch_len=c.patch_len, stride=c.patch_stride,
                padding_patch=c.patch_padding, revin=c.patch_revin, affine=c.patch_affine,
                subtract_last=c.patch_subtract_last, decomposition=c.patch_decomposition,
                kernel_size=c.patch_kernel_size,
            )
            self.model = model_type(params, act=c.activation)
        elif self.model_name == "fedformer":
            params.__dict__.update(version=c.fedformer_version, modes=c.fedformer_modes, mode_select=c.fedformer_mode_select)
            self.model = model_type(params)
        elif self.model_name == "pyraformer":
            params.__dict__.update(
                model="Pyraformer", input_size=c.lookback, predict_step=c.horizon,
                decoder=c.pyraformer_decoder, window_size=list(c.pyraformer_window_size),
                inner_size=c.pyraformer_inner_size, d_inner_hid=c.d_ff, n_head=c.heads,
                n_layer=c.depth, d_k=c.pyraformer_d_k, d_v=c.pyraformer_d_v,
                d_bottleneck=c.pyraformer_d_bottleneck, CSCM=c.pyraformer_cscm,
                truncate=c.pyraformer_truncate, use_tvm=False, embed_type="DataEmbedding",
                device=torch.device("cpu"),
            )
            self.model = model_type(params)
        else:
            self.model = model_type(params)

    def source_info(self) -> dict[str, Any]:
        source = official_source_info(self.model_name)
        source["configuration"] = asdict(self.config)
        source["head_parameter_names"] = self.head_parameter_names()
        return source

    def _heads(self) -> tuple[nn.Module, ...]:
        m = self.model
        if self.model_name == "itransformer":
            return (m.projector,)
        if self.model_name == "crossformer":
            return tuple(layer.linear_pred for layer in m.decoder.decode_layers)
        if self.model_name in ("autoformer", "fedformer"):
            return (m.decoder.projection,) + tuple(layer.projection for layer in m.decoder.layers)
        if self.model_name == "transformer":
            return (m.decoder.projection,)
        if self.model_name == "patchtst":
            return (m.model_res.head, m.model_trend.head) if m.decomposition else (m.model.head,)
        if self.model_name == "informer":
            return (m.projection,)
        return (m.predictor,)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Original output mappings only; includes seasonal/trend or multiscale heads."""
        seen = set()
        for head in self._heads():
            for parameter in head.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def head_parameter_names(self) -> tuple[str, ...]:
        ids = {id(parameter) for parameter in self.head_parameters()}
        return tuple(name for name, parameter in self.named_parameters() if id(parameter) in ids)

    def _marks(self, value: Tensor | None, x: Tensor, lengths: tuple[int, ...], name: str) -> Tensor:
        width = _TIME_FEATURES[self.config.freq]
        if value is None:
            return x.new_zeros((x.shape[0], lengths[0], width))
        if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[0] != x.shape[0] or value.shape[1] not in lengths or value.shape[2] != width:
            raise ValueError(f"{name} must have shape [B, one of {lengths}, {width}]")
        if value.device != x.device or value.dtype != x.dtype:
            raise ValueError(f"{name} must share input dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value

    def forward(self, x: Tensor, x_mark_enc: Tensor | None = None, x_mark_dec: Tensor | None = None) -> Tensor:
        c = self.config
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (c.lookback, c.n_variables):
            raise ValueError(f"expected nonempty [B, {c.lookback}, {c.n_variables}] input")
        if not x.is_floating_point():
            raise TypeError("forecast input must be floating point")
        if self.model_name == "fedformer" and x.dtype != torch.float32:
            raise ValueError("the author's Fourier weights and buffers require float32 input")
        if self.model_name in ("crossformer", "patchtst"):
            if x_mark_enc is not None or x_mark_dec is not None:
                raise ValueError(f"official {self.model_name} has no calendar-covariate input")
            if self.model_name == "patchtst" and self.training:
                patches = (c.lookback - c.patch_len) // c.patch_stride + 1 + (c.patch_padding == "end")
                if x.shape[0] * c.n_variables * patches < 2:
                    raise ValueError("PatchTST training BatchNorm requires batch * channels * patches >= 2")
            pred = self.model(x)
        elif self.model_name == "itransformer":
            marks = None if x_mark_enc is None else self._marks(x_mark_enc, x, (c.lookback,), "x_mark_enc")
            if x_mark_dec is not None:
                raise ValueError("official iTransformer consumes historical time marks only")
            pred = self.model(x, marks, None, None)
        else:
            marks = self._marks(x_mark_enc, x, (c.lookback,), "x_mark_enc")
            if self.model_name == "pyraformer":
                future = self._marks(x_mark_dec, x, (c.horizon,), "x_mark_dec")
                x_enc = x
                if c.pyraformer_decoder == "FC":
                    x_enc = torch.cat((x, x.new_zeros(x.shape[0], 1, c.n_variables)), dim=1)
                    marks = torch.cat((marks, future[:, :1]), dim=1)
                if self.training and c.pyraformer_cscm in ("Bottleneck_Construct", "Conv_Construct"):
                    top_length = x_enc.shape[1] // math.prod(c.pyraformer_window_size)
                    if x.shape[0] * top_length < 2:
                        raise ValueError("Pyraformer training BatchNorm requires batch * coarsest pyramid length >= 2")
                pred = self.model(x_enc, marks, x.new_zeros(x.shape[0], c.horizon, c.n_variables), future, False)
            else:
                future = self._marks(x_mark_dec, x, (c.horizon, c.label_len + c.horizon), "x_mark_dec")
                if future.shape[1] == c.horizon:
                    future = torch.cat((marks[:, -c.label_len:], future), dim=1)
                decoder = torch.cat((x[:, -c.label_len:], x.new_zeros(x.shape[0], c.horizon, c.n_variables)), dim=1)
                pred = self.model(x, marks, decoder, future)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape != (x.shape[0], c.horizon, c.n_variables):
            raise RuntimeError(f"official {self.model_name} returned unexpected shape {tuple(pred.shape)}")
        return pred[:, :, self.target_indices]

    def get_extra_state(self) -> dict[str, Any]:
        # The author stores randomly selected Fourier indices as Python lists,
        # which ordinary state_dict omits. Preserve them without changing the
        # source network or resampling frequencies during a checkpoint reload.
        frequency_indices = {}
        if self.model_name == "fedformer":
            for name, module in self.model.named_modules():
                entries = {key: list(getattr(module, key)) for key in ("index", "index_q", "index_kv") if hasattr(module, key)}
                if entries:
                    frequency_indices[name] = entries
        return {"format_version": 1, "model_name": self.model_name, "frequency_indices": frequency_indices}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1 or state.get("model_name") != self.model_name:
            raise ValueError("checkpoint forecaster metadata is incompatible")
        supplied = state.get("frequency_indices", {})
        expected = self.get_extra_state()["frequency_indices"]
        if set(supplied) != set(expected):
            raise ValueError("complete Fourier frequency_indices metadata is required")
        for name, entries in supplied.items():
            if set(entries) != set(expected[name]):
                raise ValueError("complete Fourier frequency index metadata is required")
            module = self.model.get_submodule(name)
            for key, values in entries.items():
                if key not in ("index", "index_q", "index_kv") or not hasattr(module, key) or len(values) != len(getattr(module, key)):
                    raise ValueError("incompatible Fourier frequency index metadata")
                length = self.config.lookback if name.startswith("encoder.") or key == "index_kv" else self.config.label_len + self.config.horizon
                if len(set(values)) != len(values) or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < length // 2 for i in values):
                    raise ValueError("invalid Fourier frequency index values")
                setattr(module, key, list(values))

    def load_pretrained(self, checkpoint: str | Path | Mapping[str, Any], *, strict: bool = True, frequency_indices: Mapping[str, Mapping[str, list[int]]] | None = None):
        """Load this adapter's state_dict or the complete author's state_dict.

        No weights are downloaded. A random-mode FEDformer raw author checkpoint
        must supply its otherwise-unsaved frequency indices; adapter checkpoints
        include them automatically. This prevents a plausible but wrong reload.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True) if isinstance(checkpoint, (str, Path)) else checkpoint
        if "state_dict" in payload:
            payload = payload["state_dict"]
        if any(key.startswith("model.") for key in payload) and "_extra_state" in payload:
            return self.load_state_dict(payload, strict=strict)
        if self.model_name == "fedformer" and self.config.fedformer_mode_select == "random":
            if frequency_indices is None:
                raise ValueError("random-mode FEDformer author checkpoints require frequency_indices metadata")
            self.set_extra_state({"format_version": 1, "model_name": self.model_name, "frequency_indices": frequency_indices})
        return self.model.load_state_dict(payload, strict=strict)
