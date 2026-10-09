# Complete author forecasting networks — implementation audit

Date: 2026-10-07. Scope: the eight full forecasting adapters and small synthetic
execution checks. No manuscript was edited, no published metric was copied into
an experiment result, and no full-dataset accuracy validation is claimed here.

## Provenance

The author repository identities were checked against their public repository
descriptions/READMEs. Source archives were downloaded at the exact commits below.
Only the complete forecast entry point and supporting Python modules are vendored,
together with the original README, LICENSE, and requirements when present.
`third_party/official_forecasters/SOURCE_MANIFEST.json` records original byte hashes;
`RUNTIME_MANIFEST.json` records the executable copies. Original files remain intact.

| Model | Repository at pinned commit | License | Complete entry point |
|---|---|---|---|
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124) | MIT | `model/iTransformer.py:Model` |
| Crossformer | [Thinklab-SJTU/Crossformer](https://github.com/Thinklab-SJTU/Crossformer/tree/c10c8eadb153d1dd9798250967747ca3ebb81383) | Apache-2.0 | `cross_models/cross_former.py:Crossformer` |
| Autoformer | [thuml/Autoformer](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff) | MIT | `models/Autoformer.py:Model` |
| Transformer | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e) | MIT | `models/Transformer.py:Model`, long-term forecast task |
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | Apache-2.0 | `PatchTST_supervised/models/PatchTST.py:Model` |
| Informer | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020/tree/29f2a739226a509202a092b464163da81fa74960) | Apache-2.0 | `models/model.py:Informer` |
| FEDformer | [DAMO-DI-ML/ICML2022-FEDformer](https://github.com/DAMO-DI-ML/ICML2022-FEDformer/tree/39f3bffd203518548c0bf3f16a8c1ae112648dfe) | MIT | `models/FEDformer.py:Model`, Fourier variant |
| Pyraformer | [ant-research/Pyraformer](https://github.com/ant-research/Pyraformer/tree/84af4dbd93b7b96975b5034f0dde412005260123) | Apache-2.0, AntGroup-RI copyright | `pyraformer/Pyraformer_LR.py:Model` |

Transformer is explicitly a complete THUML time-series reference implementation,
not a claim that the authors of the 2017 Transformer released this forecasting
code. The other seven entry points are from their paper-author repositories.
The reference Transformer retains its encoder, autoregressive-masked decoder,
value/position/time embeddings, and final projection.

## Interface and retained mechanisms

```python
from paper_impl.official_backbones import OfficialForecastConfig, OfficialForecaster

config = OfficialForecastConfig(
    model_name="iTransformer", lookback=96, horizon=24, n_variables=7,
    target_indices=(6,), hidden=64, heads=8, depth=2, dropout=0.1,
)
forecast = OfficialForecaster(config)
# x: [B, 96, 7]; result: [B, 24, 1]
y = forecast(x)
head_parameters = tuple(forecast.head_parameters())
provenance = forecast.source_info()
```

`model_name` is case-insensitive and normalized to lowercase. The adapter exposes
the complete author `nn.Module` as `.model`, supports ordinary `.to(device)`,
`deepcopy`, `state_dict`, and `torch.func.functional_call`. All prediction modules
remain the author modules. No hidden-state pooling or replacement forecast head is
introduced. All channels are forecast first; `target_indices` then selects output
columns in the requested order. Default `target_indices=None` selects all channels.

| Model | Mechanisms retained | Exact Head-GD trainable boundary, relative to `.model` |
|---|---|---|
| iTransformer | Inverted variate embedding, full encoder, author normalization/de-normalization, horizon projector | `projector` |
| Crossformer | DSW embedding, two-stage attention, hierarchical encoder and full multiscale decoder, original padding/baseline option | Every `decoder.decode_layers.*.linear_pred` |
| Autoformer | Moving-average decomposition, FFT autocorrelation, seasonal and trend decoder, author normalization and both output paths | `decoder.projection` and every `decoder.layers.*.projection` |
| Transformer | Full time-series encoder-decoder, positional/temporal embeddings and forecast projection | `decoder.projection` |
| PatchTST | Channel-independent patch transformer, RevIN and inverse transform, full flatten forecast head; optional author dual decomposition branches | `model.head`, or `model_res.head` and `model_trend.head` |
| Informer | ProbSparse attention, optional author distillation, complete decoder and projection | `projection` |
| FEDformer | Fourier block, frequency cross-attention, full decomposition encoder-decoder and seasonal/trend outputs | `decoder.projection` and every `decoder.layers.*.projection` |
| Pyraformer | Full four-scale pyramidal attention and CSCM, complete author FC or attention decoder and predictor | `predictor` |

`head_parameter_names()` returns the corresponding parameter names. In the isolated
Head-GD test, the model is in evaluation mode and only these parameters are updated.
This freezes BatchNorm running buffers as well as non-head weights. Experiments
that deliberately retain training-mode BatchNorm updates must disclose that choice;
freezing parameter gradients alone does not freeze those buffers.

## Decoder and time-feature contract

The adapter never accepts future targets. For Autoformer, Transformer, Informer,
and FEDformer it constructs decoder values as the last `label_len` observations
followed by `horizon` zeros. Autoformer/FEDformer perform their original internal
decomposition and trend initialization. `label_len` defaults to `lookback//2`
(minimum 1). `decoder_depth` is explicit and defaults to 1; `depth` controls encoder
layers. `d_ff` defaults to `4*hidden`; all resolved settings are exposed in the config.

`x_mark_enc` is `[B, lookback, F]`. For those four encoder-decoder models,
`x_mark_dec` may be `[B, horizon, F]` or the already-prefixed
`[B, label_len+horizon, F]`. The adapter prepends historical marks in the first case.
Only normalized `timeF` marks are supported: hourly width 4, minutely 5, secondly 6,
daily/business-day 3, weekly 2, monthly/yearly 1. Missing marks are explicit zeros,
not fabricated calendar timestamps. This choice must be kept consistent across
compared experiments and is not equivalent to claiming real calendar covariates.

iTransformer consumes optional historical marks as extra variate tokens; absent
marks remain `None`, preserving the no-covariate original path. It rejects decoder
marks. Official Crossformer and supervised PatchTST have no calendar input and
reject supplied marks instead of silently ignoring them.

Pyraformer uses the author's long-range `DataEmbedding` path with four time features
(`freq='h'`). Decoder marks are `[B,horizon,4]`. Its FC branch appends a zero predict
token and the first future time mark to the history, exactly as the author's
`long_range_main.py` training/evaluation code does. Its attention branch retains the
complete two-layer author decoder. The single-step/probabilistic Pyraformer variant,
sequence-ID CustomEmbedding, and optional TVM kernel are outside this adapter's scope.

## Explicit configuration limits

- Common: positive integer sizes; nonempty, unique, valid target columns; finite
  dropout in `[0,1)`; even `hidden` divisible by `heads`. Active sinusoidal positional
  embeddings have the author's 5000-step limit.
- Autoformer: odd `moving_avg`; encoder/decoder lengths must satisfy
  `1 <= int(factor*log(length)) <= length`. It does not silently clamp top-k delays.
- Crossformer: positive segment/window/router sizes; each segment-merge padding
  must fit available segments. Original input and horizon segment padding remains.
- PatchTST: explicit `patch_len <= lookback`, positive stride, author `None` or
  `'end'` padding, odd decomposition kernel. No invented short-window patch module.
  Training checks BatchNorm's `batch * channels * patches >= 2` requirement.
- Informer: all ProbSparse encoder lengths, including after optional distillation,
  must stay at least 2. ProbSparse randomness during evaluation is preserved; exact
  repeated-output comparisons require the same RNG state.
- FEDformer: this adapter supports the complete **Fourier** variant, not Wavelets.
  The author hard-codes 8 Fourier heads, so `heads=8` is required. `lookback>=4`,
  `label_len=lookback//2`, float32 inputs, and positive modes are required. The author
  clips mode count to available frequency bins internally; this behavior is retained.
  Random and low-frequency selection are explicit configuration choices. Complex
  Fourier parameters and the author's sharing of Fourier blocks across depth remain.
- Pyraformer: exactly three pyramid windows, each at least 2; an odd local attention
  window; enough encoder tokens to retain a top-level node. Default windows `(2,2,2)`
  are an explicit small-window configuration, not the author's CLI default `(4,4,4)`.
  For convolutional CSCM training, `batch * coarsest_length >= 2` is required by the
  original BatchNorm; inference may use a single top node. Four original CSCM choices
  are accepted. FC and attention decoder paths are both smoke-tested.

## Compatibility patches and checkpoint handling

Every runtime source-text change is reproducible through `prepare_runtime.py` and
stored as a per-repository unified diff in `third_party/official_forecasters/patches`.

1. Absolute author-local imports are qualified as
   `third_party.official_forecasters.runtime.<repository>...`. Relative imports stay
   relative. No global `sys.modules` aliases or `sys.path` edits are used.
2. Imports of the unrelated optional Reformer dependency are moved into the
   `ReformerLayer` constructor. The eight supported forecast paths do not instantiate
   it. No attention layer is replaced.
3. iTransformer's `x_enc /= stdev` becomes `x_enc = x_enc / stdev`. Forward values are
   mathematically unchanged; input-gradient backpropagation no longer invalidates
   the tensor saved by variance backward.
4. FEDformer's unused zero allocation follows the input device/dtype instead of a
   process-global CUDA device. Forecast computations are unchanged.
5. An unused Pyraformer graph-profiling module's sibling import is made relative to
   preserve package isolation if explicitly imported.

The adapter's `get_extra_state`/`set_extra_state` stores FEDformer's random Fourier
frequency-index lists, which the author's ordinary state dictionary omits. This is
checkpoint metadata, not a network substitution. Validation checks complete module
coverage, list lengths, uniqueness, and in-range indices. Adapter checkpoints
restore the same forecast function even when the construction RNG changes.

`load_pretrained(path_or_mapping)` accepts a full adapter state dict or a complete
author model state dict (also a mapping containing `state_dict`). It does not
download weights. A raw random-mode FEDformer state dict must be accompanied by its
frequency-index metadata; otherwise loading fails explicitly. DataParallel or other
external key prefixes should be removed by the caller before loading. The tests
verify wrapper checkpoint round trips for all eight models and a raw FEDformer load
with explicit frequency metadata; no public pretrained weights were downloaded.

## Verification actually performed

Command from the repository root:

```powershell
python -m pytest tests/paper_impl/test_official_backbones.py -q --disable-warnings
```

Environment: Python from `C:/Users/admin/anaconda3/python.exe`, PyTorch 2.7.1+cu128,
NumPy 2.4.6, einops 0.8.1, SciPy 1.16.3, SymPy 1.14.0, matplotlib 3.10.8;
CUDA device NVIDIA GeForce RTX 5060. Supporting packages were already installed.
`third_party/official_forecasters/requirements-runtime.txt` records the additional
author-code dependencies; it does not replace the main project requirements.

The smoke configuration uses `B=2,L=24,H=8,N=3,D=16,heads=8,depth=2,d_ff=32`, zero
dropout, targets `(2,0)`, moving-average kernel 3, patch length 4/stride 2,
Crossformer segment 4, FEDformer modes 4, and Pyraformer windows `(2,2,2)`.

Checks cover all eight real networks: CPU forward/backward including input gradients,
an Adam update, target-column order, complete state save/reload under changed RNG,
isolated Head-GD, CUDA forward/backward, coexistence with legacy model imports, and
original/runtime/license/patch hash verification. Additional checks cover decoder
mark construction, explicit invalid/short configurations, complete PatchTST dual
decomposition heads, Pyraformer's attention decoder and BatchNorm guard, and exact
FEDformer frequency-index restoration. `deepcopy` plus
`torch.func.functional_call` over every parameter is tested for all eight models,
with gradients reaching both a differentiable parameter-modulation scale and inputs.
The completed dedicated backbone test run reports **61 passed** in 5.37 seconds.

These checks establish executable complete-model integration only. They do not
establish forecast accuracy, publication reproduction, fair hyperparameter tuning,
multi-seed stability, full-data throughput, or LRP correctness. Subsequent expert
mixture/parameter-modulation integration and attribution rules require their own
tests and experiment evidence. The author implementations retain their original
non-differentiable selection operations where present, such as top-k delay choice;
ordinary autograd support is not proof of an LRP decomposition.
