# Operator-wise LRP through the complete author forecasters

Date: 2026-10-07. Implementation: `paper_impl/official_lrp.py`.
Verification: `tests/paper_impl/test_official_lrp.py`.

This implementation explains the actual full networks vendored in
`OFFICIAL_BACKBONES.md`: their embeddings, encoders, decoders, normalization,
decomposition, frequency operations and output heads remain present. It does not
replace the networks with pooled encoders and does not call a whole-network
input-times-gradient result “LRP”. Prediction code and source networks are unchanged
by the attribution implementation.

## Public interfaces

```python
attribution = attribute_official(
    complete_forecaster, x, epsilon=1e-6,
    freeze_head=True, sampling_seed=42,
)
```

`attribute_official` returns `OfficialAttribution`:

- `maps[B,T,N,L]`: input relevance, absolute value for each forecast step first,
  then mean over horizon.
- `channels`: each original `nn.Linear.weight` name maps to `[B,T,C_out]`.
  At that Linear's output node, absolute relevance is summed across all axes except
  output channel, then averaged over horizon. If the same module is used multiple
  times, the contributions are added. Unused or constant-input Linear branches
  receive an explicit zero channel vector.
- `residual[B,T]`: signed, horizon-averaged relevance assigned to biases,
  stabilization, constants, and specified Taylor remainders.
- `conservation_error[B,T]`: horizon mean of the **absolute** signed accounting
  error. Absolute value prevents errors from canceling between horizons.
- `residual_by_operation`: separate per-operation stabilization and
  bias/constant/Taylor-residual entries, each `[B,T]`.
- `predictions[B,H,T]`: the precise full-network predictions being explained.
- `operators`: the exact traced ATen operator names for the checked graph.

`attribute_official_mixture(model,x,freeze_head=True)` is integrated with the
parent `PaperForecaster` interface and returns its `Attribution` dataclass:
maps `[B,T,K,N,L]`, residual/error `[B,T,K]`, and complete channel keys such as
`experts.0.model.encoder.attn_layers.0.attention.query_projection.weight`.
The mixture filters channel keys to `model.expert_weights()` and raises an error
if any eligible weight lacks a channel score. The default controller excludes
every parameter returned by each expert's `head_parameters()`: complete forecast
heads stay in the relevance graph, while their channels are absent from the
default Stage II selection and Stage III generator. The standalone
`attribute_official` interface still exposes all original Linear channels.
Explicit dense update scopes can include forecast heads. Each expert is explained
independently; neither the gate nor the VAE/router is part of the relevance graph.

The model is evaluated deterministically during explanation, with its previous
training flags restored afterwards. BatchNorm running buffers are not updated.
Dropout is disabled. Informer's original stochastic sparse query sampling remains
in the graph, using a preserved fixed RNG stream. Mixture expert `k` uses
`model.config.official_sampling_seed+k`. The function preserves caller RNG states.
Ordinary predictions should use the same evaluation mode and seed for comparison.

`freeze_head=True` inserts detached views of exactly the original head parameters
exposed by `head_parameters()` into the explanation trace. Non-head parameters
remain connected to the explicit relevance formulas. A specialization loss can
therefore update the internal network without updating its forecast heads; the
ordinary prediction loss can still train them through a separate normal forward.
Under an outer `torch.no_grad()` context, the implementation temporarily enables
the local differentiation required by redistribution, then returns detached
diagnostic tensors. It does not silently return zero or gradient-free approximations.

## Trace and reverse propagation

The code uses PyTorch 2.7 `make_fx` and functionalization to trace each actual
forecast program to ATen operations. Forward hooks annotate the exact output of
every original `nn.Linear`; no guessed module-to-operator order is used. The graph
retains actual parameters, and is executed under ordinary autograd to preserve
parameter gradients of the explanation.

A separate reverse traversal distributes relevance at every operator. Local VJPs
are used only to implement explicitly selected local linear/Taylor rules. Operands
are independently cloned for a local operation while retaining their upstream
parameter graphs. This avoids accidentally traversing an entire network when
computing a local redistribution. No custom autograd backward masquerades as the
derivative of the prediction function.

The graph is recomputed for each sample so folded batch/variable dimensions do not
confuse channel scores or per-sample residuals. Discrete selections, including
top-k delays and sparse query indices, are recomputed from that sample. Their
indices are routing state, not differentiable relevance destinations.

The operator allowlist is closed. An unrecognized operation raises
`UnsupportedLRPOperator`, including when source/framework changes introduce a
different kernel. There is no generic full-network-gradient fallback. Training-mode
BatchNorm is outside this single-sample attribution protocol.

## Rules and mathematical conventions

For a real activation `z`, the epsilon stabilizer is
`s(z)=z+epsilon*sign_+(z)`, where `sign_+(0)=1`. Relevance is signed during reverse
propagation. Let `R` denote the outgoing relevance of the current local operator.

The softmax, bilinear multiplication and normalization choices follow the
[AttnLRP paper, Sections 3.2–3.3 and Appendix A](https://proceedings.mlr.press/v235/achtibat24a.html).
The local VJP implementation of an affine epsilon rule is also explained in the
[authors' LXT documentation](https://lxt.readthedocs.io/en/latest/under-the-hood.html).
The frequency and complex extensions below are this implementation's explicitly
defined real-coordinate extensions; they are not asserted to be rules experimentally
validated by that paper.

| Operator family | Explicit redistribution |
|---|---|
| Linear, convolution, add/subtract, sum/mean/cumsum, average pooling | `R_x = x * J_x(f)^T[R/s(f(x))]`, with parameters/constants fixed as relevance destinations; their values remain differentiable for the specialization loss. Bias and epsilon mass are recorded. |
| Bilinear matrix multiplication of two data operands | Apply the epsilon local rule and multiply each operand's result by `1/2`. Both query/key or attention/value paths receive relevance. If one operand is a parameter/constant matrix, the operation is linear and has no `1/2`. Repeated use of the same data tensor is still degree two. |
| Real elementwise products | Uniform relevance split across data operands, with broadcast dimensions reduced back to each operand. A constant/parameter multiplier uses the one-data-operand identity rule. |
| Softmax | For logits `x`, probabilities `p`, `R_x = x * (R - p*sum(R))` along the softmax axis. The difference is the Taylor bias, not forced back into inputs. Masked infinite logits use zero for their `x` multiplier, avoiding `0*inf`; unmasked values follow the same formula. |
| LayerNorm | First redistribute the affine gain/bias by epsilon; treat division by standard deviation by the normalization identity rule; redistribute the linear mean-centering as `x*(q-mean(q))`, where `q=R_center/s(centered_x)`. Affine/centering stabilizers and bias mass are recorded separately from numerical error. |
| Evaluation BatchNorm, CPU and CuDNN | A fixed-running-statistics affine epsilon rule. The gain is `gamma/sqrt(running_var+eps)`, with the intercept assigned to the constant/bias ledger. |
| Real GELU/ReLU/ELU/tanh/sqrt/rsqrt | Elementwise identity relevance rule. |
| Division | The divisor is held fixed for redistribution and relevance passes to the numerator. This is the chosen normalization rule; no claim is made that the variance-estimation branch is being explained through division. |
| Variance when reached through a non-division branch | Real quadratic, homogeneous-degree-two Taylor epsilon rule, hence local `x*VJP/2`. This can occur through de-normalization's scale path. |
| Shapes, transposes, repeats, slices, gathers, concatenation, patches, padding, copy/scatter | Exact routing adjoints. Repeated/overlapping coordinates add relevance. Nonzero padding and masks may retain constant relevance. |
| Max/top-k values and max pooling | Route through the selected values; no derivative or relevance is assigned to integer selection decisions. Ties follow the author/PyTorch selection. |
| Detach | Identity relevance routing. The author's ordinary forward gradient boundary remains present in the stored forward values; relevance is not automatically discarded at a detach. |

For affine/epsilon operations, the explicitly recorded stabilizer mass is
`sum(R - R*z/s(z))`. The remainder after outgoing-minus-distributed relevance and
this stabilizer is recorded as bias/constant/Taylor residual. For LayerNorm the
two epsilon sites are tracked. This ledger does not redistribute residual mass
to make an attractive heatmap and does not post-normalize explanations to force
conservation.

## FFT, complex arithmetic, and autocorrelation

Every complex activation is represented for relevance purposes by two **real**
coordinates per coefficient, `[Re(z), Im(z)]`. Relevance itself is real in this
representation. The following rules are explicit:

1. FFT/rFFT and inverse transforms are real-linear maps between these coordinates.
   Apply the affine epsilon rule through their exact local adjoints. PyTorch's
   real-FFT adjoint accounts for one-sided/Hermitian representation and normalization.
   A unit test compares rFFT relevance to a separately constructed real DFT matrix.
2. Complex linear products with fixed complex weights use their exact real block
   matrix, equivalently the local real-coordinate VJP. A separate test checks the
   four real/imaginary products analytically.
3. Complex products with two data operands use the homogeneous-degree-two epsilon
   rule, including `1/2` for each argument. Real/imaginary cross-terms remain present.
4. Complex conjugation preserves the relevance assigned to each real coordinate;
   imaginary sign reversal is a fixed nonzero scaling, not a new relevance source.
5. Complex `tanh` in FEDformer's Fourier cross-attention uses a local first-order
   real-coordinate Taylor epsilon rule. Its non-homogeneous remainder is retained
   in the bias/Taylor ledger. It is not labeled as the real activation identity rule.
6. Spectral slicing, assignment, reshaping and inverse FFT retain the author's exact
   forward function. Functionalization converts spectral writes to scatters. Because
   Torch 2.7 does not implement autograd for complex `select_scatter`, the attribution
   executor performs identical scatters separately on real and imaginary coordinates.
   The ordinary author model prediction code is not patched for this purpose.

Autoformer's correlation is therefore explained through its actual FFT,
conjugate-product, inverse FFT, selected-delay softmax and value routing.
The top-k choice is conditional routing rather than a claimed continuous explanation
of how the set of delays would change under perturbation. FEDformer's random frequency
indices come from the saved complete model, not new random frequency selection.

## Aggregation and conservation

For each sample, target `t`, and step `h`, relevance is initialized to the actual
forecast `prediction[h,t]` at that output and zero at all other outputs. Each step
is propagated independently. The input map is `mean_h(abs(R_input[h]))`; it is
**not** `abs(mean_h(R_input[h]))`. Linear channel scores use the same order.

Conservation is checked on the signed per-step relevance **before** absolute-value
aggregation:

`prediction[h,t] = sum(R_input[h]) + sum(residual_ledger[h]) + numerical_error[h]`.

The absolute aggregated map need not sum to the signed prediction. A small reported
conservation error establishes consistent bookkeeping, not explanation faithfulness:
the ledger is telescoping by construction and can include substantial legitimate
bias/Taylor mass. Residual magnitude and its operation breakdown should be inspected
alongside any explanation or specialization experiment.

## Actual verification and limits

The completed dedicated command was:

```powershell
python -m pytest tests/paper_impl/test_official_lrp.py -q --disable-warnings
```

The dedicated rule suite initially reported **29 passed** in 17.53 seconds. A
subsequent combined backbone/rule check reported **90 passed** in 20.97 seconds,
including the Fourier-weight gradient assertion and mean-absolute accounting-error
guard. The final suite also adds a parent-integration regression for Informer
identity modulation with `B=3`, nonzero gain gradients, fixed sampling, and unchanged
original model parameters/buffers. Its targeted command reported **1 passed,
29 deselected** in 1.54 seconds. No dataset score is inferred from these tests.

The default head-scope correction additionally checks all eight mixtures: eligible
channel keys exclude each complete forecast head, nonidentity Stage III gains
preserve persistent parameters, and a released-label Stage II update changes
eligible expert weights while keeping every forecast-head parameter unchanged.
The targeted regression command (`-k 'preserve_complete_forecast_heads or
full_mixture'`) reported **16 passed, 29 deselected** in 11.92 seconds. The synthetic
update test uses a finite learning rate and fixed low Fourier modes to avoid
mistaking an unresolvable float32 loss decrement for a failed scope check.
The combined adapter/LRP suite after this correction reported **106 passed in
32.16 seconds**, using `python -m pytest tests/paper_impl/test_official_backbones.py
tests/paper_impl/test_official_lrp.py -q -p no:cacheprovider`, including CPU and CUDA.

Verified checks include analytic softmax, bilinear self-matmul, affine-bias,
LayerNorm, convolution and complex-linear rules; an independent DFT-matrix comparison;
rejection of an unsupported sine operator; all eight full models on CPU at depth 2;
all eight on CUDA at depth 1; finite nonzero non-head parameter gradients; absent
head gradients under `freeze_head=True`; matching complete forward predictions under
the same seed with a batch of two; complete Linear channel keys; independent-expert
mixture integration; per-step absolute aggregation; restored training/RNG/buffer
state; and the additional complete Pyraformer attention decoder/PatchTST dual
decomposition paths. CPU conservation errors in the initial eight-model smoke check
were on the order of `1e-7`; tests use a `2e-4` bound for these small random cases.

This is a PyTorch 2.7.1 implementation using experimental FX/functionalization APIs.
Other PyTorch versions and new source commits must rerun the operator and prediction
checks. Unsupported operations fail explicitly. No TVM Pyraformer, FEDformer Wavelets,
half precision, compilation backend, multi-GPU or distributed execution is certified.
The independent numerical/gradient tests do not constitute a theoretical uniqueness
result for these frequency-domain explanations, a perturbation-faithfulness study,
or validation of the paper's specialization claims.

Cost is material: tracing and reverse redistribution are repeated per sample,
target, forecast step and expert. An initial CPU measurement at `L=16,H=2,D=8`, depth 1
took about 0.1–1 seconds per sample depending on the model, with first-trace overhead.
Keeping the explanation differentiable retains additional graphs. Full `L=96,H=24`
training or large batches requires a separate measured memory/runtime budget; the
small tests do not establish full-dataset feasibility or completion. No approximated
encoder, reduced horizon or detached specialization graph is silently substituted.
