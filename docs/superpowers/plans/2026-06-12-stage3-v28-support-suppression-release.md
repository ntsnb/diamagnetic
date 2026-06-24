# Stage III V28 Support-Suppression Release Plan

## Goal

Build a new official version line derived from the user's expert-local
invariant/correction idea and the two Chinese audit reports.

## Planned Work

1. Re-audit the Chinese reports and the strongest `V26/V27` evidence.
2. Build a cleaner next-generation line on top of the verified `V26` backbone.
3. Separate environment-conditioned suppression from delayed-change support.
4. Keep a full version namespace:
   - model
   - offline train
   - probe runner
   - generator/artifact/online wrappers
   - config
   - tests
   - audit record
5. Run deterministic checks and at least one formal experiment slice.

## Current State

Completed so far:

- `V28` model implementation
- `V28` offline training path
- `V28` probe runner
- `V28` generator/artifact/online wrappers
- `V28` config
- `V28` tests
- deterministic validation
- smoke run
- one formal `weather seed42 b60` screen
- one formal `weather seed43 b60` screen
- one formal `panama seed42 b60` transfer screen
- one narrow `support_push` tuning config
- one tuned `panama seed42 b60` support-push screen
- one tuned `weather seed43 b60` support-push screen

Not yet completed:

- broader experiment matrix
- any generator-unlocked run
## Current Decision

`V28` remains an active official version line, but not a validated mainline.

Current evidence:

- `weather seed42 b60`:
  - strong `residual_sign`
  - negative `raw_residual_vector`
  - zero `residual_change_vector`
- `weather seed43 b60`:
  - positive `raw_residual_vector`
  - positive `residual_sign`
  - zero `residual_change_vector`
- `panama seed42 b60`:
  - weak positive `residual_change_vector` MSE
  - negative raw-side signal
  - gate still fails
- `support_push panama seed42 b60`:
  - `residual_change_vector` MSE improves from `0.004641127665773605`
    to `0.0416912078672488`
  - `residual_change_vector` MAE improves from `-0.002969357117626279`
    to `0.02530271018320987`
  - raw MSE turns positive at `0.016891664820294097`
  - gate still fails because only `3/5` change windows are positive and
    counterfactual margin remains negative
- `support_push weather seed43 b60`:
  - raw MSE regresses from `0.08864664802310351` to
    `-0.07366123276356079`
  - `residual_sign` remains positive at `0.2630879282951355`
  - `residual_change_vector` remains zero
  - gate still fails

All completed runs remain at:

- `do_not_train_generator`
- `no_primary_target_passed`

Next action should be rigorous `V28` tuning, not another new structure, unless
the goal changes again.

## Tuning Discipline

The current project direction is:

- do not open `V29` yet
- do not add another mechanism before exhausting controlled `V28` tuning
- tune with matched budgets so results remain comparable
- evaluate at least one weather slice and one public-load slice before calling
  a setting better
- require both aggregate gain and window-level stability before unlocking
  generator training

The `support_push` result is not a final improvement. It is evidence that the
current architecture has a tunable change-target direction:

- positive evidence: panama change MSE and MAE both improve materially
- negative evidence: weather raw stability regresses and weather change remains
  zero
- decision: keep tuning inside `V28`; do not claim validated model improvement

The next tuning candidate should be constrained by two requirements:

- preserve the weather raw-side signal seen in base `weather seed43 b60`
- retain most of the panama `residual_change_vector` gain from `support_push`

If the next constrained setting cannot satisfy both requirements, the correct
paper-facing conclusion is that `V28` is mechanistically cleaner but not yet an
empirical replacement.
