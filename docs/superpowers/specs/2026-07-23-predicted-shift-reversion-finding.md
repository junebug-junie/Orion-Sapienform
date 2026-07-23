# `predicted_shift` reversion finding — metric-quality-gate exercise on L6 item 5

Status: **finding + shipped fix.** Not a design proposal — this documents a metric-quality-gate
pass (CLAUDE.md §0A) run against L6 item 4's already-merged `predicted_shift` field (PR #1301)
while scoping item 5, which surfaced that item 4's trend formula was empirically worse than
chance. Fixed in the same investigation, before item 5 was started.

## Context

Item 4 (`docs/superpowers/specs/2026-07-22-l6-self-model-ast-hot-active-inference-design.md`,
PR #1276/#1284/#1301) computes `predicted_shift`: whichever of the five Active-Inference domains
(execution/transport/biometrics/chat/route) has the largest-magnitude `prediction_error` trend
over a rolling window is named as "what surprises me next." The original formula (PR #1301,
`scripts/analysis/measure_ast_hot_reducer.py::compute_prediction_error_trend()`) was
`mean(recent half) - mean(prior half)` — naive continuation: if a domain's error just rose, predict
it keeps rising.

Item 5 (the higher-order/HOT piece) was proposed to build on whether item 4's own predictions come
true — a genuine second-order signal (per Rosenthal-style Higher-Order Theory: a representation of
whether a lower-order representation is accurate). Before building that, CLAUDE.md's
metric-quality-gate step 4 ("pull real data and look at it") requires checking item 4's predictions
against reality first — not assumed accurate because the code runs and produces plausible-looking
strings.

## Finding: naive continuation is worse than a coin flip

Back-tested `compute_prediction_error_trend()`'s original formula against real
`substrate_field_state` history for `node:substrate.biometrics` — the only domain with enough real
variance to test against (execution/chat/route are real but tiny; transport reads exactly 0.0 for
entire multi-hour windows, per the already-documented transport narrow-scope finding). For each
sampled tick, predicted direction (rising/falling) was checked against the domain's actual value
~60s later. Final validation run, frozen at a single reference timestamp (`2026-07-23T05:32:00Z`)
so both windows and both formulas (continuation and reversion) are scored from the identical sample
set in one pass — not re-queried across separate runs against a live, still-changing table, which
is why an earlier draft of this doc cited slightly different numbers from the same underlying
result:

| Window | n | Continuation accuracy | z (vs 50% null) | Reversion accuracy |
|---|---|---|---|---|
| A: last 3h | 332 | **37.7%** | -4.50 | **62.3%** |
| B: 44-48h before that | 454 | **41.0%** | -3.85 | **59.0%** |

Continuation and reversion accuracy sum to exactly 100% in both rows by construction — same
backtest pass, same sample set, strictly opposite predictions (not two independent runs that
happen to be close). Both continuation results are far below chance (both `p < 0.001`, two-tailed).

Consistent across every prediction horizon tested on an earlier exploratory pass against Window A
(2 to 20 field_state ticks / ~4-40s ahead, all 44-48% — below the ~60s headline number's own 37.7%,
but likewise all below chance). A random coin-flip baseline scored within noise of 50% on the same
data, confirming the below-chance continuation result isn't a methodology artifact.

**A decay-adjusted formula (compare the current value to what pure exponential continuation of the
prior half's own trajectory would predict, rather than a flat two-window-mean comparison) was also
tested in that same earlier exploratory pass and only marginally improved on continuation — 43-45%,
still well below chance.** The problem isn't the extrapolation *method*, it's the extrapolation
*direction*: `prediction_error` is spike-and-settle (a burst of real activity is more often
followed by quiet than by more activity), not momentum-carrying.

## Fix shipped

`compute_prediction_error_trend()` now returns `mean(prior half) - mean(recent half)` — the
opposite sign of the original formula. `reduce_attention_self_model()`'s own consumption contract
(positive = predicted rising, negative = predicted falling, argmax by magnitude) is unchanged; only
the upstream computation that feeds it changed. No behavioral changes to
`orion/substrate/attention_self_model.py` itself (only an illustrative docstring reference to the
old formula was corrected there).

**Validated on biometrics only, applied uniformly to all five domains.** No independent back-test
exists yet for execution/transport/chat/route — too little real variance in the available windows.
Applying the same reversion sign to them is a reasoned extrapolation (all five domains are computed
the same way, as deltas between successive states from discrete events), not an independently
confirmed one. In live replay this mostly matters for biometrics anyway — it wins the cross-domain
argmax the overwhelming majority of the time — but a future pass should back-test the other four
domains separately once enough real variance accumulates, rather than assuming this generalizes
forever.

## Implication for item 5

Item 5 has not been built yet. This finding changes what it should measure: building it directly on
top of the *original* (continuation) formula would have made item 5's "how often am I wrong"
signal trivially answer "usually" — a real finding, but a shallow one, mostly just restating this
bug rather than discovering something new. With item 4 now reversion-corrected and itself
above-chance (though still wrong ~38-41% of the time), item 5's genuine second-order question (does
the self-model's own confidence in a prediction track its actual reliability) has a non-trivial,
non-degenerate baseline to be interesting against.

## Non-goals

- Not implementing item 5 in this patch — that remains separate, per the L6 design's own phasing.
- Not re-deriving why biometrics specifically shows this spike-and-settle pattern (plausible
  candidates: individual chat/execution events triggering discrete deltas rather than a smoothly
  evolving process; not verified further here).
- Not extending this validation to execution/chat/route (too little real variance in the available
  windows to test against independently — flagged above, not resolved).

## Acceptance checks

- `pytest orion/substrate/tests/test_attention_self_model.py scripts/analysis/tests/test_measure_ast_hot_reducer.py -q`
  passes with tests updated to assert the new (validated) direction.
- Live replay (`measure_ast_hot_reducer.py --window-hours 6`) still shows non-degenerate,
  cross-domain `predicted_shift` coverage post-fix (not collapsed to a single always-predicted
  direction).
