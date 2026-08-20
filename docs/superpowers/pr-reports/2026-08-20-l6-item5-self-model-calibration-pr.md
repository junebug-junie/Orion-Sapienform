# L6 item 5 (HOT): self-model confidence-vs-reliability calibration finding

## Summary

- Built and ran the L6 design's own item 5 (the higher-order/HOT piece deferred in
  `docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`'s "Implication for item
  5" section): does the self-model's `prediction_error_confidence` track `predicted_shift`'s actual
  reliability?
- New, committed, read-only measurement script
  (`scripts/analysis/measure_self_model_calibration.py`) reads the real, already-persisted
  `substrate_attention_self_model` history directly — no replay needed, unlike the sibling
  `measure_ast_hot_reducer.py`.
- **Finding: inverted, not calibrated.** Top-confidence TEST quartile 66.4% accurate vs.
  bottom-confidence quartile 73.2% accurate (n=3843, z=-3.22, p<0.001), against 7 days of live
  production data.
- Checked both live consumers of `prediction_error_confidence` before writing this up — neither
  treats it as a `predicted_shift` trust signal today, so this is a documented guardrail against
  future work, not a live bug fix.
- **No production/service code changed.** Deliberate measure-before-minting negative result,
  matching this exact subsystem's own established precedent (this reducer was itself only wired
  live after its own offline measurement passed).
- Updated `services/orion-substrate-runtime/README.md` in the same patch so this finding doesn't go
  stale next to the code it concerns.

## Outcome moved

Before this patch, item 5 was an open, unscoped question with no measurement instrument. Now there
is a real, TEST-validated answer (confidence is anti-correlated with reliability here, not merely
untested) and a permanent, reusable script to re-check it as more history accumulates or as the
underlying formulas change.

## Current architecture

`AttentionSelfModelV1` (`orion/schemas/attention_self_model.py`) carries both
`predicted_shift` (the lower-order claim: which Active-Inference domain's `prediction_error` is
about to move, and which direction — `orion/substrate/attention_self_model.py`'s argmax-by-`|trend|`
over `prediction_error_trend_by_domain`) and `prediction_error_confidence` (`1 -
mean(prediction_error)` across `ACTIVE_INFERENCE_DOMAINS`,
`_unconditional_prediction_error_confidence`) unconditionally, on nearly every tick. Both are
persisted to `substrate_attention_self_model` by `_attention_self_model_tick()`
(`services/orion-substrate-runtime/app/worker.py`). Nothing had checked whether the second predicts
the first's accuracy.

## Architecture touched

None (production code). This patch adds a measurement instrument and its finding write-up only,
plus two documentation paragraphs in an already-existing README section.

## Files changed

- `scripts/analysis/measure_self_model_calibration.py` (new): pure-function core (predicted_shift
  parsing, sample-building with honest skip-accounting, chronological TRAIN/TEST split, TRAIN-only
  quantile binning, Pearson correlation, two-proportion z-test) + a psycopg2 read-only I/O layer +
  markdown/CSV report rendering, mirroring `measure_ast_hot_reducer.py`'s existing structure and
  conventions (read-only session enforcement, progress log, `/tmp` artifact output).
- `scripts/analysis/tests/test_measure_self_model_calibration.py` (new): 15 unit tests over the pure
  layer, no DB — parsing, correct/incorrect/ambiguous/missing-domain sample building, chronological
  split, bin-edge/assign-bin roundtrip, bin accuracy counting, Pearson correlation (including
  undefined/zero-variance cases), two-proportion z (including the sign convention and the
  empty-group `None` case).
- `docs/superpowers/specs/2026-08-20-l6-item5-self-model-calibration-finding.md` (new): full
  finding write-up — method, numbers, plausible mechanism, consumer check, non-goals, acceptance
  checks.
- `services/orion-substrate-runtime/README.md`: two new paragraphs — one next to the
  `SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS` documentation (the finding itself), one
  inside the `honesty_metrics` brain-frame region documentation (the scope caveat for that live
  consumer specifically).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none — this is a read-only measurement over already-persisted data.
- Compatibility notes: n/a.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: n/a (no service touched).
- local `.env` synced: n/a.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/analysis/tests/test_measure_self_model_calibration.py -q
15 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/analysis/tests/ orion/substrate/tests/ -q
940 passed (full pre-existing suite, unaffected -- no production code touched)
```

## Evals run

The real evaluation for this patch is the live-data measurement itself, run directly against
production Postgres (not a unit-test-shaped eval, matching item 4's own TEST-validation precedent):

```text
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
  /mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/analysis/measure_self_model_calibration.py --window-hours 168

Source: substrate_attention_self_model (Postgres, conjourney db)
19,418 rows, 2026-08-13 -> 2026-08-20 (~7 days, ~30s cadence)
Chronological 70/30 TRAIN/TEST split (no shuffle -- avoids leakage)
Horizon: 2 rows ahead, matching item 3/4's own validated horizon on this table

TRAIN: n=8966, raw reversion accuracy=61.5%
TEST:  n=3843, raw reversion accuracy=66.0%

TEST confidence bins (edges from TRAIN quantiles only):
bin 0 [-inf, 0.9125):   n=1271, 73.2% accurate
bin 1 [0.9125, 0.9655): n=1079, 59.9% accurate
bin 2 [0.9655, 0.9847): n=749,  62.1% accurate
bin 3 [0.9847, +inf):   n=744,  66.4% accurate

Top bin vs. bottom bin: z=-3.22 (p<0.001, two-tailed)
Pearson r(confidence, correct) on TEST: -0.087

Re-run after the review-driven simplification (~40 min later, more live data accumulated):
top=66.7% (n=739), bottom=73.1% (n=1282), z=-3.04 -- same conclusion, reproduces.
```

## Docker/build/smoke checks

Not applicable — no service, container, or runtime config was touched. This is an offline analysis
script run directly against the live read-only Postgres endpoint (`localhost:55432`), no deploy
required.

## Review findings fixed

- Finding: the `two_proportion_z(bottom, top)` call in `run()` required a manual sign-flip
  (`z_top_vs_bottom = -z_top_vs_bottom`) to reach the intended "positive = top bin more accurate"
  convention — correct, but one step more roundabout than necessary.
  - Fix: reordered the call to `two_proportion_z(top, bottom)` directly, removing the negation
    step.
  - Evidence: `scripts/analysis/tests/test_measure_self_model_calibration.py` 15/15 still pass; live
    re-run reproduces the same sign and conclusion (z=-3.04, "top-confidence less accurate").
- No other material findings. Reviewer independently verified the regex against the real producer
  format string, the SQL/column mapping against the real schema and INSERT call, the "same
  underlying snapshot" claim by tracing the live trend-buffer wiring, and hand-reproduced the
  reported z-score and TEST sample counts from the finding doc's own numbers — all held up.

## Restart required

```text
No restart required.
```

No service, config, or schema changed — this patch is documentation and a new offline analysis
script only.

## Risks / concerns

- Severity: low. Concern: this measures aggregate calibration across whichever domains actually won
  `predicted_shift`'s argmax during the 7-day window (execution/biometrics/bus_synaptic dominate;
  chat had only 1 TEST sample) — not a per-domain-controlled result. Mitigation: none needed for
  this patch's own claim (explicitly disclosed in the finding doc and report caveats); a
  domain-controlled re-run would be a natural follow-up if this arc continues.
- Severity: low. Concern: `prediction_error_confidence` and `predicted_shift` are computed from the
  same underlying snapshot each tick, so this can only show they're coherently related (or, here,
  anti-related) — it cannot by itself prove or disprove a genuinely independent second-order sense.
  Mitigation: disclosed explicitly in the finding doc's caveats; flagged as the real next step for
  a future item-5 pass (a confidence source built from different information than the prediction it
  rates), out of scope for this patch.
- Severity: none. No production code, schema, env, or runtime behavior changed by this patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/l6-item5-logreg-fit
