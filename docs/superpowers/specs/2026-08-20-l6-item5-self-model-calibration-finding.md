# L6 item 5 (HOT): confidence-vs-reliability calibration finding

Status: **finding, no fix shipped — negative result, documented as a guardrail.** Metric-quality-gate
exercise (CLAUDE.md §0A) on the L6 design's own item 5, the higher-order piece deferred in
`docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`'s "Implication for item 5"
section until item 4's own trend formula was validated above chance (PR #1304, shipped) and
TEST-confirmed (PR #1747, `docs/superpowers/pr-reports/2026-08-19-l6-item34-hit-it-all-pr.md`).

## Context

Item 5's question, per Rosenthal-style Higher-Order Theory: does the self-model's own confidence in
a prediction track whether that prediction is actually reliable? `AttentionSelfModelV1` already
carries both halves unconditionally, on (almost) every tick: `predicted_shift` (the lower-order
claim — which domain's `prediction_error` is about to move, and which direction) and
`prediction_error_confidence` (`1 - mean(prediction_error)` across `ACTIVE_INFERENCE_DOMAINS` —
`orion/substrate/attention_self_model.py::_unconditional_prediction_error_confidence`). Nothing had
ever checked whether the second actually predicts the first's accuracy.

## Method

`scripts/analysis/measure_self_model_calibration.py` (new, committed, read-only). Unlike
`measure_ast_hot_reducer.py` (which replays the pure reducer over raw upstream tables because no
persisted self-model history existed at Phase 1), this reads the REAL, already-persisted live
reducer output directly from `substrate_attention_self_model` — no replay, both fields are already
real production values on every row.

For each row naming domain D and a direction in `predicted_shift`, where D is present in that row's
own `prediction_error_by_domain`: look 2 rows ahead (matches item 3/4's own already-validated
horizon on this exact table) for D's actual value. `correct` = the actual direction of change
matches the predicted direction. A zero actual delta is excluded (ambiguous), not fabricated as
either outcome. Chronological 70/30 TRAIN/TEST split (no shuffle — same leakage-avoidance convention
as item 4's own TEST validation); confidence bin edges computed from TRAIN quantiles only, applied
to TEST.

Run against 7 days of live production history (`substrate_attention_self_model`, 19,418 rows,
2026-08-13 → 2026-08-20):

```text
python scripts/analysis/measure_self_model_calibration.py --window-hours 168
```

## Finding: inverted, not calibrated

```text
TRAIN: n=8966, raw reversion accuracy=61.5%
TEST:  n=3843, raw reversion accuracy=66.0%

TEST confidence bins (edges from TRAIN quantiles):
bin | confidence range     | n    | accuracy
0   | [-inf, 0.9125)       | 1271 | 73.2%
1   | [0.9125, 0.9655)     | 1079 | 59.9%
2   | [0.9655, 0.9847)     |  749 | 62.1%
3   | [0.9847, +inf)       |  744 | 66.4%

Top bin vs. bottom bin: z=-3.22 (p<0.001, two-tailed)
Pearson r(confidence, correct) on TEST: -0.087
```

The **bottom** confidence quartile is more often correct than the **top** confidence quartile, at a
sample size where this is not noise (n=3843 TEST samples, z=-3.22). Domain mix on TEST: execution
2253, biometrics 1098, bus_synaptic 491, chat 1 — the result holds across the domains that actually
won `predicted_shift`'s argmax during this window, not one narrow case.

**Plausible mechanism (not independently confirmed further).** `prediction_error_confidence` is
high exactly when the system is calm overall (`1 - mean(error)`). The reversion formula
(`orion/substrate/prediction_error_trend.py`) is accurate specifically because of spike-and-settle
dynamics — but a calm period has the least real signal to revert *from*, so the winning domain's own
trend in a high-confidence tick is more likely to be tick-level noise than a genuine spike settling
back down. The two scalars are built from the same `prediction_error_by_domain` snapshot but measure
different things (overall systemic calm vs. one domain's imminent direction), and on this data they
are anti-correlated, not merely independent.

## Consumer check (CLAUDE.md §0A step 5, existing-mechanism)

`prediction_error_confidence` has two live consumers today, checked before writing up this finding:

- `orion-equilibrium-service`'s `insight_metacog_gate.py` / `flow_metacog_gate.py` — read it for its
  own documented meaning (confidence-recovery / sustained-stability detection), not as a proxy for
  `predicted_shift`'s correctness.
- `orion-substrate-runtime`'s brain-frame `honesty_metrics` region (`_brain_frame_tick()`) — same
  scalar, displayed as an overall systemic-calm reading, not a per-prediction trust label.

Neither treats this scalar as "how much to trust the current `predicted_shift` claim" today, so this
finding does not correct an active bug — it is a guardrail against building that connection later on
the assumption it would obviously work. Both READMEs were updated in this patch to say so explicitly
(`services/orion-substrate-runtime/README.md`).

## Non-goals

- Not wiring a new `predicted_shift`-reliability field. Shipping a "calibration score" derived from
  `prediction_error_confidence` would be shipping a signal already shown not to track what it would
  claim to track — CLAUDE.md's own "no empty-shell cognition" rule.
- Not building a genuinely independent second-order confidence source (one computed from information
  other than the same `prediction_error_by_domain` snapshot `predicted_shift` itself uses). That
  would be a real next step for a future item-5 pass, but is a materially bigger patch (a new
  information source, not a re-read of an existing field) and is explicitly out of scope here.
- Not re-deriving the spike-and-settle mechanism further (e.g. conditioning on domain, or on the
  magnitude of the winning trend rather than just its sign) — flagged as a plausible follow-up, not
  resolved.

## Acceptance checks

- `pytest scripts/analysis/tests/test_measure_self_model_calibration.py -q` — pure-function coverage
  (parsing, sampling, chronological split, TRAIN-only binning, Pearson r, two-proportion z), no DB.
- Live run against `substrate_attention_self_model` (7-day window) reproduces the numbers above,
  confirming `prediction_error_confidence` is non-degenerate (min=0.5831, max=1.0000 — real variance,
  not flat) before the calibration numbers are trusted at all.
