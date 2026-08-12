# `action_warrant`: a state signal whose rest point is defined rather than guessed

Date: 2026-08-12
Branch: `feat/action-warrant-signal`

## Summary

- New signal `orion/field/action_warrant.py`. Combines each pressure dimension's
  upper-tail probability by Fisher's method into a `[0,1]` score where **0.5 means "a median
  normal day for this machine"**.
- Built only after **five** candidate statistics were measured against real history and killed. Each
  failure is recorded in the module, because re-deriving them costs a deploy cycle each.
- **Zero new persisted state, zero schema change, zero config.** Reads the `dimension_precision_zscore`
  the field-digester already produces every tick.
- Ships with an eval that re-runs the non-degeneracy checks against live data. **PASS**: 10,548 real
  ticks, 62.91% rest, 1.73% fire, median 0.408.
- **Computes a number; decides nothing.** Wiring it into a gate is a separate patch.

## Outcome moved

O1 asks that Orion's action budget rise and fall with real internal pressure. It never has: the arena
dispatched exactly 5 actions every tick and `min_priority` never rejected a single candidate.

The cause was not a mis-set threshold. `resource_pressure = 0.3255` is uninterpretable alone — calm
or busy depends on what this machine normally does — so any fixed cut on it is a guess about scale,
and the guess was wrong by enough that the gate could never bind (measured floor 0.3035 against a
threshold of 0.10).

This is the first statistic measured in this arc that can report calm:

| candidate | % of real ticks able to read calm |
| --- | --- |
| raw max over dimensions | **0.00%** |
| max \|z\| | **0.00%** — absolute value destroys the rest state |
| mean \|z\| | **0.00%** — E[\|Z\|]=√(2/π), a permanent floor |
| max signed z | 30% → **0.00%** once a pinned dimension was fixed |
| Mahalanobis p-value | 65.75%, but unstable in N and faked calm on a dead dimension |
| **`action_warrant`** | **62.91%**, with 1.73% still firing |

## Current architecture

`field_pressures()` → `precision.py` writes `dimension_precision_zscore` per tick → *(previously
consumed only by `dimension_confidence`)*. This adds a second, independent reader of that same
persisted state.

## Architecture touched

One new pure module, one eval, one test file. No service, contract, schema, config, or env change.

## Files changed

- `orion/field/action_warrant.py` — the signal
- `orion/field/evals/run_action_warrant_eval.py` — live-data non-degeneracy eval
- `orion/field/evals/__init__.py`
- `tests/test_action_warrant.py` — 15 tests, three of them property tests encoding the five prior failures

## Schema / bus / API changes

None. Added: nothing. Removed: nothing. Renamed: nothing. No table, column, channel, or event shape
changes.

## Env/config changes

- Added/removed/renamed keys: **none**
- `.env_example` updated: not applicable, no env keys touched
- local `.env` synced: not applicable, no env keys touched

## Metric quality gate (CLAUDE.md §0A)

1. **Provenance.** Traced to producing functions: `compute_ewma_update` (`orion/bus/ewma.py:63-64`) →
   `update_dimension_precision_baseline` (`digestion/precision.py:44`) → this module. Not inferred
   from a schema comment — that error cost this arc a full spec rewrite.
2. **Independence.** Verified, not assumed. Fisher's method requires independent inputs; the four
   dimensions' z-scores have **max pairwise |r| = 0.151** and a participation ratio of **3.95 of 4**
   (n=17,642). They are genuinely four signals.
3. **Theory anchor.** Fisher's combined probability test (Fisher 1925, *Statistical Methods for
   Research Workers* §21.1). Named, standard, and chosen for a specific property: the number of
   contributing dimensions enters the null as 2N degrees of freedom rather than biasing the result.
4. **Live-data sanity.** 10,548 real ticks. Non-degenerate in **both** directions — 62.91% rest,
   1.73% fire, stddev 0.255, min 0.069, max 1.0. Explicitly checked for the "can it ever read calm"
   failure that killed all five predecessors.
5. **Existing mechanism.** Searched first. Five existing/derivable statistics were measured and
   rejected before anything new was written; each is recorded in the module with its measured
   failure.
6. **Reversibility.** No persisted state, no schema, no config, no trained artifact, no env key.
   Deleting the module removes the signal completely. This directly drove the design choice below.

## Honest limits, disclosed

- **Not scale-invariant.** An all-at-median tick scores 0.5000 at N=1 and 0.1626 at N=10. This is
  correct behaviour for a combined-probability test — more independent evidence sharpens the
  inference — but it means a fixed threshold is only meaningful alongside N. That is why
  `ActionWarrant` carries `contributing` rather than returning a bare float. The drift is
  monotonically **toward calm**, the safe direction; every `max()`-shaped predecessor drifted toward
  "act" until nothing could rest.
- **The liveness guard is weaker than the alternative.** An empirical-ECDF variant tracked a pinned
  dimension's true rest fraction within 0.5–1.8pp; this stateless variant is within 5–9pp. The ECDF
  version needs a persisted per-dimension quantile history — new state, expensive to unwind. Chosen
  deliberately under gate item 6; the residual is a known, measured, bounded error.

## Tests run

```text
$ PYTHONPATH=. pytest tests/test_action_warrant.py tests/test_proposal_scoring.py \
    tests/test_proposal_frame_builder.py tests/test_feedback_extractors.py \
    tests/test_field_channel_glossary.py services/orion-field-digester/tests
236 passed in 5.27s
```

## Evals run

```text
$ PYTHONPATH=. POSTGRES_URI=... python orion/field/evals/run_action_warrant_eval.py
{ "verdict": "PASS", "scored_ticks": 10548, "rest_fraction": 0.629124,
  "fire_fraction": 0.017254, "stddev": 0.255224, "min": 0.06913,
  "median": 0.40812, "max": 1.0, "median_live_dimensions": 4.0,
  "exclusions_by_reason": {} }
```

The live median (0.408) matches the offline replay's prediction (0.4052) to within 0.003, which
cross-validates the replay against the real production code path.

## Docker/build/smoke checks

```text
Not run. Nothing is wired to a running service yet -- the module has no consumer
by design. No Dockerfile, compose, dependency, port, or health-check change.
```

## Review findings fixed

Self-caught during the build, before review:

- **Finding:** the property-2 test asserted exact scale-invariance and failed. **The test was wrong,
  not the code** — Fisher's statistic legitimately sharpens with N.
  - **Fix:** test now asserts the property that actually holds and matters — drift is monotone
    *toward calm* and never reaches a rail. Module docstring corrected to stop overclaiming
    stability.
  - **Evidence:** `test_property_2_adding_a_dimension_never_makes_an_average_tick_look_busier`.

- **Finding (significant):** the first liveness check marked a dimension "pinned" when its stored
  EWMA variance sat at its floor. Run live, it excluded **2 of 4 dimensions on every tick** —
  `reasoning_pressure` (median stored variance 9.75e-32) and `reliability_pressure` (5.59e-91).
  Neither is dead; both are *bursty*, and their z-scores stay informative because the floor rescues
  them at compute time. The check was measuring the wrong quantity — the same class of error as the
  signals this module replaces.
  - **Fix:** pinning now requires a collapsed variance **and** a z of exactly 0.0, which is what a
    genuinely stuck value produces. Live eval went from `median_live_dimensions: 2` with 21,406
    exclusions to `4.0` with zero.
  - **Evidence:** `test_bursty_dimension_with_collapsed_variance_stays_live`; eval output above.

## Separate defect found, not fixed here

`orion.bus.ewma.compute_ewma_update` applies `min_variance` when computing the z-score but **stores
the unfloored variance**. Any channel holding a constant value long enough drives its stored variance
to numerical zero — measured at **5.59e-91** for `reliability_pressure`. `scoring.py`'s own comment
warned about this exact risk and assumed the floor prevented it; it only masks it in the z. Consumers
reading the stored variance directly (as this module's first draft did) get a misleading answer. Its
own patch.

## Restart required

```text
No restart required. Nothing consumes this signal yet.
```

## Risks / concerns

- **Severity: low.** Not scale-invariant; a future gate must account for N. Mitigated by carrying
  `contributing` on the result and disclosing the drift table in the module.
- **Severity: low.** Liveness guard residual of 5–9pp on a pinned dimension. Measured and bounded;
  the ECDF upgrade path is documented if it ever matters.
- **Severity: informational.** This signal does not by itself achieve O1. It makes the *scale*
  honest. Whether the arena's behaviour then varies with state depends on wiring it to the gate,
  which is deliberately a separate patch with its own acceptance checks.

## PR link

_to be filled on open_
