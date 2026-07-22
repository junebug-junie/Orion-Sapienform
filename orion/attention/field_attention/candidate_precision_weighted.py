"""Candidate A -- precision-weighted prediction-error salience.

**Shadow, read-only candidate instrument. Not wired into any live consumer.**

Sentience Striving Program (`orion/sentience_striving_program/README.md`) §7 ("measure
before minting", "multi-theory, not single-theory") and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md`'s "Candidate A" section. This is one of two parallel candidate replacements for
the un-calibrated `compute_salience()` formula in
`orion/attention/field_attention/scoring.py` -- Candidate B (Global Workspace/
Society-of-Mind rank-aggregation) is being built separately against the same real data
window. **Do not modify `scoring.py` or `selectors.py` here** -- this module is a parallel
measurement, not a live change, per the plan doc's non-goals ("Not choosing between
Candidate A and B here... Both get built as real, replayed, comparable instruments;
integration is decided from data, later.").

## Theory anchor

Feldman & Friston 2010, "Attention, Uncertainty, and Free-Energy" (confirmed against the
primary Wikipedia summary of the Free Energy Principle, "Perceptual precision, attention
and salience" section, 2026-07-21): attention is precision-optimization on prediction
error. Precision is the inverse variance ("temperature") of a signal's own historical
fluctuation -- a large prediction error is more salient when it occurs against a
historically *quiet* (low-variance, high-precision) backdrop than when the same magnitude
error occurs against a historically noisy one.

    salience(target) = precision(target) x |prediction_error(target)|
    precision(target) = 1 / variance(target's own real historical prediction-error series)

## Real data source and scoping (2026-07-21, live-verified at build time)

`prediction_error(target)`'s *current* value comes from the five already-shipped
instruments in `orion/substrate/prediction_error.py` (`execution_prediction_error`,
`transport_prediction_error`, `biometrics_prediction_error`, `chat_prediction_error`,
`route_prediction_error`). `precision(target)` needs that same target's own *historical*
series -- FalkorDB substrate-graph nodes (`node:substrate.*`) hold only a single current
value with no history (confirmed live via `redis-cli GRAPH.QUERY` against `n.node_id`,
2026-07-21). The real historical series lives in Postgres
`substrate_reduction_receipts`, filtered by `reducer_name` (the receipt's
`state_deltas[0].reducer_id`, written by `_prediction_error_receipt()` in
`services/orion-substrate-runtime/app/worker.py`), ordered by `created_at`, with the
error value at `receipt_json->'state_deltas'->0->'after'->'pressure_hints'->>
'prediction_error'`.

**Honest scoping requirement, re-checked live at build time (not inherited from the
morning's design-doc numbers, which were already hours stale by the time this was
built):**

```
psql -h localhost -p 55432 -U postgres -d conjourney -c "
  SELECT reducer_name, count(*) FROM substrate_reduction_receipts
  WHERE reducer_name IN (
    'substrate.node_biometrics', 'substrate.execution_trajectory',
    'substrate.transport_bus', 'substrate.chat_session', 'substrate.route_arbitration'
  ) GROUP BY reducer_name;"
```

returned exactly one row: `substrate.node_biometrics` (168 receipts, real spread
0.0000-0.1656, non-degenerate). The other four reducer names -- despite their underlying
grammar reducers (`execution_trajectory`, `transport_bus`, `chat_grammar`,
`route_grammar`) all being `enabled=true` in the live `.env` -- had **zero** rows: their
prediction-error writers (`_tick`'s `if error > 0.0` gate at each of the four call sites
in `worker.py`) produced no qualifying delta within the live retention window at check
time. A second, load-bearing finding not in the earlier design doc: `substrate_
reduction_receipts` retains **success** receipts for only
`ORION_RECEIPT_RETENTION_SUCCESS_MINUTES=30` minutes (confirmed in
`services/orion-substrate-runtime/.env` and `app/settings.py`, and empirically: 6,353
total live rows spanning 2026-07-03 -> now, but all `expires_at` within 30 minutes of
their own `created_at`, so a background pruner is actively deleting older rows). This
means the "real historical series" available to this module's `precision_weighted_
salience()` is *always* bounded to a rolling ~30-minute window, for every target, not
just for whichever targets happen to be fresh today -- a structural property of the data
source, not a today-only staleness artifact.

**Scope for this patch: `substrate.node_biometrics` only.** Per the task's explicit
instruction, this module's precision computation is not silently extended to
execution/transport/chat/route -- they simply have no real, non-degenerate receipt
history to compute a meaningful variance from at build time. If/when a future live run of
one of those four lanes produces enough qualifying receipts, the replay script
(`scripts/analysis/measure_precision_weighted_salience_probe.py`) will report that
honestly rather than silently including a reducer this module was never checked against.
"""

from __future__ import annotations

from dataclasses import dataclass

# Precision = 1 / variance diverges as variance -> 0 (a target whose recent error has
# been almost perfectly constant). This is the "variance-near-zero instability risk"
# named in the tentative-plan doc's Candidate A section. Floored here at a concrete
# epsilon so precision saturates at a finite ceiling (1 / PRECISION_VARIANCE_FLOOR)
# instead of diverging to +inf or raising a ZeroDivisionError.
PRECISION_VARIANCE_FLOOR: float = 1e-6


@dataclass(frozen=True)
class PrecisionWeightedSalienceResult:
    """Everything needed to inspect *why* a salience score came out the way it did --
    not just the final scalar. `variance_floored` in particular is the concrete,
    inspectable signal for the instability risk named in the design doc: a report can
    tally how often it fires rather than trusting an unbounded score blindly."""

    salience: float
    precision: float
    variance: float
    current_error: float
    n_samples: int
    variance_floored: bool


_EMPTY_RESULT = PrecisionWeightedSalienceResult(
    salience=0.0,
    precision=0.0,
    variance=0.0,
    current_error=0.0,
    n_samples=0,
    variance_floored=False,
)


def _population_variance(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def precision_weighted_salience(
    error_history: list[float],
) -> PrecisionWeightedSalienceResult:
    """Candidate A salience for one target: precision x |current prediction error|.

    ``error_history``: a target's own real historical prediction-error magnitudes,
    oldest-first (e.g. a window of real rows from `substrate_reduction_receipts` for one
    `reducer_name`, in `created_at ASC` order -- see module docstring for the exact
    query shape and the current real scope, `substrate.node_biometrics` only). The last
    element is treated as the "current" error being weighted; the *entire* list
    (including that current point) is used to estimate the process's own variance, per
    Feldman & Friston 2010's framing of precision as a property of the signal's own
    recent fluctuation, not a leave-one-out estimate of "everything before now."

    Pure function. No I/O. Never raises.

    Edge cases:
    - Empty history -> zero salience/precision, `n_samples=0`. There is no "current"
      error to weight in the first place.
    - Single-sample history -> population variance of one point is exactly 0.0 by
      definition (no fluctuation observed yet). This is NOT treated as a separate
      "undefined" case from near-zero variance -- one real observation is real data at
      the smallest possible sample size, not missing data -- so it floors the same way
      any other near-zero-variance history does (see below), with `n_samples=1`
      reported so a caller/report can weight its confidence in the result accordingly.
    - Near-zero variance (a target whose error has been almost constant across the
      window) -> precision would otherwise diverge; floored at
      ``1 / PRECISION_VARIANCE_FLOOR`` (large but finite), and `variance_floored=True`
      is reported explicitly so a caller can flag the instability risk rather than
      silently trust an extreme score.
    """
    n = len(error_history)
    if n == 0:
        return _EMPTY_RESULT

    values = [float(v) for v in error_history]
    current_error = values[-1]
    variance = _population_variance(values)
    variance_floored = variance < PRECISION_VARIANCE_FLOOR
    effective_variance = max(variance, PRECISION_VARIANCE_FLOOR)
    precision = 1.0 / effective_variance
    salience = precision * abs(current_error)

    return PrecisionWeightedSalienceResult(
        salience=salience,
        precision=precision,
        variance=variance,
        current_error=current_error,
        n_samples=n,
        variance_floored=variance_floored,
    )
