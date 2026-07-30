"""Metacog trend reducer -- pure, incremental computation.

Hop 0 of the stream-of-consciousness hop-chain design
(`docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md`)
and the reducer named in `docs/superpowers/specs/2026-07-28-metacog-turn-scoped-
trend-reducer-design.md`. Answers "has this kept happening" over a windowed
series of repair-pressure-style readings -- a genuine trend/baseline check,
not a point-in-time cutoff.

## Scope of this patch

This module is the reducer's core computation only: pure, deterministic,
fully unit-testable, no I/O. It does NOT wire into any live poll loop yet.
That is a deliberate, disclosed deferral, not an oversight:

`orion-substrate-runtime/app/worker.py`'s `ReducerSpec`/`_grammar_reducer_poll_loop`
pattern -- the thing both design docs pointed at as this reducer's shape --
turned out, on inspection, to be a heavier, differently-scoped mechanism than
it first looked: it consumes bus-published "pressure grammar" events through a
cursor-tracked accepted-events pipeline specific to that file's five existing
reducers (biometrics/execution/transport/chat/route), not a generic "any
periodic reducer" framework. Forcing this reducer's simple "poll two Postgres
tables, fold into an EWMA" shape into that cursor/grammar-event machinery
would mean inventing a new grammar-event producer for `repair_pressure`/
`turn_effect` just to satisfy an ill-fitting pattern -- exactly the kind of
"ornamental layer" CLAUDE.md's thin-seams rule warns against. The live-wiring
follow-up (a lightweight standalone poll loop, flag-gated off, most likely
living in `services/orion-substrate-runtime` or `services/orion-cortex-exec`
since that's where `repair_pressure_appraisal_log`'s producer lives) is a
separate, smaller patch once this core computation is proven correct.

## Confidence-gating is the caller's job, not this module's

The 2026-07-30 baseline measurement (`scripts/analysis/measure_metacog_trend_
baseline.py`) found `repair_pressure_appraisal_log.level` is `FLOOR_DOMINATED`
ungated -- 76.9% of rows are the appraiser's own `confidence == 0.0` "no
evidence" default, not real readings. This module does not filter on
`reading.confidence` itself; it assumes the caller has already decided what
counts as trustworthy evidence (e.g. `confidence > 0`) before constructing a
`MetacogTrendReading`. `confidence` is carried through on the reading purely
for caller-side inspection/logging, not used in the fold math.

## Checkpointed, resumable state

`MetacogTrendStateV1` is the reducer's entire carried-forward state -- four
floats/ints. This directly answers the hop-chain design's Missing Question 4
("a hop needs to checkpoint its own state ... durable and resumable, cheaply,
every cycle it doesn't win a slot"): this state is small enough to persist
every tick and resume from exactly, unlike holding state only in a live
process.

## Import cost, disclosed

This module's own code has no I/O and no non-stdlib dependency beyond
`orion/bus/ewma.py` (pure stdlib). But `import orion.metacog.trend_reducer`
still executes `orion/metacog/__init__.py` first (Python always runs a
package's `__init__` before any of its submodules), which imports
`.service` -> `orion.schemas.metacog_entry` -> `pydantic`. So this module is
computationally pure but not import-cheap in isolation -- a future live
poll loop in a service that doesn't otherwise depend on `pydantic`/
`orion.schemas` will pull that whole chain in just by importing this file.
Not fixed here (moving `trend_reducer` out of the `orion.metacog` package
would be a bigger, separately-motivated restructuring); disclosed so the
live-wiring follow-up doesn't assume a cheap standalone import.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from orion.bus.ewma import compute_ewma_update

DEFAULT_MIN_SAMPLES: int = 20
DEFAULT_EWMA_ALPHA: float = 0.2
DEFAULT_ELEVATED_ZSCORE: float = 1.0
DEFAULT_SUSTAINED_HITS: int = 3


@dataclass(frozen=True)
class MetacogTrendReading:
    """One caller-supplied, already-confidence-gated observation."""

    at: datetime
    level: float
    confidence: float


@dataclass(frozen=True)
class MetacogTrendStateV1:
    """The reducer's entire carried-forward, checkpointable state.

    `ewma`/`variance`/`count` are exactly `orion/bus/ewma.py::compute_ewma_update`'s
    own running-baseline triple -- same shape already used live for
    `bus_synaptic_prediction_error`'s `gap_zscore` and (as of PR #1433)
    `field:recent_perturbations`' salience. Reused, not reinvented, per
    CLAUDE.md's existing-mechanism check. `min_variance` is left at that
    function's own default (1e-6): the 2026-07-30 baseline measurement's
    12-row confidence-gated sample showed variance ~0.019 (stddev ~0.14 on a
    0-1 scale), five orders of magnitude above the 1e-6 floor, so the floor
    would not dominate the way it wrongly did for `execution_prediction_
    error`'s ~4e-11 real variance (that module's own cautionary example) --
    but that same doc chain classifies this 12-row sample `INSUFFICIENT_
    DATA`, so treat this as a plausible read on the right order of
    magnitude, not a certified one; re-check once more real rows land.

    Caller responsibility, not enforced here: `min_samples`/`alpha`/
    `elevated_zscore`/`sustained_hits` (the config passed to `apply_reading`)
    are NOT part of this checkpointed state and are NOT validated for
    consistency across resumes. `is_cold_start`/`is_elevated_this_tick` are
    recomputed fresh from whatever config the *current* call passes, so
    resuming a state warmed up under one `min_samples` with a different
    value on the next call changes the classification with zero new
    evidence folded in. A live poll loop (this module's disclosed follow-up)
    must keep this config fixed across restarts, or persist it alongside
    the state and refuse to resume under a different value -- caught in
    code review, not solved here since no live caller exists yet to decide
    where that guard belongs.
    """

    ewma: float = 0.0
    variance: float = 0.0
    count: int = 0
    consecutive_elevated: int = 0


@dataclass(frozen=True)
class MetacogTrendResultV1:
    """One tick's classification, plus the new state to persist."""

    state: MetacogTrendStateV1
    latest_zscore: Optional[float]
    is_cold_start: bool
    is_elevated_this_tick: bool
    is_sustained_trend: bool


def apply_reading(
    state: MetacogTrendStateV1,
    reading: MetacogTrendReading,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    alpha: float = DEFAULT_EWMA_ALPHA,
    elevated_zscore: float = DEFAULT_ELEVATED_ZSCORE,
    sustained_hits: int = DEFAULT_SUSTAINED_HITS,
) -> MetacogTrendResultV1:
    """Fold one new reading into `state`, return the new state plus this
    tick's trend classification.

    Cold-start guarded: no reading is ever classified `is_elevated_this_tick`
    until at least `min_samples` real readings have been folded in AND a
    real z-score exists (the EWMA update itself returns `None` on the very
    first observation, since there is no baseline yet to be anomalous
    against -- same "no empty-shell cognition" discipline as
    `dimension_confidence()`'s own cold-start guard in `orion/proposals/
    scoring.py`). `min_samples=20` matches this measurement's own disclosed
    small-N threshold, not an arbitrary round number.

    `is_sustained_trend` requires `sustained_hits` (default 3) CONSECUTIVE
    elevated ticks, resetting to 0 on any non-elevated tick -- directly
    answers "has this kept happening," not "did this spike once."
    """
    update = compute_ewma_update(
        prev_ewma=state.ewma,
        prev_variance=state.variance,
        prev_count=state.count,
        value=reading.level,
        alpha=alpha,
    )
    new_count = state.count + 1
    is_cold_start = new_count < min_samples or update.zscore is None

    is_elevated_this_tick = (
        not is_cold_start
        and update.zscore is not None
        and update.zscore >= elevated_zscore
    )
    new_consecutive = state.consecutive_elevated + 1 if is_elevated_this_tick else 0
    is_sustained_trend = new_consecutive >= sustained_hits

    new_state = MetacogTrendStateV1(
        ewma=update.ewma,
        variance=update.variance,
        count=new_count,
        consecutive_elevated=new_consecutive,
    )
    return MetacogTrendResultV1(
        state=new_state,
        latest_zscore=update.zscore,
        is_cold_start=is_cold_start,
        is_elevated_this_tick=is_elevated_this_tick,
        is_sustained_trend=is_sustained_trend,
    )


def replay(
    readings: list[MetacogTrendReading],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    alpha: float = DEFAULT_EWMA_ALPHA,
    elevated_zscore: float = DEFAULT_ELEVATED_ZSCORE,
    sustained_hits: int = DEFAULT_SUSTAINED_HITS,
) -> list[MetacogTrendResultV1]:
    """Fold a batch of readings (e.g. real historical rows, ASC by time)
    through `apply_reading` in order, returning every tick's result. Used by
    both replay/measurement scripts and the eventual live poll loop (which
    would instead call `apply_reading` once per new row and persist just the
    latest `state`)."""
    state = MetacogTrendStateV1()
    results: list[MetacogTrendResultV1] = []
    for reading in readings:
        result = apply_reading(
            state,
            reading,
            min_samples=min_samples,
            alpha=alpha,
            elevated_zscore=elevated_zscore,
            sustained_hits=sustained_hits,
        )
        results.append(result)
        state = result.state
    return results
