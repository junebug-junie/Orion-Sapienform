"""Rung 6, self-revision: Orion tunes how surprised they expect to be.

Each EWMA-scored prediction-error domain (``prediction_error.py``) turns a
raw per-tick reading into surprise by comparing it with a running expectation:
an exponentially weighted mean and variance. Two settings shape that
expectation, and both were hand-picked constants:

- ``alpha``: how fast the expectation follows new readings.
- ``min_variance``: the smallest spread the expectation is allowed to assume.

This module lets Orion revise those settings from their own history. It is
pure: no I/O. Callers supply ``(observed_at, raw_value)`` readings
(``substrate_prediction_raw_reading``) and get back scores.

**Theory anchor: the logarithmic scoring rule.** For each reading, score the
log density of the reading under the Gaussian the expectation implied *before*
it saw that reading. The log score is strictly proper (Gneiting & Raftery
2007): its expected value is maximised only by the true predictive
distribution. That matters here because the cheap alternative -- "pick the
setting with the least surprise" -- is won by going numb: a huge variance makes
every reading unsurprising. The log score charges ``-0.5*log(variance)`` for
that, so the only way to score well is to be right about both where readings
land and how much they spread.

**Honesty rules.** A candidate is chosen on the older part of a window and must
then beat the current setting on the newer part it never saw (the trial). After
adoption, both settings are replayed over readings that arrived later still
(the monitor). Raw readings do not depend on the setting -- the setting only
changes the baseline they are compared with -- so both replays are exact
counterfactuals, not estimates. Every comparison is paired per reading and is
reported as a lower (or upper) confidence bound, not a bare mean, so a small
noisy difference can't pass as a real one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

# Single mutation class and surface prefix for this rung. The surface is per
# domain so each domain has its own one-live-change lock, cooldown and
# reliability history in SubstrateMutationStore.
MUTATION_CLASS = "prediction_expectation_patch"
SURFACE_PREFIX = "prediction_expectation:"
CONTROL_SURFACE_KEY_PREFIX = "prediction_expectation."

# Two standard errors: a paired difference must clear ~95% one-sided
# confidence before it counts as better (trial) or worse (monitor).
CONFIDENCE_Z = 2.0
# Per-reading differences are NOT independent: both replays carry their
# running mean/variance forward, so neighbours share ~1/alpha readings of
# memory. The standard error is therefore taken from batch means over blocks
# BATCH_MEMORY_MULTIPLE memory-lengths long (at least MIN_BATCH_SIZE), and a
# comparison needs MIN_BATCHES blocks. A naive sqrt(var/n) here was several
# times too small (code review, 2026-10-16).
BATCH_MEMORY_MULTIPLE = 2.0
MIN_BATCH_SIZE = 10
MIN_BATCHES = 8
# Smallest gain worth changing a live setting for, in nats per reading. A
# confident but tiny gain is not worth the churn (or the flip-flop risk).
MIN_EFFECT_NATS = 0.01
FIT_FRACTION = 2.0 / 3.0
# Readings replayed but never scored at the start of a window. A replay starts
# cold (variance 0, so the first comparisons run at the bare floor), and one
# cold reading can cost ~1e5 nats -- enough to decide the whole comparison on
# an artifact the live domain, warm for 100k+ readings, never experiences.
WARMUP_READINGS = 30
MIN_FIT_READINGS = 300
MIN_HOLDOUT_READINGS = 150
MIN_POST_ADOPTION_READINGS = 150
# (Effective minima are higher for slow alphas: MIN_BATCHES * batch_size.)

ALPHA_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5)
MIN_VARIANCE_MULTIPLIERS: tuple[float, ...] = (0.1, 1.0, 10.0)
ALPHA_BOUNDS = (0.01, 0.6)
# Relative to the domain's hand-calibrated floor. Wide on purpose -- the grid,
# not these bounds, keeps a single step small.
MIN_VARIANCE_BOUND_FACTOR = 1000.0


@dataclass(frozen=True)
class Expectation:
    alpha: float
    min_variance: float

    def as_patch(self) -> dict[str, float]:
        return {"alpha": float(self.alpha), "min_variance": float(self.min_variance)}


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    default: Expectation
    # How long an adoption must survive the post-adoption replay before its
    # surface is released. Sized so the domain's live reading rate can reach
    # MIN_POST_ADOPTION_READINGS inside it (execution ~70/h; chat per turn,
    # so it gets the schema's 24 h maximum and may still settle unmeasured).
    rollback_window_sec: int = 6 * 3600

    @property
    def target_surface(self) -> str:
        return f"{SURFACE_PREFIX}{self.domain}"

    @property
    def control_surface_key(self) -> str:
        return f"{CONTROL_SURFACE_KEY_PREFIX}{self.domain}"

    def in_bounds(self, expectation: Expectation) -> bool:
        lo, hi = ALPHA_BOUNDS
        floor = self.default.min_variance
        return (
            lo <= expectation.alpha <= hi
            and floor / MIN_VARIANCE_BOUND_FACTOR <= expectation.min_variance <= floor * MIN_VARIANCE_BOUND_FACTOR
        )


def _domains() -> dict[str, DomainSpec]:
    # Imported here, not at module top, so the defaults have exactly one home
    # (prediction_error.py's calibrated constants) without a circular import:
    # prediction_error.py imports Expectation from this module.
    from orion.substrate import prediction_error as pe
    from orion.substrate import street_attention as st

    return {
        "execution": DomainSpec(
            "execution",
            Expectation(pe._EXECUTION_PREDICTION_ERROR_EWMA_ALPHA, pe._EXECUTION_PREDICTION_ERROR_MIN_VARIANCE),
        ),
        "chat": DomainSpec(
            "chat",
            Expectation(pe._CHAT_PREDICTION_ERROR_EWMA_ALPHA, pe._CHAT_PREDICTION_ERROR_MIN_VARIANCE),
            rollback_window_sec=24 * 3600,
        ),
        # One reading per graded street handoff (-ln p_value, nats;
        # orion/substrate/street_attention.py). Rest point under a calibrated
        # model: mean 1 nat, variance 1. The schema's 24 h rollback maximum,
        # like chat. EXPECTED TO STAY INERT: the tuner needs WARMUP +
        # MIN_FIT + MIN_HOLDOUT = 480 readings inside its 72 h window (~160
        # graded handoffs a day); the street makes a handful. Recorded now so
        # the corpus exists; the contract (mutation_contracts.py) allows the
        # target so an adoption is possible once volume is real.
        st.STREET_HANDOFF_DOMAIN: DomainSpec(
            st.STREET_HANDOFF_DOMAIN,
            Expectation(st.STREET_HANDOFF_EWMA_ALPHA, st.STREET_HANDOFF_MIN_VARIANCE),
            rollback_window_sec=24 * 3600,
        ),
    }


def domain_specs() -> dict[str, DomainSpec]:
    return _domains()


def domain_for_surface(target_surface: str) -> DomainSpec | None:
    if not target_surface.startswith(SURFACE_PREFIX):
        return None
    return _domains().get(target_surface[len(SURFACE_PREFIX):])


def expectation_from_payload(payload: dict | None, *, default: Expectation) -> Expectation:
    """Read a stored control-surface or patch payload; fall back per field."""
    payload = payload or {}
    try:
        alpha = float(payload.get("alpha", default.alpha))
    except (TypeError, ValueError):
        alpha = default.alpha
    try:
        min_variance = float(payload.get("min_variance", default.min_variance))
    except (TypeError, ValueError):
        min_variance = default.min_variance
    if not (math.isfinite(alpha) and math.isfinite(min_variance)) or min_variance <= 0.0:
        return default
    return Expectation(alpha=alpha, min_variance=min_variance)


def per_reading_log_scores(values: Sequence[float], expectation: Expectation) -> list[float | None]:
    """Log score of each reading under the expectation held *before* it.

    Mirrors ``orion.bus.ewma.compute_ewma_update`` exactly (same update order,
    same variance floor), so a replay reproduces what the live domain would
    have computed under ``expectation``. The first reading has no prior
    expectation and scores ``None``.
    """
    scores: list[float | None] = []
    ewma = 0.0
    variance = 0.0
    count = 0
    alpha = expectation.alpha
    for value in values:
        if count == 0:
            ewma, variance = float(value), 0.0
            scores.append(None)
        else:
            spread = max(variance, expectation.min_variance)
            deviation = float(value) - ewma
            scores.append(-0.5 * math.log(2.0 * math.pi * spread) - (deviation * deviation) / (2.0 * spread))
            ewma = alpha * float(value) + (1.0 - alpha) * ewma
            variance = alpha * deviation * deviation + (1.0 - alpha) * variance
        count += 1
    return scores


@dataclass(frozen=True)
class PairedGain:
    """Per-reading paired log-score difference (candidate minus reference).

    ``standard_error`` is a batch-means estimate (see BATCH_MEMORY_MULTIPLE)."""

    mean: float
    standard_error: float
    count: int
    batches: int

    @property
    def lower_bound(self) -> float:
        return self.mean - CONFIDENCE_Z * self.standard_error

    @property
    def upper_bound(self) -> float:
        return self.mean + CONFIDENCE_Z * self.standard_error


def paired_gain(
    values: Sequence[float],
    *,
    candidate: Expectation,
    reference: Expectation,
    score_from: int = 0,
    score_to: int | None = None,
) -> PairedGain | None:
    """Mean log-score gain of ``candidate`` over ``reference`` on readings
    ``[score_from, score_to)``. Both replays warm up on everything before
    ``score_from``, so the comparison is about the scored span only."""
    end = len(values) if score_to is None else min(score_to, len(values))
    cand = per_reading_log_scores(values[:end], candidate)
    ref = per_reading_log_scores(values[:end], reference)
    diffs = [c - r for c, r in zip(cand[score_from:end], ref[score_from:end]) if c is not None and r is not None]
    size = batch_size(candidate, reference)
    n_batches = len(diffs) // size
    if n_batches < MIN_BATCHES:
        return None
    used = diffs[: n_batches * size]
    batch_means = [sum(used[i * size : (i + 1) * size]) / size for i in range(n_batches)]
    mean = sum(used) / len(used)
    var = sum((b - mean) ** 2 for b in batch_means) / (n_batches - 1)
    return PairedGain(mean=mean, standard_error=math.sqrt(var / n_batches), count=len(used), batches=n_batches)


def batch_size(*expectations: Expectation) -> int:
    slowest = min(e.alpha for e in expectations)
    return max(MIN_BATCH_SIZE, math.ceil(BATCH_MEMORY_MULTIPLE / slowest))


def candidate_grid(spec: DomainSpec, current: Expectation) -> list[Expectation]:
    grid = {
        Expectation(alpha=alpha, min_variance=spec.default.min_variance * mult)
        for alpha in ALPHA_GRID
        for mult in MIN_VARIANCE_MULTIPLIERS
    }
    grid.discard(current)
    # Candidates that keep the live floor come first: when the floor never
    # binds, every floor scores identically, and assess() keeps the first of
    # a tie -- so an unevidenced floor change is never proposed.
    return sorted(
        (e for e in grid if spec.in_bounds(e)),
        key=lambda e: (e.min_variance != current.min_variance, e.alpha, e.min_variance),
    )


@dataclass(frozen=True)
class Assessment:
    domain: str
    current: Expectation
    candidate: Expectation
    fit_gain: PairedGain
    fit_window_start: datetime
    fit_window_end: datetime
    reading_count: int


def assess(
    spec: DomainSpec,
    readings: Sequence[tuple[datetime, float]],
    *,
    current: Expectation,
) -> Assessment | None:
    """Best candidate on the older ``FIT_FRACTION`` of the window, or None.

    None when there is too little data to leave a real holdout behind, or when
    no candidate beats the current setting with confidence on the fit span.
    The holdout span is left untouched for the trial.
    """
    n = len(readings)
    if n < WARMUP_READINGS + MIN_FIT_READINGS + MIN_HOLDOUT_READINGS:
        return None
    split = int(n * FIT_FRACTION)
    if split - WARMUP_READINGS < MIN_FIT_READINGS or n - split < MIN_HOLDOUT_READINGS:
        return None
    values = [v for _, v in readings]
    best: tuple[Expectation, PairedGain] | None = None
    for candidate in candidate_grid(spec, current):
        gain = paired_gain(values, candidate=candidate, reference=current, score_from=WARMUP_READINGS, score_to=split)
        if gain is None or gain.lower_bound <= 0.0 or gain.mean < MIN_EFFECT_NATS:
            continue
        if best is None or gain.mean > best[1].mean + 1e-12:
            best = (candidate, gain)
    if best is None:
        return None
    return Assessment(
        domain=spec.domain,
        current=current,
        candidate=best[0],
        fit_gain=best[1],
        fit_window_start=readings[0][0],
        fit_window_end=readings[split - 1][0],
        reading_count=split,
    )


def holdout_gain(
    readings: Sequence[tuple[datetime, float]],
    *,
    candidate: Expectation,
    reference: Expectation,
    after: datetime,
    min_readings: int,
) -> PairedGain | None:
    """Paired gain on readings strictly after ``after``, warmed on the rest.

    Used twice: by the trial (``after`` = end of the fit span, so the scored
    readings were never used to pick the candidate) and by the post-adoption
    monitor (``after`` = adoption time, so they did not exist yet)."""
    values = [v for _, v in readings]
    first_scored = next((i for i, (ts, _) in enumerate(readings) if ts > after), None)
    if first_scored is not None:
        first_scored = max(first_scored, WARMUP_READINGS)
    if first_scored is None or len(readings) - first_scored < min_readings:
        return None
    return paired_gain(values, candidate=candidate, reference=reference, score_from=first_scored)
