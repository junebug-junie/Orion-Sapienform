"""Regime readout for a field channel: level, dispersion, drift, saturation,
and whether a producer wrote anything -- as SEPARATE readings over a window
declared in seconds.

R2 of the phase-5 roadmap, PR #1622 (branch docs/phase5-liveness-scope).
Cited by PR number, not path: that doc is not on main as of this commit.

WHY THIS IS NOT classify_channel_series()
-----------------------------------------
`orion.field.channel_glossary.classify_channel_series()` already exists, is
live behind Hub's glossary panel, and is NOT replaced here -- it stays the
single-label verdict and can be layered on top of this. This module exists
because a single label cannot carry four independent facts, and the collapse
loses exactly the distinction that keeps being asked for:

1. **Level and dispersion are one axis there, two here.** A channel sitting at
   0.813 with dispersion 0.001 and a channel sitting at 0.02 with dispersion
   0.001 both return `quiet`. "Very busy at near max, but steady, so actually
   peaceful" is not expressible in one word, and that reading is the whole
   point of a regime readout.

2. **There is no window there.** It takes a bare `list[float]`. The caller
   decides what window that is and nobody records it. Live cadence on
   `substrate_field_state` measured 2026-08-13 was a 2.04s median tick, so a
   "600-tick window" silently means ~20 minutes -- a fact that appeared
   nowhere. Windows here are declared in SECONDS and carried on the result.

3. **`dead` conflates a decayed remnant with a legitimately idle producer.**
   Its docstring says so outright: the folded-away, unproduced, and
   subnormal-noise cases "mean the same thing". They do not. `repair_pressure`
   and `conversation_load` are chat-gated (see their glossary meanings:
   "happening in chat", "turn volume/complexity"). When nobody is talking to
   Orion they decay toward zero and read `dead` -- but that is CORRECT
   behavior, not a defect. Calling it dead cries wolf on every idle channel.
   This module reports `refresh_state` as its own axis and deliberately makes
   NO claim about whether silence is correct: that depends on whether the
   producer had reason to fire, which is not knowable from the series.

WHAT IT DOES CLAIM
------------------
`refresh_state` is mechanical, not a judgement:

- `producer_written` -- at least one change in the window is not explained by
  the decay constant.
- `decay_only` -- every change is a multiplication by the same ratio, matching
  a known decay rate. Nothing refreshed this; the movement is the decay loop
  touching it. This is the signature of the documented `node:substrate.route`
  incident (0.92/tick for 48h) and it is why "the numbers move" is not
  evidence a metric is alive.
- `static` -- no change at all in the window.
- `insufficient_samples` -- fewer than two points.

KNOWN LIMIT of `static`: a producer writing the SAME value every tick is
indistinguishable from no producer writing at all. Both are `static`. Live
example (2026-08-13): the six HIGHER_IS_BETTER channels -- availability,
available_capacity, bus_health, confidence, delivery_confidence,
stream_backlog_health -- all sit at exactly 1.0 with zero dispersion, and this
module cannot tell you whether something is asserting 1.0 every tick or
nothing has touched them since boot. Separating those needs a producer-side
write timestamp, which no field channel carries today. Do not read `static` as
"nothing wrote it".

ON CONSTANTS
------------
`SUBNORMAL_CUTOFF` is reused from `channel_glossary` because "is this
functionally zero" is scale-free and transfers. The level and dispersion
thresholds are NOT reused, and that is deliberate -- see their definitions
below for the live case that forced the split. Borrowing a constant across
domains without re-verifying it at the new scale is a known, repeated failure
mode in this repo, and the first draft of this module committed it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from orion.field.channel_glossary import SUBNORMAL_CUTOFF

# services/orion-field-digester/app/digestion/decay.py applies this rate to
# NODE_DECAY_CHANNELS/CAPABILITY_DECAY_CHANNELS (BIOMETRICS_FIELD_DECAY_RATE).
# Confirmed live 2026-08-13 against 120,000 substrate_field_state ticks: both
# social channels stepped at an exact 0.92000 ratio for 42,000+ transitions.
KNOWN_DECAY_RATES: tuple[float, ...] = (0.92,)

# Successive-ratio tolerance for calling a step "decay". Tight on purpose --
# floating-point multiplication by a fixed constant reproduces to ~1e-12, so
# anything looser starts absorbing real producer writes that happen to land
# near the decay ratio.
DECAY_RATIO_EPSILON: float = 1e-6

# A value within this of either rail counts as saturated. Distinct from
# SUBNORMAL_CUTOFF (1e-100), which asks "is this functionally zero" -- this
# asks "is this pinned against a bound and therefore unable to report a
# further move in that direction".
RAIL_EPSILON: float = 1e-3

# Below this many samples, dispersion and drift are artifacts rather than
# readings. Matches the repo's standing lesson that short-window distribution
# statistics produce false floor/spread claims.
MIN_REGIME_SAMPLES: int = 8

# Neither threshold below reuses channel_glossary's LIVE_VARIANCE_THRESHOLD
# (0.05). That constant is calibrated for `max - median` SPREAD; the first
# draft of this module borrowed it for the LEVEL axis and the error surfaced
# immediately on live data -- cpu_pressure at a median of 0.2594 came out
# "loaded" because 0.2594 > 0.05. SUBNORMAL_CUTOFF is still imported, since
# "is this functionally zero" is scale-free and genuinely does transfer.
#
# Both thresholds are ABSOLUTE, which is legitimate here and only here: every
# field channel is clamp01'd to [0,1] by collect_field_channel_pressures(),
# and (after HIGHER_IS_BETTER_CHANNELS polarity correction) they share a
# direction. That shared scale is a real commensurability property of this
# surface, not an assumption. Do not copy these thresholds to a surface that
# lacks it.
#
# The relative readings (level_percentile, dispersion_ratio) are still
# computed and reported -- they answer "unusual for this channel right now",
# a drift question. The composed label uses the absolute pair because the
# question it exists to answer is "near max but steady", which is a statement
# about the scale, not about recent history. A second draft composed on the
# percentile instead and collapsed every live channel to one label, because a
# stationary channel always sits mid-baseline.

# DERIVED, not chosen. Measured across all 41 live channels over a 600-sample
# window (2026-08-13): dispersion values were 0.0, 0.000046, 0.000447,
# 0.000525, then jumped straight to 0.0449, 0.0715, 0.102, 0.117, 0.118,
# 0.131, 0.144, 0.190, 0.227. Nothing at all occupies the two orders of
# magnitude between 0.0005 and 0.045, so this threshold sits inside an empty
# region and no real channel is near enough to flip on noise.
STEADY_DISPERSION: float = 0.02

# DECLARED convention, NOT derived -- stated plainly because the same
# measurement found no natural gap here (levels ran 0.6906, 0.7107, 0.7335,
# 0.8125 with no space between them). Reads as "in the top 30% of the
# available range" on a [0,1] pressure scale. A channel just under it is a
# judgement call, and the raw `level` is on the result for exactly that
# reason.
LOADED_LEVEL: float = 0.70


@dataclass(frozen=True)
class ChannelRegime:
    """One channel's regime over one declared window.

    Every field is a separate reading. `regime` is a convenience composition
    of the others, NOT a substitute for them -- a consumer that only reads
    `regime` has re-created the single-label collapse this module exists to
    undo.
    """

    channel: str
    # DECLARED, never inferred. The caller states what window it sampled.
    window_seconds: float
    sample_count: int
    # Median over the window. Robust to the 1.0 spikes several channels show.
    level: float
    # Population stdev over the window.
    dispersion: float
    # Where this window's level sits inside the baseline distribution
    # (fraction of baseline samples below it). This -- not the raw level -- is
    # what "loaded" means, because channels have different natural ranges.
    # None without a baseline: explicitly not 0.5, which would read as
    # "perfectly typical".
    level_percentile: float | None
    # This window's dispersion over the baseline's. 1.0 means as variable as
    # usual, <1 steadier than usual. None without a baseline.
    dispersion_ratio: float | None
    # |mean(window) - mean(baseline)| / stdev(baseline). None when no baseline
    # was supplied -- explicitly not 0.0, which would read as "no drift".
    drift: float | None
    # Fraction of samples pinned within RAIL_EPSILON of 0.0 or 1.0.
    saturation: float
    refresh_state: str
    regime: str


def _refresh_state(values: list[float]) -> str:
    """Is this series moving because a producer wrote it, or because a decay
    loop touched it?

    A step counts as decay when `b / a` matches a known decay rate. A series
    is `decay_only` when it has at least one change and EVERY change is such a
    step -- one genuine producer write is enough to disqualify it, which is
    the conservative direction: mislabelling a live channel as decay-only
    would be the damaging error.
    """
    if len(values) < 2:
        return "insufficient_samples"
    changes = [(a, b) for a, b in zip(values, values[1:]) if a != b]
    if not changes:
        return "static"
    for a, b in changes:
        if a <= 0.0:
            return "producer_written"
        ratio = b / a
        if not any(abs(ratio - r) < DECAY_RATIO_EPSILON for r in KNOWN_DECAY_RATES):
            return "producer_written"
    return "decay_only"


def _compose(
    level: float,
    dispersion: float,
    saturation: float,
    refresh_state: str,
) -> str:
    """Compose the axes into one label, for surfaces that can only show one.

    Ordering is deliberate: refresh state dominates, because a number nothing
    wrote is not a reading about the world regardless of its level. Saturation
    comes next, because a pinned channel's level and dispersion are both
    artifacts of the rail.

    Needs no baseline: both thresholds are absolute on the shared [0,1]
    pressure scale, so a regime is readable from the window alone.
    """
    if refresh_state == "insufficient_samples":
        return "insufficient_samples"
    if refresh_state in ("decay_only", "static"):
        # No claim about whether this is correct -- see module docstring.
        return "no_input"
    if saturation > 0.5:
        return "saturated"
    steady = dispersion <= STEADY_DISPERSION
    loaded = level >= LOADED_LEVEL
    if loaded and steady:
        # The reading this module was built for: busy near the top of its
        # range, and not moving. Loaded is not the same as alarming.
        return "loaded_steady"
    if loaded:
        return "loaded_volatile"
    if steady:
        return "calm"
    return "quiet_volatile"


def channel_regime(
    channel: str,
    values: list[float],
    window_seconds: float,
    baseline: list[float] | None = None,
) -> ChannelRegime:
    """Read one channel's regime over a window the caller declares.

    `values` are the in-window samples, oldest first (ordering matters --
    `refresh_state` reads successive ratios). `baseline` is an optional longer
    series for the drift comparison; when omitted, `drift` is None rather than
    a fabricated 0.0.
    """
    n = len(values)
    if n == 0:
        return ChannelRegime(
            channel=channel,
            window_seconds=window_seconds,
            sample_count=0,
            level=0.0,
            dispersion=0.0,
            level_percentile=None,
            dispersion_ratio=None,
            drift=None,
            saturation=0.0,
            refresh_state="insufficient_samples",
            regime="insufficient_samples",
        )

    level = statistics.median(values)
    dispersion = statistics.pstdev(values) if n > 1 else 0.0
    saturation = sum(
        1
        for v in values
        if abs(v) < RAIL_EPSILON or abs(v - 1.0) < RAIL_EPSILON
    ) / n

    drift: float | None = None
    level_percentile: float | None = None
    dispersion_ratio: float | None = None
    if baseline and len(baseline) > 1:
        base_sd = statistics.pstdev(baseline)
        level_percentile = sum(1 for b in baseline if b < level) / len(baseline)
        if base_sd > SUBNORMAL_CUTOFF:
            drift = abs(statistics.fmean(values) - statistics.fmean(baseline)) / base_sd
            dispersion_ratio = dispersion / base_sd
        else:
            # A baseline with no spread cannot say whether this window is
            # steadier than usual. None, not a divide-by-near-zero.
            dispersion_ratio = None

    refresh_state = _refresh_state(values)
    if n < MIN_REGIME_SAMPLES:
        # refresh_state needs only 2 points and stays readable; dispersion and
        # drift do not mean anything yet, so the composed label refuses rather
        # than reporting a regime derived from artifacts.
        regime = "insufficient_samples"
    else:
        regime = _compose(level, dispersion, saturation, refresh_state)

    return ChannelRegime(
        channel=channel,
        window_seconds=window_seconds,
        sample_count=n,
        level=level,
        dispersion=dispersion,
        level_percentile=level_percentile,
        dispersion_ratio=dispersion_ratio,
        drift=drift,
        saturation=saturation,
        refresh_state=refresh_state,
        regime=regime,
    )
