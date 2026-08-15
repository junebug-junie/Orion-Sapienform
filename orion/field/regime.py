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
because a single label cannot carry several independent facts, and the
collapse loses exactly the distinction that keeps being asked for:

1. **Level and dispersion are one axis there, two here.** A channel sitting at
   0.813 with dispersion 0.001 and one idling at 0.02 with dispersion 0.001
   both return `quiet`. "Very busy at near max, but steady, so actually
   peaceful" is not expressible in one word.

2. **There is no window there.** It takes a bare `list[float]`. The caller
   decides what window that is and nobody records it. Live cadence on
   `substrate_field_state` is a 2.0417s median tick (measured over 125,236
   ticks, 2026-08-11 -> 2026-08-14), so a "600-tick window" silently means
   ~20 minutes -- a fact that appeared nowhere. Windows are declared in
   SECONDS here and carried on the result.

3. **`dead` conflates a decayed remnant with a legitimately idle producer.**
   Its docstring says so outright: the folded-away, unproduced, and
   subnormal-noise cases "mean the same thing". They do not.
   `repair_pressure` and `conversation_load` are chat-gated (glossary:
   "happening in chat", "turn volume/complexity"). With nobody talking to
   Orion they decay toward zero and read `dead` -- CORRECT behavior, not a
   defect. Calling it dead cries wolf on every idle channel. `refresh_state`
   is its own axis here and makes NO claim about whether silence is right:
   that depends on whether the producer had reason to fire, which the series
   cannot tell you.

REFRESH STATE: PREFER THE TIMESTAMP, INFER ONLY AS FALLBACK
-----------------------------------------------------------
`FieldStateV1.node_vector_updated_at` is a real, live-populated per-node,
per-channel write timestamp -- `apply_decay()` already reads it for this exact
fresh-vs-stale question. 74 of 149 live node channels carry one (capability
vectors carry none; there is no `capability_vector_updated_at`).

When timestamps are supplied, `refresh_state` is AUTHORITATIVE and
`refresh_evidence == "timestamp"`. Only without them does this module fall
back to inferring from value ratios, and that inference has a hard limit:

**Ratio inference does not work in the subnormal range.** Below roughly
5e-318, float quantization (ulp fixed at 4.94e-324) makes the relative error
of `b / a` exceed `DECAY_RATIO_EPSILON`, so a genuine 0.92 decay reads
`producer_written`. Measured: a decay starting at 1e-317 is detected, one
starting at 1e-318 is not (observed step ratios 0.91996, 0.919979, 0.920023).
Seven live channels transit that band -- avg_step_chars_pressure,
conversation_load, field_coherence_warning, harness_step_load,
reliability_pressure, repair_pressure, tool_failure_streak_pressure -- and it
is precisely the TERMINAL phase of the `node:substrate.route` incident this
detector cites ("went subnormal, and was heading to exactly 0.0"). It fails in
the dangerous direction: it says a producer wrote this.

An earlier draft of this docstring claimed the two social channels "stepped at
an exact 0.92000 ratio for 42,000+ transitions". The real counts are 41,762 of
42,141 and 41,725 of 42,106 -- 379 and 381 steps were NOT at 0.92, and those
exceptions ARE this limitation. The measurement that would have caught it was
run, and the residue was rounded away into a "+". Kept visible here on
purpose.

POLARITY
--------
`collect_field_channel_pressures()` does NOT polarity-correct. It only flips
the MERGE direction (`min()` instead of `max()`) for
`HIGHER_IS_BETTER_CHANNELS`; the values keep their native higher-is-better
sense. The `1 - value` inversion lives only in
`orion/attention/field_attention/selectors.py::_current_pressure_proxy` and
never touches the pressure dict.

An earlier version of this module asserted the opposite and composed labels
directly on raw level. Live consequence: `confidence` (median 0.8678,
meaning HEALTHY) came out `loaded_volatile` in 159 of 208 windows, and
`available_capacity` in 142 of 208. This module now corrects polarity itself,
by channel name, into `pressure_equivalent_level` -- raw `level` stays native
so a consumer reading it gets the real number.

ON CONSTANTS
------------
`SUBNORMAL_CUTOFF` is reused from `channel_glossary` because "is this
functionally zero" is scale-free and transfers. The level and dispersion
thresholds are NOT reused -- see their definitions for the live cases that
forced the split, and for which of them is derived and which is merely
declared. Borrowing a constant across domains without re-verifying it at the
new scale is a known, repeated failure mode in this repo, and the first draft
of this module committed it twice.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

from orion.field.channel_glossary import SUBNORMAL_CUTOFF
from orion.field.pressure import HIGHER_IS_BETTER_CHANNELS, _aware_utc

# services/orion-field-digester/app/digestion/decay.py applies this to
# NODE_DECAY_CHANNELS/CAPABILITY_DECAY_CHANNELS.
#
# CONFIG DRIFT RISK, stated rather than hidden: the producer reads this from
# BIOMETRICS_FIELD_DECAY_RATE (services/orion-field-digester/settings.py,
# docker-compose.yml `${BIOMETRICS_FIELD_DECAY_RATE:-0.92}`, .env_example).
# If an operator changes it, ratio inference silently degrades to "always
# producer_written" with no failing gate. The timestamp path above is immune,
# which is another reason to prefer it.
KNOWN_DECAY_RATES: tuple[float, ...] = (0.92,)

# Successive-ratio tolerance for calling a step "decay", used ONLY by the
# fallback path. Normal-range float multiplication by a fixed constant
# reproduces to ~1e-12; live subnormal steps deviate by up to ~4e-6 and by
# ~8.6e-3 near the floor, which is exactly why the subnormal carve-out above
# exists rather than this value being loosened to cover it. Loosening it to
# 0.05 would make any ratio in [0.87, 0.97] read as decay.
DECAY_RATIO_EPSILON: float = 1e-6

# Minimum number of CHANGES before the fallback will assert `decay_only`.
# channel_glossary.py:109-114 already established this convention
# (RATCHET_MIN_SAMPLES = 4, because a 2-point monotonicity claim "is a coin
# flip, not a monotonicity signal"); an earlier draft of this module did not
# carry that lesson across and would call a single 1.0 -> 0.92 step -- an
# ordinary two-decimal move, and many channels sit pinned at 1.0 -- a
# confident `decay_only`.
MIN_DECAY_CHANGES: int = 4

# A value within this of either rail counts as saturated at that rail.
RAIL_EPSILON: float = 1e-3

# Below this many samples, dispersion and drift are artifacts rather than
# readings, and are reported as None rather than as plausible floats.
MIN_REGIME_SAMPLES: int = 8

# DECLARED, not derived -- and the earlier draft's claim that it WAS derived
# is retracted here.
#
# That draft measured one 600-sample window, saw dispersion values 0.0,
# 0.000046, 0.000447, 0.000525 then a jump to 0.0449+, and called the gap
# structural. Re-measured across all 208 non-overlapping 600-sample windows in
# the full 3-day history: 206 of 208 windows have at least one channel inside
# that supposedly-empty region. disk_pressure lands in it in 146 windows and
# its steady/volatile verdict at this threshold is a coin flip across windows
# (106 steady / 102 volatile); reliability_pressure 108/100.
#
# So there is NO natural gap, the single-window measurement was an alignment
# artifact, and this is a convention. Channels whose dispersion sits near it
# will flip between adjacent windows -- read `dispersion` and
# `dispersion_ratio` directly rather than trusting the label at the boundary.
STEADY_DISPERSION: float = 0.02

# DECLARED convention. Reads as "in the top 30% of the available range" on the
# polarity-corrected [0,1] pressure scale. Levels ran 0.6906, 0.7107, 0.7335,
# 0.8125 with no space between them, so there is no natural boundary here
# either; the raw `level` is on the result for exactly that reason.
LOADED_LEVEL: float = 0.70


@dataclass(frozen=True)
class ChannelRegime:
    """One channel's regime over one declared window.

    Every field is a separate reading. `regime` is a convenience composition
    of the others, NOT a substitute for them -- a consumer that reads only
    `regime` has re-created the single-label collapse this module exists to
    undo.
    """

    channel: str
    # DECLARED, never inferred. The caller states what window it sampled.
    window_seconds: float
    sample_count: int
    # Median over the window, in the channel's NATIVE direction. For a
    # HIGHER_IS_BETTER channel a high value here means healthy.
    level: float
    # `level` mapped so that high always means "more pressure" (1 - level for
    # HIGHER_IS_BETTER_CHANNELS). This, not `level`, is what "loaded" is
    # computed from.
    pressure_equivalent_level: float
    polarity_inverted: bool
    # Population stdev. None below MIN_REGIME_SAMPLES: an artifact, not a
    # reading.
    dispersion: float | None
    # Relative readings, None without a baseline -- explicitly not 0.0/0.5,
    # which would read as "no drift"/"perfectly typical".
    level_percentile: float | None
    dispersion_ratio: float | None
    drift: float | None
    # Split by rail, because one float cannot say WHICH bound. For a pressure
    # channel the low rail is the calm, healthy state and the high rail is the
    # alarming one; for a HIGHER_IS_BETTER channel that inverts. Live, 495 of
    # 505 `saturated` labels from a combined-rail draft were the CALM rail --
    # failure_pressure reporting no failures at all got the most alarming word
    # in the vocabulary.
    saturation_low: float
    saturation_high: float
    refresh_state: str
    # "timestamp" (authoritative) or "value_ratio" (inferred, and blind in the
    # subnormal range -- see module docstring).
    refresh_evidence: str
    regime: str


# A NEGATIVE verdict ("nobody wrote this") needs the stamps to be
# representative of the window; a POSITIVE one does not. See
# _refresh_from_timestamps for why the two directions get different bars.
MIN_STAMP_COVERAGE: float = 0.5


def _refresh_from_timestamps(
    stamps: list[datetime | None], window_start: datetime | None = None
) -> str:
    """Authoritative path: is the newest producer write INSIDE the window?

    Three rules were tried here and the first two were both wrong:

    1. `len(set(real)) > 1` -- "are the stamps different". Correct for a
       single-source series; wrong the moment the Hub supplied stamps for a
       MERGED channel, because the winner changes between ticks and two nodes
       each frozen at their own old write time yield two distinct stamps.
    2. "does the newest stamp advance" -- fixed that, and introduced worse:
       a producer that wrote EXACTLY ONCE in the window is structurally
       undetectable, since one distinct stamp can never show an advance.
       Confirmed live: `execution_friction` carried 2 stamped samples out of
       437, both at 23:58:29 -- a real write, inside the window -- and this
       rule reported `no_write_in_window` and flagged it AUTHORITATIVE. A
       confidently wrong verdict is worse than the value-ratio blindness the
       timestamp path exists to replace. Same shape as CLAUDE.md 0A's
       metric-gate item 4, inverted: structurally incapable of reading one
       write.

    3. What it does now: compare the newest stamp against the window's start.
       That is the question being asked, needs no second stamp, and cannot be
       fooled by winner churn.

    The two directions carry DIFFERENT evidential bars, deliberately:

    - `producer_written` needs ONE stamp inside the window. A single real
      observation of a write is proof a write happened.
    - `no_write_in_window` needs stamps on at least MIN_STAMP_COVERAGE of the
      samples. Absence cannot be concluded from 0.5% of the series -- the
      other 99.5% are unstamped, not silent. Below the floor this returns
      "unknown" and the caller falls back to inference, which is the honest
      answer rather than a confident one.

    Without `window_start` only the negative direction is computable, so a
    caller that cannot supply it gets "unknown" rather than a guess.
    """
    real = [_aware_utc(s) for s in stamps if s is not None]
    if not real or window_start is None:
        return "unknown"
    if max(real) >= _aware_utc(window_start):
        return "producer_written"
    if len(real) / len(stamps) < MIN_STAMP_COVERAGE:
        return "unknown"
    return "no_write_in_window"


def _refresh_from_values(values: list[float]) -> str:
    """Fallback inference. See the module docstring for its subnormal blind
    spot; this cannot be fixed by loosening DECAY_RATIO_EPSILON without
    absorbing real producer writes.

    A series is `decay_only` only when it has at least MIN_DECAY_CHANGES
    changes and EVERY change is a known-rate multiplication -- one genuine
    producer write disqualifies it, which is the conservative direction, since
    mislabelling a live channel as decay-only is the damaging error.
    """
    if len(values) < 2:
        return "insufficient_samples"
    changes = [(a, b) for a, b in zip(values, values[1:]) if a != b]
    if not changes:
        return "static"
    if len(changes) < MIN_DECAY_CHANGES:
        return "producer_written"
    for a, b in changes:
        if a <= 0.0:
            return "producer_written"
        ratio = b / a
        if not any(abs(ratio - r) < DECAY_RATIO_EPSILON for r in KNOWN_DECAY_RATES):
            return "producer_written"
    return "decay_only"


def _compose(
    pressure_level: float,
    dispersion: float | None,
    saturation_low: float,
    saturation_high: float,
    refresh_state: str,
    polarity_inverted: bool,
) -> str:
    """Compose the axes into one label, for surfaces that can only show one.

    Refresh state dominates: a number nothing wrote is not a reading about the
    world regardless of its level.
    """
    if refresh_state == "insufficient_samples" or dispersion is None:
        return "insufficient_samples"
    if refresh_state in ("decay_only", "no_write_in_window", "static"):
        # Named for what is observed, not for a conclusion. `static` in
        # particular does NOT mean "nothing wrote it" -- see the note on
        # refresh_state below.
        return "no_new_input"
    # Rails, in pressure-equivalent terms.
    pinned_max = saturation_high > 0.5 if not polarity_inverted else saturation_low > 0.5
    pinned_min = saturation_low > 0.5 if not polarity_inverted else saturation_high > 0.5
    if pinned_max:
        return "pinned_max"
    if pinned_min:
        # The floor. Uninformative, but for a pressure channel it is also the
        # healthiest possible reading -- deliberately not called "saturated".
        return "pinned_min"
    steady = dispersion <= STEADY_DISPERSION
    loaded = pressure_level >= LOADED_LEVEL
    if loaded and steady:
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
    updated_at: list[datetime | None] | None = None,
    window_start: datetime | None = None,
) -> ChannelRegime:
    """Read one channel's regime over a window the caller declares.

    `values` are in-window samples, oldest first (ordering matters).
    `baseline` is an optional longer series for the relative readings.
    `updated_at` is the per-sample producer write timestamp from
    `FieldStateV1.node_vector_updated_at`; supply it whenever available -- it
    makes `refresh_state` authoritative instead of inferred. `window_start` is
    the time the window opens, and BOTH are needed: without it the timestamp
    path can only ever say "unknown", because "was there a write in this
    window" is not answerable from the stamps alone. See
    _refresh_from_timestamps for why an earlier version that tried was wrong.
    """
    n = len(values)
    inverted = channel in HIGHER_IS_BETTER_CHANNELS
    if n == 0:
        return ChannelRegime(
            channel=channel,
            window_seconds=window_seconds,
            sample_count=0,
            level=0.0,
            pressure_equivalent_level=0.0,
            polarity_inverted=inverted,
            dispersion=None,
            level_percentile=None,
            dispersion_ratio=None,
            drift=None,
            saturation_low=0.0,
            saturation_high=0.0,
            refresh_state="insufficient_samples",
            refresh_evidence="value_ratio",
            regime="insufficient_samples",
        )

    level = statistics.median(values)
    pressure_level = (1.0 - level) if inverted else level
    dispersion = statistics.pstdev(values) if n >= MIN_REGIME_SAMPLES else None
    saturation_low = sum(1 for v in values if abs(v) < RAIL_EPSILON) / n
    saturation_high = sum(1 for v in values if abs(v - 1.0) < RAIL_EPSILON) / n

    drift: float | None = None
    level_percentile: float | None = None
    dispersion_ratio: float | None = None
    if baseline and len(baseline) >= MIN_REGIME_SAMPLES:
        base_sd = statistics.pstdev(baseline)
        level_percentile = sum(1 for b in baseline if b < level) / len(baseline)
        if base_sd > SUBNORMAL_CUTOFF:
            drift = abs(statistics.fmean(values) - statistics.fmean(baseline)) / base_sd
            if dispersion is not None:
                dispersion_ratio = dispersion / base_sd

    refresh_evidence = "value_ratio"
    refresh_state = _refresh_from_values(values)
    if updated_at:
        from_stamps = _refresh_from_timestamps(updated_at, window_start)
        if from_stamps != "unknown":
            refresh_state = from_stamps
            refresh_evidence = "timestamp"

    regime = _compose(
        pressure_level,
        dispersion,
        saturation_low,
        saturation_high,
        refresh_state,
        inverted,
    )

    return ChannelRegime(
        channel=channel,
        window_seconds=window_seconds,
        sample_count=n,
        level=level,
        pressure_equivalent_level=pressure_level,
        polarity_inverted=inverted,
        dispersion=dispersion,
        level_percentile=level_percentile,
        dispersion_ratio=dispersion_ratio,
        drift=drift,
        saturation_low=saturation_low,
        saturation_high=saturation_high,
        refresh_state=refresh_state,
        refresh_evidence=refresh_evidence,
        regime=regime,
    )
