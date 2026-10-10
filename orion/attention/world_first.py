"""World-first attention: the world by default, the body only when it is unusual.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md,
"A. The attention seam" and "Decisions (Juniper, 2026-10-10)".

Before this module every attention contest looked inward and always crowned a
winner: the field contest min-max rescaled five hardcoded prediction-error
nodes so the top one read 1.0 on 124,181 of 124,181 frames, 49% of them calm
(raw error < 0.05). A creature's attention points at the world by default and
turns inward only when the body hurts. This module is that rule, and nothing
else:

- **External** candidates (the world) are eligible when they are not absent,
  have enough history, are fresh, and are busy for themselves: band ``high``
  or above (``EXTERNAL_ELIGIBLE_BANDS``). Review 2026-10-10: a ``usual``
  floor reduced to "value > 0" for a mostly-zero source -- camera surprise is
  nonzero 26% of the time, flat around the clock, and won 24% of replayed
  ticks on readings as small as 5e-05. Busy means the top decile of its own
  week, not above its median.
- **Internal** candidates (the body) are eligible ONLY at band ``high`` or
  ``unusual`` in their bad direction. The direction comes from the semantic
  layer (``orion.metrics.semantics.derived_channel_polarity`` over the
  glossary's ``value_kind``), never invented here. A placeholder or bucket is
  never eligible: it is not a measurement.
- Eligible candidates rank by their bad-direction percentile against their
  own history -- one scale for both kinds, so no exchange rate. Ranking uses
  the MID-RANK percentile (share below + half the share equal) when the
  builder had the history: a signal pinned at its ceiling (execution at 1.0
  is 3.5% of its week) otherwise caps at ~0.96 and loses to any rare-fire
  node. Eligibility keeps the strict-below percentile, so an all-zero
  history with a current 0 still reads as rest. A tie is broken by the
  caller's secondary score (Borda in the broadcast), then id.
- An empty eligible set is an explicit **no-winner** result. A calm body
  stays silent.

- **Event-written** sources (2026-10-10, Juniper: "let event type signals
  decay"): a source the semantic layer marks as written once per event and
  carried forward until the next one is news only for
  ``EVENT_ORIENTING_WINDOW_SEC`` after the event that wrote it (its reading's
  ``age_sec``). Inside the window it is judged as usual; after
  ``EVENT_FADE_GRACE_SEC`` its rank fades linearly to the rest percentile (0:
  every selected source rests at 0.0 on a mostly-zero week); at the window
  it stops competing even though the carried value is unchanged. A new event re-arms it, so a storm that keeps
  writing stays eligible. Live case: one codebase event (0.988, 05:51:03)
  held every field frame for 18 minutes because the value sat in the top
  percentile of a mostly-zero week until the next poll wrote 0.

Band cut points are ``prediction_error_magnitude.DEFAULT_BAND_CUTS`` -- knobs
to be graded on live data, not findings.

``ATTENTION_WORLD_FIRST_ENABLED`` (default on, Juniper's ship-on rule; read by
each runtime's settings) turns this on in both background contests; false
restores the previous RANKING exactly. Three downstream bug fixes shipped with
it are not flag-gated (an empty coalition never "activates", reverie skips a
no-winner tick, the self-model's no-winner narrative): they are correct with
the flag off too.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol, Sequence

from orion.schemas.attention_candidate import (
    PERCEPTION_NODE_ID,
    WORLD_CHAT_SOURCE_ID,
    AttentionCandidateV1,
    is_world_source_id,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_freshness import PE_STALENESS_HORIZON_SEC
from orion.substrate.prediction_error_magnitude import (
    DEFAULT_BAND_CUTS,
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

logger = logging.getLogger(__name__)

# Marker carried in existing free-form fields (FieldAttentionTargetV1.
# evidence_refs, AttentionSignalV1/OpenLoopV1.provenance) -- step 1 of the
# schema rollout needs no change to any extra="forbid" model.
SOURCE_KIND_REF_PREFIX = "source_kind:"
SOURCE_KIND_KEY = "source_kind"

SUBSTRATE_NODE_PREFIX = "node:substrate."

# Internal readings older than this are not current: the same 1800 s horizon
# the self-model and endogenous curiosity use
# (orion/substrate/prediction_error_freshness.py), imported so it cannot drift.
INTERNAL_MAX_AGE_SEC: float = PE_STALENESS_HORIZON_SEC

# Chat activity: Juniper turns in a trailing window, scored against the same
# windowed count sampled every minute over the last 7 days (Juniper's
# 2026-10-10 decision: chat's "unusual" is its rate against its own 7 days).
CHAT_RATE_WINDOW = timedelta(minutes=15)
CHAT_RATE_GRID_STEP = timedelta(minutes=1)
# Fewer real turns than this in 7 days and "busier than usual" has nothing
# to stand on: band says insufficient_history instead of guessing.
CHAT_MIN_TURNS_7D = 5

# Perception's tick writes every ~10 s; a reading older than this is stale.
PERCEPTION_MAX_AGE_SEC: float = 180.0

# How long one event's reading stays news. Derived 2026-10-10 from 7 days of
# substrate_node_prediction_error_history (225 isolated onsets -- top-decile,
# nonzero, no prior onset on the same node within 30 min -- across the four
# event-written nodes): the body's response in node:substrate.biometrics'
# own percentile is +0.087 (0-60 s), +0.067 (60-120), +0.040 (120-180),
# +0.054 (180-240), then +0.029 +/- 0.037 (240-300, no longer distinguishable
# from zero) and -0.006 (300-420); bus_synaptic is back by 120 s. The source's
# own next reading is back at its pre-event level by 60-180 s (execution
# 0.823 -> 0.203 vs 0.210 before; chat 0.944 -> 0.046; route 0.880 -> 0.161
# by 180-300 s). So one event's consequence is gone by 300 s. A sustained
# storm still re-arms well inside it: after an elevated execution reading the
# next write lands at p50 68 s / p90 160 s. Shorter than, and so tighter
# than, the 1800 s PE staleness horizon, which stays the outer bound.
EVENT_ORIENTING_WINDOW_SEC: float = 300.0
# Full strength for the first 60 s, then a linear fade to 0 at the window.
# Why a grace at all: a per-tick level competitor's reading is itself up to
# one write interval old and is never faded -- p90 gap between writes over the
# same 7 days: biometrics 62.6 s, bus_synaptic 60.7 s, perception 60.5 s,
# cabinet 60.6 s. Fading an event reading inside that interval penalises it
# for an age its rivals carry for free; with no grace a fresh execution spike
# lost near-ties it had won before (replay: 47 of 86 spikes won vs 60 of 86
# without decay). The biometrics response is also strongest in 0-60 s.
EVENT_FADE_GRACE_SEC: float = 60.0

_INTERNAL_ELIGIBLE_BANDS = frozenset({"high", "unusual"})
EXTERNAL_ELIGIBLE_BANDS = frozenset({"high", "unusual"})
_NEVER_A_MEASUREMENT = frozenset({"placeholder", "bucket"})


class AttentionCandidateSource(Protocol):
    """Shaped like ``AttentionSignalDetector`` (orion/substrate/attention/
    detectors/base.py): one source, one call per tick, never raises."""

    source_id: str

    def candidates(self, now: datetime) -> list[AttentionCandidateV1]:
        ...


# ---------------------------------------------------------------------------
# Semantic layer: value_kind and polarity, read, never invented.
# ---------------------------------------------------------------------------


_SEMANTICS_CACHE: dict[str, tuple[str | None, str | None]] = {}


def node_prediction_error_semantics(node_id: str) -> tuple[str | None, str | None]:
    """(value_kind, polarity) for a node's prediction_error, from the glossary.

    Polarity is ``derived_channel_polarity("prediction_error", value_kind)``
    -- the same derivation the metric lock uses (PR #2579): prediction_error
    is in PRESSURE_CHANNELS, so higher is worse, except a ``trigger`` whose
    value is "it fired" and has no polarity. An unreadable glossary yields
    value_kind None and the channel's derived polarity. Only a SUCCESSFUL
    read is cached: a transient glossary error must not pin value_kind None
    (placeholder/trigger nodes would then be ranked as levels) until restart.
    """
    cached = _SEMANTICS_CACHE.get(node_id)
    if cached is not None:
        return cached
    from orion.metrics.semantics import derived_channel_polarity

    value_kind: str | None = None
    try:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry("prediction_error", node=node_id)
        if entry is not None and entry.node == node_id:
            value_kind = dict(entry.semantics).get("value_kind")
    except Exception as exc:  # noqa: BLE001 -- a missing file must not stop attention
        logger.warning("world_first_glossary_unreadable node_id=%s err=%s", node_id, exc)
        return None, derived_channel_polarity("prediction_error", None)
    result = (value_kind, derived_channel_polarity("prediction_error", value_kind))
    _SEMANTICS_CACHE[node_id] = result
    return result


def event_fade(age_sec: float, window_sec: float, grace_sec: float | None = None) -> float:
    """1.0 through the grace period, then linear to 0 at the window."""
    grace = EVENT_FADE_GRACE_SEC if grace_sec is None else grace_sec
    grace = max(0.0, min(grace, window_sec))
    if age_sec <= grace:
        return 1.0
    if window_sec <= grace:
        return 0.0
    return max(0.0, 1.0 - (age_sec - grace) / (window_sec - grace))


_EVENT_WRITTEN_CACHE: dict[str, bool] = {}

# A reading only when an event happens (orion/metrics/semantics.py SPARSITIES).
_EVENT_SPARSITY = "event_gated"
# designed_sparse wins over event_gated when the zeros are the point, and the
# write cadence then goes in absent_means (the precedence rule stated with
# SPARSITIES in orion/metrics/semantics.py). Every such entry that is written
# per event says so as "only written ..." (execution, chat, codebase, route);
# per-tick writers say "written as 0.0 every tick" (perception) or describe a
# skipped tick (cabinet). "carried forward" alone is NOT the test: per-tick
# biometrics and bus_synaptic are carried forward between ticks too.
_EVENT_CADENCE_PREFIX = "only written"


def prediction_error_is_event_written(sparsity: str | None, absent_means: str | None) -> bool:
    """True when the semantic layer says a reading is written once per event
    and carried forward until the next one (so its age is the event's age)."""
    if sparsity == _EVENT_SPARSITY:
        return True
    if sparsity == "designed_sparse":
        return str(absent_means or "").strip().lower().startswith(_EVENT_CADENCE_PREFIX)
    return False


def node_prediction_error_event_written(node_id: str) -> bool:
    """``prediction_error_is_event_written`` over the node's glossary entry.

    An unreadable glossary reads False (no decay: the pre-decay behaviour,
    never a silently dropped source) and is not cached, same rule as
    ``node_prediction_error_semantics``."""
    cached = _EVENT_WRITTEN_CACHE.get(node_id)
    if cached is not None:
        return cached
    try:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry("prediction_error", node=node_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("world_first_glossary_unreadable node_id=%s err=%s", node_id, exc)
        return False
    sem = dict(entry.semantics) if entry is not None and entry.node == node_id else {}
    result = prediction_error_is_event_written(sem.get("sparsity"), sem.get("absent_means"))
    _EVENT_WRITTEN_CACHE[node_id] = result
    return result


def bad_direction_percentile(
    candidate: AttentionCandidateV1, *, for_ranking: bool = False
) -> float | None:
    """Percentile in the candidate's bad direction (higher = more alarming
    for the body, busier for the world). None when there is no reading or no
    declared direction. ``for_ranking`` uses the mid-rank percentile when the
    builder supplied one (ceiling ties); eligibility uses strict-below."""
    pct = candidate.unusualness.percentile_now
    if for_ranking and candidate.rank_percentile is not None:
        pct = candidate.rank_percentile
    if pct is None:
        return None
    if candidate.source_kind == "external":
        return float(pct)
    if candidate.value_kind == "trigger":
        # A trigger's value is "it fired"; firing is the event.
        return float(pct)
    if candidate.polarity == "higher_is_worse":
        return float(pct)
    if candidate.polarity == "higher_is_better":
        return 1.0 - float(pct)
    return None


def _band_for(pct: float, cuts: tuple[float, float, float]) -> str:
    quiet_cut, usual_cut, high_cut = cuts
    if pct < quiet_cut:
        return "quiet"
    if pct < usual_cut:
        return "usual"
    if pct < high_cut:
        return "high"
    return "unusual"


@dataclass(frozen=True)
class CandidateVerdict:
    candidate: AttentionCandidateV1
    eligible: bool
    reason: str
    score: float | None  # bad-direction percentile (strict-below): the band
    band: str
    rank_score: float | None = None  # mid-rank when available: the ranking key
    # Event-written sources only: seconds since the event that wrote the
    # reading, and the linear fade (1.0 at onset, 0 at the window) applied
    # to rank_score and salience. 1.0 / None for everything else.
    event_decay: float = 1.0
    event_age_sec: float | None = None

    @property
    def salience(self) -> float:
        """Band percentile faded by event age: what a frame reports as the
        target's strength. Equal to ``score`` for non-event sources."""
        return float(self.score or 0.0) * self.event_decay

    def trace(self) -> dict:
        mag = self.candidate.unusualness
        return {
            "source_id": self.candidate.source_id,
            "source_kind": self.candidate.source_kind,
            "eligible": self.eligible,
            "reason": self.reason,
            "score": None if self.score is None else round(self.score, 6),
            "rank_score": None if self.rank_score is None else round(self.rank_score, 6),
            "band": self.band,
            "value": mag.value,
            "percentile_now": mag.percentile_now,
            "n_readings_7d": mag.n_readings_7d,
            "age_sec": mag.age_sec,
            "absent": self.candidate.absent,
            "value_kind": self.candidate.value_kind,
            "polarity": self.candidate.polarity,
            "event_window_sec": self.candidate.event_window_sec,
            "event_age_sec": self.event_age_sec,
            "event_decay": round(self.event_decay, 6),
        }


def judge_candidate(
    candidate: AttentionCandidateV1,
    *,
    band_cuts: tuple[float, float, float] = DEFAULT_BAND_CUTS,
    internal_max_age_sec: float = INTERNAL_MAX_AGE_SEC,
) -> CandidateVerdict:
    """The whole eligibility rule for one candidate, with a stated reason."""
    mag = candidate.unusualness
    if candidate.absent:
        return CandidateVerdict(
            candidate, False, f"absent: {candidate.absent_reason or 'no reading'}", None, "absent"
        )
    if mag.band == "insufficient_history" or mag.percentile_now is None:
        return CandidateVerdict(
            candidate, False, "insufficient_history", None, "insufficient_history"
        )
    if candidate.source_kind == "internal":
        if candidate.value_kind in _NEVER_A_MEASUREMENT:
            return CandidateVerdict(
                candidate, False, f"{candidate.value_kind}: not a measurement", None, mag.band
            )
        if mag.age_sec > internal_max_age_sec:
            return CandidateVerdict(
                candidate, False, f"stale: reading {mag.age_sec:.0f}s old", None, mag.band
            )
    score = bad_direction_percentile(candidate)
    if score is None:
        return CandidateVerdict(
            candidate, False, "no declared polarity in the semantic layer", None, mag.band
        )
    band = _band_for(score, band_cuts)
    rank_score = bad_direction_percentile(candidate, for_ranking=True)
    decay, event_age, suffix = 1.0, None, ""
    window = candidate.event_window_sec
    if window is not None:
        # The reading is carried forward unchanged between events, so its age
        # is the event's age. Past the window it is old news, whatever its
        # percentile: one event orients, it does not hold.
        event_age = float(mag.age_sec)
        if event_age >= window:
            return CandidateVerdict(
                candidate, False,
                f"event {event_age:.0f}s old: past the {window:.0f}s orienting window "
                f"(value carried forward since)",
                score, band, None, 0.0, event_age,
            )
        decay = event_fade(event_age, window)
        if rank_score is not None:
            rank_score *= decay
        suffix = f", event {event_age:.0f}s old (fade {decay:.2f})"
    if candidate.source_kind == "internal":
        if band in _INTERNAL_ELIGIBLE_BANDS:
            return CandidateVerdict(
                candidate, True, f"body unusual for itself ({band}){suffix}", score, band,
                rank_score, decay, event_age,
            )
        return CandidateVerdict(
            candidate, False, f"body {band} for itself", score, band, None, decay, event_age
        )
    if band in EXTERNAL_ELIGIBLE_BANDS:
        return CandidateVerdict(
            candidate, True, f"world busy for itself ({band}){suffix}", score, band,
            rank_score, decay, event_age,
        )
    return CandidateVerdict(
        candidate, False, f"world {band} for itself: not busy", score, band, None, decay, event_age
    )


@dataclass
class WorldFirstRanking:
    eligible: list[CandidateVerdict] = field(default_factory=list)  # best first
    ineligible: list[CandidateVerdict] = field(default_factory=list)

    @property
    def winner(self) -> CandidateVerdict | None:
        return self.eligible[0] if self.eligible else None

    @property
    def no_winner(self) -> bool:
        return not self.eligible

    def trace(self) -> dict:
        return {
            "enabled": True,
            "no_winner": self.no_winner,
            "winner": self.winner.candidate.source_id if self.winner else None,
            "winner_source_kind": self.winner.candidate.source_kind if self.winner else None,
            # A no-winner tick where sources could not be read is NOT a calm
            # tick; consumers can tell the two apart from this list.
            "absent_sources": sorted(
                v.candidate.source_id for v in self.ineligible if v.candidate.absent
            ),
            "candidates": [v.trace() for v in (*self.eligible, *self.ineligible)],
        }


def rank_candidates(
    candidates: Iterable[AttentionCandidateV1],
    *,
    tie_break: dict[str, float] | None = None,
    band_cuts: tuple[float, float, float] = DEFAULT_BAND_CUTS,
    internal_max_age_sec: float = INTERNAL_MAX_AGE_SEC,
) -> WorldFirstRanking:
    """Judge every candidate; rank the eligible ones by bad-direction
    percentile, then ``tie_break[source_id]`` (Borda in the broadcast), then
    source_id for determinism. Empty eligible set -> ``no_winner``."""
    ranking = WorldFirstRanking()
    tb = tie_break or {}
    for cand in candidates:
        verdict = judge_candidate(
            cand, band_cuts=band_cuts, internal_max_age_sec=internal_max_age_sec
        )
        (ranking.eligible if verdict.eligible else ranking.ineligible).append(verdict)
    ranking.eligible.sort(
        key=lambda v: (
            -(v.rank_score if v.rank_score is not None else v.salience),
            -tb.get(v.candidate.source_id, 0.0),
            v.candidate.source_id,
        )
    )
    return ranking


# ---------------------------------------------------------------------------
# Candidate builders (pure). Callers do the I/O.
# ---------------------------------------------------------------------------


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def source_kind_for_node(node_id: str) -> str:
    """Camera surprise is the world; every other substrate node is the body."""
    return "external" if is_world_source_id(node_id) else "internal"


def midrank_percentile(value: float, values: Sequence[float]) -> float | None:
    """Share of ``values`` strictly below ``value`` plus half the share equal
    to it. None for an empty history."""
    if not values:
        return None
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return (below + 0.5 * equal) / len(values)


def perception_absent_reason(
    *,
    embedding_staleness: float | None = None,
    vision_frame_staleness: float | None = None,
    vision_measured: bool = True,
) -> str | None:
    """ONE definition of "the camera cannot see right now", shared by both
    contests (review 2026-10-10: they used different signals and could
    disagree). Perception writes 0.0 while stale or warming (glossary), so
    absence must come from staleness, never from the value. Each contest
    passes what it can see: the broadcast has the perception node's own
    ``embedding_staleness`` (and the vision organ node when present); the
    field contest has the vision organ's ``vision_frame_staleness``
    (``vision_measured=False`` when that key was dropped as unmeasured).
    Not caught by either: a scorer stuck at exact 0 while embeddings arrive
    (seen live 2026-10-04/05) -- that reads as quiet, never as a win."""
    for name, val in (
        ("camera embeddings stale (embedding_staleness", embedding_staleness),
        ("camera frames stale (vision_frame_staleness", vision_frame_staleness),
    ):
        if val is None:
            continue
        try:
            if float(val) >= 1.0:
                return f"{name} 1.0)"
        except (TypeError, ValueError):
            return "camera staleness unreadable"
    if not vision_measured:
        return "camera health unmeasured (vision_frame_staleness absent)"
    return None


def _missing_magnitude() -> PredictionErrorMagnitudeV1:
    return PredictionErrorMagnitudeV1(value=0.0, age_sec=0.0)


def node_candidate(
    *,
    node_id: str,
    label: str,
    magnitude: PredictionErrorMagnitudeV1 | None,
    observed_at: datetime | None,
    now: datetime,
    absent_reason: str | None = None,
    history_values: Sequence[float] | None = None,
    rank_percentile: float | None = None,
    event_decay: bool = True,
) -> AttentionCandidateV1:
    """A ``node:substrate.*`` prediction-error node as a candidate.

    ``event_decay`` (``ATTENTION_EVENT_DECAY_ENABLED``, default on): when the
    glossary marks the node event-written, the candidate carries
    ``event_window_sec`` and fades with the age of the event that wrote it.

    ``history_values`` (the node's 7-day readings) lets the candidate carry a
    mid-rank ``rank_percentile`` for ranking ceiling ties.

    ``absent_reason`` is the caller's evidence that the source cannot measure
    right now (perception: camera/embedding staleness). For perception the
    stored value is written as 0.0 while stale or warming (glossary), so
    absence must come from staleness, never from the value.
    """
    kind = source_kind_for_node(node_id)
    value_kind, polarity = node_prediction_error_semantics(node_id)
    if magnitude is None:
        absent_reason = absent_reason or "no magnitude (no stored history)"
    elif kind == "external" and magnitude.age_sec > PERCEPTION_MAX_AGE_SEC and not absent_reason:
        absent_reason = f"reading {magnitude.age_sec:.0f}s old"
    return AttentionCandidateV1(
        candidate_id=f"attention-candidate:{node_id}",
        source_id=node_id,
        source_kind=kind,  # type: ignore[arg-type]
        label=label,
        unusualness=magnitude if magnitude is not None else _missing_magnitude(),
        rank_percentile=(
            rank_percentile
            if rank_percentile is not None or magnitude is None or not history_values
            else midrank_percentile(float(magnitude.value), history_values)
        ),
        observed_at=observed_at,
        absent=absent_reason is not None,
        absent_reason=absent_reason,
        value_kind=value_kind,
        polarity=polarity,  # type: ignore[arg-type]
        event_window_sec=(
            EVENT_ORIENTING_WINDOW_SEC
            if event_decay and node_prediction_error_event_written(node_id)
            else None
        ),
        evidence_refs=[node_id],
    )


_CHAT_MEMO: dict[tuple, tuple[PredictionErrorMagnitudeV1, float | None]] = {}
_CHAT_MEMO_MAX = 8


def chat_rate_reading(
    turn_times: Sequence[datetime],
    *,
    now: datetime,
    window: timedelta = CHAT_RATE_WINDOW,
    grid_step: timedelta = CHAT_RATE_GRID_STEP,
    min_turns: int = CHAT_MIN_TURNS_7D,
) -> tuple[PredictionErrorMagnitudeV1, float | None]:
    """(magnitude, mid-rank percentile) for Juniper's turns in the trailing
    ``window``, against the same windowed count sampled every ``grid_step``
    over the last 7 days.

    Rest is reachable: no turn in the window reads 0, percentile 0.0, band
    ``quiet``. Fewer than ``min_turns`` real turns in 7 days -> band
    ``insufficient_history`` (the minute grid would otherwise always look
    like 10,080 readings).

    The 10,081-point grid is rebuilt at most once per minute per distinct
    (turns, value): the field contest calls this every ~2 s and only the
    reading's age changes in between (review 2026-10-10).
    """
    now = _aware(now)
    times = sorted(_aware(t) for t in turn_times if _aware(t) <= now)
    horizon = now - WINDOW_7D
    in_7d = [t for t in times if t >= horizon]

    def count_at(t: datetime) -> int:
        lo = bisect.bisect_right(times, t - window)
        hi = bisect.bisect_right(times, t)
        return hi - lo

    value = float(count_at(now))
    last = times[-1] if times else now
    bucket = now.replace(second=0, microsecond=0)
    key = (bucket, tuple(t for t in times if t >= horizon - window), value, window, grid_step, min_turns)
    hit = _CHAT_MEMO.get(key)
    if hit is not None:
        mag, mid = hit
        age = max(0.0, (now - last).total_seconds())
        return mag.model_copy(update={"age_sec": round(age, 3)}), mid

    history: list[tuple[datetime, float]] = []
    steps = int(WINDOW_7D / grid_step)
    for i in range(steps + 1):
        t = horizon + i * grid_step
        history.append((t, float(count_at(t))))
    mag = compute_prediction_error_magnitude(
        value=value, observed_at=last, history=history, now=now, min_readings=1
    )
    if len(in_7d) < max(1, int(min_turns)):
        mag = mag.model_copy(update={"band": "insufficient_history", "trend": "insufficient_history"})
    mid = midrank_percentile(value, [v for _, v in history])
    if len(_CHAT_MEMO) >= _CHAT_MEMO_MAX:
        _CHAT_MEMO.clear()
    _CHAT_MEMO[key] = (mag, mid)
    return mag, mid


def chat_rate_magnitude(turn_times: Sequence[datetime], *, now: datetime, **kw) -> PredictionErrorMagnitudeV1:
    return chat_rate_reading(turn_times, now=now, **kw)[0]


def chat_candidate(
    turn_times: Sequence[datetime] | None,
    *,
    now: datetime,
    absent_reason: str | None = None,
) -> AttentionCandidateV1:
    """Chat as the first external source. ``turn_times`` None = the read
    failed: absent, never calm."""
    now = _aware(now)
    if turn_times is None:
        mag = _missing_magnitude()
        mid = None
        absent_reason = absent_reason or "chat log unreadable"
        last = None
    else:
        mag, mid = chat_rate_reading(turn_times, now=now)
        last = max((_aware(t) for t in turn_times), default=None)
    n = int(mag.value)
    return AttentionCandidateV1(
        candidate_id=f"attention-candidate:{WORLD_CHAT_SOURCE_ID}",
        source_id=WORLD_CHAT_SOURCE_ID,
        source_kind="external",
        label=(
            f"Juniper is talking ({n} message{'s' if n != 1 else ''} in the last "
            f"{int(CHAT_RATE_WINDOW.total_seconds() // 60)} min)"
            if n
            else "chat is quiet"
        ),
        unusualness=mag,
        rank_percentile=mid,
        observed_at=last,
        absent=absent_reason is not None,
        absent_reason=absent_reason,
        value_kind="count",
        polarity=None,
        evidence_refs=[WORLD_CHAT_SOURCE_ID, "chat_history_log"],
    )


# Juniper's turns only: Orion-initiated rows (outreach, metacog background)
# carry no source and an empty prompt. One query, shared by every caller so
# the definition cannot drift between contests.
CHAT_TURN_TIMES_SQL = (
    "SELECT created_at FROM chat_history_log "
    "WHERE created_at >= :since AND source LIKE 'hub%' "
    "AND coalesce(prompt, '') <> '' ORDER BY created_at"
)
