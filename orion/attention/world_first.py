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

- **Continuous** body sources (2026-10-16, spec docs/superpowers/specs/
  2026-10-16-attention-continuous-signal-quota-design.md, option (b)): an
  internal source the semantic layer marks as read every tick
  (``sparsity: per_tick`` -- biometrics, bus_synaptic) is in the top decile of
  its own week ~10% of the time whatever happens. It competes only when it has
  been hot for MORE than ``SUSTAIN_DWELL_SEC`` (300 s, the orienting window)
  of the trailing ``SUSTAIN_WINDOW_SEC`` (600 s) and is hot now. The dwell is
  measured over the source's own readings (``sustained_hot_dwell``), each
  judged by ``judge_candidate`` against its 7 days. Perception (the world) is
  out of scope: orienting to a brief change in the world is correct.
  ``ATTENTION_SUSTAINED_BODY_ENABLED`` (default on) turns it off per runtime.

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
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol, Sequence

from orion.attention.sustained import (
    EVENT_ORIENTING_WINDOW_SEC as _ORIENTING_WINDOW_SEC,
    SUSTAIN_DWELL_SEC,
    SUSTAIN_WINDOW_SEC,
    hot_dwell_sec,
    is_sustained,
)
from orion.schemas.attention_candidate import (
    ATTENTION_MIN_BAND_BY_NODE_ID,
    PERCEPTION_NODE_ID,
    WORLD_CHAT_SOURCE_ID,
    WORLD_READING_SOURCE_ID,
    AttentionCandidateV1,
    is_world_source_id,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_freshness import PE_STALENESS_HORIZON_SEC
from orion.substrate.prediction_error_magnitude import (
    DEFAULT_BAND_CUTS,
    DEFAULT_MIN_READINGS,
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

# Accepted reading links use the same trailing-rate model as chat, but keep a
# distinct name and source-specific history floor so the two event streams
# cannot silently drift apart.
READING_RATE_WINDOW = CHAT_RATE_WINDOW
READING_RATE_GRID_STEP = CHAT_RATE_GRID_STEP
READING_MIN_LINKS_7D = 5

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
# The value lives in orion.attention.sustained (stdlib-only, so the metacog
# episode gate can import it); this is the same object.
EVENT_ORIENTING_WINDOW_SEC: float = _ORIENTING_WINDOW_SEC
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
# per-tick writers say "per-tick writer" (perception) or describe a
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


_CONTINUOUS_CACHE: dict[str, bool] = {}
_CONTINUOUS_WARNED: set[str] = set()
# A real reading every tick (orion/metrics/semantics.py SPARSITIES).
_PER_TICK_SPARSITY = "per_tick"
_CONTINUOUS_VALUE_KINDS = frozenset({"level", "count"})


def prediction_error_is_continuous(
    *, source_kind: str, sparsity: str | None, absent_means: str | None, value_kind: str | None
) -> bool:
    """True for a BODY source that reads every tick: the sustained-dwell rule
    applies. Not the world (perception keeps orienting to brief changes), not
    an event-written or trigger source (their own rules), not a placeholder."""
    return (
        source_kind == "internal"
        and sparsity == _PER_TICK_SPARSITY
        and value_kind in _CONTINUOUS_VALUE_KINDS
        and not prediction_error_is_event_written(sparsity, absent_means)
    )


def node_prediction_error_continuous(node_id: str) -> bool:
    """``prediction_error_is_continuous`` over the node's glossary entry
    (biometrics and bus_synaptic as of 2026-10-16). An unreadable glossary
    reads False (no dwell rule: today's verdict, never a silenced source) and
    is not cached."""
    cached = _CONTINUOUS_CACHE.get(node_id)
    if cached is not None:
        return cached
    try:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry("prediction_error", node=node_id)
    except Exception as exc:  # noqa: BLE001
        if node_id not in _CONTINUOUS_WARNED:
            _CONTINUOUS_WARNED.add(node_id)
            logger.warning("world_first_glossary_unreadable node_id=%s err=%s", node_id, exc)
        return False
    sem = dict(entry.semantics) if entry is not None and entry.node == node_id else {}
    result = prediction_error_is_continuous(
        source_kind=source_kind_for_node(node_id),
        sparsity=sem.get("sparsity"),
        absent_means=sem.get("absent_means"),
        value_kind=sem.get("value_kind"),
    )
    _CONTINUOUS_CACHE[node_id] = result
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


# Bands from calm to busy, for comparing a band against a per-node floor.
_BAND_ORDER: tuple[str, ...] = ("quiet", "usual", "high", "unusual")


def below_band_floor(source_id: str, band: str) -> str | None:
    """The node's band floor when ``band`` is under it, else None
    (``ATTENTION_MIN_BAND_BY_NODE_ID``; an unknown band or floor never blocks)."""
    floor = ATTENTION_MIN_BAND_BY_NODE_ID.get(str(source_id or ""))
    if floor is None or floor not in _BAND_ORDER or band not in _BAND_ORDER:
        return None
    return floor if _BAND_ORDER.index(band) < _BAND_ORDER.index(floor) else None


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
    # Continuous body sources only (candidate carries sustain_window_sec):
    # seconds hot in the trailing window, and whether the sustained rule held.
    # ``sustained`` False = hot now for itself but not for long enough, so not
    # eligible; the metacog episode gate still reads it as a hot poll.
    # None / None for every other source.
    hot_dwell_sec: float | None = None
    sustained: bool | None = None

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
            "sustain_window_sec": self.candidate.sustain_window_sec,
            "hot_dwell_sec": None if self.hot_dwell_sec is None else round(self.hot_dwell_sec, 3),
            "sustained": self.sustained,
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
    # A node admitted only for its own tail (walkway camera: p99+). Judged on
    # the mid-rank percentile: the camera score saturates at 1.0 for ~1.3% of
    # readings, so strict-below puts every 1.0 at ~p98.7 and "p99+" would be
    # unreachable however rare the moment is.
    floor_band = _band_for(rank_score, band_cuts) if rank_score is not None else band
    floor = below_band_floor(candidate.source_id, floor_band)
    if floor is not None:
        return CandidateVerdict(
            candidate, False, f"{floor_band} for itself (mid-rank), below this source's {floor} floor",
            score, band,
        )
    floor_note = (
        f", mid-rank {floor_band} meets its {ATTENTION_MIN_BAND_BY_NODE_ID[candidate.source_id]} floor"
        if candidate.source_id in ATTENTION_MIN_BAND_BY_NODE_ID
        else ""
    )
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
            sustain_window = candidate.sustain_window_sec
            dwell = candidate.hot_dwell_sec
            if sustain_window is not None and dwell is not None:
                # A per-tick body source is high for itself ~10% of the time by
                # construction: it competes only when the high has lasted.
                said = f"hot {dwell:.0f}s of last {sustain_window:.0f}s"
                if not is_sustained(dwell, hot_now=True):
                    return CandidateVerdict(
                        candidate, False, f"body {band} for itself, not sustained ({said})",
                        score, band, None, decay, event_age, dwell, False,
                    )
                return CandidateVerdict(
                    candidate, True, f"body unusual for itself ({band}), sustained ({said})",
                    score, band, rank_score, decay, event_age, dwell, True,
                )
            return CandidateVerdict(
                candidate, True, f"body unusual for itself ({band}){floor_note}{suffix}", score, band,
                rank_score, decay, event_age,
            )
        return CandidateVerdict(
            candidate, False, f"body {band} for itself", score, band, None, decay, event_age
        )
    if band in EXTERNAL_ELIGIBLE_BANDS:
        return CandidateVerdict(
            candidate, True, f"world busy for itself ({band}){floor_note}{suffix}", score, band,
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


def yield_reading_to_chat(ranking: WorldFirstRanking) -> WorldFirstRanking:
    """Make Juniper's eligible chat source outrank Orion's own reading.

    Both background attention contests use this same post-ranking rule so a
    reading candidate cannot win one contest while chat wins the other.
    """
    if not any(
        verdict.candidate.source_id == WORLD_CHAT_SOURCE_ID
        for verdict in ranking.eligible
    ):
        return ranking
    yielded = [
        replace(
            verdict,
            eligible=False,
            reason="yields to Juniper (world:chat eligible)",
        )
        for verdict in ranking.eligible
        if verdict.candidate.source_id == WORLD_READING_SOURCE_ID
    ]
    if yielded:
        ranking.eligible[:] = [
            verdict
            for verdict in ranking.eligible
            if verdict.candidate.source_id != WORLD_READING_SOURCE_ID
        ]
        ranking.ineligible.extend(yielded)
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
    disagree). Perception writes a 0.0 placeholder while stale or warming
    (marked ``prediction_error_missing_reason``, never recorded in history),
    so absence must come from staleness, never from the value. Each contest
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


# node_id -> (content key, times, rows, (horizon index, sorted finite values)).
# The field contest ticks every ~2 s and a node's readings move every ~60 s;
# rebuilding a ~14k-reading week each tick cost ~10 ms per node (review
# 2026-10-16). The key carries the sum of every value, not just the length and
# ends, so two histories that differ only in the middle never share an entry.
_HISTORY_MEMO: dict[str, tuple[tuple, list[datetime], list[tuple[datetime, float]], tuple[int, list[float]]]] = {}
_HISTORY_MEMO_MAX = 32


def _history_key(history: Sequence[tuple[datetime, float]]) -> tuple | None:
    try:
        return (len(history), history[0], history[-1], math.fsum(float(v) for _, v in history))
    except (TypeError, ValueError, IndexError):
        return None


def _prepared_history(
    node_id: str, history: Sequence[tuple[datetime, float]], horizon: datetime
) -> tuple[list[datetime], list[tuple[datetime, float]], list[float]]:
    """(times, rows, sorted finite values) of the readings at or after
    ``horizon``, oldest first."""
    if not history:
        return [], [], []
    key = _history_key(history)
    memo = _HISTORY_MEMO.get(node_id) if key is not None else None
    if memo is None or memo[0] != key:
        rows: list[tuple[datetime, float]] = []
        for ts, v in history:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            rows.append((ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc), fv))
        rows.sort(key=_first)
        memo = (key, [r[0] for r in rows], rows, (-1, []))
    _, times, rows, (cached_start, cached_values) = memo
    start = bisect.bisect_left(times, horizon)
    if start == cached_start:
        values = cached_values
    else:
        values = sorted([v for _, v in rows[start:] if math.isfinite(v)])
        memo = (memo[0], times, rows, (start, values))
    if key is not None:
        if len(_HISTORY_MEMO) >= _HISTORY_MEMO_MAX and node_id not in _HISTORY_MEMO:
            _HISTORY_MEMO.clear()
        _HISTORY_MEMO[node_id] = memo
    return times[start:], rows[start:], values


def _first(row: tuple[datetime, float]) -> datetime:
    return row[0]


# Gaps sampled for a source's own write cadence (the most recent ones).
_CADENCE_GAPS = 200


def reading_hold_sec(times: Sequence[datetime]) -> float:
    """How long one reading may speak for the body: twice the source's own
    median write gap (recent readings), never past the staleness horizon. A
    source whose writer stalls must not have its last reading fill the dwell
    (review 2026-10-16: one held spike read "sustained" after 400 s of
    silence). Derived from the data, not a knob."""
    recent = times[-(_CADENCE_GAPS + 1):]
    gaps = sorted(
        (b - a).total_seconds() for a, b in zip(recent, recent[1:]) if b > a
    )
    if not gaps:
        return 0.0
    mid = len(gaps) // 2
    median = gaps[mid] if len(gaps) % 2 else 0.5 * (gaps[mid - 1] + gaps[mid])
    return min(2.0 * median, INTERNAL_MAX_AGE_SEC)


def _flags(
    node_id: str,
    times: list[datetime],
    rows: list[tuple[datetime, float]],
    values: list[float],
    *,
    now: datetime,
    window_sec: float,
    band_cuts: tuple[float, float, float],
    min_readings: int,
) -> list[tuple[float, bool | None]]:
    if not rows:
        return []
    n = len(values)
    # The last reading before the window still speaks into it.
    first = max(0, bisect.bisect_left(times, now - timedelta(seconds=float(window_sec))) - 1)
    value_kind, polarity = node_prediction_error_semantics(node_id)
    kind = source_kind_for_node(node_id)
    out: list[tuple[float, bool | None]] = []
    for ts, v in rows[first:]:
        if not math.isfinite(v) or n == 0:
            out.append((ts.timestamp(), None))
            continue
        pct = bisect.bisect_left(values, v) / n
        mag = PredictionErrorMagnitudeV1(
            value=round(v, 6),
            age_sec=0.0,
            percentile_now=round(pct, 6),
            n_readings_7d=n,
            band="insufficient_history" if n < max(1, int(min_readings)) else _band_for(pct, band_cuts),
        )
        verdict = judge_candidate(
            AttentionCandidateV1(
                candidate_id=f"attention-candidate:{node_id}",
                source_id=node_id,
                source_kind=kind,  # type: ignore[arg-type]
                label=node_id,
                unusualness=mag,
                value_kind=value_kind,
                polarity=polarity,  # type: ignore[arg-type]
            ),
            band_cuts=band_cuts,
        )
        out.append((ts.timestamp(), True if verdict.eligible else (None if verdict.score is None else False)))
    return out


def reading_hot_flags(
    node_id: str,
    history: Sequence[tuple[datetime, float]],
    *,
    now: datetime,
    window_sec: float = SUSTAIN_WINDOW_SEC,
    band_cuts: tuple[float, float, float] = DEFAULT_BAND_CUTS,
    min_readings: int = DEFAULT_MIN_READINGS,
) -> list[tuple[float, bool | None]]:
    """``(epoch, hot)`` for every reading that can speak inside the trailing
    window (the readings in it, plus the last one before it, which speaks
    until the next). ``hot`` is ``judge_candidate``'s own verdict on that
    reading as a fresh candidate: eligible -> True, a real but calm reading ->
    False, unjudgeable (too little history, no polarity, non-finite) -> None.

    Each reading is scored against the node's CURRENT 7-day window, not the
    window ending at its own time: the two differ by the <= window_sec of
    readings at the edges (~10 of ~14,000 for a 60 s source), and the caller's
    cache has already trimmed the older edge.
    """
    now = _aware(now)
    times, rows, values = _prepared_history(node_id, history, now - WINDOW_7D)
    return _flags(node_id, times, rows, values, now=now, window_sec=window_sec,
                  band_cuts=band_cuts, min_readings=min_readings)


def sustained_hot_dwell(
    node_id: str,
    history: Sequence[tuple[datetime, float]],
    *,
    now: datetime,
    window_sec: float = SUSTAIN_WINDOW_SEC,
    band_cuts: tuple[float, float, float] = DEFAULT_BAND_CUTS,
) -> float:
    """Seconds of the trailing ``window_sec`` this node's own readings were
    hot (``orion.attention.sustained.hot_dwell_sec`` over
    ``reading_hot_flags``). A reading speaks until the next one, and at most
    ``reading_hold_sec`` (twice the source's own write cadence)."""
    now = _aware(now)
    times, rows, values = _prepared_history(node_id, history, now - WINDOW_7D)
    return hot_dwell_sec(
        _flags(node_id, times, rows, values, now=now, window_sec=window_sec,
               band_cuts=band_cuts, min_readings=DEFAULT_MIN_READINGS),
        now=now.timestamp(),
        window_sec=window_sec,
        max_hold_sec=reading_hold_sec(times),
    )


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
    pe_history: Sequence[tuple[datetime, float]] | None = None,
    sustain: bool = True,
) -> AttentionCandidateV1:
    """A ``node:substrate.*`` prediction-error node as a candidate.

    ``event_decay`` (``ATTENTION_EVENT_DECAY_ENABLED``, default on): when the
    glossary marks the node event-written, the candidate carries
    ``event_window_sec`` and fades with the age of the event that wrote it.

    ``history_values`` (the node's 7-day readings) lets the candidate carry a
    mid-rank ``rank_percentile`` for ranking ceiling ties.

    ``absent_reason`` is the caller's evidence that the source cannot measure
    right now (perception: camera/embedding staleness, or the node's own
    ``prediction_error_missing_reason``). The perception nodes write a 0.0
    placeholder while stale or warming, marked missing and never recorded in
    the history, so absence must come from that marker or staleness, never
    from the value.

    ``sustain`` (``ATTENTION_SUSTAINED_BODY_ENABLED``, default on) with
    ``pe_history`` (the node's 7-day ``(observed_at, value)`` readings): when
    the glossary marks the node a continuous body source, the candidate
    carries ``sustain_window_sec`` and its ``hot_dwell_sec`` from those
    readings, and ``judge_candidate`` requires the high to have lasted. No
    history (or a failed dwell computation) = no rule: today's verdict, never
    a silenced source.
    """
    kind = source_kind_for_node(node_id)
    value_kind, polarity = node_prediction_error_semantics(node_id)
    if magnitude is None:
        absent_reason = absent_reason or "no magnitude (no stored history)"
    elif kind == "external" and magnitude.age_sec > PERCEPTION_MAX_AGE_SEC and not absent_reason:
        absent_reason = f"reading {magnitude.age_sec:.0f}s old"
    sustain_window: float | None = None
    dwell: float | None = None
    if (
        sustain
        and pe_history
        and magnitude is not None
        and absent_reason is None
        and node_prediction_error_continuous(node_id)
    ):
        try:
            hist = pe_history
            if observed_at is not None:
                obs = _aware(observed_at)
                last = hist[-1][0] if len(hist) else None
                if last is None or obs > _aware(last):
                    # The caller's history can lag the reading being judged
                    # (e.g. a failed history write): the dwell must end on it.
                    hist = [*hist, (obs, float(magnitude.value))]
            dwell = round(sustained_hot_dwell(node_id, hist, now=now), 3)
            sustain_window = SUSTAIN_WINDOW_SEC
        except Exception as exc:  # noqa: BLE001 -- a dwell fault must not cost the candidate
            logger.warning("world_first_sustained_dwell_failed node_id=%s err=%s", node_id, exc)
            dwell, sustain_window = None, None
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
        sustain_window_sec=sustain_window,
        hot_dwell_sec=dwell,
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


def reading_candidate(
    link_events: Sequence[tuple[datetime, str, str]] | None,
    *,
    now: datetime,
    absent_reason: str | None = None,
) -> AttentionCandidateV1:
    """Build the external reading candidate from accepted link events.

    ``link_events`` is ``(materialized_at, subject_node_id, object_node_id)``.
    ``None`` means the journal read failed and is absent, never calm. Endpoint
    ids are evidence only and are carried to the winning open loop so curiosity
    can expand the concepts that were actually read.
    """
    now = _aware(now)
    if link_events is None:
        return AttentionCandidateV1(
            candidate_id=f"attention-candidate:{WORLD_READING_SOURCE_ID}",
            source_id=WORLD_READING_SOURCE_ID,
            source_kind="external",
            label="reading activity unavailable",
            unusualness=_missing_magnitude(),
            rank_percentile=None,
            observed_at=None,
            absent=True,
            absent_reason=absent_reason or "reading journal unreadable",
            value_kind="count",
            polarity=None,
            evidence_refs=[WORLD_READING_SOURCE_ID],
        )

    events = []
    for ts, subject, obj in link_events:
        subject = str(subject or "")
        obj = str(obj or "")
        if (
            not isinstance(ts, datetime)
            or _aware(ts) > now
            or not subject
            or not obj
            or subject == obj
        ):
            continue
        events.append((_aware(ts), subject, obj))
    times = [ts for ts, _, _ in events]
    mag, mid = chat_rate_reading(
        times,
        now=now,
        window=READING_RATE_WINDOW,
        grid_step=READING_RATE_GRID_STEP,
        min_turns=READING_MIN_LINKS_7D,
    )
    endpoint_refs: list[str] = []
    recent_since = now - READING_RATE_WINDOW
    for ts, subject, obj in reversed(events):
        if ts < recent_since or ts > now:
            continue
        for endpoint in (subject, obj):
            if endpoint and endpoint not in endpoint_refs:
                endpoint_refs.append(endpoint)
            if len(endpoint_refs) >= 8:
                break
        if len(endpoint_refs) >= 8:
            break
    n = int(mag.value)
    return AttentionCandidateV1(
        candidate_id=f"attention-candidate:{WORLD_READING_SOURCE_ID}",
        source_id=WORLD_READING_SOURCE_ID,
        source_kind="external",
        label=(
            f"Orion is reading ({n} accepted link{'s' if n != 1 else ''} in the last "
            f"{int(READING_RATE_WINDOW.total_seconds() // 60)} min)"
            if n
            else "reading is quiet"
        ),
        unusualness=mag,
        rank_percentile=mid,
        observed_at=max(times, default=None),
        absent=absent_reason is not None,
        absent_reason=absent_reason,
        value_kind="count",
        polarity=None,
        evidence_refs=[WORLD_READING_SOURCE_ID, *endpoint_refs],
    )


# Juniper's turns only: Orion-initiated rows (outreach, metacog background)
# carry no source and an empty prompt. One query, shared by every caller so
# the definition cannot drift between contests.
CHAT_TURN_TIMES_SQL = (
    "SELECT created_at FROM chat_history_log "
    "WHERE created_at >= :since AND source LIKE 'hub%' "
    "AND coalesce(prompt, '') <> '' ORDER BY created_at"
)
