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
  have enough history, are fresh, and are busier than their own usual
  (band ``usual`` or above, i.e. at or above their own 7-day median).
- **Internal** candidates (the body) are eligible ONLY at band ``high`` or
  ``unusual`` in their bad direction. The direction comes from the semantic
  layer (``orion.metrics.semantics.derived_channel_polarity`` over the
  glossary's ``value_kind``), never invented here. A placeholder or bucket is
  never eligible: it is not a measurement.
- Eligible candidates rank by their bad-direction percentile against their
  own history -- one scale for both kinds, so no exchange rate. A tie is
  broken by the caller's secondary score (Borda in the broadcast), then id.
- An empty eligible set is an explicit **no-winner** result. A calm body
  stays silent.

Band cut points are ``prediction_error_magnitude.DEFAULT_BAND_CUTS`` -- knobs
to be graded on live data, not findings.

``ATTENTION_WORLD_FIRST_ENABLED`` (default on, Juniper's ship-on rule) turns
this on in every contest; false restores the previous ranking exactly.
"""

from __future__ import annotations

import bisect
import functools
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol, Sequence

from orion.schemas.attention_candidate import AttentionCandidateV1
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_freshness import PE_STALENESS_HORIZON_SEC
from orion.substrate.prediction_error_magnitude import (
    DEFAULT_BAND_CUTS,
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

logger = logging.getLogger(__name__)

WORLD_FIRST_FLAG = "ATTENTION_WORLD_FIRST_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}

# Marker carried in existing free-form fields (FieldAttentionTargetV1.
# evidence_refs, AttentionSignalV1/OpenLoopV1.provenance) -- step 1 of the
# schema rollout needs no change to any extra="forbid" model.
SOURCE_KIND_REF_PREFIX = "source_kind:"
SOURCE_KIND_KEY = "source_kind"

WORLD_CHAT_SOURCE_ID = "world:chat"
PERCEPTION_NODE_ID = "node:substrate.perception"
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

_INTERNAL_ELIGIBLE_BANDS = frozenset({"high", "unusual"})
_EXTERNAL_ELIGIBLE_BANDS = frozenset({"usual", "high", "unusual"})
_NEVER_A_MEASUREMENT = frozenset({"placeholder", "bucket"})


def world_first_enabled(env: dict[str, str] | None = None) -> bool:
    """Default ON (ships on, per Juniper's standing rule); false restores the
    previous ranking in every contest."""
    raw = (env if env is not None else os.environ).get(WORLD_FIRST_FLAG, "true")
    return str(raw).strip().lower() in _TRUTHY


class AttentionCandidateSource(Protocol):
    """Shaped like ``AttentionSignalDetector`` (orion/substrate/attention/
    detectors/base.py): one source, one call per tick, never raises."""

    source_id: str

    def candidates(self, now: datetime) -> list[AttentionCandidateV1]:
        ...


# ---------------------------------------------------------------------------
# Semantic layer: value_kind and polarity, read, never invented.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def node_prediction_error_semantics(node_id: str) -> tuple[str | None, str | None]:
    """(value_kind, polarity) for a node's prediction_error, from the glossary.

    Polarity is ``derived_channel_polarity("prediction_error", value_kind)``
    -- the same derivation the metric lock uses (PR #2579): prediction_error
    is in PRESSURE_CHANNELS, so higher is worse, except a ``trigger`` whose
    value is "it fired" and has no polarity. An unreadable glossary yields
    value_kind None and the channel's derived polarity (logged once).
    """
    from orion.metrics.semantics import derived_channel_polarity

    value_kind: str | None = None
    try:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry("prediction_error", node=node_id)
        if entry is not None and entry.node == node_id:
            value_kind = dict(entry.semantics).get("value_kind")
    except Exception as exc:  # noqa: BLE001 -- a missing file must not stop attention
        logger.warning("world_first_glossary_unreadable node_id=%s err=%s", node_id, exc)
    return value_kind, derived_channel_polarity("prediction_error", value_kind)


def bad_direction_percentile(candidate: AttentionCandidateV1) -> float | None:
    """Percentile in the candidate's bad direction (higher = more alarming
    for the body, busier for the world). None when there is no reading or no
    declared direction."""
    pct = candidate.unusualness.percentile_now
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
    score: float | None  # bad-direction percentile; the ranking key
    band: str

    def trace(self) -> dict:
        mag = self.candidate.unusualness
        return {
            "source_id": self.candidate.source_id,
            "source_kind": self.candidate.source_kind,
            "eligible": self.eligible,
            "reason": self.reason,
            "score": None if self.score is None else round(self.score, 6),
            "band": self.band,
            "value": mag.value,
            "percentile_now": mag.percentile_now,
            "n_readings_7d": mag.n_readings_7d,
            "age_sec": mag.age_sec,
            "absent": self.candidate.absent,
            "value_kind": self.candidate.value_kind,
            "polarity": self.candidate.polarity,
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
    if candidate.source_kind == "internal":
        if band in _INTERNAL_ELIGIBLE_BANDS:
            return CandidateVerdict(candidate, True, f"body unusual for itself ({band})", score, band)
        return CandidateVerdict(candidate, False, f"body {band} for itself", score, band)
    if band in _EXTERNAL_ELIGIBLE_BANDS:
        return CandidateVerdict(candidate, True, f"world busier than its usual ({band})", score, band)
    return CandidateVerdict(candidate, False, f"world {band}: at or below its usual", score, band)


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
        key=lambda v: (-(v.score or 0.0), -tb.get(v.candidate.source_id, 0.0), v.candidate.source_id)
    )
    return ranking


# ---------------------------------------------------------------------------
# Candidate builders (pure). Callers do the I/O.
# ---------------------------------------------------------------------------


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def source_kind_for_node(node_id: str) -> str:
    """Camera surprise is the world; every other substrate node is the body."""
    return "external" if node_id == PERCEPTION_NODE_ID else "internal"


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
) -> AttentionCandidateV1:
    """A ``node:substrate.*`` prediction-error node as a candidate.

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
        observed_at=observed_at,
        absent=absent_reason is not None,
        absent_reason=absent_reason,
        value_kind=value_kind,
        polarity=polarity,  # type: ignore[arg-type]
        evidence_refs=[node_id],
    )


def chat_rate_magnitude(
    turn_times: Sequence[datetime],
    *,
    now: datetime,
    window: timedelta = CHAT_RATE_WINDOW,
    grid_step: timedelta = CHAT_RATE_GRID_STEP,
    min_turns: int = CHAT_MIN_TURNS_7D,
) -> PredictionErrorMagnitudeV1:
    """Juniper turns in the trailing ``window``, against the same windowed
    count sampled every ``grid_step`` over the last 7 days.

    Rest is reachable: no turn in the window reads 0, percentile 0.0, band
    ``quiet``. Fewer than ``min_turns`` real turns in 7 days -> band
    ``insufficient_history`` (the minute grid would otherwise always look
    like 10,080 readings).
    """
    now = _aware(now)
    times = sorted(_aware(t) for t in turn_times if _aware(t) <= now)
    horizon = now - WINDOW_7D
    in_7d = [t for t in times if t >= horizon]

    def count_at(t: datetime) -> int:
        lo = bisect.bisect_right(times, t - window)
        hi = bisect.bisect_right(times, t)
        return hi - lo

    history: list[tuple[datetime, float]] = []
    steps = int(WINDOW_7D / grid_step)
    for i in range(steps + 1):
        t = horizon + i * grid_step
        history.append((t, float(count_at(t))))
    value = float(count_at(now))
    last = times[-1] if times else now
    mag = compute_prediction_error_magnitude(
        value=value, observed_at=last, history=history, now=now, min_readings=1
    )
    if len(in_7d) < max(1, int(min_turns)):
        mag = mag.model_copy(update={"band": "insufficient_history", "trend": "insufficient_history"})
    return mag


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
        absent_reason = absent_reason or "chat log unreadable"
        last = None
    else:
        mag = chat_rate_magnitude(turn_times, now=now)
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
