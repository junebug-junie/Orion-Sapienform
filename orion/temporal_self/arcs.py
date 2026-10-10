"""The chronology reducer: broadcast ticks + events in, arcs out. Pure, no I/O, no LLM.

Contract (Temporal Self spec rev 4, "The reducer contract"):

* ``fold(state, ticks, events, cfg)`` merges both streams into ONE order --
  ``(available_at, stream_rank, ref)`` -- where ``available_at`` is the latest timestamp an
  item carries (a process event is emitted only once complete, so it becomes available at
  its end). Same inputs + same watermarks => byte-identical state. Items sorting at or
  before ``state.last_key`` were already folded and are skipped, so re-folding is a no-op.
* ``advance_clock(state, now, cfg)`` declares that nothing strictly before ``now`` will arrive:
  it expires suspended arcs, suspends a silent attention lane, and closes any day that
  ended. The driver must call it only with its read watermark (rows are read up to it).
* Closed days queue in ``state.pending_closed_days``; ``drain_closed_days`` hands them out.

Identity is an exact reference, never a label or an embedding (rule 6): returns are
counted on ``subject_ref`` equality within one lane. Every arc carries the refs that built
it (``evidence_refs``, ``<source_table>:<pk>``), and only refs whose own row names the same
subject are evidence. Subject-less events bind by time (rule 7) or, for reverie attention
rows, by correlation id (rule 8); a process arc that completes later binds the buffered
context that happened inside it.

Lanes and their rules are in ``LANE_RULES`` below; the numbers behind K and R are in the PR
report (docs/superpowers/pr-reports/2026-10-10-temporal-self-chronology-reducer-pr.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from orion.schemas.temporal_self import (
    CONTEXT_CAP,
    EVIDENCE_CAP,
    PERCEPT_CAP,
    ArcAttentionSummaryV1,
    ArcSegmentV1,
    ExpectationRefV1,
    TemporalSelfArcV1,
    TemporalSelfContextItemV1,
    TemporalSelfDayV1,
    TemporalSelfEventV1,
    TemporalSelfStateV1,
)
from orion.temporal_self.broadcast import BroadcastTickView
from orion.temporal_self.day import DEFAULT_TZ, day_id_for, day_window

INTERRUPTIONS_CAP = 64
DAY_LIST_CAP = 256
CORRELATION_CAP = 256
TICK_REFS_CAP = 64

# Lanes whose open arc is exclusive: a new subject suspends the current one (rule 3).
EXCLUSIVE_LANES = ("attention", "interoception", "conversation")
# Process arcs are born closed from one completed event.
PROCESS_KINDS = {
    "curiosity_run": "curiosity",
    "reverie_chain": "reverie",
    "visual_run": "imagery",
    "dream_cycle": "sleep",
}
CONTEXT_KINDS = {"metacog_observation", "memory_episode", "vision_percept", "attention_row"}
CONSTRAINT_KINDS = {"visual_deferral", "gpu_wait"}
SELF_CHANGE_KINDS = {"action_outcome", "expectation_verdict"}


@dataclass(frozen=True)
class ReducerConfig:
    tz_name: str = DEFAULT_TZ
    # K: consecutive broadcast ticks for a subject to open/resume an attention arc, and
    # consecutive no-winner ticks to suspend one. 3 ticks ~ 110 s at the live ~37 s cadence.
    arc_min_ticks: int = 3
    # R: a suspended arc may resume within this window, else it closes. 30 min, measured.
    return_window_sec: float = 1800.0
    # A broadcast gap longer than this is a recorder outage/restart, not attention.
    # Live 7-day tick gaps: median 37 s, p99 64 s.
    max_tick_gap_sec: float = 180.0
    # Same bar the dream service uses for "Juniper has been quiet" (DREAM_IDLE_MINUTES=45).
    conversation_idle_sec: float = 2700.0
    # A conversation suspends after 45 min idle, so its return window must be longer than
    # that or it could never return. The spec's default R (180 min) is kept for this lane.
    conversation_return_window_sec: float = 10800.0
    # A pause in field-attention runs longer than this suspends the interoception arc.
    interoception_gap_sec: float = 180.0
    # How long subject-less context waits for a process arc that completes later.
    context_buffer_sec: float = 6 * 3600.0
    # A single interoception segment longer than this is flagged as a possible stuck reading.
    stuck_segment_sec: float = 4 * 3600.0
    # Rule 1's "broadcast winner unchanged all day" warning needs at least this many winners.
    unchanged_winner_min_ticks: int = 100


LANE_RULES = {
    "attention": "broadcast winner source_refs[0]; K ticks open/resume; K no-winner ticks suspend",
    "interoception": "field_dominance_run target_id; run qualifies at tick_count >= min_streak_at_run",
    "conversation": "chat_turn session_id; idle 45 min suspends; resume within R",
    "concern": "chat-scope attention_loop_raised loop_id; closes on attention_loop_verdict; spans days",
    "curiosity": "one completed curiosity run",
    "reverie": "one reverie chain with >= 1 thought",
    "imagery": "one visual reverie chain",
    "sleep": "one completed dream cycle",
}


# --------------------------------------------------------------------------- helpers


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def arc_id_for(day_id: str, kind: str, subject_ref: str, first_ref: str) -> str:
    raw = f"{day_id}|{kind}|{subject_ref}|{first_ref}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _secs(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds()


def _return_window(kind: str, cfg: "ReducerConfig") -> float:
    return cfg.conversation_return_window_sec if kind == "conversation" else cfg.return_window_sec


def _add_capped(items: list[str], value: str, cap: int) -> bool:
    """Append if new and under cap. Returns False when the cap dropped it."""
    if value in items:
        return True
    if len(items) >= cap:
        return False
    items.append(value)
    return True


def _add_evidence(arc: TemporalSelfArcV1, ref: str) -> None:
    if ref in arc.evidence_refs:
        return
    if len(arc.evidence_refs) >= EVIDENCE_CAP:
        arc.evidence_overflow += 1
    else:
        arc.evidence_refs.append(ref)


def _add_context(arc: TemporalSelfArcV1, event_id: str) -> None:
    if event_id in arc.context_event_ids:
        return
    if len(arc.context_event_ids) >= CONTEXT_CAP:
        arc.context_overflow += 1
    else:
        arc.context_event_ids.append(event_id)


def _event_available_at(e: TemporalSelfEventV1) -> datetime:
    return e.ended_at or e.occurred_at


def _ref(e: TemporalSelfEventV1) -> str:
    return f"{e.source_table}:{e.source_ref}"


# --------------------------------------------------------------------------- arc lifecycle


def _suspend(state: TemporalSelfStateV1, arc: TemporalSelfArcV1, by: str | None) -> None:
    if arc.status != "open":
        return
    arc.status = "suspended"
    if by:
        _add_capped(arc.interruptions, by, INTERRUPTIONS_CAP)
    if state.active.get(arc.kind) == arc.arc_id:
        del state.active[arc.kind]


def _close(arc: TemporalSelfArcV1, reason: str, at: datetime) -> None:
    arc.status = "closed"
    arc.closed_reason = reason  # type: ignore[assignment]
    arc.ended_at = max(at, arc.began_at)


def _find_resumable(
    state: TemporalSelfStateV1, kind: str, subject: str, at: datetime, cfg: ReducerConfig
) -> TemporalSelfArcV1 | None:
    best = None
    for arc in state.arcs.values():
        if (
            arc.kind == kind
            and arc.subject_ref == subject
            and arc.status != "closed"
            and arc.day_id == state.day_id
            and _secs(arc.last_seen_at, at) <= _return_window(kind, cfg)
        ):
            if best is None or arc.last_seen_at > best.last_seen_at:
                best = arc
    return best


def _open_or_resume(
    state: TemporalSelfStateV1,
    cfg: ReducerConfig,
    *,
    kind: str,
    subject: str,
    label: str,
    began: datetime,
    now: datetime,
    evidence: Sequence[str],
    exclusive: bool = True,
) -> TemporalSelfArcV1:
    """Start a segment [began, now] for ``subject``: resume within R, else a new arc."""
    current = state.arcs.get(state.active.get(kind, "")) if exclusive else None
    target = _find_resumable(state, kind, subject, began, cfg)
    if target is not None and target is current:
        # Same subject still active after a pause the caller decided was a break.
        _suspend(state, target, by=None)
    if target is not None:
        if current is not None and current is not target:
            _suspend(state, current, by=target.arc_id)
        target.status = "open"
        target.attention_returns += 1
        target.segments.append(ArcSegmentV1(began_at=began, ended_at=now))
        target.cumulative_dwell_sec += max(0.0, _secs(began, now))
        target.last_seen_at = now
        if label and not target.subject_label:
            target.subject_label = label
        for ref in evidence:
            _add_evidence(target, ref)
        arc = target
    else:
        assert state.day_id is not None
        arc_id = arc_id_for(state.day_id, kind, subject, evidence[0] if evidence else subject)
        carry = state.carry.pop(f"{kind}|{subject}", None)
        carried = None
        if carry and _secs(datetime.fromisoformat(carry["last_seen"]), began) <= _return_window(kind, cfg):
            carried = carry["arc_id"]
        arc = TemporalSelfArcV1(
            arc_id=arc_id, day_id=state.day_id, kind=kind, subject_ref=subject,  # type: ignore[arg-type]
            subject_label=label, began_at=began, last_seen_at=now, status="open",
            segments=[ArcSegmentV1(began_at=began, ended_at=now)],
            cumulative_dwell_sec=max(0.0, _secs(began, now)),
            carried_from_previous_day=carried is not None, carried_from_arc_id=carried,
        )
        for ref in evidence:
            _add_evidence(arc, ref)
        if current is not None:
            _suspend(state, current, by=arc_id)
        state.arcs[arc_id] = arc
    if exclusive:
        state.active[kind] = arc.arc_id
    _retro_bind(state, arc, began, now)
    return arc


def _extend(arc: TemporalSelfArcV1, now: datetime, ref: str | None, *, accrue: bool) -> None:
    if accrue and arc.segments:
        seg = arc.segments[-1]
        arc.cumulative_dwell_sec += max(0.0, _secs(seg.ended_at, now))
        seg.ended_at = now
    elif arc.segments:
        arc.segments.append(ArcSegmentV1(began_at=now, ended_at=now))
    arc.last_seen_at = now
    if ref:
        _add_evidence(arc, ref)


# --------------------------------------------------------------------------- binding

# Kinds whose OPEN arc covers time up to "now" (the stretch is ongoing between observations).
# Interoception runs and process arcs cover only their recorded segments; concern arcs are
# threads, not stretches of time, so nothing binds to them by time.
_FORWARD_KINDS = {"attention", "conversation"}


def _forward_reach(arc: TemporalSelfArcV1, cfg: "ReducerConfig") -> float:
    # How far past its last observation an open arc still covers time. Bounded so that an
    # outage never binds context to an arc nobody observed, however late it is noticed.
    return cfg.max_tick_gap_sec if arc.kind == "attention" else cfg.conversation_idle_sec


def _covers(arc: TemporalSelfArcV1, at: datetime, cfg: "ReducerConfig") -> bool:
    if arc.kind == "concern" or at < arc.began_at:
        return False
    if (
        arc.status == "open" and arc.kind in _FORWARD_KINDS and arc.segments
        and at >= arc.segments[-1].began_at and _secs(arc.last_seen_at, at) <= _forward_reach(arc, cfg)
    ):
        return True
    return any(s.began_at <= at <= s.ended_at for s in arc.segments)


def _by_ref(item: TemporalSelfContextItemV1) -> bool:
    # Rule 8: reverie attention rows attach to their chain by correlation id, never by time.
    return item.source_kind == "attention_row" and item.process == "reverie"


def _bindable(state: TemporalSelfStateV1, arc: TemporalSelfArcV1, item: TemporalSelfContextItemV1, cfg: "ReducerConfig") -> bool:
    if _by_ref(item):
        return arc.kind == "reverie" and item.correlation_id in state.arc_correlations.get(arc.arc_id, [])
    return _covers(arc, item.at, cfg)


def _bind_item(arc: TemporalSelfArcV1, item: TemporalSelfContextItemV1) -> None:
    if arc.arc_id in item.bound_arc_ids:
        return
    item.bound_arc_ids.append(arc.arc_id)
    kind = item.source_kind
    if kind == "attention_row":
        summary = arc.attention or ArcAttentionSummaryV1()
        lane = item.process or "unknown"
        summary.rows_by_lane[lane] = summary.rows_by_lane.get(lane, 0) + 1
        if item.reason:
            words = summary.reasons_by_lane.setdefault(lane, [])
            if item.reason not in words and len(words) < 16:
                words.append(item.reason)
                words.sort()
        arc.attention = summary
    elif kind == "vision_percept":
        for ent in item.entities:
            if ent not in arc.percept_entities and len(arc.percept_entities) < PERCEPT_CAP:
                arc.percept_entities.append(ent)
        arc.percept_entities.sort()
        _add_context(arc, item.event_id)
    elif item.constraint:
        _add_capped(arc.constraint_event_ids, item.event_id, CONTEXT_CAP)
    elif kind in SELF_CHANGE_KINDS:
        _add_capped(arc.expectation_event_ids, item.event_id, CONTEXT_CAP)
    else:
        _add_context(arc, item.event_id)


def _bind_new_item(state: TemporalSelfStateV1, item: TemporalSelfContextItemV1, cfg: "ReducerConfig") -> None:
    for arc in state.arcs.values():
        if arc.day_id == state.day_id and _bindable(state, arc, item, cfg):
            _bind_item(arc, item)


def _retro_bind(state: TemporalSelfStateV1, arc: TemporalSelfArcV1, began: datetime, until: datetime) -> None:
    """Bind buffered items inside newly covered time [began, until). Items at ``until``
    itself sort after this arc's own item and bind on arrival."""
    for item in state.context_buffer:
        if _by_ref(item):
            if arc.kind == "reverie" and item.correlation_id in state.arc_correlations.get(arc.arc_id, []):
                _bind_item(arc, item)
        elif arc.kind != "concern" and began <= item.at < until:
            _bind_item(arc, item)


_DAY_CONTEXT_KINDS = {"metacog_observation", "memory_episode", "chat_turn"}


def _flush_buffer(state: TemporalSelfStateV1, before: datetime | None) -> None:
    """Drop buffered items older than ``before`` (all when None). Context events that no arc
    took land in the day-level context list (rule 7: "or in the day when none is open")."""
    # Context items carry no ended_at, so they arrive (and sit) in occurred_at order: if the
    # oldest is still inside the horizon, nothing is due.
    if before is not None and (not state.context_buffer or state.context_buffer[0].at >= before):
        return
    keep: list[TemporalSelfContextItemV1] = []
    for item in state.context_buffer:
        if before is not None and item.at >= before:
            keep.append(item)
            continue
        if item.source_kind in _DAY_CONTEXT_KINDS and not item.bound_arc_ids:
            _add_day_context(state, item.event_id)
    state.context_buffer = keep


def _add_day_context(state: TemporalSelfStateV1, event_id: str) -> None:
    """Canonical (sorted, smallest kept, overflow counted) so the list does not depend on
    how the fold was chunked."""
    if event_id in state.day_context_event_ids:
        return
    state.day_context_event_ids.append(event_id)
    state.day_context_event_ids.sort()
    if len(state.day_context_event_ids) > DAY_LIST_CAP:
        state.day_context_event_ids.pop()
        state.day_context_overflow += 1


# --------------------------------------------------------------------------- per-item folds


def _fold_tick(state: TemporalSelfStateV1, tick: BroadcastTickView, cfg: ReducerConfig) -> None:
    t = tick.generated_at
    log_ref = f"substrate_attention_broadcast_log:{tick.log_id}"
    active = state.arcs.get(state.active.get("attention", ""))
    if state.tick_prev_at is not None and _secs(state.tick_prev_at, t) > cfg.max_tick_gap_sec:
        state.tick_source_gaps_today += 1
        if active is not None:
            _suspend(state, active, by=None)
            active = None
        _reset_candidate(state)
        state.tick_prev_ref = None
    state.tick_count_today += 1
    if tick.ref is not None:
        state.tick_winner_count_today += 1
        _add_capped(state.tick_winner_refs_today, tick.ref, TICK_REFS_CAP)

    if active is not None and tick.ref == active.subject_ref:
        _extend(active, t, log_ref, accrue=state.tick_prev_ref == active.subject_ref)
        _reset_candidate(state)
    else:
        is_none = tick.ref is None
        if state.tick_candidate_count and state.tick_candidate_ref == tick.ref and state.tick_candidate_is_none == is_none:
            state.tick_candidate_count += 1
            if not is_none:
                state.tick_candidate_log_ids.append(log_ref)
        else:
            state.tick_candidate_ref = tick.ref
            state.tick_candidate_is_none = is_none
            state.tick_candidate_count = 1
            state.tick_candidate_first_at = t
            state.tick_candidate_log_ids = [] if is_none else [log_ref]
        if state.tick_candidate_count >= cfg.arc_min_ticks:
            if is_none:
                if active is not None:
                    _suspend(state, active, by=None)
            else:
                assert state.tick_candidate_first_at is not None
                _open_or_resume(
                    state, cfg, kind="attention", subject=tick.ref, label=tick.label,  # type: ignore[arg-type]
                    began=state.tick_candidate_first_at, now=t, evidence=list(state.tick_candidate_log_ids),
                )
            _reset_candidate(state)
    state.tick_prev_at = t
    state.tick_prev_ref = tick.ref


def _reset_candidate(state: TemporalSelfStateV1) -> None:
    state.tick_candidate_ref = None
    state.tick_candidate_is_none = False
    state.tick_candidate_count = 0
    state.tick_candidate_first_at = None
    state.tick_candidate_log_ids = []


def _fold_dominance_run(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    ticks = int(e.payload.get("tick_count") or 0)
    bar = int(e.payload.get("min_streak_at_run") or 1)
    if ticks < bar or e.subject_ref is None:
        return  # a run below the live minimum streak is flicker, not focus
    start, end = e.occurred_at, e.ended_at or e.occurred_at
    active = state.arcs.get(state.active.get("interoception", ""))
    if active is not None and active.subject_ref == e.subject_ref and _secs(active.last_seen_at, start) <= cfg.interoception_gap_sec:
        active.segments.append(ArcSegmentV1(began_at=start, ended_at=end))
        active.cumulative_dwell_sec += max(0.0, _secs(start, end))
        active.last_seen_at = end
        _add_evidence(active, _ref(e))
        _retro_bind(state, active, start, end)
        arc = active
    else:
        arc = _open_or_resume(
            state, cfg, kind="interoception", subject=e.subject_ref, label=e.label,
            began=start, now=end, evidence=[_ref(e)],
        )
    if _secs(start, end) > cfg.stuck_segment_sec:
        _add_capped(arc.warnings, "single run longer than 4 h: may be a stuck reading", 8)


def _fold_chat_turn(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    if e.subject_ref is None:
        _buffer(state, e, cfg)  # Orion's own row: context by time, never a conversation
        return
    t = e.occurred_at
    active = state.arcs.get(state.active.get("conversation", ""))
    if active is not None and active.subject_ref == e.subject_ref and _secs(active.last_seen_at, t) <= cfg.conversation_idle_sec:
        _extend(active, t, _ref(e), accrue=True)
        arc = active
    else:
        arc = _open_or_resume(
            state, cfg, kind="conversation", subject=e.subject_ref, label="",
            began=t, now=t, evidence=[_ref(e)],
        )
    if e.correlation_id:
        _add_capped(state.arc_correlations.setdefault(arc.arc_id, []), e.correlation_id, CORRELATION_CAP)


def _concern_arc(state: TemporalSelfStateV1, loop_id: str) -> TemporalSelfArcV1 | None:
    for arc in state.arcs.values():
        if arc.kind == "concern" and arc.subject_ref == loop_id and arc.status != "closed":
            return arc
    return None


def _fold_loop_raised(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    assert e.subject_ref is not None
    t = e.occurred_at
    arc = _concern_arc(state, e.subject_ref)
    if arc is None:
        assert state.day_id is not None
        arc = TemporalSelfArcV1(
            arc_id=arc_id_for(state.day_id, "concern", e.subject_ref, _ref(e)), day_id=state.day_id,
            kind="concern", subject_ref=e.subject_ref, subject_label=e.label, began_at=t,
            last_seen_at=t, status="open", segments=[ArcSegmentV1(began_at=t, ended_at=t)],
        )
        state.arcs[arc.arc_id] = arc
        _add_evidence(arc, _ref(e))
        return
    # A raise is a point, so a concern arc carries no dwell; returns count raises that
    # come back after a quiet stretch (the conversation idle bar).
    if _secs(arc.last_seen_at, t) > cfg.conversation_idle_sec:
        arc.attention_returns += 1
    _extend(arc, t, _ref(e), accrue=False)
    if e.label and not arc.subject_label:
        arc.subject_label = e.label


def _fold_loop_verdict(state: TemporalSelfStateV1, e: TemporalSelfEventV1) -> None:
    assert e.subject_ref is not None
    arc = _concern_arc(state, e.subject_ref)
    if arc is None:
        return  # a verdict for a loop never raised in chat (e.g. reverie scope): no arc
    _add_evidence(arc, _ref(e))
    arc.last_seen_at = max(arc.last_seen_at, e.occurred_at)
    _close(arc, "verdict", e.occurred_at)


def _fold_process(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    kind = PROCESS_KINDS[e.source_kind]
    assert state.day_id is not None and e.subject_ref is not None
    start, end = e.occurred_at, e.ended_at or e.occurred_at
    evidence = [_ref(e)]
    if e.source_kind == "reverie_chain":
        evidence += [f"substrate_reverie_thought:{tid}" for tid in e.payload.get("thought_ids") or []]
    arc_id = arc_id_for(state.day_id, kind, e.subject_ref, evidence[0])
    if arc_id in state.arcs:
        return
    arc = TemporalSelfArcV1(
        arc_id=arc_id, day_id=state.day_id, kind=kind, subject_ref=e.subject_ref,  # type: ignore[arg-type]
        subject_label=e.label, began_at=start, last_seen_at=end, status="open",
        segments=[ArcSegmentV1(began_at=start, ended_at=end)],
        cumulative_dwell_sec=max(0.0, _secs(start, end)), related_refs=list(e.related_refs),
    )
    for ref in evidence:
        _add_evidence(arc, ref)
    if _secs(start, end) > cfg.stuck_segment_sec:
        arc.warnings.append("process longer than 4 h: its start column may predate a stall or retry")
    state.arcs[arc_id] = arc
    if e.source_kind == "reverie_chain":
        state.arc_correlations[arc_id] = list(e.related_refs)[:CORRELATION_CAP]
    for event_id in state.awaiting_arc.pop(f"{kind}|{e.subject_ref}", []):
        _add_capped(arc.expectation_event_ids, event_id, CONTEXT_CAP)
    # Bind what happened inside the process, inclusive of its end, then close it.
    _retro_bind(state, arc, start, end + timedelta(microseconds=1))
    _close(arc, "process_ended", end)


def _attach_by_ref(state: TemporalSelfStateV1, kind: str, e: TemporalSelfEventV1) -> None:
    """Attach ``e`` to the ``kind`` arc whose subject is one of its related refs; if that arc
    does not exist yet, park the link until it does (``_fold_process`` claims it)."""
    found = False
    for arc in state.arcs.values():
        if arc.kind == kind and arc.subject_ref in e.related_refs:
            _add_capped(arc.expectation_event_ids, e.event_id, CONTEXT_CAP)
            found = True
    if not found:
        for ref in e.related_refs:
            _add_capped(state.awaiting_arc.setdefault(f"{kind}|{ref}", []), e.event_id, CONTEXT_CAP)


def _fold_expectation(state: TemporalSelfStateV1, e: TemporalSelfEventV1) -> None:
    if e.source_kind == "dream_hypothesis":
        expires = e.payload.get("expires_at")
        state.expectations[e.event_id] = ExpectationRefV1(
            event_id=e.event_id, source_kind=e.source_kind, committed_at=e.occurred_at,
            expires_at=datetime.fromisoformat(expires) if expires else None,
        )
        _attach_by_ref(state, "sleep", e)
        return
    if e.source_kind == "expectation_verdict":
        committed = e.payload.get("committed_at")
        state.expectations[e.event_id] = ExpectationRefV1(
            event_id=e.event_id, source_kind=e.source_kind,
            committed_at=datetime.fromisoformat(committed) if committed else e.occurred_at,
            resolved_at=e.occurred_at, verdict=e.verdict,
        )
        # Late evidence attaches by reference and never reopens a closed arc.
        _attach_by_ref(state, "reverie", e)
    elif e.source_kind == "action_outcome":
        state.expectations[e.event_id] = ExpectationRefV1(
            event_id=e.event_id, source_kind=e.source_kind, committed_at=e.occurred_at,
            resolved_at=e.occurred_at, verdict=e.verdict,
        )
    # "unscored" is the reverie scorer's own word for "no verdict reached": it stays in the
    # resolved expectations, but it is not a change in Orion (251 of 256 on 10-09 were it).
    if e.verdict == "unscored":
        return
    if not _add_capped(state.self_change_event_ids, e.event_id, DAY_LIST_CAP):
        state.self_change_overflow += 1


def _fold_consolidation_close(state: TemporalSelfStateV1, e: TemporalSelfEventV1) -> None:
    refs = set(e.related_refs)
    bound = False
    for arc in state.arcs.values():
        if arc.kind == "conversation" and refs.intersection(state.arc_correlations.get(arc.arc_id, [])):
            _add_context(arc, e.event_id)
            bound = True
    if not bound:
        _add_day_context(state, e.event_id)


def _buffer(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    item = TemporalSelfContextItemV1(
        event_id=e.event_id, at=e.occurred_at, source_kind=e.source_kind,
        correlation_id=e.correlation_id, process=e.payload.get("process"),
        reason=e.payload.get("reason") if e.source_kind == "attention_row" else None,
        entities=list(e.payload.get("entities") or []), constraint=e.source_kind in CONSTRAINT_KINDS,
    )
    _bind_new_item(state, item, cfg)
    state.context_buffer.append(item)


def _fold_event(state: TemporalSelfStateV1, e: TemporalSelfEventV1, cfg: ReducerConfig) -> None:
    k = e.source_kind
    if k == "chat_turn":
        _fold_chat_turn(state, e, cfg)
    elif k == "field_dominance_run":
        _fold_dominance_run(state, e, cfg)
    elif k == "attention_loop_raised":
        _fold_loop_raised(state, e, cfg)
    elif k == "attention_loop_verdict":
        _fold_loop_verdict(state, e)
    elif k in PROCESS_KINDS:
        _fold_process(state, e, cfg)
    elif k == "consolidation_window_close":
        _fold_consolidation_close(state, e)
    elif k in ("dream_hypothesis", "expectation_verdict", "action_outcome"):
        _fold_expectation(state, e)
        if k == "action_outcome":
            _buffer(state, e, cfg)
    elif k in CONSTRAINT_KINDS:
        if not _add_capped(state.constraint_event_ids, e.event_id, DAY_LIST_CAP):
            state.constraint_overflow += 1
        _buffer(state, e, cfg)
    elif k in CONTEXT_KINDS:
        _buffer(state, e, cfg)


# --------------------------------------------------------------------------- clock and days


def _expire(state: TemporalSelfStateV1, at: datetime, cfg: ReducerConfig) -> None:
    for arc in state.arcs.values():
        if arc.status == "closed":
            continue
        if arc.kind == "conversation" and arc.status == "open" and _secs(arc.last_seen_at, at) > cfg.conversation_idle_sec:
            _suspend(state, arc, by=None)
        if arc.kind in EXCLUSIVE_LANES and arc.status == "suspended" and _secs(arc.last_seen_at, at) > _return_window(arc.kind, cfg):
            _close(arc, "return_window_expired", arc.last_seen_at)


def _roll_day(state: TemporalSelfStateV1, cfg: ReducerConfig) -> None:
    from orion.temporal_self.frame import build_frame  # local: frame imports this module

    assert state.day_id is not None
    old_day = state.day_id
    _, day_end = day_window(old_day, cfg.tz_name)
    carried: list[str] = []
    continuations: list[TemporalSelfArcV1] = []
    next_day = day_id_for(day_end, cfg.tz_name)
    for arc in list(state.arcs.values()):
        if arc.day_id != old_day or arc.status == "closed":
            continue
        _close(arc, "day_boundary", arc.last_seen_at)
        carried.append(arc.arc_id)
        if arc.kind == "concern":
            cont = TemporalSelfArcV1(
                arc_id=arc_id_for(next_day, "concern", arc.subject_ref, arc.arc_id), day_id=next_day,
                kind="concern", subject_ref=arc.subject_ref, subject_label=arc.subject_label,
                began_at=day_end, last_seen_at=day_end, status="open",
                segments=[ArcSegmentV1(began_at=day_end, ended_at=day_end)],
                carried_from_previous_day=True, carried_from_arc_id=arc.arc_id,
            )
            continuations.append(cont)
        else:
            state.carry[f"{arc.kind}|{arc.subject_ref}"] = {"arc_id": arc.arc_id, "last_seen": arc.last_seen_at.isoformat()}
    state.active = {}
    _flush_buffer(state, before=None)
    frame = build_frame(state, day_end, cfg)
    day_arcs = sorted((a for a in state.arcs.values() if a.day_id == old_day), key=lambda a: (a.began_at, a.arc_id))
    state.pending_closed_days.append(
        TemporalSelfDayV1(day_id=old_day, closed_at=day_end, frame=frame, arcs=[a.model_copy(deep=True) for a in day_arcs])
    )
    # Start the next day.
    state.arcs = {c.arc_id: c for c in continuations}
    state.arc_correlations = {}
    state.day_id = next_day
    state.entered_day_with = sorted(carried)
    state.carry = {
        k: v for k, v in state.carry.items()
        if _secs(datetime.fromisoformat(v["last_seen"]), day_end) <= _return_window(k.split("|", 1)[0], cfg)
    }
    _reset_candidate(state)
    state.tick_count_today = 0
    state.tick_winner_count_today = 0
    state.tick_source_gaps_today = 0
    state.tick_winner_refs_today = []
    state.day_context_event_ids = []
    state.day_context_overflow = 0
    state.self_change_event_ids = []
    state.constraint_event_ids = []
    state.self_change_overflow = 0
    state.constraint_overflow = 0
    state.awaiting_arc = {}
    state.expectations = {
        k: v for k, v in state.expectations.items()
        if v.resolved_at is None and (v.expires_at is None or v.expires_at > day_end)
    }


def _roll_to(state: TemporalSelfStateV1, at: datetime, cfg: ReducerConfig) -> None:
    if state.day_id is None:
        state.day_id = day_id_for(at, cfg.tz_name)
        return
    while day_id_for(at, cfg.tz_name) > state.day_id:
        _expire(state, day_window(state.day_id, cfg.tz_name)[1], cfg)
        _roll_day(state, cfg)


# --------------------------------------------------------------------------- public API


def initial_state() -> TemporalSelfStateV1:
    return TemporalSelfStateV1()


def _key(at: datetime, rank: int, ref: str) -> list[str]:
    return [_iso(at), str(rank), ref]


def fold(
    state: TemporalSelfStateV1,
    ticks: Iterable[BroadcastTickView] = (),
    events: Iterable[TemporalSelfEventV1] = (),
    cfg: ReducerConfig = ReducerConfig(),
) -> TemporalSelfStateV1:
    """Fold broadcast ticks and events into a NEW state (the input is not mutated)."""
    s = state.model_copy(deep=True)
    items: list[tuple[list[str], datetime, object]] = []
    for tick in ticks:
        items.append((_key(tick.generated_at, 0, tick.log_id), tick.generated_at, tick))
    for e in events:
        at = _event_available_at(e)
        items.append((_key(at, 1, e.event_id), at, e))
    items.sort(key=lambda x: x[0])
    for key, at, item in items:
        if s.last_key and key <= s.last_key:
            continue
        _roll_to(s, at, cfg)
        _expire(s, at, cfg)
        _flush_buffer(s, before=at - timedelta(seconds=cfg.context_buffer_sec))
        if isinstance(item, BroadcastTickView):
            _fold_tick(s, item, cfg)
            s.cursors["broadcast_tick"] = f"{key[0]}|{item.log_id}"
        else:
            assert isinstance(item, TemporalSelfEventV1)
            _fold_event(s, item, cfg)
            s.cursors[item.source_kind] = f"{key[0]}|{item.source_ref}"
        s.last_key = key
        s.watermark = at
    return s


def fold_broadcast_ticks(state, ticks, cfg: ReducerConfig = ReducerConfig()) -> TemporalSelfStateV1:
    return fold(state, ticks=ticks, cfg=cfg)


def fold_events(state, events, cfg: ReducerConfig = ReducerConfig()) -> TemporalSelfStateV1:
    return fold(state, events=events, cfg=cfg)


def advance_clock(state: TemporalSelfStateV1, now: datetime, cfg: ReducerConfig = ReducerConfig()) -> TemporalSelfStateV1:
    """Declare that nothing strictly before ``now`` will arrive; expire, roll days.

    ``now`` must be the driver's read watermark. Rows stamped before it that are written
    later are dropped as late (the driver should read with a small lag)."""
    s = state.model_copy(deep=True)
    if s.watermark is not None and now < s.watermark:
        return s
    _roll_to(s, now, cfg)
    _expire(s, now, cfg)
    _flush_buffer(s, before=now - timedelta(seconds=cfg.context_buffer_sec))
    active = s.arcs.get(s.active.get("attention", ""))
    if active is not None and s.tick_prev_at is not None and _secs(s.tick_prev_at, now) > cfg.max_tick_gap_sec:
        _suspend(s, active, by=None)
    s.watermark = now
    # Strictly before now is in the past; an item stamped exactly ``now`` may still arrive
    # (rank "" sorts before every real rank).
    s.last_key = [_iso(now), "", ""]
    return s


def close_day(state: TemporalSelfStateV1, day_id: str, cfg: ReducerConfig = ReducerConfig()) -> tuple[TemporalSelfStateV1, TemporalSelfDayV1 | None]:
    """Close ``day_id`` at its local midnight (if not already) and return it."""
    _, end = day_window(day_id, cfg.tz_name)
    s = advance_clock(state, end, cfg)
    s, days = drain_closed_days(s)
    match = [d for d in days if d.day_id == day_id]
    # Days other than the requested one stay queued for the driver.
    s.pending_closed_days = [d for d in days if d.day_id != day_id]
    return s, (match[0] if match else None)


def drain_closed_days(state: TemporalSelfStateV1) -> tuple[TemporalSelfStateV1, list[TemporalSelfDayV1]]:
    s = state.model_copy(deep=True)
    days, s.pending_closed_days = s.pending_closed_days, []
    return s, days
