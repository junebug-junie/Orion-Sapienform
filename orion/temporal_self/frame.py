"""``build_frame``: the bounded "where am I in my day" projection. Pure; never mutates state.

No narrative. Every field is a reference, a count, or a difference of source timestamps.

``active_arc`` is the most recently seen OPEN arc among the foreground lanes (workspace
attention and conversation): what Orion is on. ``previous_arc`` is the most recently
ended foreground arc, where foreground also includes curiosity, sleep and imagery.
Reverie chains (~170 a day with content, live 10-10) and interoception (body) are not
foreground: they are listed in ``arcs_today`` and ``active_by_kind`` but never displace
what Orion was on.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from orion.schemas.temporal_self import (
    ARCS_IN_FRAME_CAP,
    ArcSummaryV1,
    OpenThreadV1,
    TemporalSelfArcV1,
    TemporalSelfFrameV1,
    TemporalSelfStateV1,
)
from orion.temporal_self.arcs import DAY_LIST_CAP, ReducerConfig
from orion.temporal_self.day import day_phase_for, day_window

ACTIVE_KINDS = ("attention", "conversation")
PREVIOUS_KINDS = ("attention", "conversation", "curiosity", "sleep", "imagery")


def summarize(arc: TemporalSelfArcV1) -> ArcSummaryV1:
    return ArcSummaryV1(
        arc_id=arc.arc_id, kind=arc.kind, subject_ref=arc.subject_ref, subject_label=arc.subject_label,
        began_at=arc.began_at, ended_at=arc.ended_at, status=arc.status,
        attention_returns=arc.attention_returns, cumulative_dwell_sec=round(arc.cumulative_dwell_sec, 3),
    )


def _frame_id(day_id: str, as_of: datetime) -> str:
    return hashlib.sha256(f"{day_id}|{as_of.astimezone(timezone.utc).isoformat()}".encode()).hexdigest()[:16]


def build_frame(state: TemporalSelfStateV1, now: datetime, cfg: ReducerConfig = ReducerConfig()) -> TemporalSelfFrameV1:
    now = now.astimezone(timezone.utc)
    day_id = state.day_id or ""
    arcs = sorted((a for a in state.arcs.values() if a.day_id == day_id), key=lambda a: (a.began_at, a.arc_id))

    open_fg = [a for a in arcs if a.status == "open" and a.kind in ACTIVE_KINDS]
    active = max(open_fg, key=lambda a: (a.last_seen_at, a.arc_id), default=None)
    ended_fg = [
        a for a in arcs
        if a.kind in PREVIOUS_KINDS and a.status != "open" and (active is None or a.arc_id != active.arc_id)
    ]
    previous = max(ended_fg, key=lambda a: (a.ended_at or a.last_seen_at, a.arc_id), default=None)

    # Non-reverie arcs first (most recent), reverie fills what is left; then chronological.
    ranked = sorted(arcs, key=lambda a: (a.kind == "reverie", -a.began_at.timestamp(), a.arc_id))[:ARCS_IN_FRAME_CAP]
    listed = sorted(ranked, key=lambda a: (a.began_at, a.arc_id))

    threads = [
        OpenThreadV1(
            subject_ref=a.subject_ref, subject_label=a.subject_label, first_seen_today=a.began_at,
            last_returned=a.last_seen_at, returns_today=a.attention_returns,
            carried_from_previous_day=a.carried_from_previous_day,
        )
        for a in arcs if a.kind == "concern" and a.status != "closed"
    ]

    start, _ = day_window(day_id, cfg.tz_name) if day_id else (now, now)
    pending = sorted(
        (x for x in state.expectations.values() if x.resolved_at is None and (x.expires_at is None or x.expires_at > now)),
        key=lambda x: (x.committed_at, x.event_id),
    )[:DAY_LIST_CAP]
    resolved_all = sorted(
        (x for x in state.expectations.values() if x.resolved_at is not None and start <= x.resolved_at <= now),
        key=lambda x: (x.resolved_at, x.event_id),
    )
    resolved = resolved_all[:DAY_LIST_CAP]

    # Flushed unbound context plus what is still buffered and unbound, without mutating state.
    unbound = set(state.day_context_event_ids)
    unbound.update(
        i.event_id for i in state.context_buffer
        if i.source_kind in ("metacog_observation", "memory_episode", "chat_turn") and not i.bound_arc_ids
        and start <= i.at <= now
    )
    unbound_sorted = sorted(unbound)

    return TemporalSelfFrameV1(
        frame_id=_frame_id(day_id, now), day_id=day_id, as_of=now, day_phase=day_phase_for(now, cfg.tz_name),
        active_arc=summarize(active) if active else None,
        previous_arc=summarize(previous) if previous else None,
        active_by_kind={k: summarize(state.arcs[v]) for k, v in sorted(state.active.items()) if v in state.arcs},
        arcs_today=[summarize(a) for a in listed], arcs_today_total=len(arcs),
        open_threads=threads, expectations_pending=pending, expectations_resolved_today=resolved,
        self_change_event_ids=list(state.self_change_event_ids),
        self_change_overflow=state.self_change_overflow,
        constraint_event_ids=list(state.constraint_event_ids),
        constraint_overflow=state.constraint_overflow,
        expectations_resolved_total=len(resolved_all),
        sleep_arc_ids=[a.arc_id for a in arcs if a.kind == "sleep"],
        unbound_context_event_ids=unbound_sorted[:DAY_LIST_CAP],
        unbound_context_overflow=state.day_context_overflow + max(0, len(unbound_sorted) - DAY_LIST_CAP),
        entered_day_with=list(state.entered_day_with),
        source_cursors=dict(sorted(state.cursors.items())),
        skipped_at_or_before_watermark=state.skipped_today,
        warnings=_warnings(state, arcs, now, cfg),
    )


def _warnings(state: TemporalSelfStateV1, arcs: list[TemporalSelfArcV1], now: datetime, cfg: ReducerConfig) -> list[str]:
    out: list[str] = []
    if len(state.tick_winner_refs_today) == 1 and state.tick_winner_count_today >= cfg.unchanged_winner_min_ticks:
        out.append(f"broadcast winner unchanged all day: {state.tick_winner_refs_today[0]}")
    if state.skipped_today:
        out.append(f"{state.skipped_today} rows at or before the watermark were not folded (re-read or late)")
    if state.correlation_overflow:
        out.append(f"{state.correlation_overflow} chat correlation ids over the per-arc cap (consolidation closes may fall to the day)")
    if state.tick_source_gaps_today:
        out.append(f"broadcast log gaps today: {state.tick_source_gaps_today} (recorder outage or restart)")
    if state.tick_prev_at is not None and (now - state.tick_prev_at).total_seconds() > cfg.max_tick_gap_sec:
        out.append(f"broadcast log silent for {int((now - state.tick_prev_at).total_seconds())} s")
    for a in arcs:
        for w in a.warnings:
            out.append(f"{a.kind} arc {a.arc_id} ({a.subject_ref}): {w}")
    return out
