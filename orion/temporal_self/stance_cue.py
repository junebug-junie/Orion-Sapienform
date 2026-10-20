"""``project_stance_cue``: where Orion is in its day, one short paragraph for stance_react. Pure.

A fixed template over the current ``TemporalSelfFrameV1`` (orion-durable-runs
``GET /temporal-self/frame``). No model writes it. Every clause after the clock line carries the
``[arc:<id>]`` it came from; a clause whose data is missing is left out, never filled in.

Render nothing (``text == ""``, with ``skip_reason``) when the frame is missing or invalid, is
for another local day, or is older than ``max_age_sec``. A fresh frame with no open arc is a
real rest state and renders as one ("No attention or conversation arc is open right now.").
A frame warning "broadcast log silent" drops the attention clauses (the record of what Orion
attends to has stopped) and keeps the rest.

An open ``company`` arc (seam S1: someone at the room camera) adds "Someone has been here since
21:02 [arc:<id>]", naming the person only when the arc carries a confirmed identity label.

A strain arc (strain loop part 1) adds "Strained since Sat 12:07 (GPU jobs backed up, cabinet too
hot) [arc:<id>]" while it is open, or "Was strained 12:07 to 14:30 (...) [arc:<id>]" for today's
latest closed one. The time is when the episode began, even on an earlier day.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from orion.schemas.temporal_self import (
    STANCE_CUE_MAX_CHARS,
    ArcSummaryV1,
    OpenThreadV1,
    TemporalSelfFrameV1,
    TemporalSelfStanceCueV1,
)
from orion.temporal_self.day import DEFAULT_TZ, as_utc, day_id_for, day_phase_for, day_window

DEFAULT_MAX_AGE_SEC = 900.0
MAX_FUTURE_SKEW_SEC = 60.0
MAX_THREAD_REFS = 5
SILENT_BROADCAST_WARNING = "broadcast log silent"

PHASE_WORDS = {
    "pre_dawn": "Pre-dawn",
    "dawn": "Dawn",
    "morning": "Morning",
    "midday": "Midday",
    "afternoon": "Afternoon",
    "dusk": "Dusk",
    "night": "Night",
}


def span_words(seconds: float) -> str:
    """A duration in plain words: "under a minute", "25 min", "2 h", "5 h 13 min"."""
    minutes = int(max(0.0, seconds) // 60)
    if minutes < 1:
        return "under a minute"
    hours, rest = divmod(minutes, 60)
    if hours == 0:
        return f"{rest} min"
    return f"{hours} h" if rest == 0 else f"{hours} h {rest} min"


def _ago(now: datetime, at: datetime | None) -> str | None:
    if at is None:
        return None
    return span_words((now - at).total_seconds())


def _ref(arc: ArcSummaryV1) -> str:
    return f"[arc:{arc.arc_id}]"


def _subject(arc: ArcSummaryV1) -> str:
    return arc.subject_label.strip() or arc.subject_ref


def _skip(now: datetime, reason: str, frame: TemporalSelfFrameV1 | None = None) -> TemporalSelfStanceCueV1:
    return TemporalSelfStanceCueV1(
        rendered_at=now, skip_reason=reason,
        day_id=frame.day_id if frame else None, frame_as_of=frame.as_of if frame else None,
    )


def _active_clause(now: datetime, arc: ArcSummaryV1 | None) -> str:
    if arc is None:
        return "No attention or conversation arc is open right now."
    if arc.kind == "conversation":
        last = _ago(now, arc.last_seen_at)
        if last is not None:
            return f"On now: talking with Juniper, last turn {last} ago {_ref(arc)}."
        return f"On now: a conversation with Juniper, open for {span_words((now - arc.began_at).total_seconds())} {_ref(arc)}."
    returns = ""
    if arc.attention_returns > 0:
        returns = f", back to it {arc.attention_returns} time{'s' if arc.attention_returns != 1 else ''} today"
    return (f"On now: attention on {_subject(arc)} for {span_words((now - arc.began_at).total_seconds())}"
            f"{returns} {_ref(arc)}.")


_PREVIOUS_WORDS = {
    "conversation": "a conversation with Juniper",
    "curiosity": "a curiosity run",
    "sleep": "a sleep cycle",
    "imagery": "an image attempt",
}


def _previous_clause(now: datetime, arc: ArcSummaryV1) -> str:
    what = f"attention on {_subject(arc)}" if arc.kind == "attention" else _PREVIOUS_WORDS.get(arc.kind, f"a {arc.kind} arc")
    if arc.ended_at is not None:
        return f"Before that: {what}, ended {_ago(now, arc.ended_at)} ago {_ref(arc)}."
    if arc.last_seen_at is not None:  # suspended: set aside, may still return
        return f"Before that: {what}, set aside {_ago(now, arc.last_seen_at)} ago {_ref(arc)}."
    return f"Before that: {what} {_ref(arc)}."


def _company_clause(frame: TemporalSelfFrameV1, tz_name: str) -> tuple[str, list[ArcSummaryV1]] | None:
    """Someone at the room camera right now (an open company arc). Names the person only when the
    arc's label exists, and the reducer sets it only from a confirmed identity row; otherwise
    "Someone". A stretch carried over midnight began before the day did: said so, not "00:00"."""
    arc = frame.active_by_kind.get("company")
    if arc is None or arc.status != "open":
        opened = [a for a in frame.arcs_today if a.kind == "company" and a.status == "open"]
        arc = max(opened, key=lambda a: (a.began_at, a.arc_id), default=None)
    if arc is None:
        return None
    label = arc.subject_label.strip()
    who = (label[:1].upper() + label[1:]) if label else "Someone"
    began = as_utc(arc.began_at)
    assert began is not None
    if began <= day_window(frame.day_id, tz_name)[0]:
        since = "before midnight"
    else:
        since = began.astimezone(ZoneInfo(tz_name)).strftime("%H:%M")
    return f"{who} has been here since {since} {_ref(arc)}.", [arc]


# The regulation step's strain causes (``orion.regulation.arousal.STRAIN_CAUSES``) in plain words.
STRAIN_CAUSE_WORDS = {"cabinet_hot": "cabinet too hot", "gpu_queue_sustained": "GPU jobs backed up"}


def strain_cause_text(label: str) -> str:
    """A strain arc's label ("gpu_queue_sustained+cabinet_hot") as words; an unknown code stays as
    it is, never dropped."""
    return ", ".join(STRAIN_CAUSE_WORDS.get(c, c) for c in label.split("+") if c)


def _strain_clause(frame: TemporalSelfFrameV1, tz_name: str) -> tuple[str, list[ArcSummaryV1]] | None:
    tz = ZoneInfo(tz_name)
    strains = [a for a in frame.arcs_today if a.kind == "strain"]
    active = frame.active_by_kind.get("strain")
    arc = active if active is not None and active.status == "open" else None
    if arc is None:
        opened = [a for a in strains if a.status == "open"]
        arc = max(opened, key=lambda a: (a.began_at, a.arc_id), default=None)
    closed = arc is None
    if closed:
        arc = max((a for a in strains if a.status == "closed"),
                  key=lambda a: (a.ended_at or a.began_at, a.arc_id), default=None)
    if arc is None:
        return None
    began = as_utc(arc.episode_began_at or arc.began_at)
    assert began is not None

    def clock(at) -> str:
        local = as_utc(at).astimezone(tz)
        return local.strftime("%H:%M") if local.date().isoformat() == frame.day_id else local.strftime("%a %H:%M")

    causes = strain_cause_text(arc.subject_label)
    why = f" ({causes})" if causes else ""
    if closed:
        return f"Was strained {clock(began)} to {clock(arc.ended_at or arc.began_at)}{why} {_ref(arc)}.", [arc]
    return f"Strained since {clock(began)}{why} {_ref(arc)}.", [arc]


def _thread_arc(thread: OpenThreadV1, concern: dict[str, ArcSummaryV1]) -> ArcSummaryV1 | None:
    """The arc a thread cites: its own ``arc_id`` (frames since thread arc ids), else the open
    concern arc of the same subject in ``arcs_today`` (older frames; may have fallen off the cap)."""
    if thread.arc_id:
        return ArcSummaryV1(
            arc_id=thread.arc_id, kind="concern", subject_ref=thread.subject_ref,
            subject_label=thread.subject_label, began_at=thread.first_seen_today, status="open",
            attention_returns=thread.returns_today, last_seen_at=thread.last_returned,
        )
    return concern.get(thread.subject_ref)


def _threads_clause(frame: TemporalSelfFrameV1, limit: int = MAX_THREAD_REFS) -> tuple[str, list[ArcSummaryV1]] | None:
    """Every open thread with its own citation, most recently returned first.

    Over ``MAX_THREAD_REFS`` threads the clause says how many it lists ("the 5 most recent"), so
    every thread it renders is cited and the count stays true. A thread with no arc to cite (an
    older frame whose concern arc fell off ``arcs_today``) is counted, not cited. ``limit`` shrinks
    when the cue is over its length cap, before the clause itself would be dropped."""
    if not frame.open_threads:
        return None
    concern = {a.subject_ref: a for a in frame.arcs_today if a.kind == "concern" and a.status != "closed"}
    ordered = sorted(frame.open_threads, key=lambda t: (t.last_returned, t.subject_ref), reverse=True)
    arcs = [a for a in (_thread_arc(t, concern) for t in ordered) if a is not None]
    if not arcs:
        return None
    n = len(frame.open_threads)
    listed = arcs[:max(1, limit)]
    refs = " ".join(_ref(a) for a in listed)
    if len(listed) == n:
        cited = ""
    elif len(arcs) == n:
        cited = ", the most recent" if len(listed) == 1 else f", the {len(listed)} most recent"
    elif len(listed) == 1:
        cited = ", 1 with its record"
    else:
        cited = f", {len(listed)} with their records"
    return f"{n} open thread{'s' if n != 1 else ''} from conversation{cited} {refs}.", listed


def project_stance_cue(
    frame: TemporalSelfFrameV1 | dict[str, Any] | None,
    now: datetime,
    *,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    tz_name: str = DEFAULT_TZ,
    max_chars: int = STANCE_CUE_MAX_CHARS,
) -> TemporalSelfStanceCueV1:
    now = now.astimezone(timezone.utc)
    if frame is None:
        return _skip(now, "fetch_failed")
    if not isinstance(frame, TemporalSelfFrameV1):
        try:
            frame = TemporalSelfFrameV1.model_validate(frame)
        except ValidationError:
            return _skip(now, "invalid_frame")
    if frame.day_id != day_id_for(now, tz_name):
        return _skip(now, "wrong_day", frame)
    as_of = as_utc(frame.as_of)
    assert as_of is not None
    if (now - as_of).total_seconds() > max_age_sec:
        return _skip(now, "stale", frame)
    if (as_of - now).total_seconds() > MAX_FUTURE_SKEW_SEC:
        # A frame "from the future" means one of the two clocks is wrong: no ages can be trusted.
        return _skip(now, "clock_skew", frame)

    silent = any(w.startswith(SILENT_BROADCAST_WARNING) for w in frame.warnings)
    dropped = 0
    local = now.astimezone(ZoneInfo(tz_name))
    phase = PHASE_WORDS.get(day_phase_for(now, tz_name), day_phase_for(now, tz_name))
    head = (f"{phase}, {local.strftime('%a %H:%M')} local "
            f"(day record as of {as_of.astimezone(ZoneInfo(tz_name)).strftime('%H:%M')}).")

    # (clause, arcs it cites). Order is priority: the cap drops from the bottom.
    clauses: list[tuple[str, list[ArcSummaryV1]]] = []
    active = frame.active_arc
    if active is not None and active.kind == "attention" and silent:
        dropped += 1
        active = None
    elif active is None and silent:
        # No open arc while the attention log is silent is not rest: the instrument stopped.
        dropped += 1
    else:
        clauses.append((_active_clause(now, active), [active] if active else []))

    previous = frame.previous_arc
    if previous is not None and not (silent and previous.kind == "attention"):
        clauses.append((_previous_clause(now, previous), [previous]))
    elif previous is not None:
        dropped += 1

    if active is None or active.kind != "conversation":
        talks = [a for a in frame.arcs_today if a.kind == "conversation" and a.last_seen_at is not None]
        if talks:
            last = max(talks, key=lambda a: (a.last_seen_at, a.arc_id))
            if previous is None or last.arc_id != previous.arc_id:
                clauses.append((f"Juniper last spoke {_ago(now, last.last_seen_at)} ago {_ref(last)}.", [last]))

    strain = _strain_clause(frame, tz_name)
    if strain is not None:
        clauses.append(strain)

    company = _company_clause(frame, tz_name)
    if company is not None:
        clauses.append(company)

    thread_clause = _threads_clause(frame)
    if thread_clause is not None:
        clauses.append(thread_clause)

    sleeps = [a for a in frame.arcs_today if a.kind == "sleep"]
    if sleeps:
        last_sleep = max(sleeps, key=lambda a: (a.ended_at or a.last_seen_at or a.began_at, a.arc_id))
        if last_sleep.status == "open":
            clauses.append((f"Asleep since {_ago(now, last_sleep.began_at)} ago {_ref(last_sleep)}.", [last_sleep]))
        else:
            woke = last_sleep.ended_at or last_sleep.last_seen_at or last_sleep.began_at
            clauses.append((f"Last slept {_ago(now, woke)} ago {_ref(last_sleep)}.", [last_sleep]))

    kept = list(clauses)
    thread_limit = MAX_THREAD_REFS
    while kept and len(" ".join([head] + [c for c, _ in kept])) > max_chars:
        if thread_clause is not None and kept[-1] is thread_clause and len(thread_clause[1]) > 1:
            # Cite fewer threads (the count stays true) before dropping the clause.
            thread_limit = len(thread_clause[1]) - 1
            thread_clause = _threads_clause(frame, thread_limit)
            kept[-1] = thread_clause
            continue
        kept.pop()
        dropped += 1
    text = " ".join([head] + [c for c, _ in kept])
    if len(text) > max_chars:  # the head alone cannot overrun; belt and braces
        text = head[:max_chars]

    arc_ids: list[str] = []
    labels: dict[str, list[str]] = {}
    for _, arcs in kept:
        for arc in arcs:
            if arc.arc_id not in arc_ids:
                arc_ids.append(arc.arc_id)
            if arc.kind == "company":
                # A person's name is not a label of Orion's day: matching it would mark every
                # uncited mention of Juniper as an unverified day claim.
                continue
            label = arc.subject_label.strip()
            if label and label in text:  # only what Orion was actually shown is a match label
                labels.setdefault(label, [])
                if arc.arc_id not in labels[label]:
                    labels[label].append(arc.arc_id)
    return TemporalSelfStanceCueV1(
        text=text, day_id=frame.day_id, frame_as_of=as_of, rendered_at=now,
        arc_ids=arc_ids, labels=labels, clauses_dropped=dropped,
    )
