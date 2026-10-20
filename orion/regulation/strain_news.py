"""Strain news: rare, budgeted pulses that let Orion's attention notice a strain. Pure.

Spec: docs/superpowers/specs/2026-10-20-strain-loop-design.md, Part 2 ("Notice").

The strain level (``orion.regulation.arousal.classify_arousal``) is a reflex: 7 gates read it,
nothing attends to it. This module turns the level into a few news events that the attention
contest can see, without letting a 54 h strain win every 30 s tick (attention has one winner per
tick, no per-signal cap and no habituation). The level itself never competes.

Events (one at most per call):

* ``onset`` -- a new strained episode (episode identity is the reading's own ``since``, which
  resets on every level change and survives a durable-runs restart).
* ``escalation`` -- it got worse in kind: a strain cause that was not yet in this episode's cause
  set joins it (``STRAIN_CAUSE_REASONS``), or the cabinet first crosses critical in this episode
  (the thermal gate's own ``DEFAULT_CRITICAL_C``; once per episode, no re-arm: a cabinet hovering
  around 34 C re-crosses it many times a day and that is not worse in kind).
* ``reminder`` -- still strained 30 min, 1 h, 2 h, 4 h, 8 h, then every 8 h after the onset was
  sent (``reminder_offsets``): it fades and comes back, less often the longer it lasts. Anchored
  on the sent onset, not the reading's ``since``, so the 30 min reminder clears the 30 min gap.

Relief (strained -> idle/engaged) is not an event: attention scores only the bad direction.
It ends the episode.

Budget (hard rules; ``scripts/analysis/replay_strain_news.py`` replays them over live history):

1. At least ``MIN_GAP`` (30 min) between any two events. A gap-blocked onset or escalation is
   DEFERRED (kept pending and fired at the first tick the gap allows). A gap-blocked reminder is
   DROPPED: it waits for the next reminder slot.
2. At most ``MAX_EVENTS_PER_DAY`` (8) per local day (``tz``; resets at local midnight). Reminders
   may not use the last ``RESERVED_FOR_NEWS`` (2) slots, so a day of reminders still leaves room
   for a real change. Past the cap nothing more is written that day. A cap-blocked onset or
   reminder is DROPPED (a day-old onset is not news). A cap-blocked ESCALATION is carried to the
   next local day and sent only if the episode is still open and the change still holds (the
   cause is still present / the cabinet is still critical).
3. An unknown, stale, future-stamped or missing reading never fires and never counts as relief:
   the state is kept as it was (only the local-day counter rolls over).

Restart safety: the caller persists each event (``substrate_node_prediction_error_history``) and
rebuilds the state with ``seed_state`` from those rows: an onset already recorded for the
current episode is never fired again, today's count and the gap carry over, and reminder slots
that already elapsed are consumed. What a restart loses (documented, conservative): a pending
deferred escalation, and an escalation whose cause joined while the process was down.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Iterable, Literal, Optional, Tuple

from orion.autonomy.cabinet_heat import REFLEX_CABINET_HOT
from orion.autonomy.thermal_gate import DEFAULT_CRITICAL_C, DEFAULT_CRITICAL_REARM_C
from orion.regulation.arousal_view import DEFAULT_MAX_AGE_SEC, MAX_FUTURE_SKEW_SEC
from orion.schemas.regulation import ArousalReadingV1

STRAIN_NODE_ID = "node:substrate.strain"
# The value written on an event. A trigger: "it fired", not a size.
EVENT_VALUE = 1.0
REST_VALUE = 0.0

MIN_GAP = timedelta(minutes=30)
MAX_EVENTS_PER_DAY = 8
RESERVED_FOR_NEWS = 2  # the last slots of a day: onset/escalation only
FIRST_REMINDERS = (
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=2),
    timedelta(hours=4),
    timedelta(hours=8),
)
REMINDER_PERIOD = timedelta(hours=8)

# The strain causes classify_arousal names on a strained reading (orion/regulation/arousal.py:220).
# Other reasons on a strained reading (strain_holding, strain_clearing, prev_dropped:gap,
# gpu_fault:...) describe the latch or the inputs, not a new cause.
STRAIN_CAUSE_REASONS: frozenset[str] = frozenset({REFLEX_CABINET_HOT, "gpu_queue_sustained"})

EventKind = Literal["onset", "escalation", "reminder"]


@dataclass(frozen=True)
class StrainNewsEvent:
    kind: EventKind
    at: datetime
    episode_since: datetime
    causes: Tuple[str, ...]          # the episode's cause set after this event
    new_causes: Tuple[str, ...]      # escalation only: causes that joined
    critical: bool                   # escalation only: the cabinet crossed critical
    cabinet_temp_c: Optional[float]
    reminder_index: Optional[int]    # reminder only: 0 = 30 min, 1 = 1 h, ...
    events_today: int                # including this one
    value: float = EVENT_VALUE

    def as_log_fields(self) -> str:
        return (
            f"kind={self.kind} episode_since={self.episode_since.isoformat()} "
            f"causes={'+'.join(self.causes) or '-'} new_causes={'+'.join(self.new_causes) or '-'} "
            f"critical={self.critical} cabinet_temp_c={self.cabinet_temp_c} "
            f"reminder_index={self.reminder_index} events_today={self.events_today}"
        )


@dataclass(frozen=True)
class StrainNewsState:
    """Everything the next call needs. Empty state = no episode, nothing sent today."""

    day: Optional[date] = None
    events_today: int = 0
    last_event_at: Optional[datetime] = None
    episode_since: Optional[datetime] = None
    episode_causes: frozenset[str] = field(default_factory=frozenset)
    critical_announced: bool = False
    onset_pending: bool = False
    escalation_pending: bool = False
    pending_new_causes: Tuple[str, ...] = ()
    pending_critical: bool = False
    reminders_consumed: int = 0
    # When this episode's onset was actually sent. Reminders count from here, not from the
    # reading's ``since``: the onset lands one regulate step (and up to one tick) after ``since``,
    # so a schedule anchored on ``since`` put the 30 min reminder inside the 30 min gap.
    episode_noticed_at: Optional[datetime] = None


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def local_day(now: datetime, tz: tzinfo) -> date:
    return _utc(now).astimezone(tz).date()


def reminder_offsets(elapsed: timedelta) -> int:
    """How many reminder slots have come due after ``elapsed`` of strain."""
    if elapsed < FIRST_REMINDERS[0]:
        return 0
    n = sum(1 for off in FIRST_REMINDERS if off <= elapsed)
    if n < len(FIRST_REMINDERS):
        return n
    return n + int((elapsed - FIRST_REMINDERS[-1]) // REMINDER_PERIOD)


def reminder_offset(index: int) -> timedelta:
    """The episode age at which reminder ``index`` (0-based) comes due."""
    if index < len(FIRST_REMINDERS):
        return FIRST_REMINDERS[index]
    return FIRST_REMINDERS[-1] + REMINDER_PERIOD * (index - len(FIRST_REMINDERS) + 1)


def reading_is_known(reading: Optional[ArousalReadingV1], now: datetime,
                     max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> bool:
    """Same doubt rules as ``arousal_view``: missing, stale, future-stamped or ``unknown`` is not
    evidence. (The caller normally judged the Redis value with ``arousal_view`` already; this is
    the pure function's own guard.)"""
    if reading is None or reading.arousal_level == "unknown":
        return False
    age = (_utc(now) - _utc(reading.observed_at)).total_seconds()
    return -MAX_FUTURE_SKEW_SEC <= age <= float(max_age_sec)


def _causes(reading: ArousalReadingV1) -> frozenset[str]:
    return frozenset(r for r in reading.reasons if r in STRAIN_CAUSE_REASONS)


def _critical_now(temp: Optional[float]) -> Optional[bool]:
    """True: at/above the thermal gate's critical. None: unmeasured. False otherwise."""
    if temp is None:
        return None
    return temp >= DEFAULT_CRITICAL_C


def strain_news(
    prev: Optional[StrainNewsState],
    reading: Optional[ArousalReadingV1],
    now: datetime,
    tz: tzinfo,
    *,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
) -> tuple[Optional[StrainNewsEvent], StrainNewsState]:
    """One step. Returns (event or None, new state). Never raises on a valid reading."""
    now = _utc(now)
    state = prev or StrainNewsState()
    today = local_day(now, tz)
    if state.day != today:
        state = replace(state, day=today, events_today=0)

    if not reading_is_known(reading, now, max_age_sec):
        return None, state  # unknown: no event, and not relief

    assert reading is not None
    if reading.arousal_level != "strained":
        # Relief ends the episode. Not an event: attention scores only the bad direction.
        return None, replace(
            state, episode_since=None, episode_causes=frozenset(), critical_announced=False,
            onset_pending=False, escalation_pending=False, pending_new_causes=(),
            pending_critical=False, reminders_consumed=0, episode_noticed_at=None,
        )

    since = _utc(reading.since)
    causes = _causes(reading)
    temp = reading.cabinet_temp_c
    crit = _critical_now(temp)

    if state.episode_since != since:
        # A new episode: onset pending. The causes and critical state at onset are part of the
        # onset, not escalations. Reminder slots that already elapsed (a long episode first seen
        # now) are consumed: the onset is the first notice.
        state = replace(
            state, episode_since=since, episode_causes=causes,
            critical_announced=bool(crit), onset_pending=True, escalation_pending=False,
            pending_new_causes=(), pending_critical=False,
            reminders_consumed=0, episode_noticed_at=None,
        )
    else:
        joined = causes - state.episode_causes
        # Once per episode: a cabinet hovering at 33-34.5 C re-crosses critical many times a day
        # (replay 2026-10-20: 5 critical escalations in one day with a 33 C re-arm). Crossing
        # again is not worse in kind.
        crossed = crit is True and not state.critical_announced
        if joined or crossed:
            merged = state.episode_causes | joined
            if state.onset_pending:
                # The onset has not been sent yet: it will carry the fuller picture.
                state = replace(state, episode_causes=merged,
                                critical_announced=state.critical_announced or crossed)
            else:
                state = replace(
                    state, episode_causes=merged,
                    critical_announced=state.critical_announced or crossed,
                    escalation_pending=True,
                    pending_new_causes=tuple(sorted(set(state.pending_new_causes) | joined)),
                    pending_critical=state.pending_critical or crossed,
                )

    noticed = state.episode_noticed_at
    due = reminder_offsets(now - noticed) if noticed is not None else 0
    reminder_due = (not state.onset_pending) and noticed is not None and due > state.reminders_consumed
    gap_ok = state.last_event_at is None or now - state.last_event_at >= MIN_GAP

    kind: Optional[EventKind] = None
    if state.onset_pending:
        kind = "onset"
    elif state.escalation_pending:
        kind = "escalation"
    elif reminder_due:
        kind = "reminder"
    if kind is None:
        return None, state

    cap = MAX_EVENTS_PER_DAY - (RESERVED_FOR_NEWS if kind == "reminder" else 0)
    if state.events_today >= cap:
        if kind == "escalation":
            # Carried, not dropped: nothing more is written today, but an open episode that got
            # worse in kind is still news tomorrow while it holds (replay 2026-10-18: the first
            # critical-heat escalation was the 9th event of the day and would never have been
            # sent). Sent at the first tick of the next local day if still true.
            return None, state
        # Onset / reminder: dropped. A dropped onset still marks the episode noticed (its
        # reminders run from here), so it is not re-sent tomorrow as day-old news.
        return None, replace(
            state, onset_pending=False, reminders_consumed=max(state.reminders_consumed, due),
            episode_noticed_at=noticed or now,
        )
    if not gap_ok:
        if kind == "reminder":
            return None, replace(state, reminders_consumed=due)  # waits for the next slot
        return None, state  # onset / escalation: deferred

    if kind == "escalation" and state.events_today == 0 and state.last_event_at is not None \
            and local_day(state.last_event_at, tz) != today:
        # Carried over the day line by the cap: send only what still holds. A cause that left,
        # or a cabinet that cooled below critical, is no longer news.
        # What is dropped here was never sent: it goes back to "not announced", so it escalates
        # if it comes back.
        still = tuple(c for c in state.pending_new_causes if c in causes)
        still_crit = state.pending_critical and crit is True
        unsent = frozenset(state.pending_new_causes) - frozenset(still)
        state = replace(
            state, episode_causes=state.episode_causes - unsent,
            critical_announced=state.critical_announced and not (state.pending_critical and not still_crit),
            pending_new_causes=still, pending_critical=still_crit,
            escalation_pending=bool(still or still_crit),
        )
        if not state.escalation_pending:
            return None, state

    count = state.events_today + 1
    event = StrainNewsEvent(
        kind=kind,
        at=now,
        episode_since=since,
        causes=tuple(sorted(state.episode_causes)),
        new_causes=state.pending_new_causes if kind == "escalation" else (),
        critical=state.pending_critical if kind == "escalation" else bool(state.critical_announced),
        cabinet_temp_c=temp,
        reminder_index=(due - 1) if kind == "reminder" else None,
        events_today=count,
    )
    new_state = replace(
        state, events_today=count, last_event_at=now, onset_pending=False,
        escalation_pending=False, pending_new_causes=(), pending_critical=False,
        # Any event resets nothing in the reminder schedule (it counts from the sent onset),
        # but slots already due are consumed so a reminder never follows right behind news.
        reminders_consumed=due,
        episode_noticed_at=noticed or now,
    )
    return event, new_state


def seed_state(
    event_times: Iterable[datetime],
    reading: Optional[ArousalReadingV1],
    now: datetime,
    tz: tzinfo,
    *,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
) -> StrainNewsState:
    """Rebuild the state after a restart from the persisted event rows (value > 0 rows of
    ``node:substrate.strain`` in ``substrate_node_prediction_error_history``).

    - today's count and the last event time come from the rows;
    - if the reading is a known strained episode and an event was recorded at or after its
      ``since``, its onset was already sent: never sent again. Its causes and critical state
      are taken as already announced, and every reminder slot that already elapsed is consumed.
    """
    now = _utc(now)
    times = sorted(_utc(t) for t in event_times if _utc(t) <= now)
    today = local_day(now, tz)
    state = StrainNewsState(
        day=today,
        events_today=sum(1 for t in times if local_day(t, tz) == today),
        last_event_at=times[-1] if times else None,
    )
    if not reading_is_known(reading, now, max_age_sec) or reading.arousal_level != "strained":
        return state
    since = _utc(reading.since)
    episode_times = [t for t in times if t >= since]
    if not episode_times:
        return state  # nothing sent for this episode yet: the next step fires its onset
    noticed = episode_times[0]
    temp = reading.cabinet_temp_c
    # Conservative: at or near critical (>= the thermal gate's 33 C re-arm point) a pre-restart
    # critical may already have been announced, so it counts as announced.
    announced = temp is not None and temp >= DEFAULT_CRITICAL_REARM_C
    return replace(
        state, episode_since=since, episode_causes=_causes(reading),
        critical_announced=announced, episode_noticed_at=noticed,
        reminders_consumed=reminder_offsets(now - noticed),
    )


def rest_write_allowed(state: StrainNewsState, now: datetime, orienting_window_sec: float) -> bool:
    """The calm 0.0 is not written while the last event is still inside attention's orienting
    window: the 1.0 must be carried unchanged so its age is the event's age (world-first fades
    an event-written node by that age and stops it at the window). Relief (no open episode)
    ends the hold at once: a strain that cleared is no longer news."""
    if state.last_event_at is None or state.episode_since is None:
        return True
    return (_utc(now) - state.last_event_at).total_seconds() >= float(orienting_window_sec)


__all__ = [
    "EVENT_VALUE",
    "FIRST_REMINDERS",
    "MAX_EVENTS_PER_DAY",
    "MIN_GAP",
    "REMINDER_PERIOD",
    "RESERVED_FOR_NEWS",
    "REST_VALUE",
    "STRAIN_CAUSE_REASONS",
    "STRAIN_NODE_ID",
    "StrainNewsEvent",
    "StrainNewsState",
    "local_day",
    "reading_is_known",
    "reminder_offset",
    "reminder_offsets",
    "rest_write_allowed",
    "seed_state",
    "strain_news",
]
