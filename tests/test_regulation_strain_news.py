"""Strain news (orion/regulation/strain_news.py): schedule, budget, unknown, restart."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from orion.regulation import strain_news as sn
from orion.regulation.strain_news import (
    MAX_EVENTS_PER_DAY,
    STRAIN_NODE_ID,
    StrainNewsState,
    reminder_offset,
    reminder_offsets,
    rest_write_allowed,
    seed_state,
    strain_news,
)
from orion.schemas.regulation import ArousalReadingV1

TZ = ZoneInfo("America/Denver")
# 06:07 MDT -- far from local midnight so day-cap tests are explicit about crossing it.
T0 = datetime(2026, 10, 18, 12, 7, tzinfo=timezone.utc)


def R(now, *, level="strained", since=T0, reasons=("gpu_queue_sustained",), temp=32.0, observed=None):
    return ArousalReadingV1(
        arousal_level=level, since=since, observed_at=observed or now,
        reasons=list(reasons), cabinet_temp_c=temp,
    )


def run(readings, state=None, step=timedelta(minutes=2)):
    """readings: callable(now) -> reading or None; runs from T0 for len steps."""
    events = []
    for now, reading in readings:
        ev, state = strain_news(state, reading, now, TZ)
        if ev is not None:
            events.append(ev)
    return events, state


def strained_for(hours, *, start=T0, step=timedelta(minutes=2), **kw):
    n = int(timedelta(hours=hours) / step)
    return [(start + step * i, R(start + step * i, **kw)) for i in range(n + 1)]


def test_onset_fires_once_when_level_turns_strained():
    events, _ = run(strained_for(0.2))
    assert [e.kind for e in events] == ["onset"]
    assert events[0].at == T0 and events[0].causes == ("gpu_queue_sustained",)
    assert events[0].value == 1.0


def test_reminder_schedule_30m_1h_2h_4h_8h_then_every_8h():
    assert [reminder_offset(i) for i in range(8)] == [
        timedelta(minutes=30), timedelta(hours=1), timedelta(hours=2), timedelta(hours=4),
        timedelta(hours=8), timedelta(hours=16), timedelta(hours=24), timedelta(hours=32),
    ]
    for i in range(10):
        assert reminder_offsets(reminder_offset(i)) == i + 1
        assert reminder_offsets(reminder_offset(i) - timedelta(seconds=1)) == i


def test_reminders_fire_on_schedule_within_a_day():
    events, _ = run(strained_for(9))
    got = [(e.kind, e.at - T0) for e in events]
    assert got == [
        ("onset", timedelta(0)),
        ("reminder", timedelta(minutes=30)),
        ("reminder", timedelta(hours=1)),
        ("reminder", timedelta(hours=2)),
        ("reminder", timedelta(hours=4)),
        ("reminder", timedelta(hours=8)),
    ]
    assert [e.reminder_index for e in events[1:]] == [0, 1, 2, 3, 4]


def test_every_8h_after_the_first_day_and_never_more_than_cap():
    events, _ = run(strained_for(54))
    by_day = {}
    for e in events:
        by_day.setdefault(e.at.astimezone(TZ).date(), []).append(e)
    assert all(len(v) <= MAX_EVENTS_PER_DAY for v in by_day.values())
    gaps = [b.at - a.at for a, b in zip(events, events[1:])]
    assert min(gaps) >= sn.MIN_GAP
    late = [e for e in events if e.at - T0 > timedelta(hours=9)]
    assert all((e.at - T0 - timedelta(hours=8)) % timedelta(hours=8) == timedelta(0) for e in late)


def test_new_cause_is_an_escalation_and_fires_once():
    seq = strained_for(0.2)
    t = T0 + timedelta(minutes=40)
    seq += [(t + timedelta(minutes=2 * i), R(t + timedelta(minutes=2 * i),
             reasons=("gpu_queue_sustained", "cabinet_hot"), temp=33.5)) for i in range(5)]
    events, state = run(seq)
    assert [e.kind for e in events] == ["onset", "escalation"]
    assert events[1].new_causes == ("cabinet_hot",) and events[1].at == t
    assert state.episode_causes == frozenset({"cabinet_hot", "gpu_queue_sustained"})


def test_only_strain_causes_count_not_latch_reasons():
    seq = strained_for(0.2)
    t = T0 + timedelta(minutes=20)
    seq += [(t, R(t, reasons=("strain_holding", "gpu_fault:x:y")))]
    events, _ = run(seq)
    assert [e.kind for e in events] == ["onset"]


def test_critical_crossing_is_an_escalation_once_per_episode():
    seq = strained_for(0.2, temp=32.0)
    for minutes, temp in ((40, 34.2), (42, 34.5), (44, 33.5), (80, 32.5), (120, 34.1), (200, 34.6)):
        t = T0 + timedelta(minutes=minutes)
        seq.append((t, R(t, temp=temp)))
    events, _ = run(seq)
    esc = [e for e in events if e.kind == "escalation"]
    assert [e.at - T0 for e in esc] == [timedelta(minutes=40)]
    assert esc[0].critical


def test_critical_at_onset_is_part_of_the_onset_not_an_escalation():
    events, _ = run(strained_for(0.3, temp=34.5))
    assert [e.kind for e in events] == ["onset"] and events[0].critical


def test_critical_constant_is_the_thermal_gate_one():
    from orion.autonomy.thermal_gate import DEFAULT_CRITICAL_C

    assert sn.DEFAULT_CRITICAL_C is DEFAULT_CRITICAL_C == 34.0


def test_gap_defers_an_escalation_but_drops_a_reminder():
    def seq(join_at_min, until_min):
        out = []
        for m in range(0, until_min, 2):
            t = T0 + timedelta(minutes=m)
            causes = ("gpu_queue_sustained", "cabinet_hot") if m >= join_at_min else ("gpu_queue_sustained",)
            out.append((t, R(t, reasons=causes)))
        return out

    # A cause joins at +10, inside the 30 min gap after onset: deferred to +30, not dropped.
    events, _ = run(seq(10, 64))
    assert [(e.kind, e.at - T0) for e in events] == [
        ("onset", timedelta(0)),
        ("escalation", timedelta(minutes=30)),
        ("reminder", timedelta(hours=1)),
    ]
    # A cause joins at +96: the 2 h reminder is 24 min after it, gap-blocked, and DROPPED: the
    # next reminder is the 4 h slot.
    events, _ = run(seq(96, 250))
    assert [(e.kind, e.at - T0) for e in events] == [
        ("onset", timedelta(0)),
        ("reminder", timedelta(minutes=30)),
        ("reminder", timedelta(hours=1)),
        ("escalation", timedelta(minutes=96)),
        ("reminder", timedelta(hours=4)),
    ]


def test_daily_cap_and_reminders_leave_the_last_two_slots_for_news():
    state = StrainNewsState(day=T0.astimezone(TZ).date(), events_today=6,
                            last_event_at=T0 - timedelta(hours=1), episode_since=T0,
                            episode_causes=frozenset({"gpu_queue_sustained"}), reminders_consumed=0)
    t = T0 + timedelta(minutes=31)
    ev, state = strain_news(state, R(t), t, TZ)
    assert ev is None                       # reminder may not take slot 7 or 8
    ev, state = strain_news(state, R(t, reasons=("gpu_queue_sustained", "cabinet_hot")), t, TZ)
    assert ev is not None and ev.kind == "escalation" and ev.events_today == 7
    state = sn.replace(state, events_today=8, last_event_at=t - timedelta(hours=1))
    t2 = t + timedelta(minutes=5)
    ev, state = strain_news(state, R(t2, temp=34.5), t2, TZ)
    assert ev is None and state.escalation_pending      # blocked by the cap, carried


def test_cap_blocked_escalation_is_sent_next_local_day_only_if_it_still_holds():
    late = datetime(2026, 10, 18, 23, 0, tzinfo=TZ).astimezone(timezone.utc)
    full = StrainNewsState(day=late.astimezone(TZ).date(), events_today=8,
                           last_event_at=late - timedelta(hours=1), episode_since=T0,
                           episode_causes=frozenset({"gpu_queue_sustained"}),
                           episode_noticed_at=T0, reminders_consumed=10)
    ev, state = strain_news(full, R(late, temp=34.5), late, TZ)
    assert ev is None and state.escalation_pending
    after = late + timedelta(hours=1, minutes=5)          # 00:05 local, next day
    ev, _ = strain_news(state, R(after, temp=34.3), after, TZ)
    assert ev is not None and ev.kind == "escalation" and ev.critical and ev.events_today == 1
    # The cabinet cooled overnight: nothing left to say now...
    ev, st2 = strain_news(state, R(after, temp=31.0), after, TZ)
    assert ev is None and not st2.escalation_pending
    # ...but it was never said, so a later crossing still escalates.
    again = after + timedelta(hours=3)
    ev, _ = strain_news(st2, R(again, temp=34.2), again, TZ)
    assert ev is not None and ev.kind == "escalation" and ev.critical
    # The strain ended overnight: the carried escalation goes with the episode.
    ev, st3 = strain_news(state, R(after, level="idle", since=after), after, TZ)
    assert ev is None and not st3.escalation_pending


def test_local_day_resets_the_count():
    local_late = datetime(2026, 10, 18, 23, 50, tzinfo=TZ).astimezone(timezone.utc)
    state = StrainNewsState(day=local_late.astimezone(TZ).date(), events_today=8,
                            last_event_at=local_late - timedelta(hours=2))
    ev, state = strain_news(state, R(local_late, since=local_late), local_late, TZ)
    assert ev is None
    after = local_late + timedelta(minutes=20)  # 00:10 local, next day
    ev, state = strain_news(state, R(after, since=after), after, TZ)
    assert ev is not None and ev.kind == "onset" and ev.events_today == 1


@pytest.mark.parametrize("reading", [
    None,
    "unknown_level",
    "stale",
    "future",
])
def test_unknown_never_fires_and_is_not_relief(reading):
    events, state = run(strained_for(0.1))
    assert len(events) == 1
    t = T0 + timedelta(hours=1)
    bad = {
        None: None,
        "unknown_level": R(t, level="unknown", reasons=("stale:cabinet",)),
        "stale": R(t, observed=t - timedelta(minutes=10)),
        "future": R(t, observed=t + timedelta(minutes=10)),
    }[reading]
    ev, state2 = strain_news(state, bad, t, TZ)
    assert ev is None
    assert state2.episode_since == T0      # not relief: the episode is kept
    # back to known strained, same episode: no new onset (only the due reminder).
    ev, _ = strain_news(state2, R(t), t, TZ)
    assert ev is not None and ev.kind == "reminder"


def test_unknown_from_the_start_never_fires():
    seq = [(T0 + timedelta(minutes=2 * i), R(T0 + timedelta(minutes=2 * i), level="unknown"))
           for i in range(100)]
    events, _ = run(seq)
    assert events == []


def test_relief_ends_the_episode_without_an_event_and_a_new_episode_onsets():
    seq = strained_for(0.2)
    t = T0 + timedelta(minutes=20)
    seq.append((t, R(t, level="idle", since=t, reasons=("strain_cleared",))))
    t2 = T0 + timedelta(minutes=60)
    seq.append((t2, R(t2, since=t2)))
    events, _ = run(seq)
    assert [(e.kind, e.at) for e in events] == [("onset", T0), ("onset", t2)]


def test_restart_does_not_refire_onset():
    events, state = run(strained_for(0.7))
    assert [e.kind for e in events] == ["onset", "reminder"]
    now = T0 + timedelta(minutes=45)
    seeded = seed_state([e.at for e in events], R(now), now, TZ)
    assert seeded.events_today == 2 and seeded.last_event_at == events[-1].at
    assert seeded.episode_since == T0 and not seeded.onset_pending
    ev, _ = strain_news(seeded, R(now), now, TZ)
    assert ev is None                       # no onset again; next reminder is at 1 h
    t1h = T0 + timedelta(hours=1)
    ev, _ = strain_news(seeded, R(t1h), t1h, TZ)
    assert ev is not None and ev.kind == "reminder"


def test_restart_with_no_recorded_event_onsets_a_long_episode_once_and_reminds_from_the_onset():
    now = T0 + timedelta(hours=54)
    state = seed_state([], R(now), now, TZ)
    ev, state = strain_news(state, R(now), now, TZ)
    assert ev is not None and ev.kind == "onset"
    ev, state = strain_news(state, R(now + timedelta(minutes=20)), now + timedelta(minutes=20), TZ)
    assert ev is None                       # old reminder slots are not replayed
    later = now + timedelta(minutes=30)
    ev, _ = strain_news(state, R(later), later, TZ)
    assert ev is not None and ev.kind == "reminder" and ev.reminder_index == 0


def test_onset_lag_after_since_does_not_cost_the_30_min_reminder():
    """Live, the onset lands a regulate step (+ a tick) after the reading's since. Reminders
    count from the sent onset, so the 30 min one still clears the 30 min gap."""
    since = T0 - timedelta(seconds=150)
    seq = [(T0 + timedelta(seconds=30 * i), R(T0 + timedelta(seconds=30 * i), since=since))
           for i in range(int(timedelta(hours=9) / timedelta(seconds=30)))]
    events, _ = run(seq)
    assert [(e.kind, e.reminder_index, e.at - T0) for e in events] == [
        ("onset", None, timedelta(0)),
        ("reminder", 0, timedelta(minutes=30)),
        ("reminder", 1, timedelta(hours=1)),
        ("reminder", 2, timedelta(hours=2)),
        ("reminder", 3, timedelta(hours=4)),
        ("reminder", 4, timedelta(hours=8)),
    ]


def test_seed_anchors_reminders_on_the_recorded_onset():
    since = T0 - timedelta(seconds=150)
    now = T0 + timedelta(minutes=45)
    seeded = seed_state([T0], R(now, since=since), now, TZ)
    assert seeded.episode_noticed_at == T0 and seeded.reminders_consumed == 1
    t1h = T0 + timedelta(hours=1)
    ev, _ = strain_news(seeded, R(t1h, since=since), t1h, TZ)
    assert ev is not None and ev.kind == "reminder" and ev.reminder_index == 1


def test_relief_ends_the_hold_at_once():
    from orion.attention.sustained import EVENT_ORIENTING_WINDOW_SEC

    _, state = run(strained_for(0.0))
    t = T0 + timedelta(seconds=60)
    _, state = strain_news(state, R(t, level="idle", since=t, reasons=("strain_cleared",)), t, TZ)
    assert rest_write_allowed(state, t, EVENT_ORIENTING_WINDOW_SEC)


def test_rest_write_is_held_inside_the_orienting_window():
    from orion.attention.sustained import EVENT_ORIENTING_WINDOW_SEC

    events, state = run(strained_for(0.0))
    assert not rest_write_allowed(state, T0 + timedelta(seconds=299), EVENT_ORIENTING_WINDOW_SEC)
    assert rest_write_allowed(state, T0 + timedelta(seconds=300), EVENT_ORIENTING_WINDOW_SEC)
    assert rest_write_allowed(StrainNewsState(), T0, EVENT_ORIENTING_WINDOW_SEC)


def test_strain_causes_are_the_ones_classify_arousal_writes():
    """One owner: the news rule reads arousal.STRAIN_CAUSES, which classify_arousal writes from."""
    from orion.regulation import arousal

    assert sn.STRAIN_CAUSE_REASONS == frozenset(arousal.STRAIN_CAUSES)
    assert arousal.REFLEX_CABINET_HOT in sn.STRAIN_CAUSE_REASONS


# ---- glossary: the node is event-written, a trigger, prompt-reaching, not continuous ----

def test_glossary_marks_strain_event_written_trigger_not_continuous():
    from orion.attention import world_first as wf

    wf._EVENT_WRITTEN_CACHE.pop(STRAIN_NODE_ID, None)
    wf._CONTINUOUS_CACHE.pop(STRAIN_NODE_ID, None)
    assert wf.node_prediction_error_event_written(STRAIN_NODE_ID) is True
    assert wf.node_prediction_error_continuous(STRAIN_NODE_ID) is False
    value_kind, polarity = wf.node_prediction_error_semantics(STRAIN_NODE_ID)
    assert value_kind == "trigger"
    assert wf.source_kind_for_node(STRAIN_NODE_ID) == "internal"
    from orion.attention.self_readings import prompt_reaching_nodes

    assert STRAIN_NODE_ID in prompt_reaching_nodes()


def _history(now, n_zero, events):
    rows = [(now - timedelta(seconds=30 * k), 0.0) for k in range(n_zero, 0, -1)]
    return rows + [(t, 1.0) for t in events]


def _judge(value, observed_at, history, now):
    from orion.attention.world_first import judge_candidate, node_candidate
    from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

    mag = compute_prediction_error_magnitude(value=value, observed_at=observed_at, history=history, now=now)
    cand = node_candidate(node_id=STRAIN_NODE_ID, label="strain", magnitude=mag,
                          observed_at=observed_at, now=now)
    return judge_candidate(cand)


def test_a_fresh_event_over_calm_history_wins_eligibility_then_fades_out_at_300s():
    now = T0
    hist = _history(now, 2000, [now])
    v = _judge(1.0, now, hist, now)
    assert v.eligible and v.band == "unusual"
    later = now + timedelta(seconds=310)
    v = _judge(1.0, now, hist, later)
    assert not v.eligible and "orienting window" in v.reason


def test_event_only_writes_could_never_compete_which_is_why_the_tick_writes_rest():
    """Without the calm 0.0 rows (a week of events only) world-first never has the 200
    readings it needs, and an all-1.0 history puts a 1.0 at percentile 0."""
    now = T0
    only_events = [(now - timedelta(hours=3 * k), 1.0) for k in range(56)]
    v = _judge(1.0, now, only_events, now)
    assert not v.eligible and v.band == "insufficient_history"
    many_events = [(now - timedelta(minutes=30 * k), 1.0) for k in range(300)]
    v = _judge(1.0, now, many_events, now)
    assert not v.eligible
