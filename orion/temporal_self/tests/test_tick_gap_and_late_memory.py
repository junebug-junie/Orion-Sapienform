"""#3400: ordinary broadcast jitter is not a gap, and an episode memory folds when it lands.

Two defects, both read live on 2026-11-15:

* The gap bar was a fixed 180 s, set when the broadcast ticked at a median 37 s. It ticks at a
  median 137 s now (p99 213 s to 310 s, longest 335 s in 7 days), so ordinary jitter was counted as
  a gap 83 to 131 times a day and each one suspended the active attention arc. The bar is now 3
  times the median of the newest intervals, never below 180 s.
* ``episode_memory`` rows are written 0.5 to 97 h after ``occurred_at``. Read by ``occurred_at``
  they were always behind the watermark: counted late, never folded. They are now ordered by their
  write time (``payload.available_at``) and still bind by ``occurred_at``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from orion.temporal_self import ReducerConfig, advance_clock, build_frame, fold, initial_state
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import TICK_INTERVALS_CAP, TICK_INTERVALS_MIN, tick_gap_sec
from orion.temporal_self.broadcast import BroadcastTickView
from orion.temporal_self.tests.fixtures import T0, metacog

CFG = ReducerConfig()
A = "node:substrate.bus_synaptic"

# One cycle of intervals shaped like the live 11-13 day (median 137 s, the worst day's p90 224 s,
# p99 310 s, max 335 s). Every one of 190, 224, 310 and 335 crossed the old fixed 180 s bar.
LIVE_INTERVALS = [118.0, 137.0, 150.0, 190.0, 137.0, 110.0, 224.0, 137.0, 140.0, 310.0, 125.0, 137.0, 335.0, 131.0]


def ticks_at(intervals: list[float], ref: str | None = A, start: datetime = T0, prefix: str = "L") -> list[BroadcastTickView]:
    out, t = [BroadcastTickView(log_id=f"{prefix}00000", generated_at=start, ref=ref, label=ref or "")], start
    for i, sec in enumerate(intervals, start=1):
        t += timedelta(seconds=sec)
        out.append(BroadcastTickView(log_id=f"{prefix}{i:05d}", generated_at=t, ref=ref, label=ref or ""))
    return out


def attention(state):
    return sorted((a for a in state.arcs.values() if a.kind == "attention"), key=lambda a: a.began_at)


def memory_row(ref: str, occurred: datetime, created: datetime) -> dict:
    return {"memory_id": ref, "episode_id": f"ep-{ref}", "purpose": "happened", "occurred_at": occurred,
            "created_at": created, "statement": "PROSE"}


# ---------------------------------------------------------------- the gap bar


def test_live_jitter_at_137_s_counts_no_gap_and_cuts_no_arc():
    """The lifecycle the ticket names: six hours at the live pace hold ONE unbroken arc."""
    ticks = ticks_at(LIVE_INTERVALS * 11)  # ~6.4 h, 154 intervals
    s = fold(initial_state(), ticks, cfg=CFG)
    assert s.tick_source_gaps_today == 0
    (arc,) = attention(s)
    assert arc.status == "open" and len(arc.segments) == 1
    assert arc.source_gap_resumes == 0 and arc.attention_returns == 0
    # Dwell is the whole stretch: no interval was dropped from it as a hole.
    assert arc.cumulative_dwell_sec == sum(LIVE_INTERVALS) * 11
    assert not [w for w in build_frame(s, ticks[-1].generated_at, CFG).warnings if "gap" in w]


def test_a_real_long_hole_still_counts_and_still_suspends_the_arc():
    before = ticks_at(LIVE_INTERVALS * 3)
    hole = before[-1].generated_at + timedelta(minutes=20)
    after = ticks_at([137.0] * 4, start=hole, prefix="M")
    s = fold(initial_state(), before, cfg=CFG)
    arc_id = attention(s)[0].arc_id
    s = fold(s, after[:1], cfg=CFG)
    assert s.tick_source_gaps_today == 1
    assert s.arcs[arc_id].status == "suspended" and s.arcs[arc_id].suspended_by_gap
    s = fold(s, after[1:], cfg=CFG)
    arc = s.arcs[arc_id]
    # Coming back from a hole is a resume of the same arc, never counted as a return.
    assert arc.status == "open" and arc.source_gap_resumes == 1 and arc.attention_returns == 0
    assert len(arc.segments) == 2
    # The 20 minutes nobody observed are not dwell.
    assert arc.cumulative_dwell_sec < sum(LIVE_INTERVALS) * 3 + 137.0 * 4 + 1


def test_the_bar_is_never_below_the_floor_and_needs_a_few_intervals_first():
    s = initial_state()
    assert tick_gap_sec(s, CFG) == CFG.max_tick_gap_sec
    # At the 10-10 pace (37 s) 3 x median is 111 s: the 180 s floor still decides.
    fast = ticks_at([37.0] * 20)
    s = fold(s, fast, cfg=CFG)
    assert tick_gap_sec(s, CFG) == 180.0
    s = fold(s, ticks_at([200.0], start=fast[-1].generated_at, prefix="M")[1:], cfg=CFG)
    assert s.tick_source_gaps_today == 1
    # Fewer than TICK_INTERVALS_MIN intervals: the floor, whatever they are.
    young = fold(initial_state(), ticks_at([137.0] * (TICK_INTERVALS_MIN - 1)), cfg=CFG)
    assert tick_gap_sec(young, CFG) == 180.0
    assert tick_gap_sec(fold(initial_state(), ticks_at([137.0] * TICK_INTERVALS_MIN), cfg=CFG), CFG) == 411.0


def test_a_cold_start_knows_the_pace_before_an_arc_can_open():
    """An arc opens on the third tick. By then two intervals were seen, so the jitter that follows
    (310 s) is judged against the real pace: a state reset does not cut the first arc of the day."""
    s = fold(initial_state(), ticks_at([137.0, 137.0, 310.0, 137.0]), cfg=CFG)
    (arc,) = attention(s)
    assert s.tick_source_gaps_today == 0 and arc.status == "open" and len(arc.segments) == 1


def test_the_window_is_bounded_follows_a_lasting_change_of_pace_and_survives_midnight():
    s = fold(initial_state(), ticks_at([37.0] * 40), cfg=CFG)
    assert len(s.tick_recent_intervals_sec) == TICK_INTERVALS_CAP
    # The pace slows for good. Until the window turns over, 137 s stays under the 180 s floor; once
    # it has, the bar is 3 x 137 s. Not one interval on the way is a gap.
    slow = ticks_at([137.0] * 40, start=T0 + timedelta(seconds=37.0 * 40), prefix="M")[1:]
    s = fold(s, slow, cfg=CFG)
    assert s.tick_source_gaps_today == 0 and tick_gap_sec(s, CFG) == 411.0
    # Midnight (06:00Z) resets the day's counters, not the pace.
    s = advance_clock(s, slow[-1].generated_at + timedelta(seconds=60), CFG)
    late = ticks_at([137.0] * 3, start=datetime(2026, 10, 10, 6, 1, tzinfo=T0.tzinfo), prefix="N")
    s = fold(s, late, cfg=CFG)
    assert s.day_id == "2026-10-10" and len(s.tick_recent_intervals_sec) == TICK_INTERVALS_CAP


def test_silence_shorter_than_the_bar_keeps_the_arc_and_longer_suspends_it():
    """``advance_clock`` runs on the driver's read watermark, 300 s behind the wall clock: at the
    live pace the newest tick is often more than 180 s behind it."""
    ticks = ticks_at(LIVE_INTERVALS * 2)
    s = fold(initial_state(), ticks, cfg=CFG)
    last = ticks[-1].generated_at
    quiet = advance_clock(s, last + timedelta(seconds=300), CFG)
    (arc,) = attention(quiet)
    assert arc.status == "open"
    assert not [w for w in build_frame(quiet, last + timedelta(seconds=300), CFG).warnings if "silent" in w]
    gone = advance_clock(s, last + timedelta(minutes=20), CFG)
    (arc,) = attention(gone)
    assert arc.status == "suspended" and arc.suspended_by_gap
    assert [w for w in build_frame(gone, last + timedelta(minutes=20), CFG).warnings if "silent for 1200 s" in w]


def test_context_inside_a_jitter_interval_binds_to_the_arc_that_held_it():
    """An open attention arc reaches forward as far as the gap bar, so a row inside a 310 s
    interval is the arc's, not the day's."""
    ticks = ticks_at(LIVE_INTERVALS)
    upto = ticks[9]  # the next interval is 310 s
    s = fold(initial_state(), ticks[:10], cfg=CFG)
    s = fold(s, events=[metacog("m1", upto.generated_at + timedelta(seconds=250))], cfg=CFG)
    (arc,) = attention(s)
    assert arc.context_event_ids == ["metacog_observation:m1"]


def test_the_bar_never_passes_the_return_window_however_slow_the_pace():
    """A pace that keeps slowing must not raise the bar without limit (3 x 900 s would be 45 min)."""
    s = fold(initial_state(), ticks_at([900.0] * 6), cfg=CFG)
    assert tick_gap_sec(s, CFG) == CFG.return_window_sec == 1800.0


def test_chunking_ticks_and_late_memories_together_does_not_change_the_state():
    from orion.temporal_self.arcs import event_available_at

    ticks = ticks_at(LIVE_INTERVALS * 4)
    end = ticks[-1].generated_at
    events = [S.memory_episode(memory_row(f"m{i}", T0 + timedelta(minutes=7 * i), T0 + timedelta(minutes=40 + 11 * i)))
              for i in range(8)] + [metacog("mc", T0 + timedelta(minutes=33))]
    assert all(event_available_at(e) < end for e in events)
    whole = fold(initial_state(), ticks, events, CFG)
    parts = initial_state()
    for i in range(0, len(ticks), 5):
        lo = ticks[i].generated_at
        hi = ticks[i + 5].generated_at if i + 5 < len(ticks) else end + timedelta(seconds=1)
        parts = fold(parts, ticks[i:i + 5], [e for e in events if lo <= event_available_at(e) < hi], CFG)
    assert parts.model_dump_json() == whole.model_dump_json()
    bound = {c for a in whole.arcs.values() for c in a.context_event_ids if c.startswith("memory_episode:")}
    assert len(bound) == 8 and whole.skipped_today == 0


def test_chunking_does_not_change_the_state():
    ticks = ticks_at(LIVE_INTERVALS * 4)
    whole = fold(initial_state(), ticks, cfg=CFG)
    parts = initial_state()
    for i in range(0, len(ticks), 7):
        parts = fold(parts, ticks[i:i + 7], cfg=CFG)
    assert parts.model_dump_json() == whole.model_dump_json()
    assert whole.model_validate_json(whole.model_dump_json()) == whole


def test_the_gap_warning_does_not_claim_an_outage():
    before = ticks_at([137.0] * 8)
    s = fold(initial_state(), before + ticks_at([137.0], start=before[-1].generated_at + timedelta(minutes=20), prefix="M"), cfg=CFG)
    (line,) = [w for w in build_frame(s, T0 + timedelta(hours=1), CFG).warnings if "gap" in w]
    assert line == ("broadcast log gaps today: 1 (rows more than 3 times the usual interval apart, "
                    "and more than 180 s; cause not recorded)")
    assert "outage" not in line and "restart" not in line


# ---------------------------------------------------------------- episode memories land late


def test_a_memory_that_lands_hours_later_joins_the_arc_that_held_its_moment():
    ticks = ticks_at([137.0] * 30)  # A holds 15:00 to 16:08
    s = advance_clock(fold(initial_state(), ticks, cfg=CFG), T0 + timedelta(hours=3), CFG)
    occurred, created = T0 + timedelta(minutes=20), T0 + timedelta(hours=3, minutes=5)
    e = S.memory_episode(memory_row("m1", occurred, created))
    assert e.occurred_at == occurred and e.payload["available_at"] == created.isoformat()
    assert "PROSE" not in e.model_dump_json()
    s = fold(s, events=[e], cfg=CFG)
    (arc,) = attention(s)
    assert arc.context_event_ids == ["memory_episode:m1"]
    assert s.skipped_today == 0 and s.cursors["memory_episode"].endswith("|m1")
    # Folding it again (a re-read) changes nothing and is counted, as for every kind.
    again = fold(s, events=[e], cfg=CFG)
    assert again.arcs == s.arcs and again.skipped_today == 1


def test_a_memory_committed_behind_the_watermark_is_counted_never_folded_and_never_negative():
    """Read lag the other way round: the clock (the watermark) was taken, and only then did a row
    stamped BEFORE it become readable. Its age against the watermark is negative; it is counted
    late and nothing breaks."""
    s = advance_clock(fold(initial_state(), ticks_at([137.0] * 30), cfg=CFG), T0 + timedelta(hours=3), CFG)
    e = S.memory_episode(memory_row("m1", T0 + timedelta(minutes=20), T0 + timedelta(hours=3) - timedelta(seconds=2)))
    out = fold(s, events=[e], cfg=CFG)
    assert out.skipped_today == 1 and not attention(out)[0].context_event_ids
    assert out.watermark == s.watermark


def test_a_memory_with_no_arc_over_its_moment_goes_to_the_day_it_was_written_even_about_yesterday():
    # Until #3584 the one about yesterday was dropped from every day.
    s = advance_clock(initial_state(), T0, CFG)
    today = S.memory_episode(memory_row("today", T0 - timedelta(hours=2), T0 + timedelta(minutes=1)))
    yesterday = S.memory_episode(memory_row("yest", T0 - timedelta(hours=20), T0 + timedelta(minutes=2)))
    s = fold(s, events=[today, yesterday], cfg=CFG)
    assert s.skipped_today == 0
    frame = build_frame(s, T0 + timedelta(minutes=5), CFG)
    assert frame.unbound_context_event_ids == ["memory_episode:today", "memory_episode:yest"]
    # After the buffer horizon (6 h) both leave the buffer, into today's list.
    s = advance_clock(s, T0 + timedelta(hours=7), CFG)
    assert s.day_context_event_ids == ["memory_episode:today", "memory_episode:yest"]
    assert not [i for i in s.context_buffer if i.source_kind == "memory_episode"]


def test_the_buffer_stays_in_time_order_so_old_items_still_leave_it():
    """A memory about 13:00 that lands at 15:30 must not sit behind newer items and hold them in
    the buffer (``_flush_buffer`` stops at the first item inside the horizon)."""
    s = advance_clock(initial_state(), T0, CFG)
    s = fold(s, events=[metacog("m1", T0 + timedelta(minutes=10))], cfg=CFG)
    s = fold(s, events=[S.memory_episode(memory_row("old", T0 - timedelta(hours=2), T0 + timedelta(minutes=30)))], cfg=CFG)
    assert [i.event_id for i in s.context_buffer] == ["memory_episode:old", "metacog_observation:m1"]
    s = advance_clock(s, T0 + timedelta(hours=5), CFG)  # 13:00 is 7 h back, 15:10 is 4 h 50 min back
    assert [i.event_id for i in s.context_buffer] == ["metacog_observation:m1"]
    assert s.day_context_event_ids == ["memory_episode:old"]


# ---------------------------------------------------------------- #3584: a memory about a closed day

# The 6 live rows of 2026-11-16 (read-only from temporal_self_event): about 18:57-20:26Z on 11-15
# (Denver day 11-15, closed at 2026-11-16 07:00Z), all written 2026-11-16 15:10:43Z.
LIVE_3584 = [
    ("4634b009", "2026-11-15T18:57:00+00:00"), ("459c7082", "2026-11-15T19:11:00+00:00"),
    ("f19c35b7", "2026-11-15T19:45:00+00:00"), ("478b74f9", "2026-11-15T20:02:00+00:00"),
    ("609ab4a9", "2026-11-15T20:02:00+00:00"), ("f5ea4fcd", "2026-11-15T20:26:00+00:00"),
]
WRITTEN_3584 = datetime.fromisoformat("2026-11-16T15:10:43.395995+00:00")


def test_every_memory_about_a_closed_day_lands_once_in_the_day_it_was_written():
    """#3584: before the fix all 6 fell out of the buffer into no day (``f5ea4fcd`` was only the
    source cursor). Rule: a memory about a closed day joins the day it was written."""
    from orion.temporal_self.arcs import drain_closed_days

    events = [S.memory_episode(memory_row(ref, datetime.fromisoformat(occ), WRITTEN_3584)) for ref, occ in LIVE_3584]
    s = advance_clock(initial_state(), datetime.fromisoformat("2026-11-15T15:00:00+00:00"), CFG)
    s = advance_clock(s, datetime.fromisoformat("2026-11-16T15:00:00+00:00"), CFG)  # 11-15 closed
    assert s.day_id == "2026-11-16"
    s = fold(s, events=events, cfg=CFG)
    ids = sorted(e.event_id for e in events)
    assert s.skipped_today == 0
    # Visible in the live frame at once, and in the day list after leaving the buffer.
    assert build_frame(s, WRITTEN_3584 + timedelta(minutes=1), CFG).unbound_context_event_ids == ids
    s = advance_clock(s, datetime.fromisoformat("2026-11-17T07:00:00+00:00"), CFG)  # 11-16 closes
    s, closed = drain_closed_days(s)
    days = {d.day_id: d for d in closed}
    assert days["2026-11-16"].frame.unbound_context_event_ids == ids
    assert all(i not in d.frame.unbound_context_event_ids for k, d in days.items() if k != "2026-11-16" for i in ids)
    # The event still says which moment it is about.
    assert [e.occurred_at.isoformat() for e in events] == [occ for _, occ in LIVE_3584]
