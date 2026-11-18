"""Replay identity (acceptance check 1): same rows in, same arcs and frames out."""

from __future__ import annotations

from datetime import timedelta

from orion.schemas.temporal_self import TemporalSelfStateV1
from orion.temporal_self import ReducerConfig, advance_clock, build_frame, drain_closed_days, fold, initial_state
from orion.temporal_self.tests.fixtures import T0, at, attention_row, chat, ev, metacog, run, ticks

CFG = ReducerConfig()
A, B, C = "node:substrate.bus_synaptic", "node:substrate.biometrics", "node:substrate.execution"


def _day():
    pattern = ([A] * 5 + [None] * 4 + [B] * 6 + [A] * 3 + [C] * 2 + [None] * 3) * 40
    tk = ticks(pattern, start=T0 - timedelta(hours=8))
    evs = [chat(str(i), "s1", T0 + timedelta(minutes=37 * i)) for i in range(8)]
    evs += [metacog(f"m{i}", T0 - timedelta(hours=7) + timedelta(minutes=11 * i)) for i in range(60)]
    evs += [attention_row(f"a{i}", T0 - timedelta(hours=8) + timedelta(minutes=3 * i)) for i in range(300)]
    evs += [run(f"r{i}", [A, B][i % 2], T0 - timedelta(hours=6) + timedelta(minutes=4 * i), 4) for i in range(60)]
    evs.append(ev("curiosity_run", "cr", at(-120), ended=at(-60), subject="cr", table="curiosity_run_outcomes"))
    evs.append(ev("dream_cycle", "cy", at(200), ended=at(201), subject="cy", table="dream_cycle"))
    return tk, evs


def _avail(e):
    return e.ended_at or e.occurred_at


def test_fold_twice_is_identical_and_refolding_is_a_no_op():
    tk, evs = _day()
    a = fold(initial_state(), tk, evs, CFG)
    b = fold(initial_state(), list(reversed(tk)), list(reversed(evs)), CFG)  # input order is irrelevant
    assert a.model_dump_json() == b.model_dump_json()
    again = fold(a, tk, evs, CFG)
    # Re-sent rows change nothing but the visible "skipped" counter.
    assert again.skipped_today == len(tk) + len(evs)
    assert again.model_copy(update={"skipped_today": 0}).model_dump_json() == a.model_dump_json()


def test_chunked_with_checkpoint_round_trip_equals_one_pass():
    tk, evs = _day()
    end = T0 + timedelta(hours=16)
    whole = advance_clock(fold(initial_state(), tk, evs, CFG), end, CFG)

    s = initial_state()
    t = T0 - timedelta(hours=8)
    while t < end:
        nxt = t + timedelta(minutes=47)
        s = fold(s, [x for x in tk if t <= x.generated_at < nxt], [e for e in evs if t <= _avail(e) < nxt], CFG)
        s = advance_clock(s, nxt, CFG)
        s = TemporalSelfStateV1.model_validate_json(s.model_dump_json())  # mid-day checkpoint
        t = nxt
    s = advance_clock(s, end, CFG)
    s1, days1 = drain_closed_days(whole)
    s2, days2 = drain_closed_days(s)
    assert [d.model_dump_json() for d in days1] == [d.model_dump_json() for d in days2]
    assert s1.arcs == s2.arcs
    assert build_frame(s1, end, CFG) == build_frame(s2, end, CFG)


def test_restart_gap_with_reset_dwell_produces_no_duplicate_arc():
    # A substrate-runtime restart: the broadcast log stops, then resumes with the same winner
    # (its own dwell_ticks reset to 0, which the reducer ignores). One arc; the outage is a
    # source-gap resume, not a return.
    before = ticks([A] * 6)
    after = ticks([A] * 6, start=before[-1].generated_at + timedelta(minutes=8), prefix="R")
    s = fold(initial_state(), before + after, cfg=CFG)
    arcs = [a for a in s.arcs.values() if a.kind == "attention"]
    assert len(arcs) == 1 and arcs[0].attention_returns == 0 and arcs[0].source_gap_resumes == 1
    # Re-reading the overlap after a restart (the driver re-sends rows it already folded)
    # folds nothing; the re-read is counted.
    again = fold(s, before[-3:] + after, cfg=CFG)
    assert again.arcs == s.arcs and again.skipped_today == 9


def _one_pass_vs_chunks(tk, evs, start, end, step):
    whole = advance_clock(fold(initial_state(), tk, evs, CFG), end, CFG)
    s, t = initial_state(), start
    while t < end:
        nxt = min(t + step, end)
        s = fold(s, [x for x in tk if t <= x.generated_at < nxt], [e for e in evs if t <= _avail(e) < nxt], CFG)
        s = advance_clock(s, nxt, CFG)
        t = nxt
    return drain_closed_days(whole), drain_closed_days(s)


def test_broadcast_silence_before_midnight_replays_identically():
    # Review finding: 4 ticks at 22:00 local, then nothing until 00:10. One pass and 20-minute
    # chunks must close the arc the same way.
    late = T0.replace(hour=4)  # 22:00 local (MDT) on 10-09 is 04:00Z on 10-10
    tk = ticks([A] * 4, start=late) + ticks([A] * 3, start=late + timedelta(hours=2, minutes=10), prefix="N")
    (s1, d1), (s2, d2) = _one_pass_vs_chunks(tk, [], late - timedelta(minutes=5), late + timedelta(hours=3), timedelta(minutes=20))
    assert [d.model_dump_json() for d in d1] == [d.model_dump_json() for d in d2]
    assert s1.arcs == s2.arcs
    (closed,) = [a for d in d1 for a in d.arcs if a.kind == "attention"]
    assert closed.closed_reason == "return_window_expired"


def test_advance_at_the_exact_watermark_never_refolds():
    tk = ticks([A] * 3)
    s = fold(initial_state(), tk, cfg=CFG)
    s = advance_clock(s, tk[-1].generated_at, CFG)
    again = fold(s, tk[-1:], cfg=CFG)
    assert again.tick_count_today == 3 and again.skipped_today == 1


def test_non_utc_inputs_order_by_instant():
    from zoneinfo import ZoneInfo

    denver = ZoneInfo("America/Denver")
    early = metacog("m-early", T0)  # 15:00Z
    later_local = ev("metacog_observation", "m-later", (T0 + timedelta(hours=1)).astimezone(denver),
                     table="orion_metacog", verdict="degraded")
    assert later_local.occurred_at.utcoffset() == timedelta(0)  # normalised on the way in
    s = fold(initial_state(), events=[later_local, early], cfg=CFG)
    s = fold(s, events=[metacog("m-mid", T0 + timedelta(minutes=30))], cfg=CFG)
    assert s.skipped_today == 1  # m-mid is genuinely earlier than the last folded instant
    tick = ticks([A], start=T0.astimezone(denver))[0]
    assert tick.generated_at.utcoffset() == timedelta(0)


def test_arc_ids_are_deterministic_from_day_kind_subject_and_first_evidence():
    s1 = fold(initial_state(), ticks([A] * 3), cfg=CFG)
    s2 = fold(initial_state(), ticks([A] * 3), cfg=CFG)
    assert list(s1.arcs) == list(s2.arcs)
    s3 = fold(initial_state(), ticks([A] * 3, prefix="Z"), cfg=CFG)
    assert list(s3.arcs) != list(s1.arcs)
