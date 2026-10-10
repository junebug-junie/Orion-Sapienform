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
    assert fold(a, tk, evs, CFG).model_dump_json() == a.model_dump_json()


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
    # (its own dwell_ticks reset to 0, which the reducer ignores). One arc, one return.
    before = ticks([A] * 6)
    after = ticks([A] * 6, start=before[-1].generated_at + timedelta(minutes=8), prefix="R")
    s = fold(initial_state(), before + after, cfg=CFG)
    arcs = [a for a in s.arcs.values() if a.kind == "attention"]
    assert len(arcs) == 1 and arcs[0].attention_returns == 1
    # Re-reading the overlap after a restart (the driver re-sends rows it already folded) is a no-op.
    again = fold(s, before[-3:] + after, cfg=CFG)
    assert again.model_dump_json() == s.model_dump_json()


def test_arc_ids_are_deterministic_from_day_kind_subject_and_first_evidence():
    s1 = fold(initial_state(), ticks([A] * 3), cfg=CFG)
    s2 = fold(initial_state(), ticks([A] * 3), cfg=CFG)
    assert list(s1.arcs) == list(s2.arcs)
    s3 = fold(initial_state(), ticks([A] * 3, prefix="Z"), cfg=CFG)
    assert list(s3.arcs) != list(s1.arcs)
