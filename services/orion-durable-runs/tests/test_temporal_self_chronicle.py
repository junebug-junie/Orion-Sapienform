"""Temporal Self patch 3: the chronicle runs the pure reducer live (no database here; see
test_temporal_self_chronicle_postgres.py for the real SQL).

What is pinned: a restart or a failed commit neither double-folds nor drops a row; a row written
within the read lag is folded and one written after it is counted once and never folded; first
boot backfills and closes yesterday; a stale stored state resets to its own day; a second writer is
refused; body summaries attach once; the chronicle node never breaks the regulate node.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS)]

from app.temporal_self_chronicle import ChronicleConfig, Chronicler  # noqa: E402
from orion.temporal_self import ReducerConfig, advance_clock, fold, initial_state  # noqa: E402
from orion.temporal_self.sources import ADAPTERS  # noqa: E402
from orion.temporal_self.broadcast import tick_from_log_row  # noqa: E402
from temporal_self_world import TZ, FakeReader, FakeStore, World, event_ids, utc  # noqa: E402

T0 = utc(2026, 10, 10, 15, 0)  # 09:00 America/Denver
STEP = timedelta(seconds=120)


def cfg(**kw) -> ChronicleConfig:
    base = dict(reducer=ReducerConfig(tz_name=TZ), read_lag_sec=300.0, backfill_days=0)
    base.update(kw)
    return ChronicleConfig(**base)


def chat(i: int, at: datetime, session: str = "s1", juniper: bool = True) -> dict:
    return {"id": f"chat-{i}", "correlation_id": f"corr-{i}", "session_id": session, "source": "hub",
            "created_at": at.replace(tzinfo=None), "has_prompt": juniper, "unsolicited": not juniper}


def scenario() -> World:
    w = World(T0)
    w.ticks(T0, ["A"] * 10 + ["B"] * 6 + ["A"] * 10 + [None] * 5 + ["C"] * 8)
    w.add("chat_turn", chat(1, T0 + timedelta(minutes=5)))
    w.add("chat_turn", chat(2, T0 + timedelta(minutes=20)))
    for i, (a, b) in enumerate(((0, 10), (12, 20))):
        w.add("field_dominance_run", {"run_id": f"run-{i}", "target_id": "node:x", "target_kind": "node",
                                      "started_at": T0 + timedelta(minutes=a), "ended_at": T0 + timedelta(minutes=b),
                                      "tick_count": 12, "min_streak_at_run": 3, "left_censored": False})
    w.add("reverie_chain", {"chain_id": "chain-1", "created_at": T0 + timedelta(minutes=3), "theme_key": "t",
                            "terminal_reason": "settled",
                            "thoughts": [{"thought_id": "th-1", "created_at": T0 + timedelta(minutes=3), "correlation_id": "c-1"},
                                         {"thought_id": "th-2", "created_at": T0 + timedelta(minutes=4), "correlation_id": "c-2"}]})
    w.add("metacog_observation", {"id": "mc-1", "correlation_id": "mcc-1", "severity": "degraded", "trigger_kind": "x",
                                  "timestamp": (T0 + timedelta(minutes=6, seconds=5)).isoformat(),
                                  "trigger_timestamp": (T0 + timedelta(minutes=6)).replace(tzinfo=None)})
    for m in range(0, 60, 2):
        w.body["cluster"].append({"observed_at": T0 + timedelta(minutes=m), "chassis_watts": 1500.0 + m})
    return w


def run(ch: Chronicler, w: World, until: datetime) -> list[dict]:
    out = []
    while w.now < until:
        w.now += STEP
        out.append(asyncio.run(ch.step()))
    return out


def chronicler(w: World, store: FakeStore, **kw) -> Chronicler:
    return Chronicler(store=store, reader=FakeReader(w), cfg=cfg(**kw), now=lambda: w.now)


def strip_body(arc_json: str) -> dict:
    d = json.loads(arc_json)
    d.pop("body", None)
    return d


def pure_reference(w: World, watermark: datetime) -> dict:
    """One fold over everything up to the watermark: what any chunking must reproduce."""
    rc = ReducerConfig(tz_name=TZ)
    ticks = [tick_from_log_row(r.row) for r in w.rows if r.kind == "broadcast"]
    events = [e for e in (ADAPTERS[r.kind](r.row, TZ) for r in w.rows if r.kind != "broadcast") if e is not None]
    s = advance_clock(initial_state(), utc(2026, 10, 10, 6, 0), rc)
    s = advance_clock(fold(s, ticks, events, rc), watermark, rc)
    return {k: json.loads(a.model_dump_json(exclude={"body"})) for k, a in s.arcs.items()}


def test_live_steps_reproduce_one_pass_fold():
    w, store = scenario(), FakeStore()
    ch = chronicler(w, store)
    run(ch, w, T0 + timedelta(hours=2))
    assert {k: strip_body(v) for k, v in store.arcs.items()} == pure_reference(w, ch.watermark)
    kinds = sorted(json.loads(v)["kind"] for v in store.arcs.values())
    assert kinds.count("attention") >= 2 and "conversation" in kinds and "reverie" in kinds and "interoception" in kinds
    a = next(json.loads(v) for v in store.arcs.values() if json.loads(v)["subject_ref"] == "A")
    assert a["attention_returns"] == 1 and a["interruptions"]          # A -> B -> A
    assert json.loads(store.frame)["skipped_at_or_before_watermark"] == 0
    assert not event_ids(store, late=True)


def test_restart_mid_day_neither_double_folds_nor_drops():
    w1, s1 = scenario(), FakeStore()
    run(chronicler(w1, s1), w1, T0 + timedelta(hours=2))

    w2, s2 = scenario(), FakeStore()
    run(chronicler(w2, s2), w2, T0 + timedelta(minutes=13))     # mid-arc: A is open, B building
    ch = chronicler(w2, s2)                                       # restart: state comes from the store
    run(ch, w2, T0 + timedelta(hours=2))

    assert s2.arcs == s1.arcs
    assert s2.frame == s1.frame
    assert s2.state_row[0] == s1.state_row[0] and s2.state_row[2] == s1.state_row[2]
    assert set(s2.events) == set(s1.events)
    assert json.loads(s2.frame)["skipped_at_or_before_watermark"] == 0


def test_failed_commit_rereads_the_same_window():
    w1, s1 = scenario(), FakeStore()
    run(chronicler(w1, s1), w1, T0 + timedelta(hours=2))

    w2, s2 = scenario(), FakeStore()
    ch = chronicler(w2, s2)
    run(ch, w2, T0 + timedelta(minutes=10))
    held = ch.watermark
    s2.fail_next = 1
    out = run(ch, w2, w2.now + STEP)
    assert out[-1]["error"] and ch.watermark == held              # nothing advanced
    run(ch, w2, T0 + timedelta(hours=2))
    assert s2.arcs == s1.arcs and s2.frame == s1.frame and s2.state_row[2] == s1.state_row[2]


def test_row_written_inside_the_lag_is_folded():
    w, store = scenario(), FakeStore()
    w.add("chat_turn", chat(9, T0 + timedelta(minutes=40), session="s2"), delay=timedelta(seconds=200))
    run(chronicler(w, store), w, T0 + timedelta(hours=1))
    convs = [json.loads(v) for v in store.arcs.values() if json.loads(v)["subject_ref"] == "s2"]
    assert len(convs) == 1 and convs[0]["evidence_refs"] == ["chat_history_log:chat-9"]
    assert json.loads(store.frame)["skipped_at_or_before_watermark"] == 0


def test_row_written_after_the_lag_is_counted_once_and_never_folded():
    w, store = scenario(), FakeStore()
    w.add("chat_turn", chat(9, T0 + timedelta(minutes=40), session="s2"), delay=timedelta(seconds=200))
    run(chronicler(w, store, read_lag_sec=60.0), w, T0 + timedelta(hours=1, minutes=30))
    assert not [v for v in store.arcs.values() if json.loads(v)["subject_ref"] == "s2"]
    assert event_ids(store, late=True) == {"chat_turn:chat-9"}
    frame = json.loads(store.frame)
    assert frame["skipped_at_or_before_watermark"] == 1         # counted once, not once per probe
    assert any("not folded" in x for x in frame["warnings"])


def test_late_probe_never_reaches_before_where_reading_began():
    w, store = World(T0), FakeStore()
    # A row from before the origin (06:00Z, local midnight) that only becomes visible now.
    w.add("chat_turn", chat(1, utc(2026, 10, 10, 5, 50)), delay=timedelta(hours=9, minutes=20))
    run(chronicler(w, store), w, T0 + timedelta(minutes=10))
    assert not event_ids(store, late=True)


def test_first_boot_backfills_from_yesterdays_midnight_and_closes_it():
    now = utc(2026, 10, 10, 18, 0)
    w, store = World(now), FakeStore()
    w.ticks(utc(2026, 10, 9, 20, 0), ["A"] * 12)
    ch = chronicler(w, store, backfill_days=1)
    out = asyncio.run(ch.step())
    assert ch.health()["origin"] == "2026-10-09T06:00:00+00:00"   # 10-09 00:00 America/Denver
    assert ch.days_closed == ["2026-10-09"] and "2026-10-09" in store.days
    day = json.loads(store.days["2026-10-09"])
    assert [a["kind"] for a in day["arcs"]] == ["attention"] and day["arcs"][0]["status"] == "closed"
    assert out["watermark"] == (now - timedelta(seconds=300)).isoformat() and out["error"] is None
    assert json.loads(store.frame)["day_id"] == "2026-10-10"


def test_invalid_stored_state_resets_to_that_days_midnight():
    w, store = scenario(), FakeStore()
    store.state_row = (T0 - timedelta(minutes=30), utc(2026, 10, 10, 6, 0), b"not gzip")
    ch = chronicler(w, store)
    run(ch, w, T0 + timedelta(minutes=4))
    assert "stored state invalid" in ch.reset_reason
    assert ch.health()["origin"] == "2026-10-10T06:00:00+00:00" and ch.last_error is None
    assert store.commits >= 1


def test_a_second_writer_is_refused_then_reloads():
    w1, s1 = scenario(), FakeStore()
    run(chronicler(w1, s1), w1, T0 + timedelta(hours=1))

    w, store = scenario(), FakeStore()
    a, b = chronicler(w, store), chronicler(w, store)
    run(a, w, T0 + timedelta(minutes=10))
    asyncio.run(b.step())                         # b loads a's state
    w.now += STEP
    asyncio.run(a.step())                         # a moves the stored watermark
    refused = asyncio.run(b.step())               # b's window started from the old one
    assert "StaleWriterError" in refused["error"]
    while w.now < T0 + timedelta(hours=1):
        w.now += STEP
        asyncio.run(b.step())                     # b reloads and carries on alone
    assert store.arcs == s1.arcs and store.state_row[2] == s1.state_row[2]


def test_body_attaches_once_to_closed_non_reverie_arcs():
    w, store = scenario(), FakeStore()
    run(chronicler(w, store), w, T0 + timedelta(hours=2))
    arcs = [json.loads(v) for v in store.arcs.values()]
    closed = [a for a in arcs if a["status"] == "closed"]
    assert closed
    for a in closed:
        if a["kind"] == "reverie":
            assert a["body"] is None
        else:
            assert a["body"] is not None
    first_a = min((a for a in arcs if a["subject_ref"] == "A"), key=lambda a: a["began_at"])
    assert first_a["body"]["cluster_sample_count"] > 0 and first_a["body"]["chassis_watts_mean"] > 1500.0


def test_retention_runs_on_boot_then_every_six_hours():
    w, store = scenario(), FakeStore()
    ch = chronicler(w, store)
    run(ch, w, T0 + timedelta(minutes=30))
    assert len(store.retention_calls) == 1


# --- the chronicle node inside the temporal_self.update thread -------------------------------


def _driver(chronicle):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.temporal_self_driver import TemporalSelfDriver
    from test_temporal_self_regulate import World as RegWorld

    rw = RegWorld()
    return rw, TemporalSelfDriver(checkpointer=InMemorySaver(), deps=rw.deps(chronicle=chronicle), tick_sec=120.0)


def test_chronicle_failure_never_breaks_regulation():
    async def boom():
        raise RuntimeError("relation temporal_self_state does not exist")

    rw, d = _driver(boom)
    out = asyncio.run(d.step({"event_id": "tick:1", "kind": "tick"}))
    assert rw.projected and rw.projected[-1].arousal.arousal_level == "idle"
    assert "chronicle_failed" in out["warnings"] and "does not exist" in out["chronicle"]["error"]
    assert d.health()["chronicle"]["error"]


def test_chronicle_summary_rides_the_step():
    async def ok():
        return {"watermark": "2026-10-10T17:55:00+00:00", "windows": 1, "error": None}

    rw, d = _driver(ok)
    out = asyncio.run(d.step({"event_id": "tick:1", "kind": "tick"}))
    assert out["chronicle"]["windows"] == 1 and "chronicle_failed" not in out["warnings"]
    assert d.health()["chronicle"]["watermark"].startswith("2026-10-10")


def test_thread_without_chronicle_is_unchanged():
    rw, d = _driver(None)
    out = asyncio.run(d.step({"event_id": "tick:1", "kind": "tick"}))
    assert out["chronicle"] is None and rw.projected


# --- review findings (2026-10-10) ------------------------------------------------------------


def test_abandoned_deferral_finalised_ninety_minutes_later_is_counted_late():
    """`abandoned` lands ~5,400 s after started_at (live): beyond a 30-min probe it vanished."""
    w, store = scenario(), FakeStore()
    w.add("visual_deferral", {"attempt_id": "att-9", "started_at": T0 + timedelta(minutes=10), "outcome": "abandoned",
                              "result_json": {"reason": "timeout", "detail": {"state": "normal"}}},
          delay=timedelta(seconds=5400))
    run(chronicler(w, store), w, T0 + timedelta(hours=2, minutes=30))
    assert "visual_deferral:att-9" in event_ids(store, late=True)
    assert json.loads(store.frame)["skipped_at_or_before_watermark"] == 1


def test_a_row_whose_available_time_moves_later_is_not_folded_twice():
    """An upsert that rewrites completed_at (curiosity_offer_decisions.py's outcome upsert) makes
    the same run available again later; it must not become a second arc."""
    def run_row(done):
        return {"run_id": "run-x", "decided_at": T0, "turn_started_at": T0 + timedelta(minutes=1), "arm": "a",
                "offered": [], "completed_at": done, "turn_ok": True, "n_tested": 1, "n_moved": 0, "n_formed": 0}

    w, store = scenario(), FakeStore()
    w.add("curiosity_run", run_row(T0 + timedelta(minutes=20)))
    ch = chronicler(w, store)
    run(ch, w, T0 + timedelta(minutes=40))
    w.rows = [r for r in w.rows if r.row.get("run_id") != "run-x"]
    w.add("curiosity_run", run_row(T0 + timedelta(minutes=50)))
    run(ch, w, T0 + timedelta(hours=1, minutes=30))
    assert [json.loads(v)["kind"] for v in store.arcs.values()].count("curiosity") == 1
    assert ch.moved_total == 1 and ch.health()["moved_total"] == 1


def test_reset_refold_still_folds_rows_it_had_stored():
    w1, s1 = scenario(), FakeStore()
    run(chronicler(w1, s1), w1, T0 + timedelta(hours=2))

    w2, s2 = scenario(), FakeStore()
    run(chronicler(w2, s2), w2, T0 + timedelta(hours=1))
    wm, origin, _ = s2.state_row
    s2.state_row = (wm, origin, b"stale shape")                  # e.g. a reducer version bump
    ch = chronicler(w2, s2)
    run(ch, w2, T0 + timedelta(hours=2))
    assert ch.reset_reason and ch.moved_total == 0
    assert s2.arcs == s1.arcs and s2.frame == s1.frame


def test_step_budget_spreads_a_long_catch_up_over_steps(monkeypatch):
    import app.temporal_self_chronicle as mod

    monkeypatch.setattr(mod, "STEP_BUDGET_SEC", 0.0)
    w, store = World(utc(2026, 10, 10, 18, 0)), FakeStore()
    ch = chronicler(w, store, backfill_days=1)
    first = asyncio.run(ch.step())
    assert first["windows"] == 1 and first["error"] is None       # progress, but one window only
    while ch.watermark < w.now - timedelta(seconds=300):
        asyncio.run(ch.step())
    assert ch.days_closed == ["2026-10-09"]


def test_a_hung_chronicle_times_out_and_regulation_still_steps():
    async def hang():
        await asyncio.sleep(5)

    from langgraph.checkpoint.memory import InMemorySaver

    from app.temporal_self_driver import TemporalSelfDriver
    from test_temporal_self_regulate import World as RegWorld

    rw = RegWorld()
    d = TemporalSelfDriver(checkpointer=InMemorySaver(), deps=rw.deps(chronicle=hang, chronicle_timeout_sec=0.05), tick_sec=120.0)
    out = asyncio.run(d.step({"event_id": "tick:1", "kind": "tick"}))
    assert "timeout" in out["chronicle"]["error"] and "chronicle_failed" in out["warnings"]
    assert rw.projected and rw.projected[-1].arousal.arousal_level == "idle"


def test_late_deferrals_count_in_the_body_thermal_refusals():
    w, store = scenario(), FakeStore()
    w.add("visual_deferral", {"attempt_id": "att-hot", "started_at": T0 + timedelta(minutes=2), "outcome": "deferred_thermal",
                              "result_json": {"reason": "hot", "detail": {"state": "hot"}}})
    run(chronicler(w, store), w, T0 + timedelta(hours=2))
    first_a = min((json.loads(v) for v in store.arcs.values() if json.loads(v)["subject_ref"] == "A"),
                  key=lambda a: a["began_at"])
    assert first_a["body"]["thermal_refusals"] == 1
