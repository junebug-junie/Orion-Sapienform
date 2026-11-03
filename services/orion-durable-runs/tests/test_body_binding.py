"""Felt state: the physiology node's body_binding rows (app.body_binding, the graph step, the
store insert and the chronicle round trip).

The plan tests drive ``plan_constraint`` tick by tick; the node tests run the real graph with the
physiology node test's World and a fake ``price_body`` that scripts one lane's bind state. The
Postgres round trip runs when ORION_ADMISSION_TEST_DSN is set (CI provides one).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS), str(SERVICE_ROOT / "evals")]

from app.body_binding import (  # noqa: E402
    BINDING_KEEPALIVE_SEC,
    BINDING_STALE_SEC,
    binding_row,
    keep_entry,
    plan_constraint,
)
from orion.schemas.physiology import PhysiologyConstraintReadingV1, PhysiologyPriceV1  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402
from test_physiology_node import World, tick  # noqa: E402

T0 = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)
TICK = timedelta(seconds=120)


def run(states, *, start=T0, step=TICK, fail_at=()):
    """Plan a sequence of bind states one tick apart; writes succeed except at ``fail_at`` indexes.
    Returns ([(tick index, row kind)], final entry)."""
    entry: dict = {}
    rows = []
    for i, s in enumerate(states):
        p = plan_constraint(entry, s, start + i * step)
        if p.row is None:
            entry = p.written
        elif i in fail_at:
            entry = p.failed
        else:
            rows.append((i, p.row))
            entry = p.written
    return rows, entry


# --------------------------------------------------------------------------- the plan


def test_onset_after_240_s_bound_never_on_one_or_two_readings():
    assert run(["binding", "binding"])[0] == []                  # 120 s bound: not yet
    assert run(["binding", "slack", "binding", "slack"])[0] == []  # flapping never opens
    rows, entry = run(["binding"] * 3)
    assert rows == [(2, "onset")] and entry["began_at"] == T0.isoformat() and entry["open"]


def test_release_after_240_s_slack_and_the_arc_ends_at_the_first_slack_reading():
    rows, _ = run(["binding"] * 3 + ["slack", "binding", "slack", "slack"])
    assert rows == [(2, "onset")]                                # slack 120 s then bound again: held
    entry: dict = {}
    for i, s in enumerate(["binding"] * 3 + ["slack"] * 2):
        p = plan_constraint(entry, s, T0 + i * TICK)
        entry = p.written
    p = plan_constraint(entry, "slack", T0 + 5 * TICK)
    assert p.row == "release" and p.slack_since == T0 + 3 * TICK and p.began_at == T0
    assert not keep_entry(p.written)


def test_no_reading_is_never_a_release_and_breaks_a_pending_run():
    rows, entry = run(["binding"] * 3 + ["no_reading"] * 10)
    assert rows == [(2, "onset")] and entry["open"]               # still open, nothing written
    assert run(["binding", "no_reading", "binding", "binding"])[0] == []   # the run restarted
    rows, _ = run(["binding"] * 3 + ["slack", "no_reading", "slack", "slack"])
    assert rows == [(2, "onset")]                                 # the slack run restarted too


def test_a_gap_in_readings_breaks_a_pending_run():
    entry = plan_constraint({}, "binding", T0).written
    p = plan_constraint(entry, "binding", T0 + timedelta(seconds=400))   # 400 s gap > 360 s
    assert p.row is None and p.written["bound_since"] == (T0 + timedelta(seconds=400)).isoformat()


def test_keepalive_every_30_min_and_change_rate_limited():
    n = int(BINDING_KEEPALIVE_SEC // 120) + 3
    rows, _ = run(["binding"] * n)
    assert rows == [(2, "onset"), (2 + 15, "keepalive")]
    states = ["binding"] * 3 + ["stuck_at_ceiling", "binding", "stuck_at_ceiling", "binding", "stuck_at_ceiling"]
    rows, _ = run(states)
    assert rows == [(2, "onset"), (7, "change")]                   # 600 s after the onset row, not every flap


def test_a_failed_write_keeps_the_episode_so_the_next_tick_retries():
    rows, entry = run(["binding"] * 4, fail_at={2})
    assert rows == [(3, "onset")] and entry["began_at"] == T0.isoformat()
    rows, entry = run(["binding"] * 3 + ["slack"] * 4, fail_at={5})
    assert rows == [(2, "onset"), (6, "release")] and not keep_entry(entry)


def test_an_episode_older_than_the_stale_timeout_restarts():
    entry = {"open": True, "began_at": T0.isoformat(), "state": "binding", "row_at": T0.isoformat(),
             "seen_at": (T0 + timedelta(seconds=BINDING_STALE_SEC)).isoformat()}
    now = T0 + timedelta(seconds=BINDING_STALE_SEC + 100)
    p = plan_constraint(entry, "binding", now)
    assert p.row is None and not p.written.get("open") and p.written["bound_since"] == now.isoformat()


# --------------------------------------------------------------------------- the row


def _reading(state="binding", constraint="heat") -> PhysiologyConstraintReadingV1:
    return PhysiologyConstraintReadingV1(
        constraint=constraint, unit="c", level=32.4, budget=31.0, budget_kind="soft_line",
        price=PhysiologyPriceV1(value=0.4, ceiling=1.0, bind_state=state, why="room 31.8 C vs line 31.0 C"))


def test_row_shape_round_trips_through_the_chronicle_adapter_with_its_own_id():
    entry: dict = {}
    for i in range(2):
        entry = plan_constraint(entry, "binding", T0 + i * TICK).written
    now = T0 + 2 * TICK
    p = plan_constraint(entry, "binding", now)
    row = binding_row(p, _reading(), now, "2026-10-23", self_spent=False)
    assert row["event_id"] == f"body_binding:heat:{T0.isoformat()}:onset:{now.isoformat()}"
    assert row["subject_ref"] == "orion:body:heat" and row["occurred_at"] == now
    e = S.body_binding(dict(row, payload_json=row["payload"]))
    assert e.event_id == row["event_id"]   # never a new id: the chronicle re-inserts what it folds
    assert e.payload["level"] == 32.4 and e.payload["line"] == 31.0 and e.payload["price"] == 0.4
    assert e.payload["began_at"] == T0.isoformat() and e.payload["why"].startswith("room")


# --------------------------------------------------------------------------- the node


class Bindings:
    def __init__(self):
        self.rows: list[dict] = []
        self.fail = False

    async def __call__(self, row):
        if self.fail:
            return False
        self.rows.append(row)
        return True


def _scripted(w, states: dict):
    """price_body: every reading priced slack, except the scripted constraints' current state."""

    def price_body(model, body, carry, now):
        out = []
        for c in model.constraints:
            st = states.get(c.constraint, "slack")
            if c.level is None and st != "no_reading":
                out.append(c)
                continue
            out.append(c.model_copy(update={"price": PhysiologyPriceV1(value=0.5 if st != "slack" else 0.0,
                                                                       ceiling=1.0, bind_state=st, why=f"{st} why")}))
        return model.model_copy(update={"constraints": out}), {}

    return price_body


def test_node_writes_onset_then_release_and_carries_the_episode():
    w, rec = World(), Bindings()
    states = {"lane:fast": "binding"}
    d = w.driver(price_body=_scripted(w, states), record_binding=rec)
    for _ in range(3):
        tick(d)
        w.now += TICK
    assert [json.loads(r["payload"])["row"] for r in rec.rows] == ["onset"]
    assert rec.rows[0]["subject_ref"] == "orion:body:lane:fast"
    assert d.health() is not None
    states["lane:fast"] = "slack"
    for _ in range(3):
        tick(d)
        w.now += TICK
    kinds = [json.loads(r["payload"])["row"] for r in rec.rows]
    assert kinds == ["onset", "release"]
    assert json.loads(rec.rows[1]["payload"])["slack_since"] == (T0 + 3 * TICK).isoformat()


def test_node_write_failure_is_a_warning_and_retried():
    w, rec = World(), Bindings()
    d = w.driver(price_body=_scripted(w, {"lane:fast": "binding"}), record_binding=rec)
    rec.fail = True
    for _ in range(3):
        out = tick(d)
        w.now += TICK
    assert rec.rows == [] and "binding_write_failed" in out["warnings"]
    rec.fail = False
    tick(d)
    assert [json.loads(r["payload"])["row"] for r in rec.rows] == ["onset"]
    assert json.loads(rec.rows[0]["payload"])["began_at"] == T0.isoformat()


def test_node_writes_nothing_with_the_flag_off_or_without_prices():
    w = World()
    d = w.driver(price_body=_scripted(w, {"lane:fast": "binding"}))          # record_binding None
    for _ in range(4):
        out = tick(d)
        w.now += TICK
    assert "binding_episodes" not in out
    w2, rec = World(), Bindings()
    d2 = w2.driver(record_binding=rec)                                         # no prices at all
    for _ in range(4):
        tick(d2)
        w2.now += TICK
    assert rec.rows == []


def test_binding_episodes_cross_local_midnight():
    from app.temporal_self_driver import CARRIED_KEYS

    assert "binding_episodes" in CARRIED_KEYS
    w, rec = World(), Bindings()
    d = w.driver(price_body=_scripted(w, {"lane:fast": "binding"}), record_binding=rec)
    w.now = datetime(2026, 10, 24, 5, 56, tzinfo=timezone.utc)       # 23:56 MDT
    for _ in range(6):                                                # across 00:00 MDT
        tick(d)
        w.now += TICK
    assert [json.loads(r["payload"])["row"] for r in rec.rows] == ["onset"]   # one onset, no re-onset


# --------------------------------------------------------------------------- Postgres round trip

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_binding_rows_fold_into_one_arc_per_constraint_and_are_never_copied():
    from test_temporal_self_chronicle import T0 as CT0, scenario
    from test_temporal_self_chronicle_postgres import canon, run_pg, with_db

    from app import regulation_store
    from app.temporal_self_sources import SourceReader
    from app.temporal_self_store import ChronicleStore
    from temporal_self_world import TZ

    async def scenario_fn(pool):
        w = scenario()
        rows, entry = [], {}
        seq = [("heat", "binding")] * 3 + [("heat", "slack")] * 3
        for i, (c, st) in enumerate(seq):
            now = CT0 + timedelta(minutes=1) + i * TICK
            p = plan_constraint(entry, st, now)
            entry = p.written
            if p.row is not None:
                rows.append(binding_row(p, _reading(st, c), now, "2026-10-10", self_spent=False))
        assert [json.loads(r["payload"])["row"] for r in rows] == ["onset", "release"]
        for r in rows:
            assert await regulation_store.record_binding(pool, r)   # the production insert
        await run_pg(pool, w, CT0 + timedelta(hours=1))
        async with pool.connection() as conn:
            n = (await (await conn.execute(
                "SELECT count(*) AS n FROM temporal_self_event WHERE source_kind = 'body_binding'")).fetchone())["n"]
        assert n == 2   # folded over many windows, still only the physiology node's rows
        (arc,) = [canon(a) for a in await ChronicleStore(pool).arcs("2026-10-10") if a["kind"] == "binding"]
        assert arc["subject_ref"] == "orion:body:heat" and arc["closed_reason"] == "process_ended"
        lo, hi = CT0 - timedelta(minutes=5), CT0 + timedelta(minutes=50)
        _, off = await SourceReader(pool, TZ, exclude_kinds=("body_binding",)).read(lo, hi, ticks=False)
        assert not [e for e in off if e.source_kind == "body_binding"]   # the flag's off path

    with_db(scenario_fn)


# --------------------------------------------------------------------------- review fixes


def test_bound_again_after_an_unread_stretch_writes_a_flagged_keepalive_at_once():
    entry: dict = {}
    for i, s in enumerate(["binding"] * 3 + ["no_reading"] * 20):    # 40 min unread
        entry = plan_constraint(entry, s, T0 + i * TICK).written
    assert entry["open"] and entry["after_gap"]
    p = plan_constraint(entry, "binding", T0 + 23 * TICK)
    assert p.row == "keepalive" and p.after_gap and "after_gap" not in p.written
    row = binding_row(p, _reading(), T0 + 23 * TICK, "d", self_spent=False)
    assert json.loads(row["payload"])["after_gap"] is True
    assert S.body_binding(dict(row, payload_json=row["payload"])).payload["after_gap"] is True
    rows, _ = run(["binding"] * 3 + ["no_reading"] * 3 + ["slack"] * 3)
    assert rows == [(2, "onset"), (8, "release")]


def test_a_malformed_open_carry_starts_over_instead_of_failing_every_tick():
    bad = {"open": True, "row_at": T0.isoformat(), "state": "binding", "seen_at": T0.isoformat()}  # no began_at
    p = plan_constraint(bad, "binding", T0 + TICK)
    assert p.row is None and not p.written.get("open") and p.written["bound_since"] == (T0 + TICK).isoformat()


def test_the_gap_limit_follows_the_tick_length():
    entry = plan_constraint({}, "binding", T0).written
    slow = timedelta(seconds=400)                                    # a 180 s tick with one late step
    assert plan_constraint(entry, "binding", T0 + slow).written["bound_since"] == (T0 + slow).isoformat()
    kept = plan_constraint(entry, "binding", T0 + slow, max_gap_sec=540.0)
    assert kept.row == "onset" and kept.began_at == T0                # three 180 s ticks: still one run


def test_a_vanished_constraint_leaves_the_carry_once_stale():
    from app.body_binding import keep_entry as keep

    entry = {"open": True, "began_at": T0.isoformat(), "state": "binding", "row_at": T0.isoformat(),
             "seen_at": T0.isoformat()}
    p = plan_constraint(entry, None, T0 + timedelta(seconds=BINDING_STALE_SEC + 1))
    assert not keep(p.written)
