"""The curiosity spend log (P1 of the 2026-09-25 attention-with-stakes design):
what each investigation run was offered, and what it moved, in nats.

Drives the real loop: a tick (in-process turn) and a durable-runner turn
request. Fakes are the shared ones from test_curiosity_investigation plus an
in-memory stand-in for the two spend tables.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from orion.curiosity.journal import INVESTIGATION_TAG
from orion.curiosity.self_inquiry import SELF_INQUIRY_TAG
from orion.curiosity.value import ARM_UNCERTAINTY_ORDER, ARM_VALUE_ORDER, kl_nats
from orion.schemas.durable_run import CuriosityTurnRequestV1
from scripts.curiosity_offer_decisions import (
    INSERT_DECISION_SQL,
    LOAD_HISTORY_SQL,
    LOAD_TURN_SNAPSHOT_SQL,
    SET_TURN_SNAPSHOT_SQL,
    UPSERT_OUTCOME_SQL,
)
from test_curiosity_investigation import _FakeBus, _FakeConn, _FakeReader, _graph_loop

ATLAS_NEEDLE = "p.last_run_id AS last_run_id, p.why AS why"
LIVE_NEEDLE = "p.status IS NULL OR NOT p.status IN"
REVISION_NEEDLE = "MATCH (n:PriorRevision)"
MIGRATION = Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_curiosity_spend_v1.sql"


class _SpendConn(_FakeConn):
    """`_FakeConn` plus the two spend tables, in memory."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.decisions: dict[str, dict] = {}
        self.outcomes: dict[str, dict] = {}
        self.history_rows: list[list[dict]] = []

    async def execute(self, sql, *args):
        if sql == INSERT_DECISION_SQL:
            self.decisions.setdefault(
                args[0],
                {
                    "arm": args[1],
                    "value_arm_propensity": args[2],
                    "offered": json.loads(args[3]),
                    "stale_offered": json.loads(args[4]),
                    "material_ids": json.loads(args[5]),
                    "constants": json.loads(args[6]),
                    "turn_snapshot": None,
                },
            )
        elif sql == SET_TURN_SNAPSHOT_SQL:
            row = self.decisions.get(args[0])
            if row is not None and row["turn_snapshot"] is None:
                row["turn_snapshot"] = args[1]  # asyncpg hands jsonb back as text
        elif sql == UPSERT_OUTCOME_SQL:
            keys = (
                "realized_nats", "n_tested", "n_moved", "n_formed", "n_moved_untested",
                "n_unattributed", "n_invalid_confidence", "per_prior", "revision_agreement",
            )
            row = dict(zip(keys, args[1:]))
            row["per_prior"] = json.loads(row["per_prior"])
            self.outcomes[args[0]] = row
        return "OK"

    async def fetchrow(self, sql, *args):
        if sql == LOAD_TURN_SNAPSHOT_SQL:
            row = self.decisions.get(args[0])
            return None if row is None else {"turn_snapshot": row["turn_snapshot"]}
        return None

    async def fetch(self, sql, *args):
        if sql == LOAD_HISTORY_SQL:
            return [{"per_prior": json.dumps(rows)} for rows in self.history_rows]
        return await super().fetch(sql, *args)


def _live(pid, confidence, tested):
    return {
        "prior_id": pid, "claim": f"claim {pid}", "confidence": str(confidence),
        "status": "open", "times_tested": tested, "formed_from": "", "last_tested_at": "",
    }


def _state(pid, confidence, tested, *, last_run_id="", run_id=""):
    return {
        "prior_id": pid, "claim": f"claim {pid}", "confidence": str(confidence),
        "status": "open", "times_tested": tested, "run_id": run_id, "last_run_id": last_run_id,
    }


def _reader(states, live=None, revisions=None) -> _FakeReader:
    # Insertion order matters: the most specific needle first.
    return _FakeReader(
        answers={
            ATLAS_NEEDLE: states,
            REVISION_NEEDLE: revisions or [],
            LIVE_NEEDLE: live
            if live is not None
            else [_live(s["prior_id"], s["confidence"], s["times_tested"]) for s in states],
        }
    )


def _moves_during_turn(loop, reader, after_for_run):
    """Replace the turn with one that changes the graph the way Orion would:
    `after_for_run(run_id)` returns the post-turn prior states."""

    async def _generate(prompt, correlation_id, source=None, require_lookup=True,
                        parent_run_id=None, session_id=None, **_kw):
        loop.seen_prompt = prompt
        reader.answers[ATLAS_NEEDLE] = after_for_run(parent_run_id)
        return "found it", {"elapsed_sec": 1.0, "harness_step_count": 14,
                            "harness_grounding_status": "grounded"}

    loop._generate = _generate


# --- one real tick ------------------------------------------------------------


def test_a_tick_records_the_offer_the_start_snapshot_and_what_the_run_moved() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    reader = _reader([_state("p1", 0.95, 2), _state("p2", 0.5, 0), _state("p3", 0.6, 1)])
    loop = _graph_loop(bus, reader=reader, conn=conn)

    def after(run_id):
        return [
            _state("p1", 0.92, 3, last_run_id=run_id),  # this run tested and moved it
            _state("p2", 0.8, 1, last_run_id="0ther0run0id"),  # someone else's test
            _state("p3", 0.6, 1),
            _state("p4", 0.55, 0, run_id=run_id),  # formed by this run
        ]

    _moves_during_turn(loop, reader, after)
    assert asyncio.run(loop.tick()) is None

    assert len(conn.decisions) == 1
    run_id, decision = next(iter(conn.decisions.items()))
    assert decision["arm"] == ARM_UNCERTAINTY_ORDER  # the value switch is off by default
    assert decision["value_arm_propensity"] == 0.0
    assert [o["prior_id"] for o in decision["offered"]] == ["p2", "p3", "p1"]
    assert all(o["expected_nats"] > 0 for o in decision["offered"])
    assert decision["turn_snapshot"] is not None
    assert decision["constants"]["pool_yield"] == 1.0

    outcome = conn.outcomes[run_id]
    assert outcome["realized_nats"] == pytest.approx(kl_nats(0.92, 0.95))
    assert (outcome["n_tested"], outcome["n_moved"], outcome["n_formed"]) == (1, 1, 1)
    assert outcome["n_unattributed"] == 1


def test_revision_agreement_is_recorded_against_orions_own_revisions() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)

    def after(run_id):
        reader.answers[REVISION_NEEDLE] = [
            {"run_id": run_id, "prior_id": "p1", "from_confidence": "0.5",
             "to_confidence": "0.8", "written_at": 1}
        ]
        return [_state("p1", 0.8, 1, last_run_id=run_id)]

    _moves_during_turn(loop, reader, after)
    assert asyncio.run(loop.tick()) is None
    (outcome,) = conn.outcomes.values()
    assert outcome["revision_agreement"] == 1.0


def test_a_test_that_moved_nothing_is_recorded_as_zero_not_skipped() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    reader = _reader([_state("p1", 0.55, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    _moves_during_turn(loop, reader, lambda run_id: [_state("p1", 0.55, 1, last_run_id=run_id)])
    assert asyncio.run(loop.tick()) is None
    (outcome,) = conn.outcomes.values()
    assert outcome["realized_nats"] == 0.0
    assert outcome["n_tested"] == 1


def test_the_log_switched_off_writes_nothing() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn, spend_log_enabled=False)
    _moves_during_turn(loop, reader, lambda run_id: [_state("p1", 0.8, 1, last_run_id=run_id)])
    assert asyncio.run(loop.tick()) is None
    assert conn.decisions == {}
    assert conn.outcomes == {}


def test_the_value_arm_orders_by_measured_progress_and_says_so() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    # "half" has been tested three times and never moved; "lean" kept moving.
    conn.history_rows = [
        [{"prior_id": "half", "kind": "tested", "before": 0.5, "after": 0.5, "nats": 0.0}],
        [{"prior_id": "half", "kind": "tested", "before": 0.5, "after": 0.5, "nats": 0.0}],
        [{"prior_id": "half", "kind": "tested", "before": 0.5, "after": 0.5, "nats": 0.0}],
        [{"prior_id": "lean", "kind": "tested", "before": 0.55, "after": 0.62, "nats": 0.01}],
        [{"prior_id": "lean", "kind": "tested", "before": 0.62, "after": 0.7, "nats": 0.01}],
    ]
    reader = _reader([_state("half", 0.5, 3), _state("lean", 0.7, 2)])
    # stale_prior_tests above 3, so "half" competes in the ordered list rather
    # than being moved to the separate stale bucket.
    loop = _graph_loop(
        bus, reader=reader, conn=conn, value_order_enabled=True, value_order_propensity=1.0,
        stale_prior_tests=5,
    )
    _moves_during_turn(loop, reader, lambda run_id: [_state("half", 0.5, 3), _state("lean", 0.7, 2)])
    assert asyncio.run(loop.tick()) is None
    (decision,) = conn.decisions.values()
    assert decision["arm"] == ARM_VALUE_ORDER
    assert decision["value_arm_propensity"] == 1.0
    assert [o["prior_id"] for o in decision["offered"]] == ["lean", "half"]
    assert decision["offered"][0]["expected_nats"] > decision["offered"][1]["expected_nats"]
    assert decision["constants"]["history_tests"] == 5
    # Orion is shown the same order it was offered in.
    assert loop.seen_prompt.index("claim lean") < loop.seen_prompt.index("claim half")


# --- the durable runner's turn request ------------------------------------------


def _request(run_id, source_tag=INVESTIGATION_TAG, attempt=1):
    return CuriosityTurnRequestV1(
        run_id=run_id, correlation_id=f"corr-{run_id}", prompt="investigate",
        timeout_sec=60.0, source_tag=source_tag, attempt=attempt,
    )


def _dispatched(conn, run_id):
    conn.decisions[run_id] = {"arm": ARM_UNCERTAINTY_ORDER, "turn_snapshot": None}


def test_a_durable_investigation_turn_is_measured_at_turn_start_not_dispatch() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    run_id = "a1b2c3d4e5f6"
    _dispatched(conn, run_id)
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    # Between dispatch and turn start, another run moved p1. The start
    # snapshot is taken now, so that move is not credited to this run.
    reader.answers[ATLAS_NEEDLE] = [_state("p1", 0.7, 1, last_run_id="0ther0run0id")]
    _moves_during_turn(loop, reader, lambda rid: [_state("p1", 0.9, 2, last_run_id=rid)])
    result = asyncio.run(loop._turn_result_for(_request(run_id), hold_lock=False))
    assert result.ok
    outcome = conn.outcomes[run_id]
    assert outcome["realized_nats"] == pytest.approx(kl_nats(0.9, 0.7))


def test_a_retried_run_is_scored_from_where_its_first_attempt_began() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    run_id = "b1b2c3d4e5f6"
    _dispatched(conn, run_id)
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    _moves_during_turn(loop, reader, lambda rid: [_state("p1", 0.7, 1, last_run_id=rid)])
    asyncio.run(loop._turn_result_for(_request(run_id, attempt=1), hold_lock=False))
    _moves_during_turn(loop, reader, lambda rid: [_state("p1", 0.9, 2, last_run_id=rid)])
    loop._turn_results.clear()
    asyncio.run(loop._turn_result_for(_request(run_id, attempt=2), hold_lock=False))
    assert conn.outcomes[run_id]["realized_nats"] == pytest.approx(kl_nats(0.9, 0.5))


def test_self_inquiry_turns_are_not_measured() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    run_id = "c1b2c3d4e5f6"
    _dispatched(conn, run_id)
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    _moves_during_turn(loop, reader, lambda rid: [_state("p1", 0.9, 1, last_run_id=rid)])
    asyncio.run(loop._turn_result_for(_request(run_id, source_tag=SELF_INQUIRY_TAG), hold_lock=False))
    assert conn.decisions[run_id]["turn_snapshot"] is None
    assert conn.outcomes == {}


def test_a_run_with_no_decision_row_records_nothing() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    _moves_during_turn(loop, reader, lambda rid: [_state("p1", 0.9, 1, last_run_id=rid)])
    asyncio.run(loop._turn_result_for(_request("d1b2c3d4e5f6"), hold_lock=False))
    assert conn.outcomes == {}


def test_an_unreadable_graph_at_turn_start_is_scored_unknown_not_zero() -> None:
    bus = _FakeBus()
    conn = _SpendConn()
    run_id = "e1b2c3d4e5f6"
    _dispatched(conn, run_id)
    loop = _graph_loop(bus, reader=_FakeReader(raises=True), conn=conn)

    async def _generate(*_a, **_kw):
        return "found it", {"elapsed_sec": 1.0}

    loop._generate = _generate
    asyncio.run(loop._turn_result_for(_request(run_id), hold_lock=False))
    assert conn.outcomes[run_id]["realized_nats"] is None


# --- the migration is the contract --------------------------------------------------


def _columns(sql: str) -> set[str]:
    return set(re.findall(r"\b([a-z_]+)\b", sql))


def test_the_migration_declares_every_column_the_writer_uses() -> None:
    ddl = MIGRATION.read_text(encoding="utf-8")
    decisions_ddl = ddl.split("CREATE TABLE IF NOT EXISTS curiosity_offer_decisions", 1)[1].split(");", 1)[0]
    outcomes_ddl = ddl.split("CREATE TABLE IF NOT EXISTS curiosity_run_outcomes", 1)[1].split(");", 1)[0]
    insert_cols = re.search(r"\(\s*([a-z_,\s]+)\)\s*VALUES", INSERT_DECISION_SQL).group(1)
    upsert_cols = re.search(r"\(\s*([a-z_,\s]+)\)\s*VALUES", UPSERT_OUTCOME_SQL).group(1)
    for col in [c.strip() for c in insert_cols.split(",")] + ["turn_snapshot", "turn_started_at", "decided_at"]:
        assert re.search(rf"^\s*{col}\s", decisions_ddl, re.M), f"curiosity_offer_decisions.{col}"
    for col in [c.strip() for c in upsert_cols.split(",")] + ["completed_at"]:
        assert re.search(rf"^\s*{col}\s", outcomes_ddl, re.M), f"curiosity_run_outcomes.{col}"
    assert "run_id text PRIMARY KEY" in decisions_ddl
    assert "run_id text PRIMARY KEY" in outcomes_ddl
