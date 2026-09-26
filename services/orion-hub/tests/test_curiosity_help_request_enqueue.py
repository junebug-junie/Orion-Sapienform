"""Post-run HelpRequest enqueue — Acceptance 1 + durable completion path.

Fake reader + fake bus only. Flag off or zero HelpRequests → zero publishes.
Durable completed state enqueues; dispatched-only tick does not.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.curiosity.peer_briefs import publish_help_requests_for_run
from orion.schemas.curiosity_peer import HELP_REQUEST_CHANNEL, HELP_REQUEST_KIND, PEER_BRIEF_CONSUMED_CHANNEL
from scripts.curiosity_investigation import CuriosityInvestigation


SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
RUN_ID = "abcd1234abcd"


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []
        self.codec = OrionCodec()
        self.redis = _FakeRedis()
        self.rpc_calls: list = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec=60.0):
        self.rpc_calls.append((channel, envelope, reply_channel))
        out = BaseEnvelope(
            kind="cortex.orch.result",
            source=SOURCE,
            correlation_id=envelope.correlation_id,
            payload={"status": "accepted"},
        )
        return {"channel": reply_channel, "data": self.codec.encode(out)}


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.values.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self.values[key] = value


class _FakeReader:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        hop_count: int = 0,
    ) -> None:
        self.rows = list(rows or [])
        self.hop_count = hop_count
        self.queries: list[str] = []

    def query(self, cypher: str) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if ":Hop" in cypher:
            return [
                {"n": i + 1, "note": f"hop {i + 1}"}
                for i in range(self.hop_count)
            ]
        return list(self.rows)


class _FakeConn:
    def __init__(self) -> None:
        self.rows = [
            {
                "crystallization_id": f"c{i}",
                "kind": "semantic",
                "subject": "a real thought",
                "summary": "a real thought",
                "salience": 0.6,
                "created_at": None,
            }
            for i in range(4)
        ]
        self.relations = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if "GROUP BY m.kind" in sql or "GROUP BY kind" in sql:
            return [{"kind": "semantic", "n": 268, "manual_n": 12}]
        if "FROM memory_crystallizations" in sql and "random()" in sql:
            return self.rows
        if "GROUP BY d.relation" in sql or "GROUP BY relation" in sql:
            return [{"relation": "same", "n": 0}]
        if "memory_concept_relation_decisions" in sql:
            return self.relations
        if "journal_entries" in sql:
            return []
        return []

    async def fetchval(self, sql, *args):
        if "pg_roles" in sql:
            return 1
        return 356


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


def _help_row(**over: Any) -> dict[str, Any]:
    base = {
        "help_id": "help-1",
        "run_id": RUN_ID,
        "prior_id": None,
        "mode": "world_curiosity",
        "question": "Why is RO_QUERY the only Hub path?",
        "tried_summary": "Looked at worldview.py header",
        "success_criteria": "A file:line pointer and one open question",
    }
    base.update(over)
    return base


def _help_publishes(bus: _FakeBus) -> list[tuple[str, Any]]:
    return [(c, e) for c, e in bus.published if c == HELP_REQUEST_CHANNEL]


def test_no_help_requests_publishes_nothing() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=True,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 0
    assert bus.published == []
    assert reader.queries  # still queried when enabled


def test_one_help_request_publishes_one_payload() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[_help_row()])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=True,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 1
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == HELP_REQUEST_CHANNEL
    assert envelope.kind == HELP_REQUEST_KIND
    payload = envelope.payload
    assert payload["help_id"] == "help-1"
    assert payload["run_id"] == RUN_ID
    assert payload["question"].startswith("Why is RO_QUERY")
    assert "HelpRequest" in reader.queries[0]
    assert RUN_ID in reader.queries[0]


@pytest.mark.asyncio
async def test_skips_empty_tried_summary_when_hops_exist() -> None:
    reader = _FakeReader(
        rows=[
            {
                "help_id": "h1",
                "run_id": RUN_ID,
                "mode": "world_curiosity",
                "question": "Why?",
                "tried_summary": "   ",
                "success_criteria": "A pointer",
            }
        ],
        hop_count=2,
    )
    bus = _FakeBus()
    n = await publish_help_requests_for_run(
        enabled=True, run_id=RUN_ID, reader=reader, bus=bus
    )
    assert n == 0
    assert bus.published == []


def test_flag_off_publishes_nothing_even_if_nodes_exist() -> None:
    bus = _FakeBus()
    reader = _FakeReader(rows=[_help_row()])
    n = asyncio.run(
        publish_help_requests_for_run(
            enabled=False,
            run_id=RUN_ID,
            reader=reader,
            bus=bus,
            source_ref=SOURCE,
        )
    )
    assert n == 0
    assert bus.published == []
    assert reader.queries == []  # must not even read when flag off


def _peer_loop(bus: _FakeBus, reader: _FakeReader, **over: Any) -> CuriosityInvestigation:
    loop = CuriosityInvestigation(
        enabled=True,
        tick_interval_sec=60.0,
        min_cooldown_sec=0.0,
        daily_cap=3,
        timeout_sec=1500.0,
        session_id="orion_curiosity",
        crystallization_sample=12,
        relation_sample=6,
        pool_provider=lambda: _FakePool(_FakeConn()),
        source_ref=SOURCE,
        reader=reader,
        kickoff_via_cortex=True,
        durable_admission_enabled=True,
        contractor_peer_enabled=True,
        **over,
    )
    loop._bus = bus
    loop._harness_rpc_bus = bus
    return loop


def test_durable_completed_state_with_one_help_request_publishes_once() -> None:
    """Live default: Hub returns after dispatch; enqueue happens on completed."""
    bus = _FakeBus()
    reader = _FakeReader(rows=[_help_row()])
    loop = _peer_loop(bus, reader)
    state = {
        "run_id": RUN_ID,
        "workflow": "curiosity.investigate",
        "thread_id": RUN_ID,
        "node": "finish",
        "status": "completed",
        "correlation_id": "c",
        "detail": {"reach_out": False},
    }
    env = BaseEnvelope(kind="durable.run.state.v1", source=SOURCE, payload=state)
    asyncio.run(loop._handle_run_state({"data": bus.codec.encode(env)}))
    helps = _help_publishes(bus)
    assert len(helps) == 1
    assert helps[0][1].payload["help_id"] == "help-1"
    completions = [e.payload for c, e in bus.published if c == PEER_BRIEF_CONSUMED_CHANNEL]
    assert len(completions) == 1
    assert completions[0]["phase"] == "completed"
    assert completions[0]["consumer_run_id"] == RUN_ID
    assert completions[0]["brief_ids"] == []
    # Same completed state again must not double-publish.
    asyncio.run(loop._handle_run_state({"data": bus.codec.encode(env)}))
    assert len(_help_publishes(bus)) == 1


def test_dispatched_only_tick_publishes_nothing() -> None:
    """Dispatch success returns before journal; no HelpRequest until completed."""
    bus = _FakeBus()
    # No reader: graph half off so tick is not blocked by ACL; HelpRequest
    # rows are irrelevant — dispatch must not enqueue regardless.
    loop = _peer_loop(bus, reader=None)  # type: ignore[arg-type]

    async def _fake_generate(*a, **k):
        raise AssertionError("turn must not run in-process on durable dispatch")

    loop._generate = _fake_generate  # type: ignore[assignment]
    assert asyncio.run(loop.tick()) == "dispatched"
    assert _help_publishes(bus) == []
    assert not [e for c, e in bus.published if c == PEER_BRIEF_CONSUMED_CHANNEL]
