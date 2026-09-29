"""Queue overflow sheds to bus_fallback_log; the drain replays it when lanes are idle."""
from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone

import pytest

from app import grammar_fallback_drain as drain
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1


class _LiveWorker:
    """Resolve app.worker at use time: other tests swap it in sys.modules, and the drain
    imports it lazily, so a module object bound at collection time can be a stale copy."""

    def __getattr__(self, name):
        return getattr(importlib.import_module("app.worker"), name)

    def __setattr__(self, name, value):
        setattr(importlib.import_module("app.worker"), name, value)


worker_mod = _LiveWorker()


def _event(n: int, trace: str = "t1") -> GrammarEventV1:
    return GrammarEventV1(
        event_id=f"gev_{trace}_{n}",
        event_kind="trace_started",
        trace_id=trace,
        emitted_at=datetime.now(timezone.utc),
        provenance=GrammarProvenanceV1(source_service="test", source_component="t", source_event_id=str(n)),
    )


@pytest.fixture
def idle(monkeypatch):
    monkeypatch.setattr(importlib.import_module('app.worker'), "grammar_queue_snapshot", lambda: {"total_depth": 0})
    monkeypatch.setattr(importlib.import_module('app.worker'), "_get_grammar_executors", lambda: [None] * 8)
    monkeypatch.setattr(importlib.import_module('app.worker'), "_grammar_shard_index", lambda t: 0)


def _wire(monkeypatch, rows, persist):
    finished: dict = {}
    monkeypatch.setattr(drain, "_fetch_rows", lambda limit: rows[:limit])
    monkeypatch.setattr(drain, "_finish_rows", lambda d, i: finished.update(deleted=d, invalid=i))
    import app.grammar_ledger_handler as h

    monkeypatch.setattr(h, "persist_grammar_trace_batch", persist)
    return finished


def test_drain_replays_and_deletes_only_applied_rows(monkeypatch, idle):
    rows = [(1, _event(1).model_dump(mode="json")), (2, _event(2).model_dump(mode="json"))]
    seen = []
    fin = _wire(monkeypatch, rows, lambda events, shard: seen.extend(e.event_id for e in events) or len(events))
    out = asyncio.run(drain.drain_once(10))
    assert out["replayed"] == 2 and fin["deleted"] == [1, 2]
    assert seen == ["gev_t1_1", "gev_t1_2"]


def test_drain_keeps_rows_when_persist_fails(monkeypatch, idle):
    def boom(events, shard):
        raise RuntimeError("db down")

    fin = _wire(monkeypatch, [(1, _event(1).model_dump(mode="json"))], boom)
    out = asyncio.run(drain.drain_once(10))
    assert out["failed"] == 1 and out["replayed"] == 0 and not fin  # nothing deleted


def test_drain_relabels_unparseable_payload(monkeypatch, idle):
    fin = _wire(monkeypatch, [(7, {"garbage": True})], lambda e, s: len(e))
    out = asyncio.run(drain.drain_once(10))
    assert out["invalid"] == 1 and fin == {"deleted": [], "invalid": [7]}


def test_drain_skips_while_lanes_busy(monkeypatch):
    monkeypatch.setattr(importlib.import_module('app.worker'), "grammar_queue_snapshot", lambda: {"total_depth": 3})
    monkeypatch.setattr(drain, "_fetch_rows", lambda limit: pytest.fail("must not read while busy"))
    assert asyncio.run(drain.drain_once(10))["skipped_busy"] == 1


def test_overflow_sheds_with_the_drain_filter_error_and_tracks_high_water(monkeypatch):
    async def go():
        monkeypatch.setattr(importlib.import_module('app.worker').settings, "sql_writer_grammar_queue_maxsize", 1)
        monkeypatch.setattr(importlib.import_module('app.worker'), "_GRAMMAR_QUEUES", [asyncio.Queue(maxsize=1)])
        monkeypatch.setattr(importlib.import_module('app.worker'), "_GRAMMAR_QUEUE_HIGH_WATER", {})
        monkeypatch.setattr(importlib.import_module('app.worker'), "_ensure_grammar_workers", lambda: None)
        monkeypatch.setattr(importlib.import_module('app.worker'), "_grammar_shard_count", lambda: 1)
        shed = []
        monkeypatch.setattr(importlib.import_module('app.worker'), "_write_fallback", lambda kind, corr, payload, err=None: shed.append(err))
        env = BaseEnvelope(kind="grammar.event.v1", source=ServiceRef(name="t"), payload={})
        for n in (1, 2):
            worker_mod._spawn_grammar_persist(env, event=_event(n), payload={}, corr_id="c")
        await asyncio.sleep(0.05)
        assert shed == [worker_mod.GRAMMAR_QUEUE_FULL_ERROR]
        assert worker_mod._GRAMMAR_QUEUE_HIGH_WATER[0] == 1

    asyncio.run(go())
