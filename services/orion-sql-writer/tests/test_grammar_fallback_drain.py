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


def _wire(monkeypatch, rows, persist, present=None):
    """`present`: event_ids the ledger reports as stored (default: everything persist saw)."""
    finished: dict = {}
    stored: set = set()
    monkeypatch.setattr(drain, "_fetch_rows", lambda limit, after_id=0: [r for r in rows if r[0] > after_id][:limit])
    monkeypatch.setattr(drain, "_present_event_ids", lambda ids: set(present) if present is not None else stored | set(ids))
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
    monkeypatch.setattr(drain, "_fetch_rows", lambda limit, after_id=0: pytest.fail("must not read while busy"))
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


def test_returning_zero_without_a_stored_row_does_not_delete(monkeypatch, idle):
    """persist_grammar_trace_batch returns 0 on a rolled-back conflict or a cancelled query
    WITHOUT raising -- the same 0 it returns for 'all deduped'. Only the table can tell them
    apart, so a row nothing stored must stay in bus_fallback_log."""
    fin = _wire(monkeypatch, [(1, _event(1).model_dump(mode="json")), (2, _event(2).model_dump(mode="json"))],
                lambda events, shard: 0, present={"gev_t1_2"})  # only event 2 really landed
    out = asyncio.run(drain.drain_once(10))
    assert fin["deleted"] == [2] and out["replayed"] == 1 and out["failed"] == 1


def test_deduped_rows_are_deleted_because_the_ledger_already_has_them(monkeypatch, idle):
    fin = _wire(monkeypatch, [(1, _event(1).model_dump(mode="json"))], lambda e, s: 0, present={"gev_t1_1"})
    assert asyncio.run(drain.drain_once(10))["replayed"] == 1 and fin["deleted"] == [1]


def test_cursor_advances_past_failing_rows_and_resets_when_empty(monkeypatch, idle):
    def boom(events, shard):
        raise RuntimeError("bad trace")

    rows = [(i, _event(i, trace=f"t{i}").model_dump(mode="json")) for i in (1, 2, 3)]
    _wire(monkeypatch, rows, boom)
    first = asyncio.run(drain.drain_once(2, 0))
    assert first["cursor"] == 2 and first["failed"] == 2  # head rows failed, cursor moved on
    second = asyncio.run(drain.drain_once(2, first["cursor"]))
    assert second["cursor"] == 3  # row 3 reached even though 1-2 keep failing
    assert asyncio.run(drain.drain_once(2, second["cursor"]))["cursor"] == 0  # sweep done -> retry from top


def test_timeout_while_queued_does_not_cancel_a_live_query(monkeypatch, idle):
    import concurrent.futures
    import importlib
    import threading

    import app.grammar_ledger_handler as h

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    ex.submit(gate.wait)  # a live lane batch occupying the shard's only thread
    w = importlib.import_module("app.worker")
    monkeypatch.setattr(w, "_get_grammar_executors", lambda: [ex] * 8)
    monkeypatch.setattr(w.settings, "sql_writer_grammar_persist_timeout_sec", 0.05)
    monkeypatch.setattr(w.settings, "sql_writer_grammar_trace_batch_timeout_sec", 0.05)
    cancelled = []
    monkeypatch.setattr(h, "cancel_active_grammar_persist", lambda shard: cancelled.append(shard))
    fin = _wire(monkeypatch, [(1, _event(1).model_dump(mode="json"))], lambda e, s: len(e))
    out = asyncio.run(drain.drain_once(10))
    gate.set()
    ex.shutdown(wait=True)
    assert out["failed"] == 1 and not cancelled and not fin
