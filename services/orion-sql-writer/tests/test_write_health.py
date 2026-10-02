"""Storage-write organ: the sql-writer counting its own write outcomes.

Classifier inputs are the real error texts recorded in bus_fallback_log for the
three live failure bursts (2026-09-07 cockpit validation, 2026-09-21 Postgres
restart, 2026-09-26 home-cooling serialization). The envelope tests drive the
real ``handle_envelope`` with the DB calls stubbed, so a hook that is not on the
live path fails here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SQL_WRITER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_write_health_tests", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
wh = worker.write_health

HOME_COOLING_2026_09_26 = (
    "(builtins.TypeError) Object of type datetime is not JSON serializable "
    "[SQL: INSERT INTO home_cooling_sample (ts, node, role, cooling_watts) VALUES (%(ts)s ...)] "
    "[parameters: {'role': 'statement timeout connection refused'}]"
)
COCKPIT_2026_09_07 = (
    "1 validation error for CockpitHopV1\nstage\n  Input should be 'ingress', 'association', "
    "'stance_inputs' [type=literal_error, input_value='could not connect timeout', input_type=str]"
)
DB_RESTART_2026_09_21 = (
    '(psycopg2.OperationalError) connection to server at "orion-athena-sql-db" (172.18.0.16), '
    "port 5432 failed: FATAL:  the database system is not yet accepting connections"
)


@pytest.fixture(autouse=True)
def _fresh_organ():
    wh.reset_for_tests()
    wh.set_enabled(True)
    yield
    wh.reset_for_tests()


@pytest.mark.parametrize(
    "error,expected",
    [
        (HOME_COOLING_2026_09_26, "serialization"),
        (COCKPIT_2026_09_07, "validation"),
        (DB_RESTART_2026_09_21, "db_unavailable"),
        ("(psycopg2.errors.QueryCanceled) canceling statement due to statement timeout", "timeout"),
        ('(psycopg2.errors.NotNullViolation) null value in column "x" violates not-null constraint', "constraint"),
        ("IntegrityError writing to journal_entries: violates foreign key constraint", "constraint"),
        ("grammar queue full", "backpressure"),
        ("grammar persist timeout after 15.0s", "timeout"),
        ("Unknown kind", "unrouted"),
        ("spark snapshot normalization failed", "validation"),
        ('(psycopg2.errors.UndefinedColumn) column "zz" does not exist', "db_error"),
        ("KeyError: 'run_id'", "other"),
        (None, "other"),
        (asyncio.TimeoutError(), "timeout"),
    ],
)
def test_classifier_on_real_error_texts(error, expected):
    assert wh.classify_write_error(error) == expected


def test_classifier_ignores_words_inside_the_payload():
    # "statement timeout connection refused" sits in [parameters: ...]: must not win
    assert wh.classify_write_error(HOME_COOLING_2026_09_26) == "serialization"
    assert wh.classify_write_error(COCKPIT_2026_09_07) == "validation"


def test_pydantic_validation_error_object():
    class M(BaseModel):
        x: int

    with pytest.raises(ValidationError) as info:
        M(x="nope")
    assert wh.classify_write_error(info.value) == "validation"


def test_outcome_precedence_and_late_marks_ignored():
    o = wh.EnvelopeOutcome("t")
    o.mark_written(True)
    o.mark_failed("Unknown kind")
    o.mark_failed(HOME_COOLING_2026_09_26)
    o.mark_failed(COCKPIT_2026_09_07)
    assert o.close() == "serialization"  # first real failure wins over commit
    o.mark_failed(DB_RESTART_2026_09_21)
    assert o.failure == "serialization"
    assert wh.EnvelopeOutcome("t").close() == "skipped"
    dup = wh.EnvelopeOutcome("t")
    dup.mark_written(False)
    assert dup.close() == "duplicate"


def test_family_cap_folds_overflow_into_other():
    rec = wh.WriteHealthRecorder()
    for i in range(40):
        rec.record(f"t{i}", "committed")
    _s, _e, buckets, _q = rec.drain()
    assert len(buckets) == 33 and buckets["_other"].attempted == 8


def test_window_events_are_bounded_and_counts_only():
    rec = wh.WriteHealthRecorder()
    rec.record("home_cooling_sample", "serialization", count=7)
    rec.record("grammar_events", "committed", count=265, latency_ms=12.0)
    rec.record("unrouted", "unrouted", count=2)
    rec.note_grammar_queue_depth(9)
    s, e, buckets, q = rec.drain()
    events = wh.build_window_events(writer_node="athena", window_start=s, window_end=e, buckets=buckets,
                                    grammar_queue_max=q)
    assert len(events) == 4
    assert all(ev.provenance.source_service == "orion-sql-writer" for ev in events)
    assert all(ev.trace_id.startswith("sql_writer.storage:athena:") for ev in events)
    by_family = {ev.atom.text_value: ev.atom.summary for ev in events[:-1]}
    assert "classes=serialization:7" in by_family["home_cooling_sample"]
    assert "attempted=7 committed=0" in by_family["home_cooling_sample"]
    assert "attempted=0" in by_family["unrouted"] and "unrouted=2" in by_family["unrouted"]
    closing = events[-1].atom.summary
    assert "attempted=272" in closing and "failed=7" in closing and "grammar_queue_max=9" in closing


def test_off_flag_records_nothing():
    wh.set_enabled(False)
    wh.record_grammar(["t1"], "committed")
    assert wh.begin_envelope("x") == (None, None)
    assert wh.get_recorder().drain()[2] == {}


def test_own_reports_are_never_counted():
    wh.record_grammar(["sql_writer.storage:athena:20261002T120000Z", "hub.chat:1"], "committed")
    buckets = wh.get_recorder().drain()[2]
    assert buckets["grammar_events"].classes == {"committed": 1}


# ---- live path: handle_envelope with the DB stubbed --------------------------


def _env(kind: str, payload: dict | None = None) -> BaseEnvelope:
    return BaseEnvelope(kind=kind, source=ServiceRef(name="test"), correlation_id=uuid4(), payload=payload or {"a": 1})


class _FakeSession:
    def __init__(self, sink: list):
        self.sink = sink

    def add(self, row):
        self.sink.append(row)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def stub_db(monkeypatch):
    fallback_rows: list = []
    monkeypatch.setattr(worker, "get_session", lambda: _FakeSession(fallback_rows))
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    monkeypatch.setattr(
        worker, "_coerce_payload", lambda cls, p: SimpleNamespace(model_dump=lambda: dict(p))
    )
    return fallback_rows


def _drain_families():
    return wh.get_recorder().drain()[2]


@pytest.mark.asyncio
async def test_committed_write_counts_with_latency(stub_db, monkeypatch):
    monkeypatch.setattr(worker, "_write_row", lambda cls, data: True)
    await worker.handle_envelope(_env("home.cooling.sample.v1"))
    fam = _drain_families()["home_cooling_sample"]
    assert fam.classes == {"committed": 1} and len(fam.latencies_ms) == 1


@pytest.mark.asyncio
async def test_duplicate_write_is_not_a_failure(stub_db, monkeypatch):
    monkeypatch.setattr(worker, "_write_row", lambda cls, data: False)
    await worker.handle_envelope(_env("home.cooling.sample.v1"))
    assert _drain_families()["home_cooling_sample"].classes == {"duplicate": 1}


@pytest.mark.asyncio
async def test_serialization_failure_lands_in_fallback_and_is_classified(stub_db, monkeypatch):
    def _boom(cls, data):
        raise TypeError("(builtins.TypeError) Object of type datetime is not JSON serializable")

    monkeypatch.setattr(worker, "_write_row", _boom)
    await worker.handle_envelope(_env("home.cooling.sample.v1"))
    assert len(stub_db) == 1  # the existing fallback row still written
    fam = _drain_families()["home_cooling_sample"]
    assert fam.classes == {"serialization": 1} and fam.failed == 1 and fam.latencies_ms == []


@pytest.mark.asyncio
async def test_validation_reject_is_classified(stub_db, monkeypatch):
    class Strict(BaseModel):
        model_config = {"extra": "forbid"}
        x: int

    def _reject(cls, p):
        Strict.model_validate({"x": 1, "surprise": True})

    monkeypatch.setattr(worker, "_coerce_payload", _reject)
    await worker.handle_envelope(_env("home.cooling.sample.v1"))
    assert _drain_families()["home_cooling_sample"].classes == {"validation": 1}


@pytest.mark.asyncio
async def test_unrouted_kind_is_counted_but_not_attempted(stub_db, monkeypatch):
    monkeypatch.setattr(worker, "build_evidence_units", lambda *a, **k: [])
    await worker.handle_envelope(_env("no.such.kind.v1"))
    fam = _drain_families()["unrouted"]
    assert fam.classes == {"unrouted": 1} and fam.attempted == 0 and fam.failed == 0


@pytest.mark.asyncio
async def test_grammar_persist_success_and_failure_are_recorded(monkeypatch):
    calls = {"n": 0}

    def _persist(event, shard=0):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError('(psycopg2.OperationalError) the database system is starting up')
        return True

    monkeypatch.setattr("app.grammar_ledger_handler.persist_grammar_event", _persist)
    monkeypatch.setattr(worker, "_write_fallback", lambda *a, **k: None)
    now = datetime.now(timezone.utc)
    for trace in ("hub.chat:a", "hub.chat:b", "sql_writer.storage:athena:x"):
        event = GrammarEventV1(event_id=f"e-{trace}", event_kind="trace_started", trace_id=trace, emitted_at=now,
                               provenance=GrammarProvenanceV1(source_service="test"))
        await worker._persist_grammar_event_envelope(_env("grammar.event.v1"), event=event, payload={}, corr_id="c")
    fam = _drain_families()["grammar_events"]
    # the organ's own trace (third) was persisted but not counted
    assert fam.classes == {"committed": 1, "db_unavailable": 1}


@pytest.mark.asyncio
async def test_flush_window_publishes_and_survives_a_dead_bus():
    published: list = []

    class Bus:
        async def publish(self, channel, env):
            published.append((channel, env))

    wh.record_grammar(["hub.chat:a"], "committed")
    sent = await wh.flush_window(Bus(), writer_node="athena")
    assert sent == 2 and all(ch == "orion:grammar:event" for ch, _ in published)
    assert published[0][1].kind == "grammar.event.v1"

    class DeadBus:
        async def publish(self, channel, env):
            raise ConnectionError("redis down")

    wh.record_grammar(["hub.chat:a"], "committed")
    assert await wh.flush_window(DeadBus(), writer_node="athena") == 0
    assert await wh.flush_window(None, writer_node="athena") == 0
