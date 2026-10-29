"""Real Postgres acceptance tests. Use only an isolated *_test database."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
BODY = "Orion investigated the delayed journal writes. The observed records support a delivery gap, while its cause remains uncertain and needs another check."


@pytest.fixture
def store(monkeypatch):
    from orion.curiosity import compaction_store as module
    uri = os.environ.get("CURIOSITY_COMPACTION_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("CURIOSITY_COMPACTION_TEST_POSTGRES_URI unset")
    assert make_url(uri).database.endswith("_test"), "isolated test database required"
    schema = "compaction_" + uuid4().hex
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    db = create_engine(uri, connect_args={"options": f"-c search_path={schema}"})
    sql = (Path(__file__).resolve().parents[2] / "orion-sql-db" /
           "manual_migration_curiosity_compaction.sql").read_text()
    with db.begin() as conn:
        conn.execute(text(sql))
        conn.execute(text(sql))  # migration is idempotent
        conn.execute(text("CREATE TABLE durable_admission_runs (run_id text PRIMARY KEY, terminal text)"))
        conn.execute(text("CREATE TABLE journal_entries (entry_id text PRIMARY KEY, source_ref text, "
                          "body text, created_at timestamptz)"))
        conn.execute(text("INSERT INTO journal_entries VALUES "
                          "('sep', 'curiosity:sep', :body, '2026-09-16T12:00Z'), "
                          "('oct', 'curiosity:oct', :body, '2026-10-06T12:00Z'), "
                          "('irrelevant', 'other:run', :body, '2026-10-06T12:00Z')"), {"body": BODY})
    instance = module.Store(uri)
    instance.engine = db
    yield instance, db, module
    db.dispose()
    admin.dispose()


def _answer(row):
    from orion.curiosity.compaction import next_prompt
    if row["state"]["phase"] == "overall":
        return json.dumps({"body": BODY})
    records = json.loads(next_prompt(row).split("SOURCE RECORDS:\n")[1])
    return json.dumps({"record_topics": {r["ref"]: 0 for r in records}, "topics": [{"topic": "Delivery", "body": BODY}]})


def test_real_checkpoint_locks_delivery_ack_and_no_duplicate_windows(store):
    from app.curiosity_compaction_submit import submit_due
    from orion.curiosity.compaction import record_answer
    store, db, module = store
    first = store.acquire("weekly")
    assert first is not None and store.acquire("weekly") is None
    first.close()
    admissions, deliveries = [], []

    async def submit(request):
        admissions.append(request)

    async def publish(entry):
        deliveries.append(entry)

    asyncio.run(submit_due(store, submit, publish, now=NOW, cycle_id="dc-test"))
    assert len(admissions) == 2 and deliveries == []
    assert {r["status"] for r in store.recent()} == {"pending"}
    # Simulate each durable run's two successful topic/overall nodes against real SQL.
    for req in admissions:
        row = store.load(req.brief.compaction_id, req.run_id)
        while row["status"] != "completed":
            row = record_answer(row, _answer(row))
            store.persist(row)
    asyncio.run(submit_due(store, submit, publish, now=NOW, cycle_id="dc-again"))
    assert len(admissions) == len(deliveries) == 2
    result = store.recent()
    assert {r["status"] for r in result} == {"completed"}
    assert {r["sources"][0]["run_ref"] for r in result} == {"curiosity:sep", "curiosity:oct"}
    asyncio.run(submit_due(store, submit, publish, now=NOW, cycle_id="dc-delivery-retry"))
    assert len(deliveries) == 4
    assert [e.entry_id for e in deliveries[:2]] == [e.entry_id for e in deliveries[2:]]
    with db.begin() as conn:
        for e in deliveries[:2]:
            conn.execute(text("INSERT INTO journal_entries VALUES (:id, :ref, :body, :created)"),
                         {"id": e.entry_id, "ref": e.source_ref, "body": e.body, "created": e.created_at})
    asyncio.run(submit_due(store, submit, publish, now=NOW, cycle_id="dc-acked"))
    assert len(deliveries) == 4 and len(store.recent()) == 2


def test_failed_run_retries_saved_topics_and_fences_old_generation(store):
    from orion.curiosity.compaction import record_answer
    store, db, module = store
    session = store.acquire("weekly")
    row = session.prepare(NOW, "dc-original")
    session.close()
    row = record_answer(row, _answer(row))
    store.persist(row)
    assert row["state"]["phase"] == "overall"
    with db.begin() as conn:
        conn.execute(text("INSERT INTO durable_admission_runs VALUES (:id, 'failed')"), {"id": row["run_id"]})
    session = store.acquire("weekly")
    resumed = session.prepare(NOW, "dc-resume")
    session.close()
    assert resumed["run_id"] != row["run_id"] and resumed["state"]["phase"] == "overall"
    with pytest.raises(RuntimeError, match="fenced"):
        store.persist(row)
    with pytest.raises(Exception):
        store.load(row["compaction_id"], row["run_id"])
    store.persist(record_answer(resumed, _answer(resumed)))
    assert store.recent()[0]["status"] == "completed"


def _fail_and_resume(store, db, run_id, *, rejected):
    from orion.curiosity.compaction import mark_rejected
    if rejected:   # what the durable graph persists when the model's answer is rejected
        store.persist(mark_rejected(store.load(run_id.rsplit(":r", 1)[0], run_id)))
    with db.begin() as conn:
        conn.execute(text("INSERT INTO durable_admission_runs VALUES (:id, 'failed')"), {"id": run_id})
    session = store.acquire("weekly")
    try:
        return session.prepare(NOW, "dc-retry")
    finally:
        session.close()


def test_window_failing_at_one_cursor_is_given_up_after_bounded_generations(store):
    """Live 2026-10-14..28: weekly 2026-10-05 reached r21 and monthly 2026-09-01 r20 -- every
    sleep re-admitted a failed run with no limit. Generations whose model answers are rejected
    at one cursor are now bounded; a saved batch resets the count; a deadline does not count."""
    from orion.curiosity.compaction import MAX_STALLED_GENERATIONS, record_answer
    store, db, _ = store
    session = store.acquire("weekly")
    row = session.prepare(NOW, "dc-original")
    session.close()
    row = _fail_and_resume(store, db, row["run_id"], rejected=True)
    assert row["run_id"].endswith(":r1") and row["state"]["stall"]["count"] == 1
    assert store.load(row["compaction_id"], row["run_id"])["state"]["stall"]["count"] == 1   # persisted
    store.persist(record_answer(row, _answer(row)))         # this generation saved a batch
    row = _fail_and_resume(store, db, row["run_id"], rejected=True)
    assert row["state"]["stall"]["count"] == 1 and row["state"]["phase"] == "overall"
    for _ in range(5):                                        # GPU deadlines / outages never count
        row = _fail_and_resume(store, db, row["run_id"], rejected=False)
    assert row["state"]["stall"]["count"] == 1
    for _ in range(MAX_STALLED_GENERATIONS - 2):
        row = _fail_and_resume(store, db, row["run_id"], rejected=True)
    assert row["state"]["stall"]["count"] == MAX_STALLED_GENERATIONS - 1
    assert _fail_and_resume(store, db, row["run_id"], rejected=True) is None   # given up
    assert store.recent()[0]["status"] == "cancelled"


def test_given_up_window_can_be_reopened_after_the_next_window_exists(store):
    store, db, _ = store
    later = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    session = store.acquire("weekly")
    try:
        first = session.prepare(NOW, "dc-1")
        session.save({**first, "status": "cancelled"})
        second = session.prepare(later, "dc-2")
        assert second["compaction_id"] != first["compaction_id"]
        with db.begin() as conn:   # operator re-open, as the dream README describes
            conn.execute(text("UPDATE curiosity_compaction SET status='pending', state=state - 'stall' "
                              "WHERE compaction_id=:id"), {"id": first["compaction_id"]})
        assert session.prepare(later, "dc-3")["compaction_id"] == first["compaction_id"]
    finally:
        session.close()


def test_cancelled_window_is_not_resurrected(store):
    store, db, _ = store
    session = store.acquire("weekly")
    row = session.prepare(NOW, "dc-original")
    with db.begin() as conn:
        conn.execute(text("INSERT INTO durable_admission_runs VALUES (:id, 'cancelled')"), {"id": row["run_id"]})
    assert session.prepare(NOW, "dc-later") is None
    assert session.prepare(NOW, "dc-later-again") is None
    session.close()
    assert store.recent()[0]["status"] == "cancelled"


def test_month_only_reuses_wholly_contained_week_topics(store):
    store, db, _ = store
    from orion.curiosity.compaction import window_id
    session = store.acquire("weekly")
    try:
        for start, end, ref in [("2026-09-14T06:00:00+00:00", "2026-09-21T06:00:00+00:00", "sep"),
                                ("2026-09-28T06:00:00+00:00", "2026-10-05T06:00:00+00:00", "outside")]:
            start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
            key = window_id("weekly", start)
            session.save(dict(compaction_id=key, cadence="weekly", window_start=start, window_end=end,
                              cycle_id="dc-seed", status="completed", run_id=key+":r0", generation=0,
                              state={"sources": [{"entry_id": ref, "run_ref": f"curiosity:{ref}"}],
                                     "body": BODY, "batches": [], "partials": [],
                                     "topics": [{"topic": "Delivery", "body": BODY, "sources": [ref]}]}))
    finally:
        session.close()
    month = store.acquire("monthly")
    try:
        row = month.prepare(NOW, "dc-month")
        refs = [item["ref"] for batch in row["state"]["batches"] for item in batch]
        assert refs == ["curiosity-compaction:weekly:2026-09-14:topic:0"]
    finally:
        month.close()


def test_sql_failure_is_not_empty_success_and_releases_session(store):
    store, db, _ = store
    session = store.acquire("weekly")
    with pytest.raises(Exception):
        session.read("SELECT * FROM nonexistent_source")
    session.close()
    again = store.acquire("weekly")
    assert again is not None
    again.close()
    assert store.recent() == []
