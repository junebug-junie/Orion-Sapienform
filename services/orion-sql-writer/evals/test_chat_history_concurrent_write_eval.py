"""Eval: chat_history_log survives concurrent same-primary-key writes.

Lives in evals/, not tests/, because it needs a real Postgres -- and that is the
entire point. The unit tests assert on compiled SQL, which cannot tell you
whether two transactions racing on one primary key actually merge. The bug this
guards against passed every mocked test in the suite for months.

Shape mirrors production exactly: one Hub turn publishes three events (user
message, assistant message, turn) that are dispatched as independent chassis
tasks and executed in parallel threads with thread-local sessions, all writing
the SAME chat_history_log row with different subsets of columns.

Synthetic ids only (``evaltest-`` prefix), removed in a fixture teardown.

Run: pytest services/orion-sql-writer/evals -q
"""
from __future__ import annotations

import threading
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DSN = dict(host="localhost", port=55432, user="postgres",
           password="postgres", dbname="conjourney")
MARKER = "evaltest-"
TRIALS = 15

# The three real contributors to a single turn. Only the turn carries `source`,
# which is why `source IS NULL` was the fingerprint of every corrupted row.
CONTRIBUTIONS = [
    {"prompt": "the user prompt"},
    {"response": "the assistant response"},
    {"source": "hub_orion", "prompt": "the user prompt",
     "response": "the assistant response"},
]


def _connect():
    try:
        return psycopg2.connect(connect_timeout=5, **DSN)
    except psycopg2.OperationalError as exc:  # pragma: no cover - infra gate
        pytest.skip(f"conjourney Postgres unavailable: {exc}")


@pytest.fixture
def cleanup_synthetic_rows():
    yield
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("delete from chat_history_log where id like %s", (MARKER + "%",))
        conn.commit()
    finally:
        conn.close()


def _upsert(row_id: str, cols: dict, barrier: threading.Barrier) -> None:
    """The production statement: INSERT ... ON CONFLICT DO UPDATE, per-column."""
    conn = _connect()
    try:
        cur = conn.cursor()
        # Every writer waits here, so all of them are past the point where the
        # old SELECT-then-INSERT would already have missed. Without this the
        # race is timing-dependent and the eval would be flaky-green.
        barrier.wait(timeout=30)
        names = ", ".join(cols)
        marks = ", ".join(["%s"] * len(cols))
        updates = ", ".join(
            f"{c} = coalesce(nullif(excluded.{c}, ''), chat_history_log.{c})"
            for c in cols
        )
        cur.execute(
            f"insert into chat_history_log (id, {names}) values (%s, {marks}) "
            f"on conflict (id) do update set {updates}",
            [row_id] + list(cols.values()),
        )
        conn.commit()
    finally:
        conn.close()


def test_concurrent_contributors_merge_into_one_complete_row(cleanup_synthetic_rows):
    """No contributor's columns may be lost, however the commits interleave."""
    incomplete: list[tuple] = []

    for _ in range(TRIALS):
        row_id = f"{MARKER}{uuid.uuid4()}"
        barrier = threading.Barrier(len(CONTRIBUTIONS))
        threads = [
            threading.Thread(target=_upsert, args=(row_id, cols, barrier))
            for cols in CONTRIBUTIONS
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "select coalesce(prompt,''), coalesce(response,''), coalesce(source,'') "
                "from chat_history_log where id = %s",
                (row_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row or not all(str(v).strip() for v in row):
            incomplete.append((row_id, row))

    assert not incomplete, (
        f"{len(incomplete)}/{TRIALS} turns lost a contributor's columns to the "
        f"race: {incomplete[:3]}"
    )


def test_exactly_one_row_per_turn(cleanup_synthetic_rows):
    """Concurrent writers must converge on one row, not insert duplicates."""
    row_id = f"{MARKER}{uuid.uuid4()}"
    barrier = threading.Barrier(len(CONTRIBUTIONS))
    threads = [
        threading.Thread(target=_upsert, args=(row_id, cols, barrier))
        for cols in CONTRIBUTIONS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("select count(*) from chat_history_log where id = %s", (row_id,))
        assert cur.fetchone()[0] == 1
    finally:
        conn.close()
