import asyncio
from pathlib import Path

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.world_pulse_read.queue import (
    claim_next_seed,
    enqueue_from_recent_digests,
    enqueue_seeds,
    ensure_seed_queue_schema,
    mark_seed_done,
    mark_seed_failed,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _REPO_ROOT
    / "services"
    / "orion-sql-db"
    / "manual_migration_world_pulse_read_seed_queue_v1.sql"
)


class _FakeConn:
    """Interprets the real SQL strings `queue.py` emits (Hub fake-conn style)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.executed: list[tuple[str, tuple]] = []
        self.digest_rows: list[dict] = []
        self.article_rows: list[dict] = []
        self._created_seq = 0

    def _norm(self, sql: str) -> str:
        return " ".join(sql.split())

    def _claim_pending(self) -> dict | None:
        pending = sorted(
            (r for r in self.rows.values() if r["status"] == "pending"),
            key=lambda r: (r["priority"], r.get("created_at", 0), r["seed_id"]),
        )
        if not pending:
            return None
        row = pending[0]
        row["status"] = "claimed"
        return row

    def _returning(self, row: dict) -> dict:
        return {
            "seed_id": row["seed_id"],
            "kind": row["kind"],
            "run_id": row["run_id"],
            "url": row["url"],
            "title": row["title"],
            "section": row["section"],
            "item_id": row["item_id"],
        }

    async def execute(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "INSERT INTO world_pulse_read_seed" in sql_n:
            seed_id = args[0]
            if seed_id in self.rows:
                return "INSERT 0 0"
            self._created_seq += 1
            self.rows[seed_id] = {
                "seed_id": args[0],
                "kind": args[1],
                "run_id": args[2],
                "url": args[3],
                "title": args[4],
                "section": args[5],
                "item_id": args[6],
                "priority": args[7],
                "status": "pending",
                "trace_id": None,
                "last_error": None,
                "created_at": self._created_seq,
            }
            return "INSERT 0 1"
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'claimed'" in sql_n:
            row = self._claim_pending()
            return "UPDATE 1" if row else "UPDATE 0"
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "done"
                self.rows[seed_id]["trace_id"] = trace_id
                self.rows[seed_id]["last_error"] = None
            return "UPDATE 1"
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'failed'" in sql_n:
            seed_id, error = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "failed"
                self.rows[seed_id]["last_error"] = error
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, sql: str, *args):
        sql_n = self._norm(sql)
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'claimed'" in sql_n:
            self.executed.append((sql, args))
            row = self._claim_pending()
            if not row:
                return None
            return self._returning(row)
        return await self.execute(sql, *args)

    async def fetch(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "FROM world_pulse_digest" in sql_n:
            return list(self.digest_rows)
        if "FROM world_pulse_article" in sql_n:
            return list(self.article_rows)
        return []

    async def fetchval(self, sql: str, *args):
        return None


def _finding(*, seed_id: str = "finding:r1:x") -> WorldPulseReadSeedV1:
    return WorldPulseReadSeedV1(
        seed_id=seed_id,
        kind="finding",
        run_id="r1",
        url="https://ex.com/a",
        title="A",
        section="ai_technology",
    )


def _digest_item() -> WorldPulseReadSeedV1:
    return WorldPulseReadSeedV1(
        seed_id="digest_item:r1:i1:b",
        kind="digest_item",
        run_id="r1",
        url="https://ex.com/d",
        title="D",
        section="hardware_compute_gpu",
        item_id="i1",
    )


def test_migration_defines_seed_queue_and_indexes():
    text = _MIGRATION.read_text()
    assert "create table if not exists world_pulse_read_seed" in text
    assert "idx_world_pulse_read_seed_claim" in text
    assert "idx_world_pulse_read_seed_run" in text
    assert "pending" in text and "claimed" in text


def test_enqueue_is_idempotent_on_seed_id():
    conn = _FakeConn()
    seed = _finding()

    async def _run():
        n1 = await enqueue_seeds(conn, [seed])
        n2 = await enqueue_seeds(conn, [seed])
        return n1, n2

    assert asyncio.run(_run()) == (1, 0)
    insert_sql = next(
        sql for sql, _ in conn.executed if "INSERT INTO world_pulse_read_seed" in sql
    )
    assert "ON CONFLICT" in insert_sql
    assert "DO NOTHING" in insert_sql


def test_claim_prefers_lower_priority_number():
    """Findings priority=0, digest_item priority=10."""
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_digest_item(), _finding(seed_id="finding:r1:a")])
        return await claim_next_seed(conn)

    claimed = asyncio.run(_run())
    assert claimed is not None
    assert claimed.kind == "finding"
    assert conn.rows["finding:r1:a"]["priority"] == 0
    assert conn.rows["digest_item:r1:i1:b"]["priority"] == 10


def test_claim_returns_none_when_empty():
    async def _run():
        return await claim_next_seed(_FakeConn())

    assert asyncio.run(_run()) is None


def test_mark_seed_done_and_failed():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding(), _digest_item()])
        await mark_seed_done(conn, "finding:r1:x", trace_id="trace-1")
        await mark_seed_failed(conn, "digest_item:r1:i1:b", error="boom")

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["status"] == "done"
    assert conn.rows["finding:r1:x"]["trace_id"] == "trace-1"
    assert conn.rows["digest_item:r1:i1:b"]["status"] == "failed"
    assert conn.rows["digest_item:r1:i1:b"]["last_error"] == "boom"


def test_ensure_seed_queue_schema_emits_create():
    conn = _FakeConn()

    async def _run():
        await ensure_seed_queue_schema(conn)

    asyncio.run(_run())
    executed_sql = [sql for sql, _ in conn.executed]
    assert len(executed_sql) == 3
    assert any("create table if not exists world_pulse_read_seed" in sql for sql in executed_sql)
    assert any("idx_world_pulse_read_seed_claim" in sql for sql in executed_sql)
    assert any("idx_world_pulse_read_seed_run" in sql for sql in executed_sql)


def test_enqueue_from_recent_digests_loads_digest_and_article_rows():
    conn = _FakeConn()
    conn.digest_rows = [
        {
            "run_id": "r1",
            "payload_json": {
                "run_id": "r1",
                "curiosity_followups": [],
                "items": [
                    {
                        "item_id": "item-2",
                        "run_id": "r1",
                        "title": "No worth_reading",
                        "category": "ai_technology",
                        "worth_reading": [],
                        "article_ids": ["art-9"],
                    }
                ],
            },
        }
    ]
    conn.article_rows = [
        {"article_id": "art-9", "url": "https://ex.com/from-article"},
    ]

    async def _run():
        return await enqueue_from_recent_digests(conn, limit_digests=5)

    n = asyncio.run(_run())
    assert n == 1
    stored = next(iter(conn.rows.values()))
    assert stored["kind"] == "digest_item"
    assert stored["url"] == "https://ex.com/from-article"
    assert stored["priority"] == 10
    assert any("FROM world_pulse_digest" in sql for sql, _ in conn.executed)
    assert any("FROM world_pulse_article" in sql for sql, _ in conn.executed)
