import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.world_pulse_read.queue import (
    RECLAIM_REASON_PROCESS_RESTART,
    RECLAIM_REASON_STALE_TIMEOUT,
    claim_next_seed,
    claim_next_stage2_seed,
    count_seeds_by_status,
    count_stage2_by_status,
    enqueue_from_recent_digests,
    enqueue_seeds,
    ensure_seed_queue_schema,
    last_stage_timestamps,
    mark_seed_done,
    mark_seed_failed,
    mark_stage2_done,
    mark_stage2_failed,
    reclaim_stale_claimed,
    reclaim_stale_stage2_claimed,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _REPO_ROOT
    / "services"
    / "orion-sql-db"
    / "manual_migration_world_pulse_read_seed_queue_v1.sql"
)
_STAGE2_MIGRATION = (
    _REPO_ROOT
    / "services"
    / "orion-sql-db"
    / "manual_migration_world_pulse_read_stage2_v1.sql"
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
        row["claimed_at"] = datetime.now(timezone.utc)
        return row

    def _claim_stage2(self) -> dict | None:
        pending = sorted(
            (
                r
                for r in self.rows.values()
                if r["status"] == "done"
                and r.get("handoff_json") is not None
                and r.get("stage2_status", "pending") == "pending"
            ),
            key=lambda r: (
                r["priority"],
                r.get("handoff_at") or r.get("created_at", 0),
                r["seed_id"],
            ),
        )
        if not pending:
            return None
        row = pending[0]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc)
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
                "claimed_at": None,
                "completed_at": None,
                "handoff_json": None,
                "handoff_at": None,
                "stage2_status": "pending",
                "stage2_claimed_at": None,
                "stage2_completed_at": None,
                "stage2_error": None,
                "stage2_trace_id": None,
            }
            return "INSERT 0 1"
        if "SET stage2_status = 'pending'" in sql_n and "stage2_status = 'claimed'" in sql_n:
            older = float(args[0]) if args else 0.0
            reason = args[1] if len(args) > 1 else None
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=older)
            n = 0
            for row in self.rows.values():
                claimed_at = row.get("stage2_claimed_at")
                if (
                    row.get("stage2_status") == "claimed"
                    and claimed_at is not None
                    and claimed_at < cutoff
                ):
                    row["stage2_status"] = "pending"
                    row["stage2_claimed_at"] = None
                    row["stage2_error"] = reason
                    n += 1
            return f"UPDATE {n}"
        if (
            "UPDATE world_pulse_read_seed" in sql_n
            and "SET status = 'pending'" in sql_n
            and "status = 'claimed'" in sql_n
        ):
            older = float(args[0]) if args else 0.0
            reason = args[1] if len(args) > 1 else None
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=older)
            n = 0
            for row in self.rows.values():
                claimed_at = row.get("claimed_at")
                if (
                    row["status"] == "claimed"
                    and claimed_at is not None
                    and claimed_at < cutoff
                ):
                    row["status"] = "pending"
                    row["claimed_at"] = None
                    row["last_error"] = reason
                    n += 1
            return f"UPDATE {n}"
        if "SET stage2_status = 'claimed'" in sql_n:
            row = self._claim_stage2()
            return "UPDATE 1" if row else "UPDATE 0"
        if "UPDATE world_pulse_read_seed" in sql_n and "SET status = 'claimed'" in sql_n:
            row = self._claim_pending()
            return "UPDATE 1" if row else "UPDATE 0"
        if "SET stage2_status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["stage2_status"] = "done"
                self.rows[seed_id]["stage2_trace_id"] = trace_id
                self.rows[seed_id]["stage2_error"] = None
                self.rows[seed_id]["stage2_completed_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "SET stage2_status = 'failed'" in sql_n:
            seed_id, error = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["stage2_status"] = "failed"
                self.rows[seed_id]["stage2_error"] = error
                if len(args) > 2 and args[2]:
                    self.rows[seed_id]["stage2_trace_id"] = args[2]
                self.rows[seed_id]["stage2_completed_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "SET status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            handoff_json = args[2] if len(args) > 2 else None
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "done"
                self.rows[seed_id]["trace_id"] = trace_id
                self.rows[seed_id]["last_error"] = None
                self.rows[seed_id]["completed_at"] = datetime.now(timezone.utc)
                if handoff_json is not None:
                    payload = handoff_json
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    self.rows[seed_id]["handoff_json"] = payload
                    self.rows[seed_id]["handoff_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "SET status = 'failed'" in sql_n:
            seed_id, error = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "failed"
                self.rows[seed_id]["last_error"] = error
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, sql: str, *args):
        sql_n = self._norm(sql)
        if "SET stage2_status = 'claimed'" in sql_n:
            self.executed.append((sql, args))
            row = self._claim_stage2()
            if not row:
                return None
            out = self._returning(row)
            out["handoff_json"] = row.get("handoff_json")
            out["trace_id"] = row.get("trace_id")
            return out
        if "UPDATE world_pulse_read_seed" in sql_n and "SET status = 'claimed'" in sql_n:
            self.executed.append((sql, args))
            row = self._claim_pending()
            if not row:
                return None
            return self._returning(row)
        if "max(completed_at)" in sql_n:
            self.executed.append((sql, args))
            last1 = None
            last2 = None
            for row in self.rows.values():
                if row.get("completed_at") and (last1 is None or row["completed_at"] > last1):
                    last1 = row["completed_at"]
                if row.get("stage2_completed_at") and (
                    last2 is None or row["stage2_completed_at"] > last2
                ):
                    last2 = row["stage2_completed_at"]
            return {"last_stage1_at": last1, "last_stage2_at": last2}
        return await self.execute(sql, *args)

    async def fetch(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "FROM world_pulse_digest" in sql_n:
            return list(self.digest_rows)
        if "FROM world_pulse_article" in sql_n:
            return list(self.article_rows)
        if "GROUP BY stage2_status" in sql_n:
            counts: dict[str, int] = {}
            for row in self.rows.values():
                st = row.get("stage2_status") or "pending"
                counts[st] = counts.get(st, 0) + 1
            return [{"stage2_status": k, "n": v} for k, v in counts.items()]
        if "GROUP BY status" in sql_n:
            counts = {}
            for row in self.rows.values():
                st = row["status"]
                counts[st] = counts.get(st, 0) + 1
            return [{"status": k, "n": v} for k, v in counts.items()]
        if "SELECT seed_id FROM world_pulse_read_seed WHERE seed_id = ANY" in sql_n:
            ids = args[0] if args else []
            return [{"seed_id": i} for i in ids if i in self.rows]
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


def test_reclaim_stale_claimed_makes_seed_claimable_again():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        assert await claim_next_seed(conn) is None
        fresh_id = "finding:r1:fresh"
        await enqueue_seeds(conn, [_finding(seed_id=fresh_id)])
        conn.rows[fresh_id]["status"] = "claimed"
        conn.rows[fresh_id]["claimed_at"] = datetime.now(timezone.utc)
        n = await reclaim_stale_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )
        claimed = await claim_next_seed(conn)
        return n, claimed

    n, claimed = asyncio.run(_run())
    assert n == 1
    assert claimed is not None
    assert claimed.seed_id == "finding:r1:x"
    assert conn.rows["finding:r1:fresh"]["status"] == "claimed"


def test_reclaim_stale_claimed_writes_stale_timeout_reason():
    """Periodic in-loop reclaim (a turn that ran past its own timeout) leaves
    a real trace instead of silently resetting the row (confirmed live
    2026-09-10: a reclaim reset a seed with zero trace of it happening)."""
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        return await reclaim_stale_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )

    n = asyncio.run(_run())
    assert n == 1
    assert conn.rows["finding:r1:x"]["status"] == "pending"
    assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:stale_timeout"


def test_reclaim_stale_claimed_writes_process_restart_reason():
    """Startup reclaim (older_than_sec=0.0, catching a claim orphaned by a
    dead process) gets a distinct label from the periodic case."""
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        return await reclaim_stale_claimed(
            conn, older_than_sec=0.0, reason=RECLAIM_REASON_PROCESS_RESTART
        )

    n = asyncio.run(_run())
    assert n == 1
    assert conn.rows["finding:r1:x"]["status"] == "pending"
    assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:process_restart"


def test_reclaimed_seed_error_cleared_by_later_successful_completion():
    """A reclaim's `interrupted:*` marker must not survive a later real
    success -- MARK_DONE_SQL already nulls `last_error` on success; confirm
    that still holds once the reclaim itself writes a non-null value."""
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        await reclaim_stale_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )
        assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:stale_timeout"
        claimed = await claim_next_seed(conn)
        assert claimed is not None
        await mark_seed_done(conn, claimed.seed_id, trace_id="tr-retry")

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["status"] == "done"
    assert conn.rows["finding:r1:x"]["last_error"] is None


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
    assert any("create table if not exists world_pulse_read_seed" in sql for sql in executed_sql)
    assert any("idx_world_pulse_read_seed_claim" in sql for sql in executed_sql)
    assert any("idx_world_pulse_read_seed_run" in sql for sql in executed_sql)
    assert any("idx_world_pulse_read_seed_stage2_claim" in sql for sql in executed_sql)
    assert any("handoff_json" in sql for sql in executed_sql)


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


def test_stage2_migration_adds_handoff_and_stage2_columns():
    text = _STAGE2_MIGRATION.read_text()
    assert "handoff_json" in text
    assert "handoff_at" in text
    assert "stage2_status" in text
    assert "stage2_claimed_at" in text
    assert "stage2_completed_at" in text
    assert "stage2_error" in text
    assert "stage2_trace_id" in text
    assert "idx_world_pulse_read_seed_stage2_claim" in text
    assert "pending" in text and "claimed" in text


def _handoff_for(seed: WorldPulseReadSeedV1) -> WorldPulseReadHandoffV1:
    return WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned a thing.",
        trace_id="tr-s1",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )


def test_mark_seed_done_persists_handoff_json():
    conn = _FakeConn()
    seed = _finding()

    async def _run():
        await enqueue_seeds(conn, [seed])
        await mark_seed_done(conn, seed.seed_id, trace_id="tr-s1", handoff=_handoff_for(seed))

    asyncio.run(_run())
    row = conn.rows[seed.seed_id]
    assert row["status"] == "done"
    assert row["trace_id"] == "tr-s1"
    assert row["handoff_json"]["what_i_learned"] == "Learned a thing."
    assert row["handoff_json"]["trace_id"] == "tr-s1"
    assert row["handoff_at"] is not None


def test_claim_next_stage2_seed_requires_done_handoff():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        empty = await claim_next_stage2_seed(conn)
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        claimed = await claim_next_stage2_seed(conn)
        again = await claim_next_stage2_seed(conn)
        return empty, claimed, again

    empty, claimed, again = asyncio.run(_run())
    assert empty is None
    assert claimed is not None
    assert claimed.seed.seed_id == "finding:r1:x"
    assert claimed.handoff_json["what_i_learned"] == "Learned a thing."
    assert claimed.stage1_trace_id == "tr-s1"
    assert again is None
    assert conn.rows["finding:r1:x"]["stage2_status"] == "claimed"


def test_mark_stage2_done_and_failed():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding(), _digest_item()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-a", handoff=_handoff_for(_finding())
        )
        await mark_seed_done(
            conn, "digest_item:r1:i1:b", trace_id="tr-b", handoff=_handoff_for(_digest_item())
        )
        await mark_stage2_done(conn, "finding:r1:x", stage2_trace_id="tr-s2")
        await mark_stage2_failed(conn, "digest_item:r1:i1:b", error="boom")

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"
    assert conn.rows["finding:r1:x"]["stage2_trace_id"] == "tr-s2"
    assert conn.rows["digest_item:r1:i1:b"]["stage2_status"] == "failed"
    assert conn.rows["digest_item:r1:i1:b"]["stage2_error"] == "boom"


def test_reclaim_stale_stage2_claimed():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        assert await claim_next_stage2_seed(conn) is None
        n = await reclaim_stale_stage2_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )
        claimed = await claim_next_stage2_seed(conn)
        return n, claimed

    n, claimed = asyncio.run(_run())
    assert n == 1
    assert claimed is not None
    assert claimed.seed.seed_id == "finding:r1:x"


def test_reclaim_stale_stage2_claimed_writes_stale_timeout_reason():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        return await reclaim_stale_stage2_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )

    n = asyncio.run(_run())
    assert n == 1
    assert conn.rows["finding:r1:x"]["stage2_status"] == "pending"
    assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:stale_timeout"


def test_reclaim_stale_stage2_claimed_writes_process_restart_reason():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        return await reclaim_stale_stage2_claimed(
            conn, older_than_sec=0.0, reason=RECLAIM_REASON_PROCESS_RESTART
        )

    n = asyncio.run(_run())
    assert n == 1
    assert conn.rows["finding:r1:x"]["stage2_status"] == "pending"
    assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:process_restart"


def test_reclaimed_stage2_seed_error_cleared_by_later_successful_completion():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        await reclaim_stale_stage2_claimed(
            conn, older_than_sec=3500, reason=RECLAIM_REASON_STALE_TIMEOUT
        )
        assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:stale_timeout"
        claimed = await claim_next_stage2_seed(conn)
        assert claimed is not None
        await mark_stage2_done(conn, claimed.seed.seed_id, stage2_trace_id="tr-s2-retry")

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"
    assert conn.rows["finding:r1:x"]["stage2_error"] is None


def test_queue_count_helpers():
    conn = _FakeConn()

    async def _run():
        await enqueue_seeds(conn, [_finding(), _digest_item()])
        await mark_seed_done(
            conn, "finding:r1:x", trace_id="tr-s1", handoff=_handoff_for(_finding())
        )
        await mark_stage2_done(conn, "finding:r1:x", stage2_trace_id="tr-s2")
        await mark_seed_failed(conn, "digest_item:r1:i1:b", error="x")
        s1 = await count_seeds_by_status(conn)
        s2 = await count_stage2_by_status(conn)
        ts = await last_stage_timestamps(conn)
        return s1, s2, ts

    s1, s2, ts = asyncio.run(_run())
    assert s1["done"] == 1
    assert s1["failed"] == 1
    assert s1["pending"] == 0
    assert s2["done"] == 1
    assert s2["pending"] == 1
    assert ts["last_stage1_at"] is not None
    assert ts["last_stage2_at"] is not None


def test_enqueue_findings_only_and_dry_run_idempotent():
    conn = _FakeConn()
    conn.digest_rows = [
        {
            "run_id": "r1",
            "payload_json": {
                "run_id": "r1",
                "curiosity_followups": [
                    {
                        "section": "ai_technology",
                        "articles": [
                            {"url": "https://ex.com/find", "title": "Find"},
                        ],
                    }
                ],
                "items": [
                    {
                        "item_id": "item-2",
                        "run_id": "r1",
                        "title": "Digest",
                        "category": "ai_technology",
                        "worth_reading": ["https://ex.com/digest"],
                        "article_ids": [],
                    }
                ],
            },
        }
    ]

    async def _run():
        dry = await enqueue_from_recent_digests(
            conn, limit_digests=5, findings_only=True, dry_run=True
        )
        real = await enqueue_from_recent_digests(
            conn, limit_digests=5, findings_only=True, dry_run=False
        )
        again = await enqueue_from_recent_digests(
            conn, limit_digests=5, findings_only=True, dry_run=False
        )
        dry_after = await enqueue_from_recent_digests(
            conn, limit_digests=5, findings_only=True, dry_run=True
        )
        return dry, real, again, dry_after, list(conn.rows.values())

    dry, real, again, dry_after, rows = asyncio.run(_run())
    assert dry == 1
    assert real == 1
    assert again == 0
    assert dry_after == 0
    assert len(rows) == 1
    assert rows[0]["kind"] == "finding"
    assert rows[0]["url"] == "https://ex.com/find"
