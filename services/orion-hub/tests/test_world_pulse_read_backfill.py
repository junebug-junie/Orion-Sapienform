"""Backfill CLI: enqueue seeds from digests, no FCC."""

from __future__ import annotations

import asyncio

from scripts.world_pulse_read_backfill import parse_args, run_backfill


class _FakeConn:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.executed: list[tuple[str, tuple]] = []
        self.digest_rows: list[dict] = []
        self.article_rows: list[dict] = []
        self._created_seq = 0

    def _norm(self, sql: str) -> str:
        return " ".join(sql.split())

    async def execute(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "INSERT INTO world_pulse_read_seed" in sql_n:
            seed_id = args[0]
            if seed_id in self.rows:
                return "INSERT 0 0"
            self._created_seq += 1
            self.rows[seed_id] = {
                "seed_id": seed_id,
                "kind": args[1],
                "url": args[3],
                "status": "pending",
            }
            return "INSERT 0 1"
        return "OK"

    async def fetch(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "FROM world_pulse_digest" in sql_n:
            return list(self.digest_rows)
        if "FROM world_pulse_article" in sql_n:
            return list(self.article_rows)
        if "SELECT seed_id FROM world_pulse_read_seed WHERE seed_id = ANY" in sql_n:
            ids = args[0] if args else []
            return [{"seed_id": i} for i in ids if i in self.rows]
        return []


def test_parse_args_flags():
    args = parse_args(["--dry-run", "--findings-only", "--limit-digests", "7"])
    assert args.dry_run is True
    assert args.findings_only is True
    assert args.limit_digests == 7
    default = parse_args([])
    assert default.dry_run is False
    assert default.findings_only is False
    assert default.limit_digests is None


def test_run_backfill_dry_run_does_not_insert():
    conn = _FakeConn()
    conn.digest_rows = [
        {
            "run_id": "r1",
            "payload_json": {
                "run_id": "r1",
                "curiosity_followups": [
                    {
                        "section": "ai_technology",
                        "articles": [{"url": "https://ex.com/find", "title": "F"}],
                    }
                ],
                "items": [
                    {
                        "item_id": "i1",
                        "run_id": "r1",
                        "title": "D",
                        "category": "ai_technology",
                        "worth_reading": ["https://ex.com/d"],
                        "article_ids": [],
                    }
                ],
            },
        }
    ]

    async def _run():
        dry = await run_backfill(
            conn, dry_run=True, findings_only=False, limit_digests=5
        )
        empty = dict(conn.rows)
        real = await run_backfill(
            conn, dry_run=False, findings_only=False, limit_digests=5
        )
        again = await run_backfill(
            conn, dry_run=False, findings_only=False, limit_digests=5
        )
        return dry, empty, real, again

    dry, empty, real, again = asyncio.run(_run())
    assert dry == 2
    assert empty == {}
    assert real == 2
    assert again == 0
    assert len(conn.rows) == 2
