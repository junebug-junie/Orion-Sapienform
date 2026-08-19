#!/usr/bin/env python3
"""One-off backfill: move existing AI-Town rows out of chat_history_log into
aitown_chat_history_log (Track B cutover, 2026-08-19,
docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md).

Why an atomic move now, not the phased dual-write bridge originally
designed: AI Town's own backend is confirmed dead
(orion.embodiment.aitown_client.AitownClientError: Connection refused,
every call) and chat_history_log's own history shows zero AI-Town writes
in 18+ days -- zero concurrent-write risk, so there is no live producer to
race against. worker.py's routing (SQL_WRITER_AITOWN_ROUTING_ENABLED,
default true) already sends every NEW AI-Town row to the mirror table
going forward; this script is purely for the historical rows that predate
that fix.

Move, not copy: a single atomic transaction (INSERT ... SELECT into the
mirror table, then DELETE the moved rows from chat_history_log, same
transaction) so a given row ends up in exactly one table, never briefly in
both and never lost between the two statements -- either the whole move
commits or none of it does.

Snapshot-first per AGENTS.md section 14 backfill protocol: full row dump
written to /tmp/aitown_chat_history_backfill/snapshot_before.json before
any mutation, well under the 100k-row/100MB stop-and-ask threshold (1,577
rows as of 2026-08-19).

Usage:
    python scripts/backfill_aitown_chat_history_move_to_split_table.py --dry-run
    python scripts/backfill_aitown_chat_history_move_to_split_table.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

# Running as `python scripts/backfill_....py` puts scripts/ on sys.path[0],
# which shadows stdlib `platform` via scripts/platform/ and breaks asyncpg
# (same issue documented in scripts/backfill_phi_corpus.py and
# scripts/backfill_recall_falkor_chat_tags_snapshot.py).
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import asyncpg

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"
JOB_DIR = Path("/tmp/aitown_chat_history_backfill")

# Every column on both tables, in a stable order -- both are column-for-
# column identical (see app/models/aitown_chat_history_log.py's own
# docstring; test_aitown_chat_history_dual_write.py's
# TestMirrorTableSchemaParity gates this staying true).
_COLUMNS = [
    "id", "correlation_id", "source", "prompt", "response", "user_id",
    "session_id", "spark_meta", "memory_status", "memory_tier",
    "memory_reason", "thought_process", "client_meta",
    "llm_uncertainty_source", "llm_mean_logprob", "llm_min_logprob",
    "llm_mean_top1_margin", "llm_low_margin_token_count",
    "llm_low_logprob_token_count", "llm_unstable_span_count",
    "llm_uncertainty_available", "created_at",
]
_COLUMN_LIST_SQL = ", ".join(_COLUMNS)

_AITOWN_WHERE_SQL = "(client_meta -> 'external_room' ->> 'platform') = 'aitown'"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


async def _snapshot(conn: asyncpg.Connection, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = await conn.fetch(
        f"SELECT {_COLUMN_LIST_SQL} FROM chat_history_log WHERE {_AITOWN_WHERE_SQL} "
        "ORDER BY created_at ASC"
    )
    payload = [dict(r) for r in rows]
    with open(out_dir / "snapshot_before.json", "w") as fh:
        json.dump(payload, fh, default=_json_default, indent=2)
    return payload


async def _move(conn: asyncpg.Connection) -> int:
    """Atomic INSERT + DELETE in one transaction. Returns rows moved."""
    async with conn.transaction():
        inserted = await conn.fetchval(
            f"""
            WITH moved AS (
                INSERT INTO aitown_chat_history_log ({_COLUMN_LIST_SQL})
                SELECT {_COLUMN_LIST_SQL} FROM chat_history_log
                WHERE {_AITOWN_WHERE_SQL}
                ON CONFLICT (id) DO NOTHING
                RETURNING id
            )
            SELECT count(*) FROM moved
            """
        )
        deleted = await conn.fetchval(
            f"""
            WITH removed AS (
                DELETE FROM chat_history_log
                WHERE {_AITOWN_WHERE_SQL}
                RETURNING id
            )
            SELECT count(*) FROM removed
            """
        )
        if inserted != deleted:
            # Should be structurally impossible (same WHERE clause, same
            # transaction, no concurrent writer per this script's own
            # premise) -- if it ever happens, the transaction's own ROLLBACK
            # (raising here inside `async with conn.transaction()`) is what
            # keeps a partial move from ever committing.
            raise RuntimeError(
                f"insert/delete count mismatch: inserted={inserted} deleted={deleted} "
                "-- rolling back, nothing moved"
            )
        return deleted


async def _counts(conn: asyncpg.Connection) -> dict[str, int]:
    primary_total = await conn.fetchval("SELECT count(*) FROM chat_history_log")
    primary_aitown = await conn.fetchval(
        f"SELECT count(*) FROM chat_history_log WHERE {_AITOWN_WHERE_SQL}"
    )
    mirror_total = await conn.fetchval("SELECT count(*) FROM aitown_chat_history_log")
    return {
        "chat_history_log_total": primary_total,
        "chat_history_log_aitown_rows": primary_aitown,
        "aitown_chat_history_log_total": mirror_total,
    }


async def _run(dsn: str, dry_run: bool) -> int:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(dsn)
    try:
        before = await _counts(conn)
        print(f"before: {json.dumps(before)}")

        snapshot_rows = await _snapshot(conn, JOB_DIR)
        print(f"snapshot: {len(snapshot_rows)} rows -> {JOB_DIR / 'snapshot_before.json'}")

        if before["chat_history_log_aitown_rows"] != len(snapshot_rows):
            print(
                "ERROR: snapshot row count does not match the aitown-row count -- "
                "aborting before any mutation."
            )
            return 1

        if len(snapshot_rows) == 0:
            print("Nothing to move -- 0 AI-Town rows in chat_history_log. Done.")
            return 0

        if dry_run:
            print(f"DRY RUN: would move {len(snapshot_rows)} rows. No changes made.")
            return 0

        moved = await _move(conn)
        after = await _counts(conn)
        print(f"moved: {moved} rows")
        print(f"after: {json.dumps(after)}")

        report = {
            "job": "aitown_chat_history_move_to_split_table",
            "before": before,
            "moved": moved,
            "after": after,
            "snapshot_path": str(JOB_DIR / "snapshot_before.json"),
            "verdict": "ok" if after["chat_history_log_aitown_rows"] == 0 and after[
                "aitown_chat_history_log_total"
            ] >= moved else "needs_review",
        }
        with open(JOB_DIR / "report.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"report: {JOB_DIR / 'report.json'}")
        print(f"verdict: {report['verdict']}")
        return 0 if report["verdict"] == "ok" else 1
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move existing AI-Town chat_history_log rows into aitown_chat_history_log"
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--dry-run", action="store_true", help="snapshot + report only, no mutation")
    args = parser.parse_args()
    return asyncio.run(_run(args.dsn, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
