#!/usr/bin/env python3
"""Snapshot stage for the collapse-triage Falkor backfill
(feat/falkor-collapse-triage-consumption, PR #1271).

Runs on the HOST (not inside a container) -- needs asyncpg/psycopg2 against
the real Postgres instance, not spaCy. Snapshots every real Juniper-observed
`collapse_mirror` row (the durable store `orion-sql-writer` writes via
`orion:collapse:sql-write`, NOT the tiny `orion:enrichment` Fuseki graph --
live-checked 2026-07-22: only 2 of these ever reached Fuseki, out of 68 real
Juniper-observed rows, because until this backfill nothing had ever run
tag/entity extraction against the other 66).

Deliberately excludes `observer='orion'` rows (8,385 of 8,499 total,
2026-07-22 count) -- Orion's own self-observations are excluded from
tagging by `handle_triage_event`'s own `_is_orion` gate in production, and
this backfill must match that live gate exactly, not backfill data the live
pipeline would never have processed.

Per AGENTS.md section 14 (backfill protocol): 68 rows is trivially under
the 100k-row/100MB stop-and-ask threshold. Snapshot written to
/tmp/backfill-collapse-triage-falkor/snapshot.json before any write.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg2
import psycopg2.extras

OUT_DIR = Path("/tmp/backfill-collapse-triage-falkor")
OUT_PATH = OUT_DIR / "snapshot.json"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host="localhost", port=55432, user="postgres", password="postgres", dbname="conjourney"
    )
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            select id, correlation_id, summary, trigger, timestamp, observer
            from collapse_mirror
            where lower(observer) = 'juniper'
            order by timestamp asc nulls last
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        # EventIn.prepare_and_hydrate_text's own field priority is
        # ["summary", "trigger", "text", "text_content", "content"] --
        # CollapseMirrorEntryV1 only ever has summary/trigger, so mirroring
        # just those two in the same order matches the live text-hydration
        # path exactly for this row shape.
        text = (r["summary"] or r["trigger"] or "").strip()
        out.append(
            {
                "id": r["id"],
                "correlation_id": r["correlation_id"],
                "text": text,
                "timestamp": r["timestamp"],
                "observer": r["observer"],
            }
        )

    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    empty_text = [r["id"] for r in out if not r["text"]]
    print(f"snapshot_rows={len(out)} empty_text_rows={len(empty_text)} path={OUT_PATH}")
    if empty_text:
        print(f"WARNING: rows with no text (will be skipped by extract stage): {empty_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
