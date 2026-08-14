#!/usr/bin/env python3
"""Restore chat_history_log turns that lost their turn-level write to the
concurrent same-primary-key race in orion-sql-writer.

Background
----------
One Hub turn publishes four bus events in a ~10ms burst: the user message, the
assistant message, the turn row, and a legacy raw dict. orion-sql-writer runs
handlers on a thread pool (``asyncio.to_thread`` at services/orion-sql-writer/
app/worker.py:1539), so all four hit the database concurrently in independent
transactions. Three of them write the *same* ``chat_history_log`` primary key.
Whichever commits first wins; the other two raise a 23505 unique violation that
``_write_row`` swallows as "Duplicate entry ... skipping (idempotent write)"
(worker.py:1332-1342). That is correct for append-only tables and wrong here,
because ``chat_history_log`` is a row progressively assembled by three events.

The casualty is always the turn-level write -- the only one carrying ``source``
plus both halves of the conversation -- so the surviving row has ``source IS
NULL`` and exactly one of prompt/response.

The legacy raw dict is not a registered envelope kind, so sql-writer files it in
``bus_fallback_log`` with ``error='Unknown kind'`` instead of routing it. That
accident is what makes recovery possible: it is a complete, untouched copy of
prompt + response + session + spark_meta for every affected turn.

Safety
------
Fill-only. A field is written only when the stored value is empty; a row that
already holds text is never overwritten. Rows are matched by correlation_id and
only ``source IS NULL`` rows are considered, so a healthy row can never be
touched. Dry-run by default -- pass --apply to write.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

JOB_DIR = Path("/tmp/chat-history-backfill")

# `legacy.message` is NOT Hub-exclusive: at least six services accept that kind
# (orion-llm-gateway, orion-state-service, orion-cortex-orch, orion-context-exec,
# orion-harness-governor, orion-whisper-tts), and correlation_id is turn-scoped,
# so it propagates across all of them. Matching on kind + correlation_id alone
# would let a foreign envelope that happens to carry a `prompt` key -- an LLM
# request dict, say -- be spliced into the chat log as if it were Juniper's turn.
# So the payload must also *look* like a Hub turn: a known Hub source label plus
# both halves present. Requiring `source` additionally makes idempotency a
# property rather than an accident -- a repaired row always gets `source` set, so
# it stops matching `source is null` on the next run.
_HUB_SOURCES = ("hub_orion", "hub_ws", "hub_http", "hub")

SELECT_CANDIDATES = """
    select distinct on (h.id)
           h.id,
           h.correlation_id,
           coalesce(h.prompt, '')   as cur_prompt,
           coalesce(h.response, '') as cur_response,
           h.source                 as cur_source,
           h.session_id             as cur_session_id,
           h.spark_meta             as cur_spark_meta,
           h.created_at,
           b.payload                as fallback_payload,
           b.id                     as fallback_id
    from chat_history_log h
    join bus_fallback_log b
      on b.correlation_id = h.correlation_id
     and b.kind = 'legacy.message'
    where h.source is null
      -- _write_fallback stores a correlation-less envelope as '' rather than
      -- NULL (worker.py:2368), and '' = '' would cross-product every such row.
      and coalesce(b.correlation_id, '') <> ''
      and b.payload::jsonb ->> 'source' in %(hub_sources)s
      and b.payload::jsonb ? 'prompt'
      and b.payload::jsonb ? 'response'
    -- distinct on + this ordering picks the most recent fallback per row, so two
    -- fallbacks sharing a correlation_id cannot silently double-count or let row
    -- order decide which payload wins.
    order by h.id, b.created_at_ts desc
"""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _connect(dsn: str | None):
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("ORION_PG_HOST", "localhost"),
        port=int(os.getenv("ORION_PG_PORT", "55432")),
        user=os.getenv("ORION_PG_USER", "postgres"),
        password=os.getenv("ORION_PG_PASSWORD", "postgres"),
        dbname=os.getenv("ORION_PG_DB", "conjourney"),
    )


def plan_updates(rows) -> list[dict]:
    """Decide, per row, which empty columns the fallback payload can fill."""
    planned: list[dict] = []
    for row in rows:
        payload = row["fallback_payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue

        updates: dict[str, object] = {}

        # Fill-only: never clobber text that is already stored.
        if not _text(row["cur_prompt"]) and _text(payload.get("prompt")):
            updates["prompt"] = payload["prompt"]
        if not _text(row["cur_response"]) and _text(payload.get("response")):
            updates["response"] = payload["response"]
        if not _text(row["cur_source"]) and _text(payload.get("source")):
            updates["source"] = payload["source"]
        if not _text(row["cur_session_id"]) and _text(payload.get("session_id")):
            updates["session_id"] = payload["session_id"]
        if not row["cur_spark_meta"] and isinstance(payload.get("spark_meta"), dict):
            updates["spark_meta"] = json.dumps(payload["spark_meta"])

        if not updates:
            continue

        planned.append(
            {
                "id": row["id"],
                "correlation_id": row["correlation_id"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "fallback_id": row["fallback_id"],
                "before": {
                    "prompt_len": len(row["cur_prompt"]),
                    "response_len": len(row["cur_response"]),
                    "source": row["cur_source"],
                    "session_id": row["cur_session_id"],
                    # Recorded verbatim, not as a length: spark_meta is the one
                    # column the UPDATE can replace wholesale, so the snapshot
                    # has to be enough to reconstruct it (CLAUDE.md section 14).
                    "spark_meta": row["cur_spark_meta"],
                },
                "updates": updates,
            }
        )
    return planned


def build_fill_only_guard(cols: list[str]) -> str:
    """SQL re-asserting 'this column is still empty' for every column we write.

    Lives inside the UPDATE so a concurrent writer that filled the row between
    planning and applying wins over this backfill. Every planned column must be
    represented: an earlier version skipped spark_meta by name, which let a
    spark_meta-only plan collapse to `true` and clobber a concurrent write.
    """
    if not cols:
        raise ValueError("refusing to build an unguarded UPDATE")
    return " and ".join(
        # spark_meta is JSONB -- coalesce(col,'') would not even typecheck.
        f"{c} is null" if c == "spark_meta" else f"coalesce({c}, '') = ''"
        for c in cols
    )


def write_snapshot(planned: list[dict]) -> Path:
    """CLAUDE.md section 14: snapshot exactly what will change before changing it."""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = JOB_DIR / f"snapshot_{stamp}.json"
    path.write_text(json.dumps(planned, indent=2, default=str))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually write; omit for a dry run")
    ap.add_argument("--dsn", default=os.getenv("ORION_PG_DSN"),
                    help="postgres DSN (defaults to local conjourney)")
    args = ap.parse_args()

    conn = _connect(args.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SELECT_CANDIDATES, {"hub_sources": _HUB_SOURCES})
            rows = cur.fetchall()

        planned = plan_updates(rows)
        print(f"broken rows with a recoverable fallback payload: {len(rows)}")
        print(f"rows with at least one field to restore:          {len(planned)}\n")

        for item in planned:
            restored = ", ".join(
                f"{k}(+{len(str(v))}ch)" if k in ("prompt", "response") else k
                for k, v in item["updates"].items()
            )
            # created_at is server_default, not NOT NULL, so a legacy row can
            # carry NULL here -- plan_updates already tolerates it, and slicing
            # None would crash before the snapshot is even written.
            when = (item["created_at"] or "unknown-date")[:19]
            print(f"  {item['id'][:8]}  {when:19}  restore: {restored}")

        if not planned:
            print("\nnothing to do.")
            return 0

        snapshot = write_snapshot(planned)
        print(f"\nsnapshot written: {snapshot}")

        if not args.apply:
            print("DRY RUN -- no rows written. Re-run with --apply to commit.")
            return 0

        applied = 0
        skipped: list[str] = []
        with conn.cursor() as cur:
            for item in planned:
                cols = list(item["updates"].keys())
                assignments = ", ".join(f"{c} = %s" for c in cols)
                values = [item["updates"][c] for c in cols]
                guards = build_fill_only_guard(cols)
                cur.execute(
                    f"update chat_history_log set {assignments} "
                    f"where id = %s and source is null and ({guards})",
                    values + [item["id"]],
                )
                if cur.rowcount:
                    applied += cur.rowcount
                else:
                    # Planned but not written: the row changed under us between
                    # plan and apply. Silence here would look like success.
                    skipped.append(item["id"])
        conn.commit()

        print(f"\napplied: {applied} row(s) updated")
        if skipped:
            print(f"SKIPPED {len(skipped)} planned row(s) -- changed since planning:")
            for row_id in skipped:
                print(f"  {row_id}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
