"""The day dream eval (day-dream spec Patch 2, "Patch 2 eval"): every ``channel='dream'`` memory beside the text of the
records it cites. Read-only, live.

The danger this watches (spec, proposal mode): memories that sound right but do not match the records. Code already
refuses any unresolved ref at write time; this puts each statement next to its refs' words so a reader (or, later, a
judge on the local lane) can mark it supported or not. It also re-resolves every ref NOW: a ref that resolved at
write time and no longer does (retention) is said, never hidden.

    python3 services/orion-durable-runs/evals/run_dream_day_eval.py               # every dream memory
    python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-17
    python3 services/orion-durable-runs/evals/run_dream_day_eval.py --json

Reads through ``docker exec orion-athena-sql-db psql`` with ``default_transaction_read_only=on`` (nothing can be
written), or ``--dsn`` with a read-only psycopg session. Not a gate yet: the 90%-supported gate needs a judge and
7 nights of rows (spec acceptance); this prints the evidence that judge would read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from orion.memory.dream_day import REF_KIND_ALIASES  # noqa: E402

CONTAINER = os.environ.get("ORION_SQL_CONTAINER", "orion-athena-sql-db")

DAYS_SQL = """
SELECT day::text, run_id, produced, note, memories_written, memory_ids, items_dropped, refs_dropped, kept, dropped,
       new, day_text, day_refs, read->'unread' AS unread, read->>'digs_total' AS digs_total, now_shown_id
  FROM dream_day {where} ORDER BY day
"""
MEMORIES_SQL = """
SELECT m.memory_id::text, m.episode_id, m.purpose, m.voice, m.statement, m.status, m.run_id, m.created_at::text,
       coalesce((SELECT json_agg(json_build_object('ref', e.source_id, 'quote', e.quote, 'verified', e.verified)
                                 ORDER BY e.source_id)
                   FROM episode_memory_evidence e WHERE e.memory_id = m.memory_id), '[]'::json) AS evidence
  FROM episode_memory m WHERE m.channel = 'dream' {where} ORDER BY m.episode_id, m.created_at, m.memory_id
"""
# Does each ref still name a row? One existence query per kind (ids are quoted literals from our own rows).
EXISTS_SQL = {
    "experience": "SELECT event_id AS id FROM temporal_self_event WHERE event_id IN ({ids})",
    "act": "SELECT act_id AS id FROM grammar_acts WHERE act_id IN ({ids})",
    "dream_cycle": "SELECT cycle_id AS id FROM dream_cycle WHERE cycle_id IN ({ids})",
    "dream_hypothesis": "SELECT hypothesis_id AS id FROM dream_hypothesis WHERE hypothesis_id IN ({ids})",
    "harness_turn_trace": "SELECT correlation_id AS id FROM harness_turn_trace WHERE correlation_id IN ({ids})",
    "chat_turn": "SELECT id FROM chat_history_log WHERE id IN ({ids}) UNION "
                 "SELECT correlation_id FROM chat_history_log WHERE correlation_id IN ({ids})",
}


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def query(sql: str, dsn: str | None) -> list[dict[str, Any]]:
    """Rows of one read-only SELECT."""
    if dsn:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.read_only = True
            return [dict(r) for r in conn.execute(sql).fetchall()]
    wrapped = f"SELECT coalesce(json_agg(q), '[]'::json) FROM ({sql}) q"
    out = subprocess.run(["docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on", CONTAINER, "psql",
                          "-U", "postgres", "-d", "conjourney", "-Atqc", wrapped],
                         capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout.strip() or "[]")


def still_resolving(refs: list[str], dsn: str | None) -> set[str]:
    by_kind: dict[str, list[str]] = {}
    for r in refs:
        kind, _, rid = r.partition(":")
        by_kind.setdefault(REF_KIND_ALIASES.get(kind, kind), []).append(rid)
    alive: set[str] = set()
    for kind, ids in by_kind.items():
        sql = EXISTS_SQL.get(kind)
        if not sql or not ids:
            continue
        for row in query(sql.format(ids=", ".join(_lit(i) for i in ids)), dsn):
            alive.add(f"{kind}:{row['id']}")
    return alive


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--day", help="one Denver day (YYYY-MM-DD); default every night")
    ap.add_argument("--dsn", help="a Postgres DSN (read-only session); default docker exec psql")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    day_where = f"WHERE day = {_lit(ns.day)}::date" if ns.day else ""
    mem_where = f"AND m.episode_id = {_lit('dream_day:' + ns.day)}" if ns.day else ""
    present = query("SELECT to_regclass('dream_day') IS NOT NULL AS present", ns.dsn)[0]["present"]
    if not present:
        print("UNREAD dream_day: the table does not exist here (apply "
              "services/orion-sql-db/manual_migration_dream_day.sql); dream memories are still listed below")
    days = query(DAYS_SQL.format(where=day_where), ns.dsn) if present else []
    memories = query(MEMORIES_SQL.format(where=mem_where), ns.dsn)
    refs = sorted({e["ref"] for m in memories for e in (m.get("evidence") or [])})
    alive = still_resolving(refs, ns.dsn)
    for m in memories:
        for e in m.get("evidence") or []:
            kind, _, rid = e["ref"].partition(":")
            e["resolves_now"] = f"{REF_KIND_ALIASES.get(kind, kind)}:{rid}" in alive
    if ns.json:
        print(json.dumps({"days": days, "memories": memories}, indent=1, default=str))
        return 0
    print(f"dream_day rows: {len(days)}")
    for d in days:
        unread = d.get("unread") or {}
        print(f"  {d['day']} run={d['run_id']} produced={d['produced']} ({d['note']}); memories={d['memories_written']} "
              f"items_dropped={d['items_dropped']} refs_dropped={d['refs_dropped']} plan keep/drop/new="
              f"{d['kept']}/{d['dropped']}/{d['new']} digs={d['digs_total']} unread={sorted(unread) or 'none'} "
              f"now_shown={d['now_shown_id']}")
        if d.get("day_text"):
            print(f"    day: {d['day_text']}")
    print(f"\nchannel='dream' memories: {len(memories)} (each statement, then the text of every record it cites)")
    gone = 0
    for m in memories:
        print(f"\n[{m['episode_id']}] {m['memory_id']} ({m['purpose']}, {m['voice']}, {m['status']})")
        print(f"  STATEMENT: {m['statement']}")
        for e in m.get("evidence") or []:
            flag = "" if e["resolves_now"] else "  <-- NO LONGER RESOLVES"
            gone += 0 if e["resolves_now"] else 1
            print(f"  REF {e['ref']}{flag}\n      {e['quote']}")
    print(f"\nrefs: {len(refs)} cited, {sum(1 for r in refs if _alias(r) not in alive)} no longer resolve; "
          f"evidence lines not resolving now: {gone}")
    return 0


def _alias(ref: str) -> str:
    kind, _, rid = ref.partition(":")
    return f"{REF_KIND_ALIASES.get(kind, kind)}:{rid}"


if __name__ == "__main__":
    raise SystemExit(main())
