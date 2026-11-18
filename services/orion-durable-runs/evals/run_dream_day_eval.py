"""The day dream eval (day-dream spec Patch 2, "Patch 2 eval"): every ``channel='dream'`` memory beside the text of the
records it cites. Read-only, live.

The danger this watches (spec, proposal mode): memories that sound right but do not match the records. Code already
refuses any unresolved ref at write time; this puts each statement next to its refs' words so a reader (or, later, a
judge on the local lane) can mark it supported or not. It also re-resolves every ref NOW: a ref that resolved at
write time and no longer does (retention) is said, never hidden.

    python3 services/orion-durable-runs/evals/run_dream_day_eval.py               # every dream memory
    python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-17
    python3 services/orion-durable-runs/evals/run_dream_day_eval.py --json
    python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-17 --judge   # re-judge now

Reads through ``docker exec orion-athena-sql-db psql`` with ``default_transaction_read_only=on`` (nothing can be
written), or ``--dsn`` with a read-only psycopg session.

The grounding gate (2026-11-18): each night's row carries the gate's verdicts (``dream_day.grounding``: how many
memories were judged supported, unsupported or unjudged, and each refused statement with the claims no record
shows); they are printed under the night. ``--judge`` re-runs the same gate NOW on every written dream memory:
the same ref resolver (app/dream_day_store.py ``resolve_refs``, read-only session), the same prompt and strict
verdict parser (orion/memory/dream_day.py ``judge_memories``), on the same local route through the gateway's
HTTP door (``--gateway``, default $ORION_GATEWAY_URL or http://localhost:8210). It writes nothing. The 90%-supported
gate over 7 nights (spec acceptance) reads the ``supported`` share it prints.
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
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(SERVICE_ROOT)]

from orion.memory.dream_day import REF_KIND_ALIASES  # noqa: E402

CONTAINER = os.environ.get("ORION_SQL_CONTAINER", "orion-athena-sql-db")

DAYS_SQL = """
SELECT day::text, run_id, produced, note, memories_written, memory_ids, items_dropped, refs_dropped, kept, dropped,
       new, day_text, day_refs, read->'unread' AS unread, read->>'digs_total' AS digs_total, now_shown_id{grounding}
  FROM dream_day {where} ORDER BY day
"""
# The gate's columns (services/orion-sql-db/manual_migration_dream_day_grounding.sql); absent before it is applied.
GROUNDING_COLS_SQL = ("SELECT count(*) AS n FROM information_schema.columns WHERE table_name = 'dream_day' "
                      "AND column_name IN ('memories_unsupported', 'memories_unjudged', 'grounding')")
GROUNDING_SELECT = ", memories_unsupported, memories_unjudged, grounding"
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


def default_dsn() -> str:
    """$ORION_PAGES_DSN, else orion-durable-runs' own .env POSTGRES_URI on the host port (as the supersede eval)."""
    if os.environ.get("ORION_PAGES_DSN"):
        return os.environ["ORION_PAGES_DSN"]
    for base in (REPO, Path("/mnt/scripts/Orion-Sapienform")):
        env = base / "services/orion-durable-runs/.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POSTGRES_URI="):
                    return line.split("=", 1)[1].strip().replace("@orion-athena-sql-db:5432", "@localhost:55432")
    raise SystemExit("no DSN for --judge: pass --dsn or set ORION_PAGES_DSN")


class ReadOnlyPool:
    """``pool.connection()`` for app/dream_day_store.py ``resolve_refs``: every session is read-only."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def connection(self):
        from contextlib import asynccontextmanager

        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        @asynccontextmanager
        async def cm():
            conn = await AsyncConnection.connect(self.dsn, row_factory=dict_row, prepare_threshold=0)
            try:
                await conn.set_read_only(True)
                yield conn
            finally:
                await conn.close()
        return cm()


def gateway_ask(gateway: str, route: str):
    """The judge's model call through the gateway's HTTP door: the same route, options as the run's call
    (app/runner.py ``_call_frozen_evidence_llm``: JSON only, no thinking, temperature 0)."""
    import httpx

    from orion.memory.dream_day import GROUNDING_MAX_TOKENS

    async def ask(prompt: str) -> dict[str, Any]:
        import asyncio
        import time

        started = time.monotonic()
        async with httpx.AsyncClient(timeout=180.0) as client:
            while True:
                resp = await client.post(f"{gateway.rstrip('/')}/v1/chat/completions", json={
                    "model": route, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": GROUNDING_MAX_TOKENS, "temperature": 0.0, "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False}})
                # 503 = the route's seat is busy (the run asks under its own hold and never sees this): wait.
                if resp.status_code == 503 and time.monotonic() - started < 900:
                    await asyncio.sleep(10.0)
                    continue
                break
            resp.raise_for_status()
            body = resp.json()
        choice = (body.get("choices") or [{}])[0]
        return {"text": (choice.get("message") or {}).get("content") or "", "finish_reason": choice.get("finish_reason"),
                "model": body.get("model")}
    return ask


def rejudge(memories: list[dict[str, Any]], dsn: str, gateway: str, route: str) -> list[dict[str, Any]]:
    """The gate, re-run now on written memories. Read-only: resolve through a read-only session, ask the model."""
    import asyncio

    from app.dream_day_store import resolve_refs
    from orion.memory.dream_day import CheckedMemory, canonical_ref, judge_memories

    if route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("the grounding judge never runs on the fast or metacog lanes")

    async def go() -> list[dict[str, Any]]:
        checked = []
        for m in memories:
            refs = [c for c in (canonical_ref(e["ref"]) for e in (m.get("evidence") or [])) if c]
            checked.append(CheckedMemory(statement=" ".join(m["statement"].split()), purpose=m["purpose"], refs=refs))
        resolved = await resolve_refs(ReadOnlyPool(dsn), sorted({r for c in checked for r in c.refs}), full=True)
        verdicts = await judge_memories(checked, resolved, gateway_ask(gateway, route))
        for m, v in zip(memories, verdicts):
            v["memory_id"] = m["memory_id"]
            v["ref_text"] = {r: resolved.get(r) for r in v["refs"]}
        return verdicts
    return asyncio.run(go())


def print_grounding(g: Any, indent: str = "    ") -> None:
    if not g:
        print(f"{indent}grounding: none recorded (row written before the gate, or gate off)")
        return
    g = json.loads(g) if isinstance(g, str) else g
    print(f"{indent}grounding (enabled={g.get('enabled')}): judged={g.get('judged')} supported={g.get('supported')} "
          f"unsupported={g.get('unsupported')} unjudged={g.get('unjudged')}")
    for v in g.get("verdicts") or []:
        if v.get("status") != "supported":
            why = "; ".join(v.get("unsupported_claims") or []) or v.get("error") or ""
            print(f"{indent}  NOT WRITTEN ({v.get('status')}): {v.get('statement')}\n{indent}    -> {why}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--day", help="one Denver day (YYYY-MM-DD); default every night")
    ap.add_argument("--dsn", help="a Postgres DSN (read-only session); default docker exec psql")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--judge", action="store_true", help="re-run the grounding gate now on every written memory "
                    "(read-only; calls the local route through the gateway)")
    ap.add_argument("--gateway", default=os.environ.get("ORION_GATEWAY_URL", "http://localhost:8210"))
    ap.add_argument("--route", default="memory_distill")
    ns = ap.parse_args(argv)
    day_where = f"WHERE day = {_lit(ns.day)}::date" if ns.day else ""
    mem_where = f"AND m.episode_id = {_lit('dream_day:' + ns.day)}" if ns.day else ""
    present = query("SELECT to_regclass('dream_day') IS NOT NULL AS present", ns.dsn)[0]["present"]
    if not present:
        print("UNREAD dream_day: the table does not exist here (apply "
              "services/orion-sql-db/manual_migration_dream_day.sql); dream memories are still listed below")
    has_gate = present and int(query(GROUNDING_COLS_SQL, ns.dsn)[0]["n"]) == 3
    days = query(DAYS_SQL.format(where=day_where, grounding=GROUNDING_SELECT if has_gate else ""), ns.dsn) \
        if present else []
    memories = query(MEMORIES_SQL.format(where=mem_where), ns.dsn)
    refs = sorted({e["ref"] for m in memories for e in (m.get("evidence") or [])})
    alive = still_resolving(refs, ns.dsn)
    for m in memories:
        for e in m.get("evidence") or []:
            kind, _, rid = e["ref"].partition(":")
            e["resolves_now"] = f"{REF_KIND_ALIASES.get(kind, kind)}:{rid}" in alive
    verdicts = rejudge(memories, ns.dsn or default_dsn(), ns.gateway, ns.route) if ns.judge else None
    if ns.json:
        print(json.dumps({"days": days, "memories": memories, "rejudged": verdicts}, indent=1, default=str))
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
        if has_gate:
            print_grounding(d.get("grounding"))
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
    if verdicts is not None:
        n = len(verdicts)
        ok = sum(1 for v in verdicts if v["status"] == "supported")
        print(f"\ngrounding re-judged now ({ns.route}): {ok}/{n} supported"
              + (f" ({100.0 * ok / n:.0f}%)" if n else ""))
        for v in verdicts:
            print(f"  {v['memory_id']}: {v['status'].upper()}"
                  + (f" -- unsupported: {'; '.join(v['unsupported_claims'])}" if v["unsupported_claims"] else "")
                  + (f" -- {v['error']}" if v.get("error") else ""))
    return 0


def _alias(ref: str) -> str:
    kind, _, rid = ref.partition(":")
    return f"{REF_KIND_ALIASES.get(kind, kind)}:{rid}"


if __name__ == "__main__":
    raise SystemExit(main())
