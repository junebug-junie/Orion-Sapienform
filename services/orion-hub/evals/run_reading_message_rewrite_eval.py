#!/usr/bin/env python3
"""Readings, old vs new: the last N reading messages Juniper got, and what the 2026-11-10 prompt asks for.

    PYTHONPATH=.:services/orion-hub .venv/bin/python -P \
        services/orion-hub/evals/run_reading_message_rewrite_eval.py [--limit 20] [--llm-url http://127.0.0.1:8210/v1]

For each of the last ``--limit`` ``sent`` reading rows: rebuild its bundle with the production code
(``reading_items_between``, ``findings_for_reads``, ``read_origins``; worldview priors read with
GRAPH.RO_QUERY), keep only the findings that row told, and build the NEW prompt.

Deterministic checks (no model):

- provenance kept: every origin line of the new prompt (the chain of who queued what) is also in the
  OLD prompt the row was composed from (``client_meta.outreach_provenance.prompt_text``);
- the new prompt's "what stayed with you" lines carry none of the record vocabulary the old prompt
  fed Orion ("Link made:", "concept graph", "verbatim", "->" confidence arrows);
- the new prompt asks for the ``Why Juniper`` line.

With ``--llm-url`` (an OpenAI-compatible endpoint; the llm-gateway ``chat`` lane by default) it also
writes the new message from the new prompt and reports, old vs new, how many messages use record
vocabulary. That is the bare prompt on the chat lane, not the live unified turn (no persona, no
recall), so it shows what the prompt asks for, not exactly what Orion will send.

SELECT / GRAPH.RO_QUERY only. Exit 0 on a completed run; 2 when Postgres is unavailable; 1 when a
deterministic check fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

_HUB = Path(__file__).resolve().parents[1]
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# Measurement only (this eval), never a send filter: phrases that describe Orion's record-keeping
# rather than the idea. The 2026-11-09/10 messages Juniper called graph mechanics use these.
RECORD_VOCAB = re.compile(
    r"\b(graph|edge|edges|node|nodes|cluster|clusters|verbatim|word-for-word|accepted|one link|"
    r"two links|a link|links? (?:landed|came out|made))\b",
    re.IGNORECASE,
)
# What the OLD prompt fed Orion about each change.
OLD_PROMPT_VOCAB = ("Link made:", "concept graph", "verbatim", " -> ")

ROWS_SQL = """
SELECT d.decided_at, d.correlation_id, d.result_json->>'seed_id' AS seed_id,
       d.result_json->'finding_ids' AS finding_ids, c.response,
       c.client_meta->'outreach_provenance'->>'prompt_text' AS old_prompt
FROM endogenous_outreach_decisions d
JOIN chat_history_log c ON c.correlation_id = d.correlation_id
WHERE d.reason = 'sent' AND d.result_json->>'source' = 'reading_finding_outreach'
ORDER BY d.decided_at DESC LIMIT $1
"""


def changed_section(prompt: str) -> str:
    """The prompt's list of changes, for the vocabulary check. Pure."""
    for head in ("What stayed with you (from your own records):", "What it changed (from your own records):"):
        if head in prompt:
            tail = prompt.split(head, 1)[1]
            return tail.split("\n\nWrite Juniper", 1)[0]
    return ""


def check_prompt(new_prompt: str, origin: list[str], old_prompt: Optional[str]) -> dict[str, Any]:
    """Deterministic checks for one rebuilt message. Pure."""
    from orion.hub.why_juniper import WHY_JUNIPER_INSTRUCTION

    chain = [line for line in origin if line[:1].isdigit()]
    section = changed_section(new_prompt)
    return {
        "provenance_kept": None if not old_prompt else all(line in old_prompt for line in chain),
        "record_vocab_in_changes": [v for v in OLD_PROMPT_VOCAB if v in section],
        "asks_why_juniper": WHY_JUNIPER_INSTRUCTION in new_prompt,
    }


def record_vocab_hits(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in RECORD_VOCAB.finditer(str(text or ""))})


async def _generate(llm_url: str, model: str, prompt: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{llm_url.rstrip('/')}/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"] or "").strip()


async def run(conn: Any, *, limit: int, prior_lookup: Any, llm_url: str, model: str) -> list[dict[str, Any]]:
    from orion.hub.why_juniper import split_why_juniper
    from orion.world_pulse_read.finding_prompt import build_reading_finding_prompt, origin_lines
    from orion.world_pulse_read.findings import ReadRefV1, findings_for_reads
    from orion.world_pulse_read.introspect import reading_items_between
    from orion.world_pulse_read.origin import read_origins
    from orion.schemas.reading_finding import ReadingFindingBundleV1

    rows = [dict(r) for r in await conn.fetch(ROWS_SQL, limit)]
    if not rows:
        return []
    since = min(r["decided_at"] for r in rows) - timedelta(days=8)
    until = max(r["decided_at"] for r in rows) + timedelta(minutes=1)
    items = await reading_items_between(conn, since=since, until=until, text_cap=None)
    reads = {i.id: ReadRefV1(seed_id=i.id, occurred_at=i.occurred_at, url=i.extra.get("url") or "",
                             request_id=i.extra.get("request_id") or "", title=i.extra.get("title") or "",
                             learned=i.text, article_summary=i.extra.get("article_summary") or "") for i in items}
    seeds = sorted({r["seed_id"] for r in rows if r["seed_id"] in reads})
    result = await findings_for_reads(conn, [reads[s] for s in seeds], until=until, prior_lookup=prior_lookup)
    origins = await read_origins(conn, seeds)
    out = []
    for r in reversed(rows):
        told = set(json.loads(r["finding_ids"]) if isinstance(r["finding_ids"], str) else (r["finding_ids"] or []))
        read = reads.get(r["seed_id"])
        findings = [f for f in result.by_seed.get(r["seed_id"], []) if f.finding_id in told]
        item: dict[str, Any] = {"at": r["decided_at"].strftime("%m-%d %H:%M"), "seed_id": r["seed_id"],
                                "old_text": r["response"] or "", "old_vocab": record_vocab_hits(r["response"])}
        if read is None or not findings:
            item["skipped"] = "read or its findings no longer rebuild (outside the read window)"
            out.append(item)
            continue
        bundle = ReadingFindingBundleV1(seed_id=read.seed_id, title=read.title, url=read.url, learned=read.learned,
                                        article_summary=read.article_summary, findings=findings,
                                        origin=origins.get(read.seed_id, []))
        prompt = build_reading_finding_prompt(bundle)
        origin = origin_lines(bundle.origin)
        item.update(title=read.title or read.url, origin=origin, checks=check_prompt(prompt, origin, r["old_prompt"]))
        if llm_url:
            try:
                raw = await _generate(llm_url, model, prompt)
                body, why = split_why_juniper(raw)
                item.update(new_text=body, why_juniper=why, new_vocab=record_vocab_hits(body))
            except Exception as exc:  # noqa: BLE001
                item["new_error"] = f"{type(exc).__name__}: {exc}"
        out.append(item)
    return out


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--graph-host", default="127.0.0.1")
    ap.add_argument("--graph-port", type=int, default=6380)
    ap.add_argument("--graph-name", default="orion_worldview")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="chat")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(a.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    from orion.curiosity.worldview import WorldviewReader
    from orion.world_pulse_read.findings import prior_lookup_from_reader

    reader = WorldviewReader(host=a.graph_host, port=a.graph_port, graph_name=a.graph_name)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        items = await run(conn, limit=a.limit, prior_lookup=prior_lookup_from_reader(lambda: reader),
                          llm_url=a.llm_url, model=a.model)
    finally:
        await conn.close()
    rebuilt = [i for i in items if "checks" in i]
    failed = [i for i in rebuilt if i["checks"]["provenance_kept"] is False
              or i["checks"]["record_vocab_in_changes"] or not i["checks"]["asks_why_juniper"]]
    summary = {
        "rows": len(items), "rebuilt": len(rebuilt), "check_failures": len(failed),
        "old_with_record_vocab": sum(1 for i in items if i["old_vocab"]),
        "new_written": sum(1 for i in rebuilt if "new_text" in i),
        "new_with_record_vocab": sum(1 for i in rebuilt if i.get("new_vocab")),
        "new_with_why_juniper": sum(1 for i in rebuilt if i.get("why_juniper")),
    }
    if a.json:
        print(json.dumps({"summary": summary, "items": items}, indent=2, default=str))
        return 1 if failed else 0
    print(json.dumps(summary))
    for i in items:
        print(f"\n=== {i['at']}  {i.get('title', i['seed_id'])}")
        if "skipped" in i:
            print(f"  skipped: {i['skipped']}")
            continue
        for line in i["origin"]:
            print(f"  started: {line}")
        print(f"  checks: {json.dumps(i['checks'])}")
        print(f"  OLD ({', '.join(i['old_vocab']) or 'no record vocab'}): {' '.join(i['old_text'].split())[:500]}")
        if "new_text" in i:
            print(f"  NEW ({', '.join(i['new_vocab']) or 'no record vocab'}): {' '.join(i['new_text'].split())[:500]}")
            print(f"  WHY JUNIPER: {i['why_juniper'] or '(missing: would not be sent)'}")
        elif "new_error" in i:
            print(f"  NEW: error {i['new_error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
