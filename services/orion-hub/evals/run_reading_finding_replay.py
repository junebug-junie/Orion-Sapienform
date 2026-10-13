#!/usr/bin/env python3
"""Live replay, read-only: over the last N days of reads, which reading-finding messages would
Orion have sent Juniper, and what would the letter's "What I read" lines say?

    PYTHONPATH=.:services/orion-hub .venv/bin/python services/orion-hub/evals/run_reading_finding_replay.py \
        [--days 7] [--dsn postgresql://...] [--graph-host 127.0.0.1 --graph-port 6380] [--json]

Runs the production code: ``reading_items_between`` (the verified reads), ``findings_for_reads``
(links accepted + projected, beliefs moved by a run citing the read; priors read with
GRAPH.RO_QUERY), ``pending_bundles`` (one message per read, unprojected links held). A read
whose findings landed in different 5-minute ticks is counted as one message per tick, as the
live loop would send them. Gates that depend on the moment (quiet hours, recent chat, a turn
in flight) only delay a message, so they do not change the count. Orion's own words are
generated at send time and are NOT reproduced here; the prompt and the recorded appendix are.

SELECT / GRAPH.RO_QUERY only. Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HUB = Path(__file__).resolve().parents[1]
# Hub's `scripts` package must win over the repo root's `scripts/` (same as the other Hub evals).
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

TICK_SEC = 300.0


def message_groups(bundle, tick_sec: float = TICK_SEC):
    """Findings of one bundle split by the live tick that would first see them."""
    groups: dict[int, list] = {}
    for f in bundle.findings:
        groups.setdefault(int(f.observed_at.timestamp() // tick_sec), []).append(f)
    return [bundle.model_copy(update={"findings": fs}) for _, fs in sorted(groups.items())]


async def replay(conn, *, days: int, prior_lookup, now: datetime) -> dict:
    from orion.world_pulse_read.finding_prompt import article_appendix, build_reading_finding_prompt
    from orion.world_pulse_read.findings import ReadRefV1, findings_for_reads, first_sentences
    from orion.world_pulse_read.introspect import reading_items_between

    from orion.orion_day.budget import NOTHING_CHANGED_LINE

    from scripts.reading_finding_outreach import pending_bundles

    items = await reading_items_between(conn, since=now - timedelta(days=days), until=now, text_cap=None)
    reads = [ReadRefV1(seed_id=i.id, occurred_at=i.occurred_at, url=i.extra.get("url") or "",
                       request_id=i.extra.get("request_id") or "", title=i.extra.get("title") or "",
                       learned=i.text, article_summary=i.extra.get("article_summary") or "") for i in items]
    result = await findings_for_reads(conn, reads, until=now, prior_lookup=prior_lookup)
    bundles = pending_bundles(reads, result.by_seed, told=set(), failures={}, max_attempts=3)
    messages = [g for b in bundles for g in message_groups(b)]
    held = [f for fs in result.by_seed.values() for f in fs if f.kind == "link_made" and not f.projected]
    letter = []
    for r in reads:
        fs = result.by_seed.get(r.seed_id, [])
        letter.append({
            "seed_id": r.seed_id, "title": r.title or r.url, "url": r.url,
            "article_summary": r.article_summary or None,
            "take": first_sentences(r.learned, 2),
            "changed": [f.headline for f in fs] or [NOTHING_CHANGED_LINE],
        })
    return {
        "days": days, "reads": len(reads),
        "reads_with_article_summary": sum(1 for r in reads if r.article_summary),
        "findings": {k: sum(1 for fs in result.by_seed.values() for f in fs if f.kind == k)
                     for k in ("link_made", "belief_moved")},
        "beliefs_unchecked": result.beliefs_unchecked or None,
        "held_unprojected_links": len(held),
        "messages": [{
            "seed_id": m.seed_id, "title": m.title, "url": m.url,
            "at": min(f.observed_at for f in m.findings).isoformat(),
            "findings": [f.headline for f in m.findings],
            "appendix": article_appendix(m),
            "prompt": build_reading_finding_prompt(m),
        } for m in messages],
        "letter": letter,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    parser.add_argument("--graph-host", default="127.0.0.1")
    parser.add_argument("--graph-port", type=int, default=6380)
    parser.add_argument("--graph-name", default="orion_worldview")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(args.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    from orion.curiosity.worldview import WorldviewReader
    from orion.world_pulse_read.findings import prior_lookup_from_reader

    reader = WorldviewReader(host=args.graph_host, port=args.graph_port, graph_name=args.graph_name)
    try:
        out = await replay(conn, days=args.days, prior_lookup=prior_lookup_from_reader(lambda: reader),
                           now=datetime.now(timezone.utc))
    finally:
        await conn.close()
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"reads={out['reads']} (with article summary: {out['reads_with_article_summary']}) "
          f"findings={out['findings']} held_unprojected={out['held_unprojected_links']} "
          f"beliefs_unchecked={out['beliefs_unchecked']}")
    print(f"messages that would have gone out: {len(out['messages'])}")
    for m in out["messages"]:
        print(f"\n- {m['at']}  {m['title']}  ({m['url']})")
        for h in m["findings"]:
            print(f"    {h}")
        print(f"    appendix: {m['appendix']}")
    print("\nletter lines:")
    for r in out["letter"]:
        print(f"- {r['title']}: {'; '.join(r['changed'])}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
