#!/usr/bin/env python3
"""Live replay, read-only: how many reading choices would recent Stage 2 passes have produced?

    PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_choice_replay.py \
        [--days 7] [--dsn postgresql://...]

SELECT only, over world_pulse_read_seed, reading_document_snapshot and harness_turn_trace. For
every Stage 2 pass that finished in the window it runs the production functions
(orion/world_pulse_read/next_read_choice.py): the link list the prompt would show
(``next_read_options`` over the parent's retained text, skipped when the read's chain already
holds ``--max-round-trips`` hops as Stage 2 does, minus pages Stage 1 fetched and links read
before the pass finished), then ``reading_choices`` against the model's stored ``need_stage1_urls``.

This is an estimate, not a record: these passes ran without the list in their prompt, so Orion
picked from page text it may only partly have seen. Prints per-day counts of passes, choice
decisions (picked / declined_all), rows, and why a pass produced none; then how many written links
were left out as not something to read (``classified_links`` reasons, #3443) with ``--examples``
of each, and sample option texts as the prompt would show them.

Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict

from orion.world_pulse_read.fetch_provenance import hop_allowlist, hop_url_key, parent_read_state, reached_pages
from orion.world_pulse_read.fetch_text import load_retained_texts
from orion.world_pulse_read.next_read_choice import classified_links, next_read_options, reading_choices
from orion.world_pulse_read.read_evidence import parse_source_fetches

SEEDS_SQL = """
SELECT s.seed_id, s.url, s.stage2_trace_id, s.handoff_json, s.stage2_result_json, s.stage2_completed_at,
       -- Hops already in this read's chain when Stage 1 handed off (about when the Stage 2 prompt
       -- is built): the count enqueue_reading's round_trip_cap reads.
       (SELECT count(*) FROM world_pulse_read_seed r
        WHERE r.root_request_id = coalesce(s.root_request_id, s.request_id)
          AND r.request_id <> coalesce(s.root_request_id, s.request_id)
          AND r.duplicate_of IS NULL AND r.created_at < s.handoff_at) AS chain_hops
FROM world_pulse_read_seed s
WHERE s.stage2_status = 'done' AND s.stage2_completed_at > now() - make_interval(days => $1)
ORDER BY s.stage2_completed_at
"""
TRACE_SQL = ("SELECT correlation_id, run_artifact->'source_fetches' AS f FROM harness_turn_trace "
             "WHERE correlation_id = ANY($1::text[])")
READ_BEFORE_SQL = """
SELECT DISTINCT url FROM world_pulse_read_seed
WHERE url = ANY($1::text[]) AND duplicate_of IS NULL AND status = 'done'
  AND (handoff_json->'read_evidence'->0) IS NOT NULL AND completed_at < $2
"""


def _obj(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def summarize(rows: list[tuple[str, str, int]]) -> dict[str, dict[str, int]]:
    """(day, reason, n_rows) per pass -> per-day counts. Pure; the eval test pins it."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for day, reason, n in rows:
        d = out[day]
        d["passes"] += 1
        d[reason] += 1
        d["rows"] += n
        if reason in ("picked", "declined_all"):
            d["decisions"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-round-trips", type=int, default=5,
                        help="Hub HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS (chain cap)")
    parser.add_argument("--examples", type=int, default=5, help="examples per drop reason / of shown options")
    parser.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    args = parser.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(args.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    per_pass: list[tuple[str, str, int]] = []
    ranks: Counter = Counter()
    unlisted: Counter = Counter()
    shown_sizes: list[int] = []
    dropped: Counter = Counter()
    drop_examples: dict[str, list[str]] = defaultdict(list)
    shown_examples: list[str] = []
    try:
        seeds = await conn.fetch(SEEDS_SQL, args.days)
        traces = {r["correlation_id"]: parse_source_fetches(_obj(r["f"]) if r["f"] else None)
                  for r in await conn.fetch(TRACE_SQL, [s["stage2_trace_id"] for s in seeds if s["stage2_trace_id"]])}
        for s in seeds:
            day = s["stage2_completed_at"].date().isoformat()
            if int(s["chain_hops"] or 0) >= args.max_round_trips:
                per_pass.append((day, "chain_full", 0))
                continue
            handoff = _obj(s["handoff_json"])
            evidence = parse_source_fetches(handoff.get("read_evidence")) or []
            rows = await load_retained_texts(conn, evidence, drop_http_errors=False)
            parent_pages, dead = parent_read_state(evidence, rows)
            if dead or not parent_pages:
                per_pass.append((day, "no_parent_text", 0))
                continue
            fetched = [s["url"], *(e.url for e in evidence)]
            classified = classified_links(parent_pages, exclude_urls=fetched)
            cands = [c for c, why in classified if why is None]
            for c, why in classified:
                if why is None:
                    continue
                dropped[why] += 1
                if len(drop_examples[why]) < args.examples:
                    drop_examples[why].append(f"{c.text} -> {c.url}")
            read = await conn.fetch(READ_BEFORE_SQL, [c.url for c in cands], s["stage2_completed_at"])
            done = {k for k in (hop_url_key(str(r["url"])) for r in read) if k}
            shown = next_read_options(parent_pages, exclude_urls=fetched, already_read_keys=done)
            shown_sizes.append(len(shown))
            if shown and len(shown_examples) < args.examples:
                shown_examples.append(f"{s['url']}: " + " | ".join(o.text for o in shown[:4]))
            reached, stage2_pages = reached_pages(traces.get(s["stage2_trace_id"]) or [])
            allow = hop_allowlist([*parent_pages, *stage2_pages], reached)
            picked = _obj(s["stage2_result_json"]).get("need_stage1_urls") or []
            choices, reason = reading_choices(
                seed_id=s["seed_id"], shown=shown, picked_urls=picked, allowlist=allow,
                occurred_at=s["stage2_completed_at"])
            parent_keys = {hop_url_key(c.url) for c in cands}
            for c in choices:
                ranks["declined" if c.declined_all else (c.chosen_market_rank or "unlisted")] += 1
            for url in picked:
                k = hop_url_key(url)
                if k and k in allow and k not in {hop_url_key(o.url) for o in shown}:
                    unlisted["in_parent_not_shown" if k in parent_keys else "found_this_turn"] += 1
            per_pass.append((day, reason, len(choices)))
    finally:
        await conn.close()
    table = summarize(per_pass)
    print(f"window={args.days}d stage2_passes={len(per_pass)}")
    for day, counts in table.items():
        print(f"  {day} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    days = max(1, len(table))
    decisions = sum(c.get("decisions", 0) for c in table.values())
    rows = sum(c.get("rows", 0) for c in table.values())
    print(f"per_day: decisions={decisions / days:.1f} rows={rows / days:.1f}")
    if shown_sizes:
        print(f"links_shown: mean={sum(shown_sizes) / len(shown_sizes):.1f} zero={shown_sizes.count(0)}")
    print(f"chosen market ranks: {dict(ranks)}")
    print(f"unlisted picks by where Orion met them: {dict(unlisted)}")
    print(f"links left out as not something to read: {dict(dropped)}")
    for why, rows_ in sorted(drop_examples.items()):
        for row in rows_:
            print(f"  [{why}] {row}")
    print("options as shown (first 4 per pass):")
    for row in shown_examples:
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
