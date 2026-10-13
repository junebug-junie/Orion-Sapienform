#!/usr/bin/env python3
"""Live replay, read-only: which recent reads were dead pages, and which Stage 2
hops would the provenance gate have refused?

    PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_dead_url_replay.py \
        [--days 14] [--dsn postgresql://...]

SELECT only, over world_pulse_read_seed, reading_document_snapshot and
harness_turn_trace (the fetches Hub actually received, with text on reading
turns since 2026-10-11). Runs the production decision functions:

* Stage 1 -- ``source_read_evidence`` / ``no_evidence_reason`` over each `done`
  seed's reported fetches. A row that now has no evidence would have been
  `failed`. Fetches with no retained text cannot be judged by the status line;
  those exactly matching a notice length are listed as UNRETAINED-LIKELY.
* Stage 2 -- every ``need_stage1_urls`` entry against ``hop_allowlist`` built
  from the parent's retained text plus the Stage 2 turn's own fetches.
* Duplicates -- repeated same-URL fetches inside one seed's evidence.

Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter

from orion.schemas.reading import SourceFetchEvidenceV1
from orion.world_pulse_read.fetch_provenance import (
    hop_allowlist,
    hop_refusal_reason,
    parent_read_state,
    reached_pages,
)
from orion.world_pulse_read.fetch_text import load_retained_texts
from orion.world_pulse_read.read_evidence import (
    no_evidence_reason,
    parse_source_fetches,
    source_read_evidence,
)

# Length of WebFetch's notice for the codes seen live (404/403 -> 215, 400 -> 217,
# 401 -> 218, 429 -> 223). Used only to flag rows with no text to judge.
NOTICE_LENGTHS = {215, 217, 218, 223}

SEEDS_SQL = """
SELECT seed_id, url, status, trace_id, stage2_status, stage2_trace_id, request_id,
       handoff_json, stage2_result_json, completed_at
FROM world_pulse_read_seed
WHERE completed_at > now() - make_interval(days => $1) AND handoff_json IS NOT NULL
ORDER BY completed_at
"""
TRACE_SQL = "SELECT correlation_id, run_artifact->'source_fetches' AS f FROM harness_turn_trace WHERE correlation_id = ANY($1::text[])"
CHILD_SQL = """
SELECT url, status, last_error, (SELECT max((e->>'content_chars')::int)
  FROM jsonb_array_elements(coalesce(handoff_json->'read_evidence','[]'::jsonb)) e) AS chars
FROM world_pulse_read_seed WHERE request_json->>'parent_request_id' = $1
"""


def _obj(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


async def _texts_by_sha(conn, evidence: list[SourceFetchEvidenceV1]) -> dict[str, str]:
    rows = await load_retained_texts(conn, evidence, drop_http_errors=False)
    return {r.sha256: r.text for r in rows}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    args = parser.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(args.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        seeds = await conn.fetch(SEEDS_SQL, args.days)
        trace_ids = [s["trace_id"] for s in seeds if s["trace_id"]]
        trace_ids += [s["stage2_trace_id"] for s in seeds if s["stage2_trace_id"]]
        traces = {r["correlation_id"]: parse_source_fetches(_obj(r["f"]) if r["f"] else None)
                  for r in await conn.fetch(TRACE_SQL, trace_ids)}

        done = [s for s in seeds if s["status"] == "done"]
        print(f"window={args.days}d seeds_with_handoff={len(seeds)} done={len(done)}")

        # --- Stage 1 ---
        flipped, likely, borderline, dup_seeds = [], [], [], []
        for s in done:
            handoff = _obj(s["handoff_json"])
            evidence = parse_source_fetches(handoff.get("read_evidence")) or []
            texts = await _texts_by_sha(conn, evidence)
            reported = traces.get(s["trace_id"])
            if reported is None:
                # No trace row: judge the stored evidence with its retained text.
                reported = [e.model_copy(update={"content_text": texts.get(e.content_sha256 or "")})
                            for e in evidence]
            kept = source_read_evidence(s["url"], reported)
            urls = Counter(e.url for e in evidence)
            if any(n > 1 for n in urls.values()):
                dup_seeds.append((s["seed_id"], s["url"], [e.content_chars for e in evidence]))
            if not kept:
                flipped.append((s["seed_id"], s["url"], no_evidence_reason(s["url"], reported),
                                [e.content_chars for e in evidence]))
                continue
            judged = [e for e in kept if e.content_text is None]
            if judged and all(e.content_chars in NOTICE_LENGTHS for e in kept):
                likely.append((s["seed_id"], s["url"], [e.content_chars for e in kept]))
            elif max(e.content_chars for e in kept) < 400:
                borderline.append((s["seed_id"], s["url"], [e.content_chars for e in kept],
                                   (kept[0].content_text or "")[:90].replace("\n", " ")))

        print(f"\n[stage1] done -> failed under the status rule: {len(flipped)}")
        for row in flipped:
            print("  FAIL", *row)
        print(f"[stage1] done, kept, no text to judge but every fetch is notice-length "
              f"(UNRETAINED-LIKELY dead): {len(likely)}")
        for row in likely:
            print("  LIKELY", *row)
        print(f"[stage1] done, kept, largest fetch < 400 chars (borderline, still done): {len(borderline)}")
        for row in borderline:
            print("  KEEP", *row)

        # --- Stage 2 hops ---
        hop_total = refused = allowed = parents = 0
        reasons: Counter = Counter()
        refused_good: list[tuple[str, str, int]] = []
        for s in seeds:
            result = _obj(s["stage2_result_json"])
            urls = result.get("need_stage1_urls") or []
            if not urls:
                continue
            parents += 1
            handoff = _obj(s["handoff_json"])
            evidence = parse_source_fetches(handoff.get("read_evidence")) or []
            rows = await load_retained_texts(conn, evidence, drop_http_errors=False)
            # The same functions Stage 2 calls (world_pulse_read_stage2._parent_read/_hop_allowlist).
            parent_pages, dead = parent_read_state(evidence, rows)
            reached, stage2_pages = reached_pages(traces.get(s["stage2_trace_id"]) or [])
            page_texts = [*parent_pages, *stage2_pages]
            allow = hop_allowlist(page_texts, reached)
            texts = rows
            # Nothing retained to look for links in (reads before 2026-10-11): the
            # live gate refuses these, but the replay cannot say they were invented.
            judgeable = bool(page_texts) or dead
            children = {r["url"]: r for r in await conn.fetch(CHILD_SQL, str(s["request_id"]))}
            print(f"\n[stage2] parent {s['seed_id']} {s['url']} dead={dead} "
                  f"parent_texts={len(texts)} stage2_fetches={len(traces.get(s['stage2_trace_id']) or [])}")
            for url in urls:
                hop_total += 1
                reason = "parent_no_valid_read" if dead else hop_refusal_reason(url, allow)
                child = children.get(url)
                outcome = (f"child={child['status']} chars={child['chars']} err={child['last_error']}"
                           if child else "child=none")
                if reason:
                    refused += 1
                    label = reason if judgeable else f"{reason}:no_retained_text"
                    reasons[label] += 1
                    if child and child["status"] == "done" and (child["chars"] or 0) >= 400:
                        refused_good.append((label, url, child["chars"]))
                    print(f"  REFUSE {label} {url} [{outcome}]")
                else:
                    allowed += 1
                    print(f"  ALLOW {url} [{outcome}]")
        print(f"\n[stage2] parents_with_hops={parents} hops={hop_total} allowed={allowed} "
              f"refused={refused} reasons={dict(reasons)}")
        print(f"[stage2] refused hops whose child read was a real page (>=400 chars): {len(refused_good)}")
        for row in refused_good:
            print("  LOST", *row)

        print(f"\n[dup] seeds whose evidence repeats a URL: {len(dup_seeds)}")
        for row in dup_seeds:
            print("  DUP", *row)
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
