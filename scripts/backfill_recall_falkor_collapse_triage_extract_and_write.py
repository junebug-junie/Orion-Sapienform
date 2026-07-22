#!/usr/bin/env python3
"""Extraction+write stage for the collapse-triage Falkor backfill
(feat/falkor-collapse-triage-consumption, PR #1271).

Reads /app/_backfill_snapshot.json (copied in from
scripts/backfill_recall_falkor_collapse_triage_snapshot.py's output), runs
the SAME extraction the live orion-meta-tags generic `handle_triage_event`
branch uses -- unfiltered `doc.ents` (NOT `app.main._named_entities`'s
type-label filter, which only wires into the chat.history/social.turn.
stored.v1 branch, a deliberate, pre-existing asymmetry this backfill must
match, not fix) + the same keyword sentiment heuristic -- and writes via
the SAME `write_collapse_triage_tags_to_falkor` Cypher writer, not a
reimplementation. Calls that function directly (bypassing
RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED's flag check, same as the
chat-turn backfill's own precedent) so this can run safely before the flag
is ever flipped live.

Must run INSIDE the orion-athena-meta-tags container (`docker cp` this file
to /app/ first, then `docker exec ... python /app/<this file>`) -- that's
the only place SPA_MODEL (en_core_web_trf) is confirmed loadable.

collapse_id: the row's real `id` column (e.g. "collapse_...") -- always
present and stable, matches what the live path would use as
`in_payload.collapse_id or in_payload.id`.

ts: the row's real `timestamp`, not "now" -- backfilling with a fresh
timestamp would misrepresent when the reflection actually happened.

Idempotent at the Cypher level (MERGE on collapse_id), but this script also
pre-checks Falkor for already-present collapse_ids so the final report's
"written" vs "already present" counts are honest, not just "ran, no crash."
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from app.falkor_recall_writer import write_collapse_triage_tags_to_falkor  # noqa: E402
from app.settings import settings  # noqa: E402
from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402

SNAPSHOT_PATH = Path("/app/_backfill_snapshot.json")
PROGRESS_EVERY = 10

POSITIVE_KEYWORDS = {"triumphant", "relief", "capable", "good", "success"}
NEGATIVE_KEYWORDS = {"anxious", "fear", "fail", "bad", "panic"}


def _sentiment_tag(text: str) -> str:
    tokens = set((text or "").lower().split())
    if tokens & POSITIVE_KEYWORDS:
        return "sentiment:positive"
    if tokens & NEGATIVE_KEYWORDS:
        return "sentiment:negative"
    return "sentiment:neutral"


def _existing_collapse_ids(client: RedisGraphQueryClient) -> set[str]:
    rows = client.graph_query("MATCH (c:CollapseEvent) RETURN c.collapse_id AS collapse_id")
    return {str(r.get("collapse_id")) for r in rows if r.get("collapse_id")}


def main() -> int:
    import app.main as app_main  # noqa: E402  (loads the real spaCy pipeline)

    nlp = app_main.nlp

    rows = json.loads(SNAPSHOT_PATH.read_text())

    client = RedisGraphQueryClient(uri=settings.FALKORDB_URI, graph_name=settings.FALKORDB_RECALL_GRAPH)
    skip_ids = _existing_collapse_ids(client)

    total = len(rows)
    written = 0
    skipped_already_present = 0
    skipped_no_text = 0
    errors: list[dict] = []
    t0 = time.time()

    for idx, row in enumerate(rows, start=1):
        collapse_id = str(row["id"])
        text = row.get("text") or ""

        if collapse_id in skip_ids:
            skipped_already_present += 1
        elif not text.strip():
            skipped_no_text += 1
        else:
            try:
                # Deliberately unfiltered -- matches handle_triage_event's
                # generic branch exactly (tags = entities = [ent.text for
                # ent in doc.ents], sentiment appended to tags separately).
                doc = nlp(text)
                entities = [ent.text for ent in doc.ents]
                sentiment_tag = _sentiment_tag(text)
                write_collapse_triage_tags_to_falkor(
                    client,
                    collapse_id=collapse_id,
                    correlation_id=row.get("correlation_id"),
                    ts=row["timestamp"],
                    tags=[sentiment_tag],
                    entities=entities,
                )
                written += 1
                skip_ids.add(collapse_id)
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                errors.append({"collapse_id": collapse_id, "error": str(exc)})

        if idx % PROGRESS_EVERY == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = total - idx
            eta_sec = remaining / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "event": "backfill_recall_falkor_collapse_triage_progress",
                        "percent": round(100.0 * idx / total, 1),
                        "rows_processed": idx,
                        "rows_total": total,
                        "written": written,
                        "skipped_already_present": skipped_already_present,
                        "skipped_no_text": skipped_no_text,
                        "error_count": len(errors),
                        "rate_rows_per_sec": round(rate, 2),
                        "eta_sec": round(eta_sec, 1),
                    }
                ),
                flush=True,
            )

    summary = {
        "event": "backfill_recall_falkor_collapse_triage_complete",
        "rows_total": total,
        "written": written,
        "skipped_already_present": skipped_already_present,
        "skipped_no_text": skipped_no_text,
        "error_count": len(errors),
        "errors": errors[:50],
        "duration_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
