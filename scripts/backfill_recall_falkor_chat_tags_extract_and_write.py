#!/usr/bin/env python3
"""Phase 3 of the recall RDF->Falkor cutover: extraction+write stage.

Reads /app/_backfill_snapshot.json (copied in from
scripts/backfill_recall_falkor_chat_tags_snapshot.py's output), runs the
SAME extraction pipeline the live orion-meta-tags writer uses (spaCy NER +
keyword sentiment heuristic), and writes via the SAME
write_chat_turn_tags_to_falkor Cypher writer -- not a reimplementation, and
not a raw copy of any pre-existing Fuseki data (per the Phase 0 spec's own
instruction; see the design note this script's PR report links to for why:
Fuseki has almost no chat_tagging data for this population either, since
the same observer-gate bug that blocked the Falkor writer also blocked the
Fuseki extraction for ~6 months).

Must run INSIDE the orion-athena-meta-tags container (`docker cp` this file
to /app/ first, then `docker exec ... python /app/<this file>`) -- that's
the only place SPA_MODEL (en_core_web_trf) is confirmed loadable; the
shared repo .venv lacks the downloaded model. Running from /app makes
`app.*`/`orion.*` importable exactly like main.py's own imports (sys.path[0]
becomes /app, the script's own directory, when invoked as `python
/app/script.py`).

text hydration matches app/models.py::EventIn.prepare_and_hydrate_text
exactly: f"User: {prompt}\\nOrion: {response}" for chat_history_log rows
(social_room_turns rows already carry this exact format in their own
`text` column, built the same way by orion-sql-writer).

turn_id: chat_history_log rows use the `id` column (== correlation_id
whenever correlation_id was present at write time, confirmed live;
`id` is always present and stable, unlike the live path's fallback of a
fresh random uuid4 for the ~19% of rows with a NULL correlation_id -- there
is no way to reconstruct what the live path *would* have generated for
those, so `id` is the only stable choice). social_room_turns rows use the
`turn_id` primary key column directly (unambiguous).

ts: the row's real `created_at`, not "now" -- backfilling with a fresh
timestamp would make historical turns look like they just happened, wrong
for the recency-based ranking fusion.py performs.

Idempotent at the Cypher level (MERGE on turn_id), but this script also
pre-checks Falkor for already-present turn_ids so the final report's
"written" vs "already present" counts are honest, not just "ran, no
crash" -- distinct from correctness (MERGE would already the same
either way).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from app.falkor_recall_writer import write_chat_turn_tags_to_falkor  # noqa: E402
from app.settings import settings  # noqa: E402
from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402

SNAPSHOT_PATH = Path("/app/_backfill_snapshot.json")
PROGRESS_EVERY = 50

POSITIVE_KEYWORDS = {"triumphant", "relief", "capable", "good", "success"}
NEGATIVE_KEYWORDS = {"anxious", "fear", "fail", "bad", "panic"}


def _sentiment_tag(text: str) -> str:
    tokens = set((text or "").lower().split())
    if tokens & POSITIVE_KEYWORDS:
        return "sentiment:positive"
    if tokens & NEGATIVE_KEYWORDS:
        return "sentiment:negative"
    return "sentiment:neutral"


def _existing_turn_ids(client: RedisGraphQueryClient) -> set[str]:
    rows = client.graph_query("MATCH (t:ChatTurn) RETURN t.turn_id AS turn_id")
    return {str(r.get("turn_id")) for r in rows if r.get("turn_id")}


def main() -> int:
    import app.main as app_main  # noqa: E402  (loads the real spaCy pipeline)

    nlp = app_main.nlp

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    chat_rows = snapshot["chat_history_log"]
    social_rows = snapshot["social_room_turns"]

    jobs: list[dict] = []
    for row in chat_rows:
        text = f"User: {row['prompt']}\nOrion: {row['response']}"
        jobs.append(
            {
                "turn_id": str(row["id"]),
                "source_kind": "chat.history",
                "session_id": row.get("session_id"),
                "ts": row["created_at"],
                "correlation_id": row.get("correlation_id"),
                "text": text,
            }
        )
    for row in social_rows:
        jobs.append(
            {
                "turn_id": str(row["turn_id"]),
                "source_kind": "social.turn.stored.v1",
                "session_id": row.get("session_id"),
                "ts": row["created_at"],
                "correlation_id": row.get("correlation_id"),
                "text": row.get("text") or f"User: {row['prompt']}\nOrion: {row['response']}",
            }
        )

    client = RedisGraphQueryClient(uri=settings.FALKORDB_URI, graph_name=settings.FALKORDB_RECALL_GRAPH)
    skip_ids = _existing_turn_ids(client)

    total = len(jobs)
    written = 0
    skipped_already_present = 0
    errors: list[dict] = []
    t0 = time.time()

    for idx, job in enumerate(jobs, start=1):
        turn_id = job["turn_id"]
        if turn_id in skip_ids:
            skipped_already_present += 1
        else:
            try:
                doc = nlp(job["text"] or "")
                entities = [ent.text for ent in doc.ents]
                sentiment_tag = _sentiment_tag(job["text"])
                write_chat_turn_tags_to_falkor(
                    client,
                    turn_id=turn_id,
                    source_kind=job["source_kind"],
                    session_id=job["session_id"],
                    ts=job["ts"],
                    correlation_id=job["correlation_id"],
                    tags=[sentiment_tag],
                    entities=entities,
                )
                written += 1
                skip_ids.add(turn_id)
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                errors.append({"turn_id": turn_id, "source_kind": job["source_kind"], "error": str(exc)})

        if idx % PROGRESS_EVERY == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = total - idx
            eta_sec = remaining / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "event": "backfill_recall_falkor_chat_tags_progress",
                        "percent": round(100.0 * idx / total, 1),
                        "rows_processed": idx,
                        "rows_total": total,
                        "written": written,
                        "skipped_already_present": skipped_already_present,
                        "error_count": len(errors),
                        "rate_rows_per_sec": round(rate, 2),
                        "eta_sec": round(eta_sec, 1),
                    }
                ),
                flush=True,
            )

    summary = {
        "event": "backfill_recall_falkor_chat_tags_complete",
        "rows_total": total,
        "written": written,
        "skipped_already_present": skipped_already_present,
        "error_count": len(errors),
        "errors": errors[:50],
        "duration_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
