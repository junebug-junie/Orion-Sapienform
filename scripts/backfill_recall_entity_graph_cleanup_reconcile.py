#!/usr/bin/env python3
"""Phase 0 of the entity-graph-reasoning work: reconcile every existing
:ChatTurn -[:MENTIONS_ENTITY]-> :Entity edge in orion_recall against the
now-fixed extraction rules (app.main._named_entities' spaCy-label filter +
app.falkor_recall_writer's diacritic-aware normalization).

Re-classifying a bare Entity node name in isolation does NOT work -- live-
verified: "P4" and "8GB" get zero entities from spaCy without sentence
context, and "day"/"a moment" silently lose their noise labels the same way.
So this replays each turn's REAL original text (from the same Postgres
snapshot used by the Phase 3 backfill) through the exact fixed pipeline,
diffs against what's currently attached to that turn in Falkor, deletes
stale edges, and adds any newly-correct ones. A final pass merges any
remaining diacritic-duplicate Entity nodes and deletes now-orphaned nodes.

Must run INSIDE orion-athena-meta-tags (same reason as the Phase 3 backfill:
SPA_MODEL is only loaded there). `docker cp` this file to /app/ first.

AGENTS.md section 14 backfill protocol: snapshot already taken (this job
reuses the Phase 3 snapshot format), progress log, report, before/after --
see the caller script for how those get assembled.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from app.falkor_recall_writer import filter_noise  # noqa: E402
from app.settings import settings  # noqa: E402
from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402

SNAPSHOT_PATH = Path("/app/_backfill_snapshot.json")
PROGRESS_EVERY = 50


def _existing_entity_names_for_turn(client: RedisGraphQueryClient, turn_id: str) -> set[str]:
    rows = client.graph_query(
        "MATCH (t:ChatTurn {turn_id: $turn_id})-[:MENTIONS_ENTITY]->(e:Entity) RETURN e.name AS name",
        {"turn_id": turn_id},
    )
    return {str(r.get("name")) for r in rows if r.get("name")}


def _delete_stale_edges(client: RedisGraphQueryClient, turn_id: str, stale: list[str]) -> None:
    client.graph_query(
        "MATCH (t:ChatTurn {turn_id: $turn_id})-[r:MENTIONS_ENTITY]->(e:Entity) "
        "WHERE e.name IN $names DELETE r",
        {"turn_id": turn_id, "names": stale},
    )


def _add_missing_edges(client: RedisGraphQueryClient, turn_id: str, ts: str, missing: list[str]) -> None:
    client.graph_query(
        "MATCH (t:ChatTurn {turn_id: $turn_id}) "
        "UNWIND $names AS name "
        "MERGE (g:Entity {name: name}) "
        "MERGE (t)-[r:MENTIONS_ENTITY]->(g) "
        "SET r.ts = $ts",
        {"turn_id": turn_id, "names": missing, "ts": ts},
    )


def _delete_orphan_entities(client: RedisGraphQueryClient) -> int:
    rows = client.graph_query(
        "MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS_ENTITY]-() "
        "WITH e, e.name AS name DELETE e RETURN count(name) AS deleted"
    )
    return int(rows[0].get("deleted") or 0) if rows else 0


def _merge_diacritic_duplicates(client: RedisGraphQueryClient) -> dict:
    rows = client.graph_query("MATCH (e:Entity) RETURN e.name AS name")
    names = [str(r.get("name")) for r in rows if r.get("name")]
    groups: dict[str, list[str]] = {}
    for name in names:
        kept, _ = filter_noise([name])
        key = kept[0] if kept else name
        groups.setdefault(key, []).append(name)

    merges = []
    for canonical_key, variants in groups.items():
        distinct = sorted(set(variants))
        if len(distinct) <= 1:
            continue
        # Prefer the variant that's already the canonical normalized form if
        # present (avoids a pointless rename when e.g. "orion" and "Orion"
        # both exist -- "orion" already matches canonical_key exactly).
        canonical = canonical_key if canonical_key in distinct else sorted(distinct)[0]
        losers = [d for d in distinct if d != canonical]
        for loser in losers:
            client.graph_query(
                "MATCH (loser:Entity {name: $loser})<-[r:MENTIONS_ENTITY]-(t:ChatTurn) "
                "MERGE (canon:Entity {name: $canonical}) "
                "MERGE (t)-[nr:MENTIONS_ENTITY]->(canon) "
                "SET nr.ts = coalesce(nr.ts, r.ts) "
                "DELETE r",
                {"loser": loser, "canonical": canonical},
            )
            client.graph_query(
                "MATCH (loser:Entity {name: $loser}) WHERE NOT (loser)<-[:MENTIONS_ENTITY]-() DELETE loser",
                {"loser": loser},
            )
        merges.append({"canonical": canonical, "merged_from": losers})
    return {"groups_merged": len(merges), "merges": merges}


def main() -> int:
    import app.main as app_main  # noqa: E402  (loads the real spaCy pipeline)

    nlp = app_main.nlp

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    chat_rows = snapshot["chat_history_log"]
    social_rows = snapshot["social_room_turns"]

    jobs: list[dict] = []
    for row in chat_rows:
        text = f"User: {row['prompt']}\nOrion: {row['response']}"
        jobs.append({"turn_id": str(row["id"]), "ts": row["created_at"], "text": text})
    for row in social_rows:
        jobs.append(
            {
                "turn_id": str(row["turn_id"]),
                "ts": row["created_at"],
                "text": row.get("text") or f"User: {row['prompt']}\nOrion: {row['response']}",
            }
        )

    client = RedisGraphQueryClient(uri=settings.FALKORDB_URI, graph_name=settings.FALKORDB_RECALL_GRAPH)

    total = len(jobs)
    reconciled = 0
    turns_changed = 0
    edges_deleted = 0
    edges_added = 0
    errors: list[dict] = []
    t0 = time.time()

    for idx, job in enumerate(jobs, start=1):
        turn_id = job["turn_id"]
        try:
            existing = _existing_entity_names_for_turn(client, turn_id)
            if not existing:
                # Turn not in Falkor yet, or genuinely has zero entities --
                # either way, nothing to reconcile for this turn.
                reconciled += 1
            else:
                doc = nlp(job["text"] or "")
                raw_entities = [ent.text for ent in doc.ents if ent.label_ in app_main._KEEP_ENTITY_LABELS]
                correct, _rejected = filter_noise(raw_entities)
                correct_set = set(correct)

                stale = sorted(existing - correct_set)
                missing = sorted(correct_set - existing)

                if stale:
                    _delete_stale_edges(client, turn_id, stale)
                    edges_deleted += len(stale)
                if missing:
                    _add_missing_edges(client, turn_id, job["ts"], missing)
                    edges_added += len(missing)
                if stale or missing:
                    turns_changed += 1
                reconciled += 1
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
            errors.append({"turn_id": turn_id, "error": str(exc)})

        if idx % PROGRESS_EVERY == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = total - idx
            eta_sec = remaining / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "event": "reconcile_progress",
                        "percent": round(100.0 * idx / total, 1),
                        "rows_processed": idx,
                        "rows_total": total,
                        "turns_changed": turns_changed,
                        "edges_deleted": edges_deleted,
                        "edges_added": edges_added,
                        "error_count": len(errors),
                        "rate_rows_per_sec": round(rate, 2),
                        "eta_sec": round(eta_sec, 1),
                    }
                ),
                flush=True,
            )

    orphans_deleted = _delete_orphan_entities(client)
    dedup_summary = _merge_diacritic_duplicates(client)
    orphans_deleted_after_dedup = _delete_orphan_entities(client)

    summary = {
        "event": "reconcile_complete",
        "rows_total": total,
        "reconciled": reconciled,
        "turns_changed": turns_changed,
        "edges_deleted": edges_deleted,
        "edges_added": edges_added,
        "error_count": len(errors),
        "errors": errors[:50],
        "orphans_deleted_first_pass": orphans_deleted,
        "diacritic_dedup": dedup_summary,
        "orphans_deleted_after_dedup": orphans_deleted_after_dedup,
        "duration_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
