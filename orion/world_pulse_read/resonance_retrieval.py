"""Build the production packet using Patch A's centered retrieval and authoritative records.

No similarity reaches the judge. Worldview vectors are derived via the existing semantic
index writer; current record hashes must match before they can be used. Mean and hub
correction are frozen with the brief, so every resumed judgment sees the same evidence.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import httpx

from orion.introspect.semantic_index import SearchConfig, SearchUnavailableError, embed, index_docs, doc_hash
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_resonance import ReadingResonanceRunBriefV1
from orion.world_pulse_read.fetch_text import load_retained_texts
from orion.world_pulse_read.resonance import PoolDoc, ReadInput, ScoredDoc, build_packet
from orion.world_pulse_read.resonance_records import Falkor, worldview_pool, atlas_hydrate
from orion.world_pulse_read.resonance_similarity import center, dot, hub_from_sims, mean_vector, rank_score, CENTERED_OFFER_FLOOR

WORLDVIEW_COLLECTION = "orion_worldview_beliefs"
COLLECTIONS = {"orion_memory_crystallizations": "crystallization", "orion_curiosity": "curiosity_finding",
               "orion_dreams": "dream", "orion_atlas_concepts": "atlas_concept"}


def obj(raw):
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


async def collection_rows(client, base: str, name: str, *, ids=None):
    resp = await client.get(f"{base.rstrip('/')}/api/v1/collections/{name}")
    resp.raise_for_status()  # unavailable is not an empty collection / no resonance
    coll = resp.json()["id"]
    rows, offset = [], 0
    while True:
        query = {"include": ["documents", "metadatas", "embeddings"], "limit": 500, "offset": offset}
        if ids is not None:
            if not ids:
                return []
            query["ids"] = ids
        resp = await client.post(f"{base.rstrip('/')}/api/v1/collections/{coll}/get", json=query)
        resp.raise_for_status()
        body = resp.json()
        columns = [body[k] for k in ("ids", "documents", "metadatas", "embeddings")]
        if len({len(c) for c in columns}) != 1:
            raise SearchUnavailableError("resonance index columns mismatch")
        rows.extend({"id": i, "doc": d or "", "meta": m or {}, "vec": v} for i, d, m, v in zip(*columns))
        if len(columns[0]) < 500:
            return rows
        offset += 500


async def worldview_index(conn, fk, cfg, client, bus, source):
    docs, _, _ = await asyncio.to_thread(worldview_pool, fk)
    docs = sorted(docs, key=lambda d: (d.ref_kind, d.ref_id))
    by_id = {f"{d.ref_kind}:{d.ref_id}": d for d in docs}
    records = [(i, d.excerpt, {"ref_kind": d.ref_kind, "ref_id": d.ref_id}) for i, d in by_id.items()]
    await index_docs(records, cfg, client=client, bus=bus, source=source, doc_prefix="resonance-worldview", batch=50)
    if not records:
        return []
    rows = await collection_rows(client, cfg.chroma_url, cfg.collection, ids=list(by_id))
    vectors = {r["id"]: r for r in rows}
    # Never silently omit a newly changed belief because the async index writer has not caught up.
    if any(i not in vectors or vectors[i]["meta"].get("content_hash") != doc_hash(text, meta)
           for i, text, meta in records):
        raise SearchUnavailableError("worldview index catching up")
    return [(d, vectors[i]["vec"]) for i, d in by_id.items()]


async def prepare_brief(conn, seed_id: str, *, cfg: SearchConfig, graph_url: str, bus, source,
                        now: datetime, lane="agent", max_tokens=900, substrate_store=None):
    row = await conn.fetchrow("SELECT *, root_request_id::text AS root FROM world_pulse_read_seed "
                              "WHERE seed_id=$1 AND stage2_status='done'", seed_id)
    if row is None:
        raise ValueError("read_not_complete")
    from orion.world_pulse_read.queue import _seed_from_row, request_for_seed
    reading_request = request_for_seed(_seed_from_row(row))
    s2, handoff = obj(row["stage2_result_json"]), obj(row["handoff_json"])
    texts = await load_retained_texts(conn, [SourceFetchEvidenceV1.model_validate(e) for e in handoff.get("read_evidence", [])])
    read = ReadInput(seed_id, row["root"], row["title"] or "", row["url"], str(s2.get("summary") or ""),
                     str(handoff.get("what_i_learned") or ""), [(t.sha256, t.text) for t in texts])
    thread_rows = await conn.fetch("SELECT request_id::text AS request_id, url FROM world_pulse_read_seed "
                                  "WHERE seed_id=$1 OR ($2::text IS NOT NULL AND root_request_id::text=$2)", seed_id, row["root"])
    excluded_evidence = {str(v) for r in thread_rows for v in (r["request_id"], r["url"]) if v}
    fk = Falkor(graph_url)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            pool = await worldview_index(conn, fk, replace(cfg, collection=WORLDVIEW_COLLECTION), client, bus, source)
            for collection, kind in COLLECTIONS.items():
                for r in await collection_rows(client, cfg.chroma_url, collection):
                    ref = r["meta"].get("crystallization_id") or r["meta"].get("node_id") or r["id"]
                    store = {"crystallization": "pg:memory_crystallizations", "atlas_concept": "falkor:orion_substrate:Concept"}.get(kind, f"chroma:{collection} (primary)")
                    pool.append((PoolDoc(kind, str(ref), store, r["doc"]), r["vec"]))
            threads = await conn.fetch("SELECT seed_id, root_request_id::text AS root, title, url, "
                "stage2_result_json->'open_threads' AS threads FROM world_pulse_read_seed "
                "WHERE stage2_status='done' AND stage2_completed_at >= $1 AND stage2_completed_at < $2 "
                "AND seed_id <> $3 AND (root_request_id IS NULL OR root_request_id::text IS DISTINCT FROM $4)",
                now - timedelta(days=60), row["stage2_completed_at"], seed_id, row["root"])
            for t in threads:
                for i, value in enumerate(obj(t["threads"]) or []):
                    text = value if isinstance(value, str) else json.dumps(value)
                    if text.strip():
                        pool.append((PoolDoc("open_thread", f"{t['seed_id']}#open_thread{i}",
                            "pg:world_pulse_read_seed.stage2_result_json.open_threads", text),
                            (await embed(client, cfg, text[:2000], doc_prefix="resonance-thread"))[0]))
            for t in await conn.fetch("SELECT arc_id, arc_json FROM temporal_self_arc WHERE kind='concern' AND status='open'"):
                arc = obj(t["arc_json"])
                text = str(arc.get("subject_label") or "").strip() or (json.dumps(arc["body"]) if arc.get("body") else "")
                if text:
                    pool.append((PoolDoc("open_thread", t["arc_id"], "pg:temporal_self_arc", text, floorless=True),
                                 (await embed(client, cfg, text[:2000], doc_prefix="resonance-concern"))[0]))
            query_texts = [read.learned] + [str(c.get("claim") or "") for c in s2.get("candidate_priors", [])]
            query_texts = [t[:2000] for t in query_texts if t.strip()][:6] or [(read.summary or read.title or read.url)[:2000]]
            queries = [(await embed(client, cfg, t, doc_prefix="resonance-query"))[0] for t in query_texts]
            history = await conn.fetch("SELECT handoff_json->>'what_i_learned' AS text FROM world_pulse_read_seed "
                "WHERE stage2_completed_at >= $1 AND stage2_completed_at < $2 AND seed_id <> $3 "
                "ORDER BY stage2_completed_at DESC LIMIT 100", now - timedelta(days=14), row["stage2_completed_at"], seed_id)
            histories = [(await embed(client, cfg, str(r["text"])[:2000], doc_prefix="resonance-history"))[0]
                         for r in history if str(r["text"] or "").strip()]
            scored, mu, hubs = await asyncio.to_thread(score_pool, pool, queries, histories)
            hydrated = await hydrate_ranked(conn, fk, scored, excluded_evidence)
            packet = build_packet(read, hydrated, built_at=now)
            if substrate_store is not None:
                from orion.world_pulse_read.assertions import resolve_reading_subjects
                from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
                handoff_model = WorldPulseReadHandoffV1.model_validate(handoff)
                landed = await asyncio.to_thread(resolve_reading_subjects, substrate_store, handoff_model)
                packet = packet.model_copy(update={"landed_concept_ids": [n.node_id for n in landed][:16]})
            counts = await conn.fetch("SELECT c->>'ref_kind' AS kind,c->>'ref_id' AS id,count(*) AS n "
                "FROM reading_resonance r CROSS JOIN LATERAL jsonb_array_elements(r.request_json->'brief'->'packet'->'candidates') c "
                "WHERE r.created_at >= $1 AND r.seed_id <> $2 GROUP BY 1,2", now - timedelta(days=14), seed_id)
            offered = {(r["kind"], r["id"]): int(r["n"]) for r in counts}
            packet = packet.model_copy(update={"candidates": [c.model_copy(update={"recent_packet_count": offered.get((c.ref_kind, c.ref_id), 0)}) for c in packet.candidates]})
            return ReadingResonanceRunBriefV1(packet=packet, reading_request=reading_request, retained_texts=read.retained_texts,
                llm_route=lane, max_tokens=max_tokens,
                retrieval_basis={"mean": mu, "hub_scores": hubs, "offer_floor": CENTERED_OFFER_FLOOR,
                                 "history_reads": len(histories)})
    finally:
        await asyncio.to_thread(fk.r.close)


def score_pool(pool, queries, histories):
    """CPU work stays off Hub's event loop; normalize each history vector exactly once."""
    mu = mean_vector([v for _, v in pool] + queries + histories)
    q = center(mean_vector([center(v, mu) for v in queries]), [0.0] * len(mu))
    history_vectors = [center(h, mu) for h in histories]
    scored, hubs = [], {}
    for d, v in pool:
        v = center(v, mu)
        hub = hub_from_sims([dot(v, h) for h in history_vectors])
        hubs[f"{d.ref_kind}:{d.ref_id}"] = hub
        scored.append(ScoredDoc(d, dot(q, v), rank_score(q, v, hub)))
    return scored, mu, hubs


async def hydrate_ranked(conn, fk, scored, excluded_evidence):
    """Fill the eligible per-kind budget after dropping stale/self/thread index hits."""
    hydrated = []
    for kind in sorted({s.doc.ref_kind for s in scored}):
        kept = 0
        for s in sorted([x for x in scored if x.doc.ref_kind == kind], key=lambda x: (-x.rank, x.doc.ref_id)):
            if s.centered_cos < CENTERED_OFFER_FLOOR and not s.doc.floorless:
                continue
            d = s.doc
            if kind == "crystallization":
                r = await conn.fetchrow("SELECT kind, summary, subject, status FROM memory_crystallizations "
                                       "WHERE crystallization_id::text=$1", d.ref_id)
                if r is None or r["status"] in ("rejected", "retired"):
                    continue
                d = replace(d, excerpt=f"[{r['kind']}] {r['summary'] or r['subject'] or ''}")
            elif kind == "atlas_concept":
                h = await asyncio.to_thread(atlas_hydrate, fk, d.ref_id)
                if h is None or excluded_evidence.intersection(h[1]):
                    continue
                d = h[0]
            hydrated.append(ScoredDoc(d, s.centered_cos, s.rank))
            kept += 1
            if kept >= 12:
                break
    return hydrated
