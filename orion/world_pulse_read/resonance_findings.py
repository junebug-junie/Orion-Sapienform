"""Read checked receipts and resolve cited records again before showing any connection."""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
import httpx
from orion.schemas.reading_resonance import ReadingResonanceResultV1
from orion.schemas.reading_finding import ReadingFindingV1
from orion.world_pulse_read.resonance_records import Falkor, worldview_pool, atlas_hydrate
from orion.world_pulse_read.resonance_retrieval import collection_rows, obj

logger = logging.getLogger(__name__)


async def resolve_citations(conn, result, *, graph_url=None, chroma_url=None):
    """Return current record excerpts keyed by (kind,id); vanished records are absent."""
    graph_url = graph_url if graph_url is not None else os.getenv("FALKORDB_URI", "")
    chroma_url = chroma_url if chroma_url is not None else os.getenv("HUB_READING_SEARCH_CHROMA_URL", "")
    citations = result.judgment.citations if result.judgment else []
    out, fk, worldview = {}, None, None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for c in citations:
                text = None
                if c.ref_kind in {"self_prior", "lived_question", "lived_answer", "atlas_concept"}:
                    if not graph_url:
                        raise RuntimeError("resonance_graph_unconfigured")
                    fk = fk or Falkor(graph_url)
                    if c.ref_kind == "atlas_concept":
                        found = await asyncio.to_thread(atlas_hydrate, fk, c.ref_id)
                        text = found[0].excerpt if found else None
                    else:
                        if worldview is None:
                            docs, _, _ = await asyncio.to_thread(worldview_pool, fk)
                            worldview = {(d.ref_kind, d.ref_id): d.excerpt for d in docs}
                        text = worldview.get((c.ref_kind, c.ref_id))
                elif c.ref_kind == "crystallization":
                    row = await conn.fetchrow("SELECT kind,summary,subject,status FROM memory_crystallizations "
                                              "WHERE crystallization_id::text=$1", c.ref_id)
                    if row and row["status"] not in {"rejected", "retired"}:
                        text = f"[{row['kind']}] {row['summary'] or row['subject'] or ''}"
                elif c.ref_kind in {"dream", "curiosity_finding"}:
                    collection = "orion_dreams" if c.ref_kind == "dream" else "orion_curiosity"
                    rows = await collection_rows(client, chroma_url, collection, ids=[c.ref_id])
                    text = next((r["doc"] for r in rows if r["id"] == c.ref_id), None)
                elif "#open_thread" in c.ref_id:
                    seed, index = c.ref_id.rsplit("#open_thread", 1)
                    raw = await conn.fetchval("SELECT stage2_result_json->'open_threads' FROM world_pulse_read_seed "
                                             "WHERE seed_id=$1 AND stage2_status='done'", seed)
                    threads = obj(raw)
                    if index.isdigit() and isinstance(threads, list) and int(index) < len(threads):
                        t = threads[int(index)]
                        text = t if isinstance(t, str) else json.dumps(t)
                else:
                    raw = await conn.fetchval("SELECT arc_json FROM temporal_self_arc WHERE arc_id=$1 "
                                             "AND kind='concern' AND status='open'", c.ref_id)
                    arc = obj(raw)
                    text = arc.get("subject_label") or (json.dumps(arc["body"]) if arc.get("body") else None)
                if text and str(text).strip():
                    out[c.ref_kind, c.ref_id] = str(text)[:600]
    finally:
        if fk:
            await asyncio.to_thread(fk.r.close)
    return out


def finding_from_result(result, excerpts):
    j = result.judgment
    if not result.check.accepted or j is None or j.decision == "none":
        return None
    # Removing just a missing citation could leave an unsupported connection. Suppress it whole.
    if not j.citations or any((c.ref_kind, c.ref_id) not in excerpts for c in j.citations):
        return None
    refs = sorted({f"{c.ref_kind}:{c.ref_id}" for c in j.citations})
    content_id = "resonance:" + hashlib.sha256("\n".join(refs).encode()).hexdigest()
    evidence = [f'Read quote: "{j.read_quote}"']
    if j.bears_on_prior:
        evidence.append(f"Relation: {j.bears_on_prior}; prior: {j.strained_prior_id or 'cited belief'}")
    evidence.extend(f"{c.ref_kind}:{c.ref_id}: {excerpts[c.ref_kind, c.ref_id]}" for c in j.citations)
    return ReadingFindingV1(finding_id="resonance:" + result.resonance_id, kind="resonance",
        seed_id=result.seed_id, observed_at=result.completed_at, headline=j.connection,
        evidence_text="\n".join(evidence), evidence_refs=refs, run_id=result.run_id,
        resonance_decision=j.decision, resonance_root=result.packet.root_request_id or result.seed_id,
        resonance_content_id=content_id, resonance_article_summary=j.article_summary)


RESONANCE_FINDINGS_SQL = (
    "SELECT result_json, outreach_note_reason FROM reading_resonance WHERE seed_id=ANY($1::text[]) "
    "AND completed_at < $2 AND result_json IS NOT NULL ORDER BY completed_at"
)

async def resonance_findings(conn, seed_ids, *, until: datetime, resolver=resolve_citations):
    rows = await conn.fetch(RESONANCE_FINDINGS_SQL, seed_ids, until)
    findings = []
    for row in rows:
        result = ReadingResonanceResultV1.model_validate(obj(row["result_json"]))
        if not result.check.accepted or result.check.decision == "none":
            continue
        excerpts = await resolver(conn, result)
        finding = finding_from_result(result, excerpts)
        if finding:
            if row.get("outreach_note_reason"):
                finding = finding.model_copy(update={"resonance_decision": "note"})
            findings.append(finding)
    return findings
