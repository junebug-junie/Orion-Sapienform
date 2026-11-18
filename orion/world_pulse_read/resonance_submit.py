"""Hub hands completed reads to durable admission. No generation or model retry lives here."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5
import httpx
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.reading_resonance import READING_RESONANCE_WORKFLOW, ReadingResonanceRunBriefV1
from orion.world_pulse_read.resonance_retrieval import prepare_brief

logger = logging.getLogger(__name__)


def request_for(brief: ReadingResonanceRunBriefV1) -> DurableRunRequestV1:
    return DurableRunRequestV1(run_id="resonance-" + brief.packet.packet_id,
        correlation_id=str(uuid5(NAMESPACE_URL, "reading-resonance:" + brief.packet.seed_id)),
        workflow=READING_RESONANCE_WORKFLOW, brief=brief, requested_at=brief.packet.built_at,
        admission=ResourceRequirementV1(resource=f"llm.route.{brief.llm_route}", preferred_lane=brief.llm_route,
            priority="background", deadline_at=brief.packet.built_at + timedelta(hours=6)))


async def bind_request(conn, brief):
    request = request_for(brief)
    await conn.execute("INSERT INTO reading_resonance(seed_id,run_id,request_json) VALUES($1,$2,$3::jsonb) "
                       "ON CONFLICT(seed_id) DO NOTHING", brief.packet.seed_id, request.run_id, request.model_dump_json())
    raw = await conn.fetchval("SELECT request_json FROM reading_resonance WHERE seed_id=$1", brief.packet.seed_id)
    return DurableRunRequestV1.model_validate(json.loads(raw) if isinstance(raw, str) else raw)


async def reconcile(conn, *, cfg, graph_url, bus, source, durable_url, lane="agent", max_tokens=900,
                    now=None, prepare=prepare_brief, client=None, substrate_store=None):
    """Recover the Stage-2-completion/submission crash gap. Request retries are byte-identical.

    Six-hour recent-read window avoids a surprise historical backfill at rollout. Once a request
    is frozen, durable admission's deadline bounds it; a restart never gives it six more hours.
    """
    now = now or datetime.now(timezone.utc)
    pending = await conn.fetch("SELECT seed_id, request_json FROM reading_resonance "
        "WHERE submitted_at IS NULL AND created_at >= $1 ORDER BY created_at LIMIT 4", now - timedelta(hours=6))
    requests = [DurableRunRequestV1.model_validate(json.loads(r["request_json"]) if isinstance(r["request_json"], str)
                 else r["request_json"]) for r in pending]
    rows = await conn.fetch("SELECT s.seed_id FROM world_pulse_read_seed s WHERE stage2_status='done' "
        "AND stage2_completed_at >= $1 AND NOT EXISTS (SELECT 1 FROM reading_resonance r WHERE r.seed_id=s.seed_id) "
        "ORDER BY stage2_completed_at LIMIT 4", now - timedelta(hours=6))
    for row in rows:
        try:
            brief = await prepare(conn, row["seed_id"], cfg=cfg, graph_url=graph_url, bus=bus, source=source,
                                  now=now, lane=lane, max_tokens=max_tokens, substrate_store=substrate_store)
            requests.append(await bind_request(conn, brief))
        except Exception:
            logger.exception("reading_resonance_prepare_failed seed=%s", row["seed_id"])
            # Index catch-up or unavailable store: no empty packet disguised as a judgment.
    async def submit(http):
        for request in requests:
            response = await http.post(durable_url.rstrip("/") + "/runs", json=request.model_dump(mode="json"))
            response.raise_for_status()
            await conn.execute("UPDATE reading_resonance SET submitted_at=coalesce(submitted_at,$2) WHERE run_id=$1",
                               request.run_id, now)
            logger.info("reading_resonance_submitted seed=%s run=%s candidates=%d", request.brief.packet.seed_id,
                        request.run_id, len(request.brief.packet.candidates))
    if client is not None:
        await submit(client)
    else:
        async with httpx.AsyncClient(timeout=15) as http:
            await submit(http)
