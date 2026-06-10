"""HTTP API for memory crystallization governance and projection.

Deliberately namespaced under /api/memory/crystallizations and
/api/memory/graphiti — the existing RDF memory_graph routes
(/api/memory/graph/*) are not touched.

All repository calls are synchronous psycopg2 and run via asyncio.to_thread
so handlers never block the event loop. Governance transitions are written
atomically (status change + audit history in one transaction) with
optimistic status guards so stale writers cannot revert governed state.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from orion.memory.crystallization import active_packet as active_packet_mod
from orion.memory.crystallization import governor
from orion.memory.crystallization import projection_cards, projection_chroma, projection_graphiti
from orion.memory.crystallization.repository import CrystallizationRepository
from orion.schemas.memory_crystallization import (
    CrystallizationLinkV1,
    MemoryCrystallizationV1,
)

logger = logging.getLogger("orion.memory-crystallizer.routes")

router = APIRouter(prefix="/api/memory")

CONFLICT_DETAIL = "concurrent_modification: state changed underneath this request"


# --- request bodies ---------------------------------------------------------


class GovernanceActionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str
    reason: Optional[str] = None
    approval_mode: Literal["auto_policy", "operator", "manual_required"] = "operator"
    salience_override: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class StatusChangeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["deprecated", "archived"]
    actor: str
    reason: Optional[str] = None


class SupersedeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    superseded_by: str
    actor: str
    reason: Optional[str] = None


class LinkBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str
    link: CrystallizationLinkV1


class ActivePacketBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    task_type: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: Optional[str] = None
    kind: Optional[str] = None
    limit: int = Field(default=25, ge=1, le=200)


# --- helpers ----------------------------------------------------------------


def _repo(request: Request) -> CrystallizationRepository:
    repo = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="repository_unavailable")
    return repo


async def _get_or_404(repo: CrystallizationRepository, cid: str) -> MemoryCrystallizationV1:
    crystallization = await asyncio.to_thread(repo.get, cid)
    if crystallization is None:
        raise HTTPException(status_code=404, detail="not_found")
    return crystallization


async def _publish(request: Request, channel: str, kind: str, payload: dict[str, Any]) -> bool:
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        return False
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    settings = request.app.state.settings
    try:
        await bus.publish(
            channel,
            BaseEnvelope(
                kind=kind,
                source=ServiceRef(name=settings.service_name, version=settings.service_version),
                payload=payload,
            ),
        )
        return True
    except Exception as exc:
        logger.warning("bus_publish_failed channel=%s reason=%s", channel, exc)
        return False


async def _publish_projection_update(request: Request, crystallization: MemoryCrystallizationV1) -> None:
    """Notify projection consumers about lifecycle changes so derived
    surfaces (cards/Chroma/Graphiti) can refresh or retract stale state."""
    settings = request.app.state.settings
    await _publish(
        request,
        settings.channel_project,
        "memory.crystallization.project.v1",
        crystallization.model_dump(mode="json"),
    )


# --- proposals --------------------------------------------------------------


@router.post("/crystallizations/propose")
async def propose(request: Request, proposal: MemoryCrystallizationV1) -> dict[str, Any]:
    if proposal.status != "proposed":
        raise HTTPException(status_code=422, detail="proposals must have status='proposed'")
    repo = _repo(request)
    existing = await asyncio.to_thread(repo.get, proposal.crystallization_id)
    if existing is not None and existing.status != "proposed":
        raise HTTPException(status_code=409, detail="crystallization already governed; cannot re-propose")
    settings = request.app.state.settings
    actor = proposal.governance.proposed_by or settings.service_name

    validated, entry = governor.validate(proposal, actor)
    # Guarded write: never overwrite a row that was governed in the meantime.
    written = await asyncio.to_thread(
        repo.apply_transition,
        validated,
        entry,
        after=validated.model_dump(mode="json"),
        expected_statuses=["proposed"],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    await _publish(
        request,
        settings.channel_proposed,
        "memory.crystallization.proposed.v1",
        validated.model_dump(mode="json"),
    )
    return {
        "crystallization_id": validated.crystallization_id,
        "status": validated.status,
        "validation_status": validated.governance.validation_status,
        "validation_errors": validated.governance.validation_errors,
    }


@router.get("/crystallizations/proposals")
async def list_proposals(request: Request, kind: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    repo = _repo(request)
    rows = await asyncio.to_thread(repo.list, status="proposed", kind=kind, limit=limit)
    return [row.model_dump(mode="json") for row in rows]


@router.get("/crystallizations/proposals/{cid}")
async def get_proposal(request: Request, cid: str) -> dict[str, Any]:
    crystallization = await _get_or_404(_repo(request), cid)
    if crystallization.status != "proposed":
        raise HTTPException(status_code=404, detail="not_a_proposal")
    return crystallization.model_dump(mode="json")


@router.post("/crystallizations/proposals/{cid}/validate")
async def validate_endpoint(request: Request, cid: str, body: GovernanceActionBody) -> dict[str, Any]:
    repo = _repo(request)
    proposal = await _get_or_404(repo, cid)
    if proposal.status != "proposed":
        raise HTTPException(status_code=409, detail=f"cannot validate from status {proposal.status!r}")
    validated, entry = governor.validate(proposal, body.actor)
    written = await asyncio.to_thread(
        repo.apply_transition,
        validated,
        entry,
        after=validated.model_dump(mode="json"),
        expected_statuses=["proposed"],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    settings = request.app.state.settings
    await _publish(
        request,
        settings.channel_validated,
        "memory.crystallization.validated.v1",
        validated.model_dump(mode="json"),
    )
    return {
        "crystallization_id": cid,
        "validation_status": validated.governance.validation_status,
        "validation_errors": validated.governance.validation_errors,
    }


@router.post("/crystallizations/proposals/{cid}/approve")
async def approve_endpoint(request: Request, cid: str, body: GovernanceActionBody) -> dict[str, Any]:
    repo = _repo(request)
    proposal = await _get_or_404(repo, cid)
    try:
        approved, entry = governor.approve(
            proposal,
            body.actor,
            approval_mode=body.approval_mode,
            salience_override=body.salience_override,
        )
    except governor.GovernanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    written = await asyncio.to_thread(
        repo.apply_transition,
        approved,
        entry,
        before=proposal.model_dump(mode="json"),
        after=approved.model_dump(mode="json"),
        expected_statuses=["proposed"],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    settings = request.app.state.settings
    await _publish(
        request,
        settings.channel_approved,
        "memory.crystallization.approved.v1",
        approved.model_dump(mode="json"),
    )
    return approved.model_dump(mode="json")


@router.post("/crystallizations/proposals/{cid}/reject")
async def reject_endpoint(request: Request, cid: str, body: GovernanceActionBody) -> dict[str, Any]:
    repo = _repo(request)
    proposal = await _get_or_404(repo, cid)
    try:
        rejected, entry = governor.reject(proposal, body.actor, reason=body.reason)
    except governor.GovernanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    written = await asyncio.to_thread(
        repo.apply_transition,
        rejected,
        entry,
        before=proposal.model_dump(mode="json"),
        after=rejected.model_dump(mode="json"),
        expected_statuses=["proposed"],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    settings = request.app.state.settings
    await _publish(
        request,
        settings.channel_rejected,
        "memory.crystallization.rejected.v1",
        rejected.model_dump(mode="json"),
    )
    return {"crystallization_id": cid, "status": rejected.status}


@router.post("/crystallizations/proposals/{cid}/quarantine")
async def quarantine_endpoint(request: Request, cid: str, body: GovernanceActionBody) -> dict[str, Any]:
    repo = _repo(request)
    proposal = await _get_or_404(repo, cid)
    try:
        quarantined, entry = governor.quarantine(proposal, body.actor, reason=body.reason)
    except governor.GovernanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    written = await asyncio.to_thread(
        repo.apply_transition,
        quarantined,
        entry,
        before=proposal.model_dump(mode="json"),
        after=quarantined.model_dump(mode="json"),
        expected_statuses=[proposal.status],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    settings = request.app.state.settings
    await _publish(
        request,
        settings.channel_quarantined,
        "memory.crystallization.quarantined.v1",
        quarantined.model_dump(mode="json"),
    )
    # Quarantine from active means derived surfaces must drop the artifact.
    await _publish_projection_update(request, quarantined)
    return {"crystallization_id": cid, "status": quarantined.status}


# --- canonical crystallizations ---------------------------------------------


@router.get("/crystallizations")
async def list_crystallizations(
    request: Request,
    status: Optional[str] = "active",
    kind: Optional[str] = None,
    scope: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    repo = _repo(request)
    rows = await asyncio.to_thread(repo.list, status=status or None, kind=kind, scope=scope, limit=limit)
    return [row.model_dump(mode="json") for row in rows]


@router.get("/crystallizations/{cid}")
async def get_crystallization(request: Request, cid: str) -> dict[str, Any]:
    crystallization = await _get_or_404(_repo(request), cid)
    return crystallization.model_dump(mode="json")


@router.get("/crystallizations/{cid}/history")
async def get_history(request: Request, cid: str, limit: int = 100) -> list[dict[str, Any]]:
    repo = _repo(request)
    await _get_or_404(repo, cid)
    return await asyncio.to_thread(repo.list_history, cid, limit)


@router.post("/crystallizations/{cid}/status")
async def change_status(request: Request, cid: str, body: StatusChangeBody) -> dict[str, Any]:
    repo = _repo(request)
    crystallization = await _get_or_404(repo, cid)
    try:
        updated, entry = governor.set_status(crystallization, body.status, body.actor, reason=body.reason)
    except governor.GovernanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    written = await asyncio.to_thread(
        repo.apply_transition,
        updated,
        entry,
        before=crystallization.model_dump(mode="json"),
        after=updated.model_dump(mode="json"),
        expected_statuses=[crystallization.status],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    await _publish_projection_update(request, updated)
    return {"crystallization_id": cid, "status": updated.status}


@router.post("/crystallizations/{cid}/supersede")
async def supersede_endpoint(request: Request, cid: str, body: SupersedeBody) -> dict[str, Any]:
    repo = _repo(request)
    old = await _get_or_404(repo, cid)
    new = await _get_or_404(repo, body.superseded_by)
    try:
        superseded_old, updated_new, entries = governor.supersede(old, new, body.actor, reason=body.reason)
    except governor.GovernanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    written = await asyncio.to_thread(
        repo.apply_supersession,
        superseded_old,
        updated_new,
        entries,
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    # Derived surfaces must re-render the old artifact as superseded.
    await _publish_projection_update(request, superseded_old)
    await _publish_projection_update(request, updated_new)
    return {
        "superseded": superseded_old.crystallization_id,
        "superseded_by": updated_new.crystallization_id,
        "old_status": superseded_old.status,
    }


@router.post("/crystallizations/{cid}/links")
async def add_link(request: Request, cid: str, body: LinkBody) -> dict[str, Any]:
    repo = _repo(request)
    crystallization = await _get_or_404(repo, cid)
    links = list(crystallization.links)
    if any(
        l.target_crystallization_id == body.link.target_crystallization_id
        and l.relation == body.link.relation
        for l in links
    ):
        raise HTTPException(status_code=409, detail="link_exists")
    links.append(body.link)
    updated = crystallization.model_copy(
        update={"links": links, "updated_at": datetime.now(timezone.utc)}
    )
    entry = governor.GovernanceHistoryEntry(
        op="link_add",
        actor=body.actor,
        crystallization_id=cid,
        detail={"relation": body.link.relation, "target": body.link.target_crystallization_id},
    )
    written = await asyncio.to_thread(
        repo.apply_transition,
        updated,
        entry,
        after=body.link.model_dump(mode="json"),
        expected_statuses=[crystallization.status],
    )
    if not written:
        raise HTTPException(status_code=409, detail=CONFLICT_DETAIL)
    return {"crystallization_id": cid, "links": [l.model_dump(mode="json") for l in updated.links]}


@router.get("/crystallizations/{cid}/links")
async def get_links(request: Request, cid: str) -> list[dict[str, Any]]:
    repo = _repo(request)
    await _get_or_404(repo, cid)
    return await asyncio.to_thread(repo.list_links, cid)


# --- projections -------------------------------------------------------------


@router.post("/crystallizations/{cid}/project/card")
async def project_card(request: Request, cid: str, force: bool = False) -> dict[str, Any]:
    repo = _repo(request)
    crystallization = await _get_or_404(repo, cid)
    try:
        card = projection_cards.crystallization_to_card(crystallization)
    except projection_cards.ProjectionNotAllowed as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    existing_ids = list(crystallization.projection_refs.memory_card_ids)
    if existing_ids and not force:
        return {
            "crystallization_id": cid,
            "card_created": False,
            "card_id": existing_ids[-1],
            "card_payload": card.model_dump(mode="json"),
            "note": "already projected; pass force=true to project again",
        }

    card_id: str | None = None
    pool = getattr(request.app.state, "cards_pool", None)
    if pool is not None:
        from orion.core.storage.memory_cards import insert_card

        settings = request.app.state.settings
        card_uuid = await insert_card(pool, card, actor=settings.service_name)
        card_id = str(card_uuid)
        await asyncio.to_thread(
            repo.merge_projection_refs, cid, {"memory_card_ids": [card_id]}
        )

    await _publish_projection_update(request, crystallization)
    return {
        "crystallization_id": cid,
        "card_created": card_id is not None,
        "card_id": card_id,
        "card_payload": card.model_dump(mode="json"),
    }


@router.post("/crystallizations/{cid}/project/chroma")
async def project_chroma(request: Request, cid: str) -> dict[str, Any]:
    repo = _repo(request)
    settings = request.app.state.settings
    crystallization = await _get_or_404(repo, cid)

    embedding: list[float] | None = None
    embedding_model: str | None = None
    if settings.embed_host_url:
        try:
            text = projection_chroma.crystallization_doc_text(crystallization)
            async with httpx.AsyncClient(timeout=settings.embed_timeout_ms / 1000.0) as client:
                resp = await client.post(
                    settings.embed_host_url.rstrip("/"),
                    json={"doc_id": cid, "text": text},
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding")
                embedding_model = data.get("embedding_model")
        except Exception as exc:
            logger.warning("embed_fetch_failed cid=%s reason=%s", cid, exc)

    try:
        payload = projection_chroma.crystallization_to_vector_upsert(
            crystallization,
            collection=settings.vector_collection,
            embedding=embedding,
            embedding_model=embedding_model,
        )
    except projection_chroma.ProjectionNotAllowed as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    published = await _publish(
        request,
        settings.channel_vector_upsert,
        projection_chroma.VECTOR_UPSERT_ENVELOPE_KIND,
        payload.model_dump(mode="json"),
    )
    if published:
        await asyncio.to_thread(
            repo.merge_projection_refs, cid, {"chroma_doc_ids": [payload.doc_id]}
        )
    return {
        "crystallization_id": cid,
        "published": published,
        "embedded": embedding is not None,
        "collection": settings.vector_collection,
    }


@router.post("/crystallizations/{cid}/project/graphiti")
async def project_graphiti_endpoint(request: Request, cid: str) -> dict[str, Any]:
    settings = request.app.state.settings
    if not settings.graphiti_enabled or not settings.graphiti_url:
        raise HTTPException(status_code=503, detail="graphiti_disabled")
    repo = _repo(request)
    crystallization = await _get_or_404(repo, cid)
    try:
        episode = projection_graphiti.build_graphiti_episode(crystallization)
    except projection_graphiti.ProjectionNotAllowed as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{settings.graphiti_url.rstrip('/')}/episodes", json=episode)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"graphiti_sync_failed: {exc}")

    ref_updates = {
        "graphiti_episode_ids": list(data.get("episode_ids") or [episode["episode_name"]]),
        "graphiti_entity_ids": list(data.get("entity_ids") or []),
        "graphiti_edge_ids": list(data.get("edge_ids") or []),
    }
    await asyncio.to_thread(repo.merge_projection_refs, cid, ref_updates)
    return {"crystallization_id": cid, "synced": True, "graphiti_refs": ref_updates}


@router.get("/graphiti/health")
async def graphiti_health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return {
        "enabled": settings.graphiti_enabled,
        "url_configured": bool(settings.graphiti_url),
        "falkordb_configured": bool(settings.falkordb_uri),
    }


# --- retrieval ----------------------------------------------------------------


@router.post("/active-packet")
async def active_packet_endpoint(request: Request, body: ActivePacketBody) -> dict[str, Any]:
    repo = _repo(request)
    settings = request.app.state.settings
    crystallizations = await asyncio.to_thread(
        repo.list, status="active", kind=body.kind, scope=body.scope, limit=body.limit
    )
    packet = active_packet_mod.build_active_packet(
        query=body.query,
        crystallizations=crystallizations,
        task_type=body.task_type,
        project_id=body.project_id,
        session_id=body.session_id,
        retrieval_trace={"backend": "postgres", "scope": body.scope, "kind": body.kind},
    )
    retrieval_event_id = f"mre_{uuid.uuid4().hex}"
    try:
        await asyncio.to_thread(repo.record_retrieval_event, retrieval_event_id, packet)
    except Exception as exc:
        logger.warning("retrieval_event_record_failed reason=%s", exc)
    await _publish(
        request,
        settings.channel_retrieved,
        "memory.crystallization.retrieved.v1",
        packet.model_dump(mode="json"),
    )
    return {"retrieval_event_id": retrieval_event_id, "packet": packet.model_dump(mode="json")}


@router.get("/retrieval-events/{retrieval_event_id}")
async def get_retrieval_event(request: Request, retrieval_event_id: str) -> dict[str, Any]:
    repo = _repo(request)
    event = await asyncio.to_thread(repo.get_retrieval_event, retrieval_event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="not_found")
    return event
