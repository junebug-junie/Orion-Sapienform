from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import ValidationError

from datetime import datetime, timezone

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.detection import detect_contradictions, detect_duplicates, merge_detection
from orion.memory.crystallization.retriever import retrieve_active_packet
from orion.memory.crystallization.bus_emit import emit_active_packet_retrieved, emit_crystallization_lifecycle
from orion.memory.crystallization.chroma_publish import publish_crystallization_to_chroma
from orion.memory.crystallization.governor import GovernorError, approve, quarantine, reject, supersede
from orion.memory.crystallization.links import insert_link, list_links, neighborhood as link_neighborhood
from orion.memory.crystallization.projection_cards import build_memory_card_projection
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.projection_rdf import build_rdf_projection_hint
from orion.memory.crystallization.projector import ProjectionConfig, project_crystallization
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.sources import resolve_crystallization_sources
from orion.memory.crystallization.repository import (
    get_crystallization,
    insert_crystallization,
    insert_history,
    insert_retrieval_event,
    list_crystallizations,
    update_crystallization,
)
from orion.memory.crystallization.schemas import CrystallizationLinkV1, MemoryCrystallizationProposeRequestV1, MemoryCrystallizationV1
from orion.memory.crystallization.validator import ValidationResult, apply_validation_to_governance, validate_proposal

from .session import ensure_session

try:
    from asyncpg.exceptions import UndefinedTableError as _AsyncpgUndefinedTableError
except ImportError:
    _AsyncpgUndefinedTableError = None  # type: ignore[misc, assignment]

logger = logging.getLogger("orion-hub.crystallization")

router = APIRouter(tags=["memory-crystallizations"])


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


def _http_if_missing_schema(exc: BaseException) -> None:
    if _AsyncpgUndefinedTableError is not None and isinstance(exc, _AsyncpgUndefinedTableError):
        raise HTTPException(status_code=503, detail="memory_crystallization_schema_missing") from exc


async def _need_session(x_orion_session_id: Optional[str]) -> str:
    from .main import bus

    return await ensure_session(x_orion_session_id, bus)


def _settings():
    from scripts.settings import settings

    return settings


def _graphiti(request: Request) -> GraphitiAdapter:
    settings = _settings()
    adapter_url = (getattr(settings, "GRAPHITI_ADAPTER_URL", "") or getattr(settings, "GRAPHITI_URL", "") or "").strip()
    return GraphitiAdapter(
        enabled=bool(getattr(settings, "GRAPHITI_ENABLED", False)) or bool(adapter_url),
        url=adapter_url or None,
        falkordb_uri=getattr(settings, "FALKORDB_URI", None),
    )


def _projection_config() -> ProjectionConfig:
    s = _settings()
    return ProjectionConfig(
        collection=getattr(s, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations"),
        embed_host_url=getattr(s, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        embed_mode=getattr(s, "CRYSTALLIZER_EMBED_MODE", "http") or "http",
        embed_timeout_ms=int(getattr(s, "CRYSTALLIZER_EMBED_TIMEOUT_MS", 8000) or 8000),
        graphiti_enabled=bool(getattr(s, "GRAPHITI_ENABLED", False)),
        graphiti_url=getattr(s, "GRAPHITI_URL", "") or "",
        falkordb_uri=getattr(s, "FALKORDB_URI", "") or "",
        service_name=getattr(s, "SERVICE_NAME", "orion-hub"),
        service_version=getattr(s, "SERVICE_VERSION", "0.1.0"),
        node_name=getattr(s, "NODE_NAME", "hub"),
    )


async def _bus():
    from .main import bus

    return bus


async def _rpc_bus():
    from .main import rpc_bus

    return rpc_bus


@router.post("/api/memory/crystallizations/propose")
async def crystallization_propose(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        req = MemoryCrystallizationProposeRequestV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e

    crystallization = propose(req)
    try:
        stored_id = await insert_crystallization(pool, crystallization)
        await insert_history(
            pool,
            crystallization_id=stored_id,
            op="propose",
            actor=req.proposed_by,
            before=None,
            after={"status": crystallization.status},
        )
    except Exception as exc:
        _http_if_missing_schema(exc)
        logger.warning("crystallization_propose_failed error=%s", exc)
        raise HTTPException(status_code=400, detail="propose_failed") from exc

    row = await get_crystallization(pool, stored_id)
    if not row:
        raise HTTPException(status_code=500, detail="propose_missing_row")

    await emit_crystallization_lifecycle(await _bus(), lifecycle="proposed", crystallization=row, service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME)
    return row.model_dump(mode="json")


@router.get("/api/memory/crystallizations/proposals")
async def crystallization_list_proposals(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        items = await list_crystallizations(pool, status="proposed", limit=limit, offset=offset)
        quarantined = await list_crystallizations(pool, status="quarantined", limit=limit, offset=offset)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="list_failed") from exc
    return {
        "items": [i.model_dump(mode="json") for i in items + quarantined],
        "count": len(items) + len(quarantined),
    }


@router.get("/api/memory/crystallizations/proposals/{crystallization_id}")
async def crystallization_get_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        row = await get_crystallization(pool, crystallization_id)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="get_failed") from exc
    if not row or row.status not in ("proposed", "quarantined"):
        raise HTTPException(status_code=404, detail="proposal_not_found")
    return row.model_dump(mode="json")


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/validate")
async def crystallization_validate_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    result = validate_proposal(row)
    source_result = await resolve_crystallization_sources(pool, row)
    existing = await list_crystallizations(pool, limit=500)
    detection = merge_detection(
        detect_duplicates(row, existing),
        detect_contradictions(row, existing),
    )
    all_errors = list(result.errors) + list(source_result.errors)
    if detection.duplicates:
        all_errors.append(f"duplicate_candidates:{','.join(detection.duplicates)}")
    if detection.contradictions:
        all_errors.append(f"contradiction_candidates:{','.join(detection.contradictions)}")
    valid = result.valid and source_result.valid and not detection.duplicates

    updated = row.model_copy(deep=True)
    if valid:
        updated.governance.validation_status = "valid"
        updated.governance.validation_errors = []
    elif source_result.unresolved:
        updated = apply_validation_to_governance(
            updated, ValidationResult(valid=False, errors=all_errors, quarantine=True)
        )
    else:
        updated.governance.validation_status = "invalid"
        updated.governance.validation_errors = all_errors

    await update_crystallization(pool, updated)
    lifecycle = "validated" if valid else ("quarantined" if updated.status == "quarantined" else "validated")
    await emit_crystallization_lifecycle(await _bus(), lifecycle=lifecycle, crystallization=updated, service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME)
    return {
        "valid": valid,
        "errors": all_errors,
        "detection": {
            "duplicates": detection.duplicates,
            "contradictions": detection.contradictions,
            "warnings": detection.warnings,
        },
        "crystallization": updated.model_dump(mode="json"),
    }


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/approve")
async def crystallization_approve_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    # Governor path requires explicit approval actor
    row.governance.approved_by = session
    reason = (body or {}).get("reason") if body else None
    try:
        updated, history = approve(row, actor=session, reason=reason)
    except GovernorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await update_crystallization(pool, updated)
    await insert_history(
        pool,
        crystallization_id=crystallization_id,
        op=history["op"],
        actor=session,
        before=history.get("before"),
        after=history.get("after"),
        reason=reason,
    )
    await emit_crystallization_lifecycle(
        await _bus(), lifecycle="approved", crystallization=updated,
        service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME,
    )

    projection_summary = None
    if bool(getattr(_settings(), "CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE", True)):
        updated, proj = await project_crystallization(
            pool, await _rpc_bus(), updated, actor=session, config=_projection_config(),
        )
        await update_crystallization(pool, updated)
        projection_summary = {
            "card_id": proj.card_id,
            "chroma": proj.chroma,
            "graphiti": proj.graphiti,
            "bus_project_emitted": proj.bus_project_emitted,
            "errors": proj.errors,
        }

    out = updated.model_dump(mode="json")
    if projection_summary:
        out["projection"] = projection_summary
    return out


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/reject")
async def crystallization_reject_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    reason = (body or {}).get("reason") if body else None
    try:
        updated, history = reject(row, actor=session, reason=reason)
    except GovernorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op=history["op"], actor=session, before=history.get("before"), after=history.get("after"), reason=reason)
    await emit_crystallization_lifecycle(await _bus(), lifecycle="rejected", crystallization=updated, service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME)
    return updated.model_dump(mode="json")


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/quarantine")
async def crystallization_quarantine_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    errors = (body or {}).get("errors") or ["operator_quarantine"]
    reason = (body or {}).get("reason") if body else None
    updated, history = quarantine(row, actor=session, errors=errors, reason=reason)
    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op=history["op"], actor=session, before=history.get("before"), after=history.get("after"), reason=reason)
    await emit_crystallization_lifecycle(await _bus(), lifecycle="quarantined", crystallization=updated, service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME)
    return updated.model_dump(mode="json")


@router.get("/api/memory/crystallizations")
async def crystallization_list(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        items = await list_crystallizations(pool, status=status, kind=kind, limit=limit, offset=offset)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="list_failed") from exc
    return {"items": [i.model_dump(mode="json") for i in items], "count": len(items)}


@router.get("/api/memory/crystallizations/{crystallization_id}")
async def crystallization_get(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return row.model_dump(mode="json")


@router.post("/api/memory/crystallizations/{crystallization_id}/project/card")
async def crystallization_project_card(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    card_create = build_memory_card_projection(row)
    if card_create is None:
        raise HTTPException(status_code=400, detail="projection_not_allowed_for_status")

    card_id = await mc_dal.insert_card(pool, card_create, actor=session, op="crystallization_project")
    updated = row.model_copy(deep=True)
    updated.projection_refs.memory_card_ids = list(updated.projection_refs.memory_card_ids) + [str(card_id)]
    await update_crystallization(pool, updated)
    return {"card_id": str(card_id), "crystallization_id": crystallization_id}


@router.post("/api/memory/crystallizations/{crystallization_id}/project/chroma")
async def crystallization_project_chroma(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    cfg = _projection_config()
    updated, chroma_result = await publish_crystallization_to_chroma(
        row, await _bus(),
        collection=cfg.collection,
        vector_channel=cfg.vector_channel,
        embed_host_url=cfg.embed_host_url,
        embed_mode=cfg.embed_mode,
        embed_timeout_ms=cfg.embed_timeout_ms,
        service_name=cfg.service_name,
    )
    await update_crystallization(pool, updated)
    return {
        "channel": cfg.vector_channel,
        "kind": "memory.vector.upsert.v1",
        "result": chroma_result,
        "crystallization_id": crystallization_id,
    }


@router.post("/api/memory/crystallizations/{crystallization_id}/project/graphiti")
async def crystallization_project_graphiti(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    adapter = _graphiti(request)
    result = adapter.sync_crystallization(row)
    updated = adapter.apply_projection_refs(row, result)
    await update_crystallization(pool, updated)
    return {
        "canonical_mutated": result.canonical_mutated,
        "projection": result.__dict__,
        "crystallization_id": crystallization_id,
    }


@router.get("/api/memory/crystallizations/projection/health")
async def crystallization_projection_health(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    adapter = _graphiti(request)
    return {
        "chroma_collection": "orion_memory_crystallizations",
        "graphiti_enabled": adapter.enabled,
        "rdf_memory_graph": "unchanged_existing_path",
    }


@router.post("/api/memory/active-packet")
async def memory_active_packet(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    task_type = body.get("task_type")
    project_id = body.get("project_id")
    session_id = body.get("session_id")
    card_refs: List[str] = list(body.get("card_refs") or [])

    try:
        active_items = await list_crystallizations(pool, status="active", limit=100)
        active_cards = await mc_dal.list_cards(pool, status="active", limit=50)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="retrieval_failed") from exc

    if not card_refs and active_cards:
        card_refs = [str(c.card_id) for c in active_cards[:20]]

    s = _settings()
    seed_id = str(body.get("seed_crystallization_id") or "") or (active_items[0].crystallization_id if active_items else "")
    packet = await retrieve_active_packet(
        query=query,
        crystallizations=active_items,
        card_refs=card_refs,
        active_cards=[c.model_dump(mode="json") for c in active_cards[:20]],
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        chroma_host=getattr(s, "CHROMA_HOST", "") or "",
        chroma_port=int(getattr(s, "CHROMA_PORT", 8000) or 8000),
        chroma_collection=getattr(s, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations"),
        embed_host_url=getattr(s, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        graphiti_adapter=_graphiti(request) if seed_id else None,
        seed_crystallization_id=seed_id or None,
    )

    event_id = await insert_retrieval_event(
        pool,
        query=query,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        crystallization_ids=packet.crystallization_refs,
        card_refs=card_refs,
        trace=packet.retrieval_trace,
    )
    await emit_active_packet_retrieved(
        await _bus(), packet, service_name=_settings().SERVICE_NAME, node_name=_settings().NODE_NAME,
    )
    out = packet.model_dump(mode="json")
    out["retrieval_event_id"] = event_id
    return out


@router.get("/api/memory/retrieval-events/{event_id}")
async def memory_retrieval_event(
    request: Request,
    event_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        UUID(event_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid_event_id") from e

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memory_crystallization_retrieval_events WHERE retrieval_event_id = $1::uuid",
            event_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return dict(row)


@router.get("/api/memory/graphiti/health")
async def graphiti_health(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    adapter = _graphiti(request)
    return {"enabled": adapter.enabled, "url_configured": bool(adapter.url)}


@router.get("/api/memory/graphiti/neighborhood/{crystallization_id}")
async def graphiti_neighborhood(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    return _graphiti(request).neighborhood(crystallization_id)


@router.patch("/api/memory/crystallizations/{crystallization_id}")
async def crystallization_patch(
    request: Request,
    crystallization_id: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    updated = row.model_copy(deep=True)
    for field in ("subject", "summary", "confidence", "tags", "planning_effects", "retrieval_affordances"):
        if field in body:
            setattr(updated, field, body[field])
    if "salience" in body:
        updated.salience = float(body["salience"])
    updated.updated_at = datetime.now(timezone.utc)
    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op="update", actor=session, before=row.model_dump(mode="json"), after=updated.model_dump(mode="json"))
    return updated.model_dump(mode="json")


@router.post("/api/memory/crystallizations/{crystallization_id}/status")
async def crystallization_status(
    request: Request,
    crystallization_id: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    new_status = str(body.get("status") or "").strip()
    if new_status not in ("active", "rejected", "superseded", "deprecated", "archived", "quarantined"):
        raise HTTPException(status_code=400, detail="invalid_status")

    before = {"status": row.status}
    updated = row.model_copy(deep=True)
    updated.status = new_status  # type: ignore[assignment]
    updated.updated_at = datetime.now(timezone.utc)
    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op="status_change", actor=session, before=before, after={"status": new_status}, reason=body.get("reason"))
    return updated.model_dump(mode="json")


@router.post("/api/memory/crystallizations/{crystallization_id}/suppress")
async def crystallization_suppress(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    return await crystallization_status(
        request, crystallization_id,
        {"status": "archived", "reason": "suppress_from_retrieval"},
        x_orion_session_id,
    )


@router.post("/api/memory/crystallizations/{crystallization_id}/deprecate")
async def crystallization_deprecate(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    return await crystallization_status(
        request, crystallization_id, {"status": "deprecated", "reason": "operator_deprecate"}, x_orion_session_id,
    )


@router.post("/api/memory/crystallizations/{crystallization_id}/links")
async def crystallization_add_link(
    request: Request,
    crystallization_id: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        link = CrystallizationLinkV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    await insert_link(pool, from_crystallization_id=crystallization_id, link=link)
    return {"ok": True, "from": crystallization_id, "link": link.model_dump(mode="json")}


@router.get("/api/memory/crystallizations/{crystallization_id}/links")
async def crystallization_list_links(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    links = await list_links(pool, crystallization_id)
    return {"items": links, "count": len(links)}


@router.get("/api/memory/crystallizations/{crystallization_id}/neighborhood")
async def crystallization_neighborhood(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    return await link_neighborhood(pool, crystallization_id)


@router.post("/api/memory/crystallizations/{crystallization_id}/project/rdf")
async def crystallization_project_rdf(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    hint = build_rdf_projection_hint(row)
    if hint.skipped:
        raise HTTPException(status_code=400, detail=hint.reason)
    updated = row.model_copy(deep=True)
    if hint.named_graph and hint.named_graph not in updated.projection_refs.rdf_named_graphs:
        updated.projection_refs.rdf_named_graphs = list(updated.projection_refs.rdf_named_graphs) + [hint.named_graph]
        await update_crystallization(pool, updated)
    return {"named_graph": hint.named_graph, "note": "use_existing_memory_graph_approve_flow", "crystallization_id": crystallization_id}


@router.post("/api/memory/crystallizations/projection/rebuild")
async def crystallization_projection_rebuild(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    items = await list_crystallizations(pool, status="active", limit=limit)
    results = []
    for item in items:
        updated, proj = await project_crystallization(pool, await _rpc_bus(), item, actor=session, config=_projection_config())
        await update_crystallization(pool, updated)
        results.append({"crystallization_id": item.crystallization_id, "card_id": proj.card_id, "chroma": proj.chroma, "errors": proj.errors})
    return {"rebuilt": len(results), "items": results}


@router.post("/api/memory/graphiti/sync/{crystallization_id}")
async def graphiti_sync(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    updated, proj = await project_crystallization(
        pool, await _rpc_bus(), row, actor=session, config=_projection_config(),
        project_card=False, project_chroma=False, project_graphiti=True,
    )
    await update_crystallization(pool, updated)
    return {"crystallization_id": crystallization_id, "graphiti": proj.graphiti, "canonical_mutated": False}
