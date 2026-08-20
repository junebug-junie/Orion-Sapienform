from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import ValidationError

from datetime import datetime, timezone

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.detection import detect_contradictions, detect_duplicates, merge_detection
from orion.memory.crystallization.dynamics import decayed_activation, should_retire
from orion.memory.crystallization.recall_eligibility import eligible_for_recall
from orion.memory.crystallization.retriever import retrieve_active_packet
from orion.memory.crystallization.bus_emit import emit_active_packet_retrieved, emit_crystallization_lifecycle
from orion.memory.crystallization.chroma_publish import publish_crystallization_to_chroma
from orion.memory.crystallization.governor import GovernorError, approve, quarantine, reject, supersede
from orion.memory.crystallization.links import insert_link, list_links, neighborhood as link_neighborhood
from orion.memory.crystallization.projection_cards import build_memory_card_projection
from orion.memory.crystallization.graphiti_config import resolve_graphiti_adapter_url
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
    normalize_crystallization_id,
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
    adapter_url = resolve_graphiti_adapter_url(settings)
    return GraphitiAdapter(
        enabled=bool(getattr(settings, "GRAPHITI_ENABLED", False)) or bool(adapter_url),
        url=adapter_url or None,
        falkordb_uri=getattr(settings, "FALKORDB_URI", None),
    )


def _projection_config() -> ProjectionConfig:
    s = _settings()
    adapter_url = resolve_graphiti_adapter_url(s)
    return ProjectionConfig(
        collection=getattr(s, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations"),
        embed_host_url=getattr(s, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        embed_mode=getattr(s, "CRYSTALLIZER_EMBED_MODE", "http") or "http",
        embed_timeout_ms=int(getattr(s, "CRYSTALLIZER_EMBED_TIMEOUT_MS", 8000) or 8000),
        graphiti_enabled=bool(getattr(s, "GRAPHITI_ENABLED", False)) or bool(adapter_url),
        graphiti_url=adapter_url,
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
    try:
        source_result = await resolve_crystallization_sources(pool, row)
    except Exception as exc:
        # Mirrors the 503 pattern used for get_crystallization above. resolve_* can now raise
        # from a probe failure, and an unhandled exception here would 500 the endpoint AND
        # leave the proposal's status untouched with no signal -- worse than the honest
        # "cannot validate right now" this returns.
        logger.warning("crystallization_validate_sources_failed id=%s: %s", crystallization_id, exc)
        raise HTTPException(status_code=503, detail="source_resolution_unavailable") from exc
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
        # Grammar refs that did not resolve, and refs that could not be checked at all.
        # Neither invalidates -- grammar_events is retention-bounded, so absence is expected
        # -- but an operator looking at a proposal with thin evidence needs to see it. Live
        # 2026-08-20: 999 of 1,167 refs are absent, across 61 of 124 crystallizations.
        "absent_grammar_refs": list(source_result.absent_grammar_refs),
        "unverified_grammar_refs": list(source_result.unverified_grammar_refs),
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


BULK_DECIDE_MAX = 500
# Approve is an order of magnitude more expensive per item than reject: each one
# re-runs get/update/history/lifecycle-emit AND (with
# CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE, default True) a chroma/card projection
# plus a second update. 500 of those serialized in one request is minutes of I/O
# with no timeout budget -- the client disconnects while the server keeps
# writing, so the operator sees a network error over a half-applied batch.
BULK_APPROVE_MAX = 50


@router.post("/api/memory/crystallizations/proposals/bulk")
async def crystallization_bulk_decide(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    """Approve or reject many proposals in one call.

    Path-collision note: the only other route shaped `/proposals/{one_segment}`
    is a GET, so this POST cannot be captured by it regardless of registration
    order. If a `POST /proposals/{crystallization_id}` is ever added, it must be
    registered AFTER this one or "bulk" starts matching as an id.

    Partial success is the contract, not an error: one bad id must not sink the
    other 199. Every id gets its own result entry with an explicit outcome, so
    the caller can re-render precisely and never has to guess which half landed.
    Approve reuses the single-item handler rather than reimplementing it, so the
    projection/lifecycle-emit side effects cannot drift apart between the two
    paths.
    """
    session = await _need_session(x_orion_session_id)
    raw_ids = body.get("ids")
    action = str(body.get("action") or "").strip()
    reason = body.get("reason")
    if action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action_must_be_approve_or_reject")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise HTTPException(status_code=400, detail="ids_required")
    cap = BULK_APPROVE_MAX if action == "approve" else BULK_DECIDE_MAX
    if len(raw_ids) > cap:
        raise HTTPException(status_code=400, detail=f"too_many_ids_max_{cap}")

    # Dedup while preserving order -- a double-click on "select all" must not
    # attempt the same row twice and report a spurious already_decided failure.
    seen: set[str] = set()
    ids: List[str] = []
    for raw in raw_ids:
        cid = str(raw)
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)

    pool = _pool(request)
    results: List[Dict[str, Any]] = []
    for cid in ids:
        try:
            row = await get_crystallization(pool, cid)
            if row is None:
                results.append({"crystallization_id": cid, "ok": False, "error": "not_found"})
                continue
            if row.status not in ("proposed", "quarantined"):
                results.append(
                    {"crystallization_id": cid, "ok": False, "error": f"already_{row.status}"}
                )
                continue
            if action == "approve":
                await crystallization_approve_proposal(
                    request, cid, x_orion_session_id=x_orion_session_id, body={"reason": reason}
                )
            else:
                updated, history = reject(row, actor=session, reason=reason)
                await update_crystallization(pool, updated)
                await insert_history(
                    pool,
                    crystallization_id=cid,
                    op=history["op"],
                    actor=session,
                    before=history.get("before"),
                    after=history.get("after"),
                    reason=reason,
                )
                # Post-commit. The decision is already durable, so a bus outage
                # must not be reported as a failed decision: the caller would
                # keep the id selected and every retry would answer
                # already_rejected, leaving a "N failed" the operator can never
                # clear except by deselecting by hand.
                try:
                    await emit_crystallization_lifecycle(
                        await _bus(),
                        lifecycle="rejected",
                        crystallization=updated,
                        service_name=_settings().SERVICE_NAME,
                        node_name=_settings().NODE_NAME,
                    )
                except Exception:
                    logger.exception("crystallization_bulk_reject_emit_failed id=%s", cid)
                    results.append(
                        {"crystallization_id": cid, "ok": True, "warning": "lifecycle_emit_failed"}
                    )
                    continue
            results.append({"crystallization_id": cid, "ok": True})
        except (HTTPException, GovernorError, Exception) as exc:  # noqa: BLE001
            if not isinstance(exc, (HTTPException, GovernorError)):
                _http_if_missing_schema(exc)
            detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            # Did the decision actually land before the failure? The approve path
            # writes the row and THEN emits/projects, so a bus or chroma outage
            # raises after the status is already durable. Reporting that as
            # failed keeps the id selected and makes every retry answer
            # already_active -- a "N failed" the operator can never clear.
            landed = False
            try:
                after = await get_crystallization(pool, cid)
                landed = after is not None and after.status not in ("proposed", "quarantined")
            except Exception:  # noqa: BLE001
                landed = False
            if landed:
                logger.warning(
                    "crystallization_bulk_%s_post_commit_failure id=%s error=%s", action, cid, exc
                )
                results.append(
                    {"crystallization_id": cid, "ok": True, "warning": f"post_commit: {detail}"}
                )
            else:
                logger.warning("crystallization_bulk_%s_failed id=%s error=%s", action, cid, exc)
                results.append(
                    {
                        "crystallization_id": cid,
                        "ok": False,
                        "error": detail if isinstance(exc, (HTTPException, GovernorError)) else "decide_failed",
                    }
                )

    succeeded = sum(1 for r in results if r["ok"])
    return {
        "action": action,
        "requested": len(ids),
        "succeeded": succeeded,
        "failed": len(ids) - succeeded,
        "results": results,
    }


@router.delete("/api/memory/crystallizations/{crystallization_id}/evidence/{source_id}")
async def crystallization_delete_evidence(
    request: Request,
    crystallization_id: str,
    source_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    """Drop one source turn from a proposal without discarding the whole thing.

    A consolidation window is assembled by a global open-window cursor, so a
    proposal routinely carries a turn that does not belong to what it is
    actually about. Before this, the only options were approve-with-the-bad-turn
    or reject-the-whole-window.

    Deliberately restricted to proposals. An `active` crystallization has already
    been projected into cards/chroma/graphiti, and silently removing evidence
    from underneath those projections would leave them citing a source the
    canonical row no longer claims. Deprecate and re-propose instead.

    Deletes only the sources row; the crystallization's own summary/claims are
    left alone, because rewriting what Orion concluded is a different, much
    heavier operation than correcting which turns it cited.
    """
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        row = await get_crystallization(pool, crystallization_id)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="get_failed") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="crystallization_not_found")
    if row.status not in ("proposed", "quarantined"):
        raise HTTPException(status_code=409, detail=f"not_editable_status_{row.status}")

    remaining = [e for e in row.evidence if e.source_id != source_id]
    if len(remaining) == len(row.evidence):
        raise HTTPException(status_code=404, detail="evidence_not_found")
    if not remaining:
        raise HTTPException(status_code=409, detail="cannot_remove_last_evidence")

    try:
        async with pool.acquire() as conn:
            deleted = await conn.execute(
                "DELETE FROM memory_crystallization_sources "
                "WHERE crystallization_id = $1::uuid AND source_id = $2",
                # Same normalization every other repository helper applies. A
                # caller passing the crys_<hex32> form new_crystallization_id()
                # mints clears the lookup and the 404/409 guards above, then
                # dies on the ::uuid cast without it.
                normalize_crystallization_id(crystallization_id),
                source_id,
            )
    except Exception as exc:
        _http_if_missing_schema(exc)
        logger.warning("crystallization_evidence_delete_failed id=%s src=%s error=%s", crystallization_id, source_id, exc)
        raise HTTPException(status_code=503, detail="evidence_delete_failed") from exc

    await insert_history(
        pool,
        crystallization_id=crystallization_id,
        op="evidence_removed",
        actor=session,
        before={"evidence_count": len(row.evidence)},
        after={"evidence_count": len(remaining), "removed_source_id": source_id},
        reason=f"operator removed source turn {source_id}",
    )
    logger.info(
        "crystallization_evidence_removed id=%s src=%s rows=%s remaining=%s actor=%s",
        crystallization_id,
        source_id,
        deleted,
        len(remaining),
        session,
    )
    refreshed = await get_crystallization(pool, crystallization_id)
    return (refreshed or row).model_dump(mode="json")


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
    now = datetime.now(timezone.utc)
    out_items: List[Dict[str, Any]] = []
    for i in items:
        row = i.model_dump(mode="json")
        # Retirement candidacy only applies to the active pool -- matches the same
        # status scoping recall_eligibility.eligible_for_recall() uses. Computed live,
        # never persisted (docs/superpowers/specs/2026-07-13-recall-followups-loop-
        # retirement-saturation-gate-spec.md section 2).
        if i.status == "active":
            row["decayed_activation"] = decayed_activation(i, now=now)
            row["retirement_candidate"] = should_retire(i, now=now)
        else:
            row["decayed_activation"] = None
            row["retirement_candidate"] = False
        out_items.append(row)
    return {"items": out_items, "count": len(out_items)}


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
        active_items = [
            c for c in await list_crystallizations(pool, status="active", limit=100)
            if eligible_for_recall(c)
        ]
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
        pool=pool,
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
