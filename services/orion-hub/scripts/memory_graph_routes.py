from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import asyncpg
import requests
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from orion.memory_graph.approve import approve_memory_graph_draft, preview_validate_only
from orion.memory_graph.consolidation_draft_hydrate import hydrate_consolidation_draft_dict
from orion.memory_graph.dto import CardProjectionDefaultsV1, SuggestDraftV1
from orion.memory_graph.utterance_text import ensure_draft_utterance_text

from .mutation_cognition_context import build_mutation_cognition_context
from .session import ensure_session

logger = logging.getLogger("orion-hub.memory_graph")

router = APIRouter(tags=["memory-graph"])


def _supplemental_utterance_text(body: Dict[str, Any], draft_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for source in (
        draft_payload.get("utterance_text_by_id") if isinstance(draft_payload, dict) else None,
        body.get("utterance_text_by_id"),
    ):
        if not isinstance(source, dict):
            continue
        for key, val in source.items():
            k = str(key or "").strip()
            v = str(val or "").strip()
            if k and v:
                out[k] = v
    return out


def _parse_card_projection_defaults(body: Dict[str, Any]) -> Optional[CardProjectionDefaultsV1]:
    raw = body.get("card_projection_defaults")
    if not isinstance(raw, dict) or not raw:
        return None
    return CardProjectionDefaultsV1.model_validate(raw)


def _parse_draft_body(body: Dict[str, Any]) -> tuple[SuggestDraftV1, Dict[str, str]]:
    payload = body.get("draft") if isinstance(body.get("draft"), dict) else body
    if not isinstance(payload, dict):
        payload = {}
    draft = SuggestDraftV1.model_validate(payload)
    supplemental = _supplemental_utterance_text(body, payload)
    return draft, supplemental


async def _consolidation_supplemental_utterance_text(pool, consolidation_draft_id: str) -> Dict[str, str]:
    from orion.memory_graph.draft_repository import get_consolidation_draft

    row = await get_consolidation_draft(pool, consolidation_draft_id)
    if not row:
        return {}
    draft = row.get("draft") if isinstance(row.get("draft"), dict) else {}
    hydrated = await hydrate_consolidation_draft_dict(pool, draft, row.get("turn_correlation_ids") or [])
    text_map = hydrated.get("utterance_text_by_id") if isinstance(hydrated.get("utterance_text_by_id"), dict) else {}
    return {str(k): str(v) for k, v in text_map.items() if str(k).strip() and str(v).strip()}


async def _parse_draft_body_with_consolidation(
    pool,
    body: Dict[str, Any],
) -> tuple[SuggestDraftV1, Dict[str, str]]:
    draft, supplemental = _parse_draft_body(body)
    consolidation_draft_id = str(body.get("consolidation_draft_id") or "").strip() or None
    if consolidation_draft_id:
        supplemental = {
            **await _consolidation_supplemental_utterance_text(pool, consolidation_draft_id),
            **supplemental,
        }
    return draft, supplemental


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


@router.post("/api/memory/graph/validate")
async def memory_graph_validate(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    from scripts.main import bus

    await ensure_session(x_orion_session_id, bus)
    pool = _pool(request)
    try:
        draft, supplemental = await _parse_draft_body_with_consolidation(pool, body if isinstance(body, dict) else {})
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    ok, violations, preview = preview_validate_only(
        draft,
        supplemental_utterance_text=supplemental,
    )
    warnings = preview.get("warnings") if isinstance(preview, dict) else []
    return {"ok": ok, "violations": violations, "warnings": warnings or [], "preview": preview}


@router.post("/api/memory/graph/suggest")
async def memory_graph_suggest(
    _request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    """Memory-graph suggest draft (grounded Quick primary, Brain escalation). Read-only: no GraphDB/Postgres writes.

    ``user_id`` in the JSON body is passed through to ``build_chat_request`` for telemetry only; it is not
    authenticated from the session header. ``diagnostic`` / ``options.diagnostic`` embed raw model text — use
    only on trusted control-plane paths.
    """
    from scripts.main import bus, cortex_client
    from scripts.memory_graph_suggest import suggest_with_escalation
    from scripts.settings import settings

    import scripts.api_routes as api_mod

    if not bus or cortex_client is None:
        raise HTTPException(status_code=503, detail="cortex_unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    payload = body if isinstance(body, dict) else {}
    mc = build_mutation_cognition_context(store=api_mod.SUBSTRATE_MUTATION_STORE)
    return await suggest_with_escalation(
        cortex_client=cortex_client,
        payload=payload,
        session_id=str(session_id),
        user_id=str(payload.get("user_id") or "").strip() or None,
        settings=settings,
        mutation_context=mc,
    )


@router.post("/api/memory/graph/approve")
async def memory_graph_approve(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    from scripts.main import bus
    from scripts.settings import settings

    await ensure_session(x_orion_session_id, bus)
    from orion.memory_graph.rdf_target import resolve_memory_graph_rdf_target

    target = resolve_memory_graph_rdf_target()
    if target is None:
        raise HTTPException(status_code=503, detail="graph_backend_unconfigured")
    pool = _pool(request)
    consolidation_draft_id = str(body.get("consolidation_draft_id") or "").strip() or None
    try:
        draft, supplemental = await _parse_draft_body_with_consolidation(pool, body if isinstance(body, dict) else {})
        draft = ensure_draft_utterance_text(draft, supplemental=supplemental)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    except ValueError as exc:
        if str(exc).startswith("utterance_text_missing:"):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise
    named = (
        str(body.get("named_graph_iri") or "").strip()
        or getattr(settings, "MEMORY_GRAPH_DEFAULT_NAMED_GRAPH", "").strip()
    )
    if not named:
        raise HTTPException(status_code=400, detail="named_graph_iri_required")

    card_defaults = _parse_card_projection_defaults(body if isinstance(body, dict) else {})
    try:
        result = await approve_memory_graph_draft(
            draft,
            pool,
            graphdb_url=str(settings.GRAPHDB_URL),
            graphdb_repo=str(settings.GRAPHDB_REPO or "collapse"),
            graphdb_user=str(settings.GRAPHDB_USER or ""),
            graphdb_pass=str(settings.GRAPHDB_PASS or ""),
            named_graph_iri=named,
            card_defaults=card_defaults,
        )
    except ValueError as exc:
        if str(exc) == "graph_backend_unconfigured":
            raise HTTPException(status_code=503, detail="graph_backend_unconfigured") from exc
        if str(exc) == "hierarchy_cycle":
            raise HTTPException(status_code=400, detail="hierarchy_cycle") from exc
        logger.warning("memory_graph_approve_failed error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncpg.PostgresError as exc:
        logger.warning("memory_graph_approve_failed postgres error=%s", exc)
        raise HTTPException(status_code=503, detail="memory_store_error") from exc
    except requests.RequestException as exc:
        from orion.graph.fuseki_http import fuseki_http_error_body, is_fuseki_lock_exhaustion

        resp = getattr(exc, "response", None)
        body = fuseki_http_error_body(resp) if resp is not None else str(exc)
        if resp is not None and is_fuseki_lock_exhaustion(resp.status_code, body):
            logger.warning("memory_graph_approve_failed fuseki_lock_exhaustion error=%s", exc)
            raise HTTPException(
                status_code=503,
                detail="fuseki_lock_exhaustion",
            ) from exc
        logger.warning("memory_graph_approve_failed rdf_http error=%s", exc)
        raise HTTPException(status_code=503, detail="rdf_graph_unavailable") from exc

    if not result.ok:
        return {"ok": False, "violations": result.violations, "card_ids": []}
    if consolidation_draft_id:
        consolidation_draft_marked = False
        try:
            from orion.memory_graph.draft_repository import update_consolidation_draft_status

            updated = await update_consolidation_draft_status(pool, consolidation_draft_id, status="approved")
            consolidation_draft_marked = updated is not None
        except Exception as exc:
            logger.warning(
                "consolidation_draft_mark_approved_failed draft_id=%s error=%s",
                consolidation_draft_id,
                exc,
            )
        if not consolidation_draft_marked:
            return {
                "ok": True,
                "violations": [],
                "card_ids": [str(x) for x in result.card_ids],
                "consolidation_draft_id": consolidation_draft_id,
                "consolidation_draft_marked": False,
                "consolidation_draft_status": "update_failed",
            }
        return {
            "ok": True,
            "violations": [],
            "card_ids": [str(x) for x in result.card_ids],
            "consolidation_draft_id": consolidation_draft_id,
            "consolidation_draft_marked": True,
        }
    return {
        "ok": True,
        "violations": [],
        "card_ids": [str(x) for x in result.card_ids],
    }
