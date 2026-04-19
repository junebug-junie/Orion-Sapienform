from __future__ import annotations

import json
import logging
import os
import ipaddress
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional, Any, List, Dict, Tuple
from urllib.parse import urlparse
import aiohttp

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
import requests

from .settings import settings
from .session import ensure_session
from .chat_history import (
    build_chat_history_envelope,
    build_chat_response_feedback_envelope,
    publish_chat_history,
    publish_chat_response_feedback,
    publish_social_room_turn,
    select_reasoning_trace_for_history,
)
from .library import scan_cognition_library
from .trace_payloads import extract_agent_trace_payload
from .autonomy_payloads import extract_autonomy_payload
from .workflow_payloads import extract_workflow_payload
from .cortex_chat_display import hub_effective_chat_text
from .cortex_request_builder import build_chat_request, build_continuity_messages, validate_single_verb_override
from .social_room import is_social_room_payload, social_room_client_meta
from .service_logs import collect_service_inventory
from orion.cognition.verb_activation import build_verb_list
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.workflow_execution import WorkflowScheduleManageRequestV1, WorkflowScheduleManageResponseV1
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1, build_feedback_category_options
from orion.schemas.notify import (
    ChatAttentionAck,
    ChatMessageReceipt,
    NotificationPreferencesUpdate,
    PreferenceResolutionRequest,
    RecipientProfileUpdate,
)

from orion.core.schemas.substrate_review_queue import GraphReviewCyclePolicyV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryQueryV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_policy_comparison import SubstratePolicyComparisonRequestV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeSurfaceV1
from orion.substrate import build_substrate_policy_store_from_env, build_substrate_store_from_env
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.policy_comparison import SubstratePolicyComparisonService
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_bootstrap import GraphReviewBootstrapper
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.review_telemetry import GraphReviewCalibrationAnalyzer, GraphReviewTelemetryRecorder

logger = logging.getLogger("orion-hub.api")
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)

router = APIRouter()


def _hub_uses_host_network_mode() -> bool:
    mode = str(os.getenv("HUB_DOCKER_NETWORK_MODE", "")).strip().lower()
    return mode == "host"


def _postgres_url_host(postgres_url: str) -> str:
    try:
        return str(urlparse(postgres_url).hostname or "").strip()
    except Exception:
        return ""


def _looks_like_docker_service_hostname(postgres_url: str) -> bool:
    host = _postgres_url_host(postgres_url)
    if not host:
        return False
    if host in {"localhost", "host.docker.internal"}:
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    return "." not in host


def _resolve_control_plane_postgres_url() -> Optional[str]:
    control_plane_url = str(os.getenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "")).strip()
    policy_url = str(os.getenv("SUBSTRATE_POLICY_POSTGRES_URL", "")).strip()
    database_url = str(os.getenv("DATABASE_URL", "")).strip()
    resolved = control_plane_url or policy_url or database_url or None
    if resolved and _hub_uses_host_network_mode() and _looks_like_docker_service_hostname(resolved):
        logger.warning(
            "Hub is configured for host networking (HUB_DOCKER_NETWORK_MODE=host), "
            "but the resolved Postgres hostname '%s' looks like a Docker service name. "
            "Use a host-reachable address (for example 127.0.0.1, LAN IP, or Tailscale IP).",
            _postgres_url_host(resolved),
        )
    return resolved


SUBSTRATE_REVIEW_QUEUE_STORE = GraphReviewQueue(
    max_items=200,
    sql_db_path=str(os.getenv("SUBSTRATE_REVIEW_QUEUE_SQL_DB_PATH", "")).strip() or None,
    postgres_url=_resolve_control_plane_postgres_url(),
)
SUBSTRATE_REVIEW_TELEMETRY_STORE = GraphReviewTelemetryRecorder(
    max_records=2000,
    sql_db_path=str(os.getenv("SUBSTRATE_REVIEW_TELEMETRY_SQL_DB_PATH", "")).strip() or None,
    postgres_url=_resolve_control_plane_postgres_url(),
)
SUBSTRATE_SEMANTIC_STORE = build_substrate_store_from_env()
SUBSTRATE_POLICY_STORE = build_substrate_policy_store_from_env()
SUBSTRATE_POLICY_COMPARISON = SubstratePolicyComparisonService(
    policy_store=SUBSTRATE_POLICY_STORE,
    telemetry_recorder=SUBSTRATE_REVIEW_TELEMETRY_STORE,
)
SUBSTRATE_REVIEW_RUNTIME_EXECUTOR = GraphReviewRuntimeExecutor(
    queue=SUBSTRATE_REVIEW_QUEUE_STORE,
    consolidation_evaluator=GraphConsolidationEvaluator(store=SUBSTRATE_SEMANTIC_STORE),
    scheduler=GraphReviewScheduler(
        queue=SUBSTRATE_REVIEW_QUEUE_STORE,
        policy_profiles=SUBSTRATE_POLICY_STORE,
    ),
    telemetry_recorder=SUBSTRATE_REVIEW_TELEMETRY_STORE,
    policy_profiles=SUBSTRATE_POLICY_STORE,
)
SUBSTRATE_REVIEW_BOOTSTRAPPER = GraphReviewBootstrapper(
    scheduler=SUBSTRATE_REVIEW_RUNTIME_EXECUTOR.scheduler,
    semantic_store=SUBSTRATE_SEMANTIC_STORE,
)


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"

class AttentionAckRequest(BaseModel):
    ack_type: str = Field("seen")
    note: Optional[str] = None


class ChatMessageReceiptRequest(BaseModel):
    session_id: str
    receipt_type: str = Field("opened")


class PreferencesResolveProxyRequest(BaseModel):
    recipient_group: str
    event_kind: str
    severity: str


class WorkflowScheduleUpdateRequest(BaseModel):
    run_at_utc: Optional[str] = None
    cadence: Optional[str] = None
    day_of_week: Optional[int] = None
    hour_local: Optional[int] = None
    minute_local: Optional[int] = None
    timezone: Optional[str] = None
    notify_on: Optional[str] = None
    expected_revision: Optional[int] = None


class SubstrateReviewExecuteRequest(BaseModel):
    explicit_queue_item_id: Optional[str] = None


class SubstrateReviewSmokeCheckRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)


class SubstrateReviewBootstrapRequest(BaseModel):
    limit: int = Field(default=12, ge=1, le=32)


class SubstratePolicyAdoptHubRequest(BaseModel):
    """Hub-facing adopt payload; defaults match Substrate Inspector operator_review cycles.

    Empty ``target_zones`` applies the profile to every review zone (see
    ``SubstratePolicyProfileStore._matches_scope``); a single zone caused
    autonomy_graph/world_ontology runs to stay on baseline while Postgres
    still showed an active profile for concept_graph only.
    """

    model_config = ConfigDict(extra="forbid")

    activate_now: bool = True
    operator_id: Optional[str] = None
    rationale: str = "substrate-inspector"
    invocation_surfaces: list[GraphReviewRuntimeSurfaceV1] = Field(default_factory=lambda: ["operator_review"])
    target_zones: list[FrontierTargetZoneV1] = Field(default_factory=list)
    operator_only: bool = False
    policy_overrides: SubstratePolicyOverridesV1 = Field(default_factory=SubstratePolicyOverridesV1)


async def _execute_workflow_schedule_management(*, session_id: str, user_id: Optional[str], request: WorkflowScheduleManageRequestV1) -> Dict[str, Any]:
    from .main import cortex_client

    if cortex_client is None:
        raise RuntimeError("Cortex client unavailable")

    corr_id = str(uuid4())
    chat_req = CortexChatRequest(
        prompt="workflow schedule management",
        mode="brain",
        route_intent="none",
        session_id=session_id,
        user_id=user_id,
        trace_id=corr_id,
        metadata={"workflow_schedule_management": request.model_dump(mode="json")},
    )
    resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=corr_id)
    metadata = resp.cortex_result.metadata if isinstance(resp.cortex_result.metadata, dict) else {}
    payload = metadata.get("workflow_schedule_management") if isinstance(metadata.get("workflow_schedule_management"), dict) else {}
    if payload:
        try:
            return WorkflowScheduleManageResponseV1.model_validate(payload).model_dump(mode="json")
        except Exception:
            return payload
    return {
        "ok": bool(resp.cortex_result.ok),
        "operation": request.operation,
        "request_id": request.request_id,
        "message": resp.final_text or "No schedule response payload.",
        "schedules": [],
        "history": [],
        "ambiguous": False,
        "error_code": "invalid_management_payload",
        "error_details": {},
    }


async def _proxy_request(request: Request, base_url: str, path: str) -> Response:
    url = f"{base_url.rstrip('/')}/{path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length"}}
    body = await request.body()
    timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method,
            url,
            params=dict(request.query_params),
            data=body if body else None,
            headers=headers,
        ) as response:
            payload = await response.read()
            content_type = response.headers.get("content-type", "application/json")
            return Response(content=payload, status_code=response.status, media_type=content_type)


async def _fetch_landing_pad(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = settings.LANDING_PAD_URL.rstrip("/")
    url = f"{base_url}{path}"
    timeout = aiohttp.ClientTimeout(total=settings.LANDING_PAD_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


async def _fetch_social_memory(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = settings.SOCIAL_MEMORY_BASE_URL.rstrip("/")
    url = f"{base_url}{path}"
    timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


def _normalize_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _log_hub_route_decision(
    *,
    corr_id: str,
    session_id: str,
    route_debug: Dict[str, Any],
    user_prompt: str,
) -> None:
    emitted_mode = route_debug.get("mode")
    emitted_verb = route_debug.get("verb")
    effective_verb = emitted_verb
    if not effective_verb and emitted_mode not in {"agent", "council"}:
        effective_verb = "chat_general"
    summary = {
        "corr_id": corr_id,
        "session_id": session_id,
        "selected_ui_route": route_debug.get("selected_ui_route"),
        "emitted_mode": emitted_mode,
        "emitted_verb": emitted_verb,
        "effective_verb": effective_verb,
        "emitted_options": route_debug.get("options") or {},
        "packs": route_debug.get("packs") or [],
        "force_agent_chain": bool(route_debug.get("force_agent_chain")),
        "supervised": bool(route_debug.get("supervised")),
        "diagnostic": bool(route_debug.get("diagnostic")),
        "last_user_head": (user_prompt or "")[:120],
    }
    logger.info("hub_route_egress %s", json.dumps(summary, sort_keys=True, default=str))


def _rec_tape_req(
    *,
    corr_id: str,
    session_id: str,
    mode: str,
    use_recall: bool,
    recall_profile: Optional[str],
    user_head: str,
    no_write: bool,
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    logger.info(
        "REC_TAPE REQ corr_id=%s sid=%s mode=%s recall=%s profile=%s user_head=%r no_write=%s",
        corr_id,
        session_id,
        mode,
        use_recall,
        recall_profile,
        user_head,
        no_write,
    )


def _rec_tape_rsp(
    *,
    corr_id: str,
    memory_used: bool,
    recall_count: int,
    backend_counts: Dict[str, Any] | None,
    memory_digest: Optional[str],
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    digest_chars = len(memory_digest or "")
    logger.info(
        "REC_TAPE RSP corr_id=%s memory_used=%s digest_chars=%s recall_count=%s backend_counts=%s",
        corr_id,
        memory_used,
        digest_chars,
        recall_count,
        backend_counts or {},
    )

# ======================================================================
# 🏠 ROOT + STATIC HTML
# ======================================================================
@router.get("/")
async def root():
    """Serves the main Hub UI (index.html)."""
    from .main import html_content
    return HTMLResponse(
        content=html_content,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": settings.SERVICE_NAME}


def _runtime_identity() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
        "git_sha": os.getenv("GIT_SHA") or os.getenv("SOURCE_COMMIT") or "unknown",
        "build_timestamp": os.getenv("BUILD_TIMESTAMP") or "unknown",
        "environment": os.getenv("ORION_ENV") or os.getenv("ENVIRONMENT") or "unknown",
        "process_started_at": PROCESS_STARTED_AT_UTC.isoformat(),
        "now_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/debug/build")
def api_debug_build():
    return {
        "hub": _runtime_identity(),
        "downstream": {
            "cortex_gateway_request_channel": settings.CORTEX_GATEWAY_REQUEST_CHANNEL,
            "notify_base_url": settings.NOTIFY_BASE_URL,
            "landing_pad_url": settings.LANDING_PAD_URL,
        },
    }



@router.get("/api/service-logs/services")
def api_service_logs_services() -> Dict[str, Any]:
    return collect_service_inventory()

@router.get("/api/notifications")
async def api_notifications(limit: int = 50):
    from .main import notification_cache
    if not notification_cache:
        return []
    return await notification_cache.get_latest(limit)


@router.get("/api/presence")
def api_presence():
    from .main import presence_state
    if not presence_state:
        return {"active": False, "last_seen": None, "active_connections": 0}
    return presence_state.snapshot()


@router.get("/api/notify/recipients")
def api_notify_recipients():
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(f"{settings.NOTIFY_BASE_URL}/recipients", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch recipient profiles: %s", exc)
        return []


@router.get("/api/notify/recipients/{recipient_group}")
def api_notify_recipient(recipient_group: str):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch recipient profile %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to fetch recipient profile") from exc


@router.api_route("/api/topic-foundry/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_topic_foundry(path: str, request: Request) -> Response:
    if not settings.TOPIC_FOUNDRY_BASE_URL:
        raise HTTPException(status_code=400, detail="Topic Foundry base URL not configured")
    try:
        return await _proxy_request(request, settings.TOPIC_FOUNDRY_BASE_URL, path)
    except aiohttp.ClientError as exc:
        logger.warning("Topic Foundry proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Topic Foundry proxy request failed") from exc


@router.get("/api/social-memory/inspection")
async def api_social_memory_inspection(
    platform: str = Query(...),
    room_id: str = Query(...),
    participant_id: str | None = Query(None),
):
    if not settings.SOCIAL_MEMORY_BASE_URL:
        raise HTTPException(status_code=400, detail="Social memory base URL not configured")
    try:
        return await _fetch_social_memory(
            "/inspection",
            {"platform": platform, "room_id": room_id, "participant_id": participant_id},
        )
    except aiohttp.ClientResponseError as exc:
        detail = exc.message or "Social memory inspection request failed"
        raise HTTPException(status_code=exc.status or 502, detail=detail) from exc
    except aiohttp.ClientError as exc:
        logger.warning("Social memory inspection proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Social memory inspection request failed") from exc


@router.put("/api/notify/recipients/{recipient_group}")
def api_notify_recipient_update(recipient_group: str, payload: RecipientProfileUpdate):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.put(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}",
            json=payload.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to update recipient profile %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to update recipient profile") from exc


@router.get("/api/notify/recipients/{recipient_group}/preferences")
def api_notify_preferences(recipient_group: str):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}/preferences",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch preferences for %s: %s", recipient_group, exc)
        return []


@router.put("/api/notify/recipients/{recipient_group}/preferences")
def api_notify_preferences_update(recipient_group: str, payload: NotificationPreferencesUpdate):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.put(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}/preferences",
            json=payload.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to update preferences for %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to update preferences") from exc


@router.post("/api/notify/preferences/resolve")
def api_notify_preferences_resolve(payload: PreferencesResolveProxyRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    req = PreferenceResolutionRequest(
        recipient_group=payload.recipient_group,
        event_kind=payload.event_kind,
        severity=payload.severity,
    )
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/preferences/resolve",
            json=req.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to resolve preferences: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to resolve preferences") from exc


@router.get("/api/attention")
def api_attention(limit: int = 50, status: Optional[str] = None):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/attention",
            params={"limit": limit, "status": status},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch attention list: %s", exc)
        return []


@router.post("/api/attention/{attention_id}/ack")
def api_attention_ack(attention_id: str, payload: AttentionAckRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    ack = ChatAttentionAck(attention_id=attention_id, ack_type=payload.ack_type, note=payload.note)
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/attention/{attention_id}/ack",
            json=ack.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to acknowledge attention %s: %s", attention_id, exc)
        raise HTTPException(status_code=502, detail="Failed to acknowledge attention") from exc


@router.get("/api/chat/messages")
def api_chat_messages(limit: int = 50, status: Optional[str] = None, session_id: Optional[str] = None):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/chat/messages",
            params={"limit": limit, "status": status, "session_id": session_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch chat messages: %s", exc)
        return []


@router.post("/api/chat/message/{message_id}/receipt")
def api_chat_message_receipt(message_id: str, payload: ChatMessageReceiptRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    receipt = ChatMessageReceipt(
        message_id=message_id,
        session_id=payload.session_id,
        receipt_type=payload.receipt_type,
    )
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/chat/message/{message_id}/receipt",
            json=receipt.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to send chat message receipt %s: %s", message_id, exc)
        raise HTTPException(status_code=502, detail="Failed to acknowledge chat message") from exc

@router.post("/api/chat/response-feedback")
async def api_chat_response_feedback(payload: Dict[str, Any]):
    from .main import bus

    if not bus or not getattr(bus, "enabled", False):
        raise HTTPException(status_code=503, detail="Bus unavailable")

    try:
        payload_data = dict(payload or {})
        payload_data.setdefault("source", "hub_ui")
        payload_data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        feedback_payload = ChatResponseFeedbackV1.model_validate(payload_data)
    except Exception as exc:
        logger.warning(
            "chat_response_feedback_rejected reason=%s feedback_id=%s corr=%s sid=%s",
            exc,
            (payload or {}).get("feedback_id"),
            (payload or {}).get("target_correlation_id"),
            (payload or {}).get("session_id"),
        )
        raise HTTPException(status_code=422, detail=f"Invalid feedback payload: {exc}") from exc

    env = build_chat_response_feedback_envelope(
        feedback_payload=feedback_payload,
        correlation_id=feedback_payload.target_correlation_id,
    )
    await publish_chat_response_feedback(bus, env)
    return {"ok": True, "feedback_id": feedback_payload.feedback_id}


@router.get("/api/chat/response-feedback/options")
def api_chat_response_feedback_options() -> Dict[str, Any]:
    return {
        "feedback_values": ["up", "down"],
        "categories": build_feedback_category_options(),
        "max_free_text_chars": 2000,
    }


# ======================================================================
# 🧠 SESSION MANAGEMENT
# ======================================================================
@router.get("/api/session")
async def api_session(x_orion_session_id: Optional[str] = Header(None)):
    """
    Called by Hub UI on load.
    Always returns a warm-started session_id.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}

# ======================================================================
# 📚 COGNITION LIBRARY (Verbs & Packs)
# ======================================================================
@router.get("/api/cognition/library")
def get_cognition_library():
    """
    Returns the scanned list of Packs and Verbs available in the system.
    Used by the UI to populate dropdowns and filters.
    """
    return scan_cognition_library()



@router.get("/api/verbs")
def api_verbs(include_inactive: int = Query(default=0, ge=0, le=1)):
    include = bool(include_inactive)
    return {"verbs": build_verb_list(node_name=settings.NODE_NAME, include_inactive=include)}

# ======================================================================
# 💬 SHARED CHAT CORE (HTTP + WS)
# ======================================================================

async def handle_chat_request(
    cortex_client,
    payload: dict,
    session_id: str,
    no_write: bool,
) -> Dict[str, Any]:
    """
    Core chat handler used by both HTTP /api/chat and (optionally) WebSocket.
    Delegate strict typed requests to orion-cortex-gateway via Bus.
    """
    user_messages = payload.get("messages", [])

    # Respect client toggle; default to True if missing
    raw_recall = payload.get("use_recall", None)
    use_recall = _normalize_bool(raw_recall, default=True)

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    logger.info(
        "HTTP Chat Request payload session_id=%s messages_len=%s last_user_len=%s last_user_head=%r",
        session_id,
        len(user_messages),
        len(user_prompt or ""),
        (user_prompt or "")[:120],
    )

    inactive = validate_single_verb_override(payload, node_name=settings.NODE_NAME)
    if inactive:
        return inactive

    corr_id = str(uuid4())
    context_turns = int(payload.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
    continuity_messages = build_continuity_messages(
        history=user_messages,
        latest_user_prompt=user_prompt,
        turns=context_turns,
    )
    routed_payload = dict(payload)
    routed_payload["no_write"] = bool(no_write)
    req, route_debug, _ = build_chat_request(
        payload=routed_payload,
        session_id=session_id,
        user_id=payload.get("user_id"),
        trace_id=None,
        default_mode="brain",
        auto_default_enabled=bool(settings.HUB_AUTO_DEFAULT_ENABLED),
        source_label="hub_http",
        prompt=user_prompt,
        messages=continuity_messages,
    )
    workflow_request = req.metadata.get("workflow_request") if isinstance(req.metadata, dict) else None
    execution_policy = workflow_request.get("execution_policy") if isinstance(workflow_request, dict) else None
    logger.info(
        "workflow_resolution_result %s",
        json.dumps(
            {
                "correlation_id": corr_id,
                "matched_workflow_id": (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                "fallback_route": route_debug.get("fallback_route"),
                "reason": route_debug.get("workflow_resolution_reason"),
            },
            sort_keys=True,
            default=str,
        ),
    )
    logger.info(
        "hub_workflow_request corr=%s sid=%s workflow_id=%s invocation_mode=%s schedule_kind=%s",
        corr_id,
        session_id,
        (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
        (execution_policy or {}).get("invocation_mode") if isinstance(execution_policy, dict) else None,
        ((execution_policy or {}).get("schedule") or {}).get("kind") if isinstance(execution_policy, dict) else None,
    )

    recall_payload = req.recall or {"enabled": use_recall}
    mode = req.mode

    logger.info(f"Chat Request recall config: {recall_payload} session_id={session_id}")

    diagnostic = bool(payload.get("diagnostic") or (isinstance(payload.get("options"), dict) and payload.get("options", {}).get("diagnostic")))
    if diagnostic:
        logger.info("HTTP outbound CortexChatRequest corr=%s payload=%s", corr_id, req.model_dump(mode="json"))
    logger.info(
        "hub_egress corr=%s sid=%s mode=%s verb=%s route_intent=%s allowed_verbs=%s packs=%s",
        corr_id,
        session_id,
        req.mode,
        req.verb,
        (req.options or {}).get("route_intent") or "none",
        len(((req.options or {}).get("allowed_verbs") or [])),
        req.packs or [],
    )
    logger.info(
        "hub_context_messages corr=%s sid=%s mode=%s count=%s roles=%s",
        corr_id,
        session_id,
        req.mode,
        len(req.messages or []),
        [m.role if hasattr(m, "role") else m.get("role") for m in (req.messages or [])][:12],
    )
    _log_hub_route_decision(
        corr_id=corr_id,
        session_id=session_id,
        route_debug=route_debug,
        user_prompt=user_prompt,
    )

    _rec_tape_req(
        corr_id=corr_id,
        session_id=session_id,
        mode=mode,
        use_recall=use_recall,
        recall_profile=recall_payload.get("profile"),
        user_head=(user_prompt or "")[:80],
        no_write=no_write,
    )
    try:
        # Call Bus RPC - Hub/Client generates correlation_id internally for RPC
        resp: CortexChatResult = await cortex_client.chat(req, correlation_id=corr_id)

        # Map raw result for UI debug (HTTP + WS clients use this to coalesce display text)
        raw_result = resp.cortex_result.model_dump(mode="json")

        # Extract Text — prefer longest answer if top-level vs nested diverge (parity with curl `raw.final_text`)
        text = hub_effective_chat_text(resp)

        # Use the correlation_id from the response (gateway) if available
        # or it might be passed back from the client logic if modified to do so.
        # Here we rely on CortexChatResult having it.
        correlation_id = resp.cortex_result.correlation_id or corr_id

        memory_digest = None
        recall_debug = None
        agent_trace = None
        workflow = None
        autonomy_payload = {}
        metacog_traces = []
        if resp.cortex_result and isinstance(resp.cortex_result.recall_debug, dict):
            recall_debug = resp.cortex_result.recall_debug
            memory_digest = recall_debug.get("memory_digest")
        agent_trace = extract_agent_trace_payload(resp.cortex_result)
        raw_traces = getattr(resp.cortex_result, "metacog_traces", None)
        if isinstance(raw_traces, list):
            metacog_traces = [t for t in raw_traces if isinstance(t, dict)]
        logger.info(
            "hub_metacog_received corr=%s source=http traces=%s",
            correlation_id,
            len(metacog_traces),
        )
        workflow = extract_workflow_payload(resp.cortex_result)
        autonomy_payload = extract_autonomy_payload(resp.cortex_result)
        if isinstance(workflow, dict):
            logger.info(
                "hub_workflow_response corr=%s workflow_id=%s status=%s scheduled_count=%s persisted_count=%s rendered_path=%s",
                correlation_id,
                workflow.get("workflow_id"),
                workflow.get("status"),
                len(workflow.get("scheduled") or []),
                len(workflow.get("persisted") or []),
                "scheduled_confirmation" if len(workflow.get("scheduled") or []) else "immediate_or_unscheduled",
            )

        recall_count = 0
        backend_counts = None
        if isinstance(recall_debug, dict):
            recall_count = int(recall_debug.get("count") or 0)
            backend_counts = recall_debug.get("backend_counts")
            if backend_counts is None and isinstance(recall_debug.get("debug"), dict):
                backend_counts = recall_debug["debug"].get("backend_counts")
        memory_used = bool(getattr(resp.cortex_result, "memory_used", False))
        if not memory_used:
            memory_used = bool(recall_count)
        logger.info(
            "hub_ingress_result corr=%s sid=%s mode=%s status=%s final_len=%s memory_used=%s recall_count=%s",
            correlation_id,
            session_id,
            mode,
            getattr(resp.cortex_result, "status", None),
            len(text or ""),
            memory_used,
            recall_count,
        )
        _rec_tape_rsp(
            corr_id=str(correlation_id),
            memory_used=memory_used,
            recall_count=recall_count,
            backend_counts=backend_counts,
            memory_digest=memory_digest,
        )

        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "text": text,
            "tokens": len(text.split()), # simple approx
            "raw": raw_result,
            "recall_debug": recall_debug,
            "agent_trace": agent_trace,
            "workflow": workflow,
            "memory_used": memory_used,
            "memory_digest": memory_digest,
            "no_write": no_write,
            "spark_meta": None,
            "correlation_id": correlation_id,
            "routing_debug": route_debug,
            "metacog_traces": metacog_traces,
            **autonomy_payload,
        }

    except Exception as e:
        logger.error(f"Chat RPC failed: {e}", exc_info=True)
        return {"error": str(e)}


# ======================================================================
# 💬 CHAT ENDPOINT (HTTP wrapper around core)
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
    x_orion_no_write: Optional[str] = Header(None),
):
    """
    Main LLM chat endpoint.
    Delegates to Cortex Gateway via Bus RPC.
    """
    from .main import bus, cortex_client
    if not bus or not cortex_client:
        raise RuntimeError("OrionBus/Client not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)
    no_write = _normalize_bool(payload.get("no_write"), default=False) or _normalize_bool(
        x_orion_no_write, default=False
    )

    # Core chat handling
    result = await handle_chat_request(cortex_client, payload, session_id, no_write)

    # ─────────────────────────────────────────────
    # 📡 Publish HTTP chat → chat history log
    # ─────────────────────────────────────────────
    text = result.get("text")
    correlation_id = result.get("correlation_id")

    if text and getattr(bus, "enabled", False) and not no_write:
        try:
            user_messages = payload.get("messages", [])
            latest_user_prompt = ""
            if isinstance(user_messages, list) and user_messages:
                latest_user_prompt = user_messages[-1].get("content", "") or ""

            use_recall = bool(result.get("use_recall", True))

            # If we didn't get a correlation_id from gateway, fallback to new UUID
            # (but ideally we got it).
            final_corr_id = correlation_id or str(uuid4())

            metacog_traces = result.get("metacog_traces") or []
            reasoning_content = (
                result.get("reasoning_content")
                or ((result.get("raw") or {}).get("reasoning_content") if isinstance(result.get("raw"), dict) else None)
            )
            selected_reasoning_trace, _ = select_reasoning_trace_for_history(
                correlation_id=final_corr_id,
                reasoning_trace=result.get("reasoning_trace"),
                metacog_traces=metacog_traces if isinstance(metacog_traces, list) else None,
                reasoning_content=reasoning_content,
                session_id=session_id,
                message_id=f"{final_corr_id}:assistant",
                model=(settings.GATEWAY_MODEL if hasattr(settings, "GATEWAY_MODEL") else None),
            )

            envelopes = []
            social_meta = {}
            if is_social_room_payload(payload):
                social_meta = social_room_client_meta(
                    payload=payload,
                    route_debug=result.get("routing_debug") or {},
                    trace_verb=(result.get("routing_debug") or {}).get("verb"),
                    memory_digest=result.get("memory_digest"),
                )
            if latest_user_prompt:
                envelopes.append(
                    build_chat_history_envelope(
                        content=latest_user_prompt,
                        role="user",
                        session_id=session_id,
                        correlation_id=final_corr_id,
                        speaker=payload.get("user_id") or "user",
                        tags=[result.get("mode", "brain")],
                        client_meta=social_meta or None,
                    )
                )
            envelopes.append(
                build_chat_history_envelope(
                    content=text,
                    role="assistant",
                    session_id=session_id,
                    correlation_id=final_corr_id,
                    speaker=settings.SERVICE_NAME,
                    tags=[result.get("mode", "brain")],
                    client_meta=social_meta or None,
                    reasoning_trace=selected_reasoning_trace,
                )
            )
            await publish_chat_history(bus, envelopes)
            if social_meta:
                await publish_social_room_turn(
                    bus,
                    prompt=latest_user_prompt,
                    response=text,
                    session_id=session_id,
                    correlation_id=final_corr_id,
                    user_id=payload.get("user_id"),
                    source_label="hub_http",
                    recall_profile=((result.get("routing_debug") or {}).get("recall_profile")),
                    trace_verb=(result.get("routing_debug") or {}).get("verb"),
                    client_meta=social_meta,
                    memory_digest=result.get("memory_digest"),
                )
            if isinstance(metacog_traces, list):
                for trace in metacog_traces:
                    if not isinstance(trace, dict):
                        continue
                    if _thought_debug_enabled():
                        logger.info(
                            "THOUGHT_DEBUG_METACOG_PUB stage=hub_http_prepare corr=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r",
                            final_corr_id,
                            trace.get("trace_role") or trace.get("role"),
                            trace.get("trace_stage") or trace.get("stage"),
                            trace.get("model"),
                            _debug_len(trace.get("content")),
                            _debug_snippet(trace.get("content")),
                        )
                    trace_env = BaseEnvelope(
                        kind="metacognitive.trace.v1",
                        source=ServiceRef(
                            name=settings.SERVICE_NAME,
                            node=settings.NODE_NAME,
                            version=settings.SERVICE_VERSION,
                        ),
                        correlation_id=final_corr_id,
                        payload=trace,
                    )
                    await bus.publish("orion:metacog:trace", trace_env)
                if _thought_debug_enabled() and not any(isinstance(t, dict) for t in metacog_traces):
                    logger.info("THOUGHT_DEBUG_METACOG_PUB stage=hub_http_skipped corr=%s reason=no_valid_trace_dicts", final_corr_id)
                logger.info(
                    "hub_metacog_published corr=%s source=http channel=%s traces=%s",
                    final_corr_id,
                    "orion:metacog:trace",
                    len([t for t in metacog_traces if isinstance(t, dict)]),
                )

            # Legacy log for downstream SQL-writer compatibility
            chat_log_payload = {
                "correlation_id": final_corr_id,
                "source": settings.SERVICE_NAME,
                "prompt": latest_user_prompt,
                "response": text,
                "session_id": session_id,
                "mode": result.get("mode", "brain"),
                "recall": use_recall,
                "user_id": None,
                "spark_meta": None,
                "reasoning_trace": selected_reasoning_trace,
            }
            if _thought_debug_enabled():
                reasoning_trace = chat_log_payload.get("reasoning_trace")
                reasoning_content = chat_log_payload.get("reasoning_content")
                thought_candidate = (
                    (reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None)
                    or reasoning_content
                )
                logger.info(
                    "THOUGHT_DEBUG_HUB stage=legacy_chat_history_publish corr=%s channel=%s target=chat.history.turn_payload reasoning_trace_exists=%s reasoning_content_exists=%s thought_candidate_len=%s thought_candidate_snippet=%r",
                    final_corr_id,
                    settings.chat_history_channel,
                    isinstance(reasoning_trace, dict),
                    bool(str(reasoning_content or "").strip()),
                    _debug_len(thought_candidate),
                    _debug_snippet(thought_candidate),
                )

            await bus.publish(
                settings.chat_history_channel,
                chat_log_payload,
            )
        except Exception as e:
            logger.warning(
                "Failed to publish HTTP chat to chat history log: %s",
                e,
                exc_info=True,
            )

    return result


@router.get("/api/workflow/schedules")
async def api_workflow_schedule_list(
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
    include_history: bool = Query(default=False),
):
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    req = WorkflowScheduleManageRequestV1(
        operation="list",
        request_id=str(uuid4()),
        include_history=include_history,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)


@router.get("/api/workflow/schedules/{schedule_id}/history")
async def api_workflow_schedule_history(
    schedule_id: str,
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
):
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    req = WorkflowScheduleManageRequestV1(
        operation="history",
        request_id=str(uuid4()),
        schedule_id=schedule_id,
        include_history=True,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)


@router.post("/api/workflow/schedules/{schedule_id}/{operation}")
async def api_workflow_schedule_action(
    schedule_id: str,
    operation: str,
    payload: Optional[WorkflowScheduleUpdateRequest] = None,
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
):
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation not in {"cancel", "pause", "resume", "update"}:
        raise HTTPException(status_code=400, detail="unsupported_operation")

    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    patch = payload.model_dump(exclude_none=True) if payload is not None else None
    req = WorkflowScheduleManageRequestV1(
        operation=normalized_operation,
        request_id=str(uuid4()),
        schedule_id=schedule_id,
        patch=patch,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)

# ======================================================================
# 📿 COLLAPSE MIRROR ENDPOINTS
# ======================================================================
@router.get("/schema/collapse")
def get_collapse_schema():
    """Exposes the CollapseMirrorEntry schema for UI templating."""
    logger.info("Fetching CollapseMirrorEntry schema")
    return JSONResponse(CollapseMirrorEntry.schema())


@router.post("/submit-collapse")
async def submit_collapse(data: dict):
    """
    Receives Collapse Mirror data and publishes it to the bus.
    """
    from .main import bus
    logger.info(f"🔥 /submit-collapse called with: {data}")

    if not bus or not bus.enabled:
        logger.error("Submission failed: OrionBus is disabled or not connected.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "OrionBus disabled or unavailable"},
        )

    try:
        # Normalize legacy ServiceRef objects coming from the UI.
        # Some clients still send {"service": "hub", "node": "..."}
        # but our canonical model uses {"name": "...", "node": "..."}.
        src = data.get("source")
        if isinstance(src, dict) and "name" not in src and "service" in src:
            src = dict(src)
            src["name"] = src.pop("service")
            data["source"] = src

        entry = CollapseMirrorEntry(**data).with_defaults()

        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        # Note: we do NOT explicitly set correlation_id here.
        # BaseEnvelope will generate a random one, but our worker heuristic (empty causality chain)
        # will treat it as ad-hoc and not persist it to DB.
        env = BaseEnvelope(
            kind="collapse.submit"
          , source=ServiceRef(name="hub", node=settings.NODE_NAME)
          , payload=entry.model_dump(mode="json")
        )

        await bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, env)
        logger.info(
            "📡 Published Collapse Mirror → %s",
            settings.CHANNEL_COLLAPSE_INTAKE,
        )

        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )

@router.get("/api/debug/spark-last")
async def api_debug_spark_last(
    x_orion_session_id: Optional[str] = Header(None),
):
    # Legacy debug endpoint - likely broken but kept safe
    return JSONResponse(
        {
            "session_id": x_orion_session_id,
            "spark_meta": None,
            "note": "Spark debug deprecated in dumb hub.",
        }
    )


def _source_meta(*, kind: str, degraded: bool, limit: int, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kind": kind,
        "degraded": degraded,
        "error": error,
        "query": {"limit": limit},
        "last_refreshed": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_slice(result) -> Dict[str, Any]:
    return {
        "nodes": [node.model_dump(mode="json") for node in result.slice.nodes],
        "edges": [edge.model_dump(mode="json") for edge in result.slice.edges],
        "truncated": bool(result.truncated),
        "limits": dict(result.limits or {}),
        "query_kind": result.query_kind,
        "generated_at": result.generated_at,
        "details": dict(result.details or {}),
    }


def _graphdb_overview_payload(*, limit: int = 10) -> Dict[str, Any]:
    hotspot = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=limit, limit_edges=max(20, limit * 2))
    contradiction = SUBSTRATE_SEMANTIC_STORE.query_contradiction_region(limit_nodes=limit, limit_edges=max(20, limit * 2))
    concept = SUBSTRATE_SEMANTIC_STORE.query_concept_region(limit_nodes=limit, limit_edges=max(20, limit * 2))

    source_kind = hotspot.source_kind
    degraded = bool(hotspot.degraded or contradiction.degraded or concept.degraded)
    error = hotspot.error or contradiction.error or concept.error

    return {
        "source": {
            **_source_meta(kind=source_kind, degraded=degraded, limit=limit, error=error),
            "query_kind": "overview",
            "semantic_backend": type(SUBSTRATE_SEMANTIC_STORE).__name__,
            "truncated": bool(hotspot.truncated or contradiction.truncated or concept.truncated),
        },
        "data": {
            "coherence": None,
            "identity_conflict": None,
            "goal_pressure": None,
            "concept_drift": None,
            "contradiction_count": len(contradiction.slice.nodes),
            "top_hotspots": [node.model_dump(mode="json") for node in hotspot.slice.nodes[:limit]],
            "top_tensions": [node.model_dump(mode="json") for node in hotspot.slice.nodes if node.node_kind == "tension"][:limit],
            "top_stabilizers": [node.model_dump(mode="json") for node in concept.slice.nodes[:limit]],
            "metacog_brief": None,
            "regions": {
                "hotspot": _serialize_slice(hotspot),
                "contradiction": _serialize_slice(contradiction),
                "concept": _serialize_slice(concept),
            },
        },
    }


def _graphdb_hotspots_payload(*, limit: int = 20) -> Dict[str, Any]:
    hotspot = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=limit, limit_edges=max(40, limit * 2))
    contradiction = SUBSTRATE_SEMANTIC_STORE.query_contradiction_region(limit_nodes=limit, limit_edges=max(40, limit * 2))
    concept = SUBSTRATE_SEMANTIC_STORE.query_concept_region(limit_nodes=limit, limit_edges=max(40, limit * 2))

    degraded = bool(hotspot.degraded or contradiction.degraded or concept.degraded)
    error = hotspot.error or contradiction.error or concept.error

    return {
        "source": {
            **_source_meta(kind=hotspot.source_kind, degraded=degraded, limit=limit, error=error),
            "query_kind": "hotspots",
            "semantic_backend": type(SUBSTRATE_SEMANTIC_STORE).__name__,
            "truncated": bool(hotspot.truncated or contradiction.truncated or concept.truncated),
        },
        "data": {
            "active_regions": _serialize_slice(hotspot),
            "contradiction_hotspots": _serialize_slice(contradiction),
            "pressure_hotspots": _serialize_slice(hotspot),
            "drift_hotspots": _serialize_slice(concept),
        },
    }


def _sql_review_queue_payload(*, limit: int = 50) -> Dict[str, Any]:
    snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=limit)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_QUEUE_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_QUEUE_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_QUEUE_STORE.last_error(),
        ),
        "data": snapshot.model_dump(mode="json"),
    }


def _sql_review_executions_payload(*, limit: int = 50) -> Dict[str, Any]:
    records = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=limit))
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": [record.model_dump(mode="json") for record in records],
    }


def _sql_telemetry_summary_payload(*, limit: int = 200) -> Dict[str, Any]:
    summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=limit))
    recommendations = GraphReviewCalibrationAnalyzer().recommend(summary=summary)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "summary": summary.model_dump(mode="json"),
            "recommendations": [rec.model_dump(mode="json") for rec in recommendations],
        },
    }


def _sql_calibration_payload(*, limit: int = 20) -> Dict[str, Any]:
    summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=200))
    recommendations = GraphReviewCalibrationAnalyzer().recommend(summary=summary)
    baseline_policy = GraphReviewCyclePolicyV1().model_dump(mode="json")
    inspection = SUBSTRATE_POLICY_STORE.inspect(audit_limit=limit)
    actives_cal = list(inspection.active_profiles)
    if actives_cal:
        actives_cal.sort(key=lambda p: p.activated_at or p.created_at, reverse=True)
        active_profile_dump = actives_cal[0].model_dump(mode="json")
    else:
        active_profile_dump = baseline_policy
    source_kind = "postgres" if (
        SUBSTRATE_POLICY_STORE.source_kind() == "postgres"
        and SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "postgres"
    ) else SUBSTRATE_POLICY_STORE.source_kind()
    return {
        "source": _source_meta(
            kind=source_kind,
            degraded=bool(SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()),
            limit=limit,
            error=SUBSTRATE_POLICY_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "active_profile": active_profile_dump,
            "staged_profiles": [item.model_dump(mode="json") for item in inspection.staged_profiles],
            "recent_audit_events": [item.model_dump(mode="json") for item in inspection.recent_audit_events],
            "rolled_back_profiles": [item.model_dump(mode="json") for item in inspection.rolled_back_profiles],
            "advisory_recommendations": [rec.model_dump(mode="json") for rec in recommendations],
        },
    }


def _sql_policy_comparison_payload(
    *,
    pair_mode: str = "baseline_vs_active",
    baseline_profile_id: str | None = None,
    candidate_profile_id: str | None = None,
    window_seconds: int = 86400,
    sample_limit: int = 500,
) -> Dict[str, Any]:
    try:
        if not isinstance(pair_mode, str):
            pair_mode = "baseline_vs_active"
        if not isinstance(baseline_profile_id, str):
            baseline_profile_id = None
        if not isinstance(candidate_profile_id, str):
            candidate_profile_id = None
        if not isinstance(window_seconds, int):
            window_seconds = 86400
        if not isinstance(sample_limit, int):
            sample_limit = 500
        request = SubstratePolicyComparisonRequestV1(
            pair_mode=pair_mode,  # type: ignore[arg-type]
            baseline_profile_id=baseline_profile_id,
            candidate_profile_id=candidate_profile_id,
            window_seconds=window_seconds,
            sample_limit=sample_limit,
        )
        data = SUBSTRATE_POLICY_COMPARISON.compare(request=request)
        source_kind = "postgres" if (
            SUBSTRATE_POLICY_STORE.source_kind() == "postgres"
            and SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "postgres"
        ) else (
            "fallback" if (SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()) else (
                "sqlite" if (
                    SUBSTRATE_POLICY_STORE.source_kind() == "sqlite"
                    or SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "sqlite"
                ) else "sql"
            )
        )
        return {
            "source": _source_meta(
                kind=source_kind,
                degraded=bool(SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()),
                limit=sample_limit,
                error=SUBSTRATE_POLICY_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
            ),
            "data": data,
        }
    except Exception as exc:
        return {
            "source": _source_meta(kind="sql", degraded=True, limit=sample_limit, error=str(exc)),
            "data": {
                "pair": {
                    "mode": pair_mode,
                    "baseline_profile_id": baseline_profile_id,
                    "candidate_profile_id": candidate_profile_id,
                },
                "report": {
                    "verdict": "insufficient_data",
                    "confidence": 0.2,
                    "notes": [f"comparison_error:{exc}"],
                },
                "advisory": {"mutating": False, "message": "comparison failed safely"},
            },
        }


def _substrate_source_posture() -> Dict[str, Any]:
    control_kind = SUBSTRATE_REVIEW_QUEUE_STORE.source_kind()
    telemetry_kind = SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind()
    semantic_kind_raw = getattr(SUBSTRATE_SEMANTIC_STORE, "source_kind", "graphdb")
    semantic_kind = semantic_kind_raw() if callable(semantic_kind_raw) else semantic_kind_raw
    policy_kind = SUBSTRATE_POLICY_STORE.source_kind()
    control_degraded = bool(SUBSTRATE_REVIEW_QUEUE_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded())
    semantic_degraded_raw = getattr(SUBSTRATE_SEMANTIC_STORE, "degraded", False)
    semantic_degraded = bool(semantic_degraded_raw() if callable(semantic_degraded_raw) else semantic_degraded_raw)
    policy_degraded = bool(SUBSTRATE_POLICY_STORE.degraded())
    control_posture = "degraded" if control_degraded else ("postgres" if control_kind == "postgres" else "fallback")
    return {
        "control_plane": {
            "kind": control_kind,
            "telemetry_kind": telemetry_kind,
            "degraded": control_degraded,
            "posture": control_posture,
            "error": SUBSTRATE_REVIEW_QUEUE_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        },
        "semantic": {
            "kind": semantic_kind,
            "degraded": semantic_degraded,
            "posture": "degraded" if semantic_degraded else semantic_kind,
            "error": getattr(SUBSTRATE_SEMANTIC_STORE, "last_error", lambda: None)(),
        },
        "policy": {
            "kind": policy_kind,
            "degraded": policy_degraded,
            "posture": "degraded" if policy_degraded else ("postgres" if policy_kind == "postgres" else "fallback"),
            "error": SUBSTRATE_POLICY_STORE.last_error(),
        },
    }


def _parse_substrate_status_instant(value: object) -> datetime | None:
    """Parse ISO-8601 strings or datetimes for substrate status clock comparisons."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _last_execution_predates_active_profile(
    last_execution: Dict[str, Any] | None,
    active_profile: Dict[str, Any],
) -> bool | None:
    """True when the newest telemetry row is older than the active policy's activation time."""
    if not last_execution or not active_profile.get("activated_at"):
        return None
    selected = _parse_substrate_status_instant(last_execution.get("selected_at"))
    activated = _parse_substrate_status_instant(active_profile.get("activated_at"))
    if selected is None or activated is None:
        return None
    return selected < activated


def _substrate_runtime_status_payload(*, queue_limit: int = 20, telemetry_limit: int = 20) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    queue_snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=queue_limit)
    due_items = SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=queue_limit)
    recent_exec = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=1))
    telemetry_summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=telemetry_limit))
    policy_inspection = SUBSTRATE_POLICY_STORE.inspect(audit_limit=5)
    actives = list(policy_inspection.active_profiles)
    if actives:
        actives.sort(key=lambda p: p.activated_at or p.created_at, reverse=True)
        active_profile = actives[0].model_dump(mode="json")
    else:
        active_profile = {}
    last_execution = recent_exec[0].model_dump(mode="json") if recent_exec else None
    last_predates = _last_execution_predates_active_profile(last_execution, active_profile)
    next_due = min((item.next_review_at for item in due_items), default=None)

    return {
        "generated_at": now.isoformat(),
        "summary": {
            "queue_count": len(queue_snapshot.queue_items),
            "due_count": len(due_items),
            "next_due_at": next_due.isoformat() if next_due else None,
            "next_item": (
                {
                    "queue_item_id": due_items[0].queue_item_id,
                    "target_zone": due_items[0].target_zone,
                    "subject_ref": due_items[0].subject_ref,
                    "remaining_cycles": due_items[0].cycle_budget.remaining_cycles,
                    "next_review_at": due_items[0].next_review_at.isoformat(),
                }
                if due_items
                else None
            ),
            "last_execution": last_execution,
            "last_execution_predates_active_profile": last_predates,
            "telemetry_count": (
                telemetry_summary.total_executions
                + telemetry_summary.total_noops
                + telemetry_summary.total_suppressed
                + telemetry_summary.total_terminated
                + telemetry_summary.total_failed
            ),
            "policy_active_profile_id": active_profile.get("profile_id"),
            "policy_active_profile": active_profile,
        },
        "source": _substrate_source_posture(),
    }


def _execute_substrate_review_cycle(*, allow_followup: bool, explicit_queue_item_id: str | None = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    SUBSTRATE_REVIEW_QUEUE_STORE.refresh_from_storage()
    before_due = len(SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=200))
    before_count = len(SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=200).queue_items)
    request = GraphReviewRuntimeRequestV1(
        invocation_surface="operator_review",
        explicit_queue_item_id=explicit_queue_item_id,
        execute_frontier_followup_allowed=allow_followup,
        operator_override_strict_zone=False,
        max_items_to_consider=20,
    )
    result = SUBSTRATE_REVIEW_RUNTIME_EXECUTOR.execute_once(request=request, now=now)
    after_now = datetime.now(timezone.utc)
    after_due = len(SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=after_now, limit=200))
    after_count = len(SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=200).queue_items)
    return {
        "request": {
            "invocation_surface": "operator_review",
            "allow_frontier_followup": allow_followup,
            "explicit_queue_item_id": explicit_queue_item_id,
            "single_cycle": True,
        },
        "result": result.model_dump(mode="json"),
        "queue_before": {"count": before_count, "due_count": before_due},
        "queue_after": {"count": after_count, "due_count": after_due},
        "source": _substrate_source_posture(),
    }


def _bootstrap_substrate_review_frontier(*, limit: int = 12) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    SUBSTRATE_REVIEW_QUEUE_STORE.refresh_from_storage()
    execution = SUBSTRATE_REVIEW_BOOTSTRAPPER.bootstrap(now=now, query_limit=limit)
    return {
        "generated_at": now.isoformat(),
        "items_before": execution.items_before,
        "items_enqueued": execution.items_enqueued,
        "items_after": execution.items_after,
        "due_after": execution.due_after,
        "source_posture": _substrate_source_posture(),
        "audit_summary": {
            "scheduled_decision_count": execution.scheduled_decision_count,
            "semantic_source": execution.semantic_source,
            "semantic_degraded": execution.semantic_degraded,
        },
        "notes": execution.notes,
    }


def _queue_count(payload: Dict[str, Any]) -> int:
    data = payload.get("data")
    if isinstance(data, dict):
        items = data.get("queue_items")
        if isinstance(items, list):
            return len(items)
    return 0


def _classify_substrate_debug_diagnosis(
    *,
    source_posture: Dict[str, Any],
    bootstrap_payload: Dict[str, Any],
    baseline_queue_payload: Dict[str, Any],
    post_bootstrap_queue_payload: Dict[str, Any],
    execute_payload: Dict[str, Any],
    final_queue_payload: Dict[str, Any],
) -> Dict[str, Any]:
    categories: list[str] = []
    highlights: list[str] = []
    severity = "ok"

    control_plane = source_posture.get("control_plane") if isinstance(source_posture, dict) else {}
    semantic = source_posture.get("semantic") if isinstance(source_posture, dict) else {}

    control_degraded = bool(control_plane.get("degraded")) if isinstance(control_plane, dict) else False
    semantic_degraded = bool(semantic.get("degraded")) if isinstance(semantic, dict) else False

    if control_degraded:
        categories.append("control plane degraded")
        highlights.append("Control-plane storage/reporting is degraded.")
        severity = "degraded"

    if semantic_degraded:
        categories.append("semantic substrate degraded")
        highlights.append("Semantic substrate queries are degraded.")
        severity = "degraded"

    bootstrap_items_enqueued = int(bootstrap_payload.get("items_enqueued") or 0)
    bootstrap_notes = bootstrap_payload.get("notes")
    if isinstance(bootstrap_notes, list) and any("seed_skipped" in str(note) for note in bootstrap_notes):
        categories.append("likely weak seed heuristics for current graph shape")
        highlights.append("Bootstrap skipped one or more seed regions.")
        severity = "warning" if severity != "degraded" else severity

    if "items_enqueued" not in bootstrap_payload and "error" in bootstrap_payload:
        categories.append("bootstrap route/path unavailable")
        highlights.append("Bootstrap path could not be invoked.")
        severity = "degraded"
    elif bootstrap_items_enqueued <= 0:
        categories.append("bootstrap produced zero items")
        highlights.append("Bootstrap completed without queue growth.")
        severity = "warning" if severity != "degraded" else severity

    baseline_queue_count = _queue_count(baseline_queue_payload)
    post_bootstrap_queue_count = _queue_count(post_bootstrap_queue_payload)
    final_queue_count = _queue_count(final_queue_payload)

    if bootstrap_items_enqueued > 0 and post_bootstrap_queue_count <= baseline_queue_count:
        categories.append("bootstrap claimed to enqueue but queue remained empty")
        highlights.append("Bootstrap reported enqueues, but queue did not increase.")
        severity = "degraded"

    execute_result = execute_payload.get("result") if isinstance(execute_payload, dict) else {}
    execute_outcome = execute_result.get("outcome") if isinstance(execute_result, dict) else None
    execute_reason = ""
    execute_audit = execute_result.get("audit_summary") if isinstance(execute_result, dict) else {}
    if isinstance(execute_audit, dict):
        execute_reason = str(execute_audit.get("selection_reason") or "").strip().lower()
    execute_notes = execute_result.get("notes") if isinstance(execute_result, dict) else []
    execute_notes_lower = " ".join(str(note).lower() for note in execute_notes) if isinstance(execute_notes, list) else ""

    if post_bootstrap_queue_count > 0 and execute_outcome == "noop" and (
        "no eligible queue items" in execute_reason or "not due yet" in execute_reason
    ):
        categories.append("queue nonempty but execute-once still noop due to no eligible items")
        highlights.append("Queue has items, but none were eligible for this cycle.")
        severity = "warning" if severity == "ok" else severity

    if bootstrap_items_enqueued > 0 and execute_outcome == "executed":
        categories.append("bootstrap produced items and execute-once succeeded")
        highlights.append("Bootstrap seeded the queue and one cycle executed successfully.")

    if "seed_skipped" in execute_notes_lower and "likely weak seed heuristics for current graph shape" not in categories:
        categories.append("likely weak seed heuristics for current graph shape")
        highlights.append("Execution notes indicate weak substrate seed coverage.")
        severity = "warning" if severity == "ok" else severity

    summary = " | ".join(highlights) if highlights else "Debug pass completed."
    return {
        "severity": severity,
        "categories": categories,
        "summary": summary,
        "facts": {
            "baseline_queue_count": baseline_queue_count,
            "post_bootstrap_queue_count": post_bootstrap_queue_count,
            "final_queue_count": final_queue_count,
            "bootstrap_items_enqueued": bootstrap_items_enqueued,
            "execute_outcome": execute_outcome,
        },
    }


def _run_substrate_review_debug_pass(*, bootstrap_limit: int = 12) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    baseline_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)
    baseline_queue = _sql_review_queue_payload(limit=50)
    baseline_overview = _graphdb_overview_payload(limit=10)
    baseline_hotspots = _graphdb_hotspots_payload(limit=20)
    baseline_executions = _sql_review_executions_payload(limit=20)

    bootstrap = _bootstrap_substrate_review_frontier(limit=bootstrap_limit)
    post_bootstrap_queue = _sql_review_queue_payload(limit=50)
    post_bootstrap_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)

    execute_once = _execute_substrate_review_cycle(allow_followup=False)
    final_queue = _sql_review_queue_payload(limit=50)
    final_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)
    final_executions = _sql_review_executions_payload(limit=20)
    source_posture = _substrate_source_posture()
    diagnosis = _classify_substrate_debug_diagnosis(
        source_posture=source_posture,
        bootstrap_payload=bootstrap,
        baseline_queue_payload=baseline_queue,
        post_bootstrap_queue_payload=post_bootstrap_queue,
        execute_payload=execute_once,
        final_queue_payload=final_queue,
    )

    return {
        "generated_at": generated_at,
        "baseline": {
            "runtime_status": baseline_status,
            "review_queue": baseline_queue,
            "overview": baseline_overview,
            "hotspots": baseline_hotspots,
            "recent_executions": baseline_executions,
        },
        "bootstrap": bootstrap,
        "post_bootstrap": {
            "review_queue": post_bootstrap_queue,
            "runtime_status": post_bootstrap_status,
        },
        "execute_once": execute_once,
        "final": {
            "review_queue": final_queue,
            "runtime_status": final_status,
            "recent_executions": final_executions,
        },
        "diagnosis": diagnosis,
        "source_posture": source_posture,
    }


@router.get("/substrate")
async def substrate_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "substrate.html").read_text(encoding="utf-8")
    rendered = template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version())
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/substrate/overview")
def api_substrate_overview(limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    return _graphdb_overview_payload(limit=limit)


@router.get("/api/substrate/hotspots")
def api_substrate_hotspots(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    return _graphdb_hotspots_payload(limit=limit)


@router.get("/api/substrate/review-queue")
def api_substrate_review_queue(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    return _sql_review_queue_payload(limit=limit)


@router.get("/api/substrate/review-executions")
def api_substrate_review_executions(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    return _sql_review_executions_payload(limit=limit)


@router.get("/api/substrate/telemetry-summary")
def api_substrate_telemetry_summary(limit: int = Query(default=200, ge=1, le=500)) -> Dict[str, Any]:
    return _sql_telemetry_summary_payload(limit=limit)


@router.get("/api/substrate/calibration")
def api_substrate_calibration(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    return _sql_calibration_payload(limit=limit)


@router.get("/api/substrate/policy-comparison")
def api_substrate_policy_comparison(
    pair_mode: str = Query(default="baseline_vs_active"),
    baseline_profile_id: str | None = Query(default=None),
    candidate_profile_id: str | None = Query(default=None),
    window_seconds: int = Query(default=86400, ge=60, le=60 * 60 * 24 * 30),
    sample_limit: int = Query(default=500, ge=1, le=500),
) -> Dict[str, Any]:
    return _sql_policy_comparison_payload(
        pair_mode=pair_mode,
        baseline_profile_id=baseline_profile_id,
        candidate_profile_id=candidate_profile_id,
        window_seconds=window_seconds,
        sample_limit=sample_limit,
    )


@router.post("/api/substrate/policy/adopt")
def api_substrate_policy_adopt(request: SubstratePolicyAdoptHubRequest) -> Dict[str, Any]:
    """Stage or activate a substrate review policy profile (operator-controlled)."""
    adoption = SubstratePolicyAdoptionRequestV1(
        rollout_scope=SubstratePolicyRolloutScopeV1(
            invocation_surfaces=list(request.invocation_surfaces),
            target_zones=list(request.target_zones),
            operator_only=request.operator_only,
        ),
        policy_overrides=request.policy_overrides,
        activate_now=request.activate_now,
        operator_id=request.operator_id,
        rationale=request.rationale,
    )
    result = SUBSTRATE_POLICY_STORE.adopt(adoption)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_POLICY_STORE.source_kind(),
            degraded=SUBSTRATE_POLICY_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_POLICY_STORE.last_error(),
        ),
        "data": result.model_dump(mode="json"),
    }


@router.get("/api/substrate/review-runtime/status")
def api_substrate_review_runtime_status(
    queue_limit: int = Query(default=20, ge=1, le=200),
    telemetry_limit: int = Query(default=20, ge=1, le=500),
) -> Dict[str, Any]:
    return _substrate_runtime_status_payload(queue_limit=queue_limit, telemetry_limit=telemetry_limit)


@router.post("/api/substrate/review-runtime/execute-once")
def api_substrate_review_runtime_execute_once(request: SubstrateReviewExecuteRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewExecuteRequest()
    return _execute_substrate_review_cycle(
        allow_followup=False,
        explicit_queue_item_id=req.explicit_queue_item_id,
    )


@router.post("/api/substrate/review-runtime/execute-once-followup")
def api_substrate_review_runtime_execute_once_followup(request: SubstrateReviewExecuteRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewExecuteRequest()
    return _execute_substrate_review_cycle(
        allow_followup=True,
        explicit_queue_item_id=req.explicit_queue_item_id,
    )


@router.post("/api/substrate/review-runtime/bootstrap")
def api_substrate_review_runtime_bootstrap(request: SubstrateReviewBootstrapRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewBootstrapRequest()
    return _bootstrap_substrate_review_frontier(limit=req.limit)


@router.post("/api/substrate/review-runtime/debug-run")
def api_substrate_review_runtime_debug_run(request: SubstrateReviewBootstrapRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewBootstrapRequest()
    return _run_substrate_review_debug_pass(bootstrap_limit=req.limit)


@router.post("/api/substrate/review-runtime/smoke-check")
def api_substrate_review_runtime_smoke_check(request: SubstrateReviewSmokeCheckRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewSmokeCheckRequest()
    now = datetime.now(timezone.utc)
    queue_snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=req.limit)
    due_items = SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=req.limit)
    semantic_probe = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=min(5, req.limit), limit_edges=max(10, req.limit * 2))
    source = _substrate_source_posture()
    return {
        "generated_at": now.isoformat(),
        "source": source,
        "checks": {
            "queue_available": len(queue_snapshot.queue_items) >= 0,
            "runtime_eligible": len(due_items) > 0,
            "semantic_available": not bool(semantic_probe.degraded),
            "control_plane_available": not bool(source["control_plane"]["degraded"]),
        },
        "summary": {
            "queue_count": len(queue_snapshot.queue_items),
            "due_count": len(due_items),
            "semantic_query_kind": semantic_probe.query_kind,
            "semantic_truncated": bool(semantic_probe.truncated),
            "semantic_error": semantic_probe.error,
        },
    }
