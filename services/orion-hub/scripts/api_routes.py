from __future__ import annotations

import json
import logging
from uuid import uuid4
from typing import Optional, Any, List, Dict, Tuple
import aiohttp

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
import requests

from .settings import settings
from .session import ensure_session
from .chat_history import build_chat_history_envelope, publish_chat_history, publish_social_room_turn
from .library import scan_cognition_library
from .trace_payloads import extract_agent_trace_payload
from .cortex_request_builder import build_chat_request, validate_single_verb_override
from .social_room import is_social_room_payload, social_room_client_meta
from orion.cognition.verb_activation import build_verb_list
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.cortex.contracts import CortexChatResult
from orion.schemas.notify import (
    ChatAttentionAck,
    ChatMessageReceipt,
    NotificationPreferencesUpdate,
    PreferenceResolutionRequest,
    RecipientProfileUpdate,
)

logger = logging.getLogger("orion-hub.api")

router = APIRouter()

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
    summary = {
        "corr_id": corr_id,
        "session_id": session_id,
        "selected_ui_route": route_debug.get("selected_ui_route"),
        "emitted_mode": route_debug.get("mode"),
        "emitted_verb": route_debug.get("verb"),
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
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": settings.SERVICE_NAME}

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
    req, route_debug, _ = build_chat_request(
        payload=payload,
        session_id=session_id,
        user_id=payload.get("user_id"),
        trace_id=None,
        default_mode="brain",
        auto_default_enabled=bool(settings.HUB_AUTO_DEFAULT_ENABLED),
        source_label="hub_http",
        prompt=user_prompt,
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

        # Extract Text
        text = resp.final_text or ""

        # Map raw result for UI debug
        raw_result = resp.cortex_result.model_dump(mode="json")

        # Use the correlation_id from the response (gateway) if available
        # or it might be passed back from the client logic if modified to do so.
        # Here we rely on CortexChatResult having it.
        correlation_id = resp.cortex_result.correlation_id or corr_id

        memory_digest = None
        recall_debug = None
        agent_trace = None
        if resp.cortex_result and isinstance(resp.cortex_result.recall_debug, dict):
            recall_debug = resp.cortex_result.recall_debug
            memory_digest = recall_debug.get("memory_digest")
        agent_trace = extract_agent_trace_payload(resp.cortex_result)

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
            "memory_used": memory_used,
            "memory_digest": memory_digest,
            "no_write": no_write,
            "spark_meta": None,
            "correlation_id": correlation_id,
            "routing_debug": route_debug,
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
            }

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
