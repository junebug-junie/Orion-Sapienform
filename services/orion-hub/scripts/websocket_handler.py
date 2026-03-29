# services/orion-hub/scripts/websocket_handler.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.cortex_request_builder import build_chat_request, build_continuity_messages, validate_single_verb_override
from scripts.biometrics_cache import BiometricsCache
from scripts.chat_history import (
    build_chat_history_envelope,
    publish_chat_history,
    build_chat_turn_envelope,
    publish_social_room_turn,
    publish_chat_turn,
)
from scripts.social_room import is_social_room_payload, social_room_client_meta
from scripts.trace_payloads import extract_agent_trace_payload
from scripts.workflow_payloads import extract_workflow_payload
from scripts.warm_start import mini_personality_summary
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload
from orion.cognition.verb_activation import is_active

logger = logging.getLogger("orion-hub.ws")


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


#________________________
# store chat turns
#________________________

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




def _truncate_text(value: Any, limit: int = 800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _compact_council_debug(payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    opinions_out = []
    opinions = payload.get("opinions")
    if isinstance(opinions, list):
        for item in opinions[:12]:
            if not isinstance(item, dict):
                continue
            opinions_out.append({
                "agent_name": _truncate_text(item.get("agent_name") or item.get("name") or "unknown", 80),
                "confidence": item.get("confidence"),
                "text": _truncate_text(item.get("text") or "", 800),
            })

    verdict_out = {}
    verdict = payload.get("verdict")
    if isinstance(verdict, dict):
        verdict_out = {
            "action": verdict.get("action"),
            "reason": _truncate_text(verdict.get("reason") or "", 500),
            "constraints": verdict.get("constraints") if isinstance(verdict.get("constraints"), dict) else {},
        }

    blink_out = {}
    blink = payload.get("blink")
    if isinstance(blink, dict):
        blink_out = {
            "proposed_answer": _truncate_text(blink.get("proposed_answer") or "", 500),
            "scores": blink.get("scores") if isinstance(blink.get("scores"), dict) else {},
        }

    if not opinions_out and not verdict_out and not blink_out:
        return None
    return {"opinions": opinions_out, "verdict": verdict_out, "blink": blink_out}


def _extract_council_debug_from_result(resp: CortexChatResult) -> Dict[str, Any] | None:
    if not resp or not getattr(resp, "cortex_result", None):
        return None

    cr = resp.cortex_result
    recall_debug = cr.recall_debug if isinstance(cr.recall_debug, dict) else {}
    metadata = cr.metadata if isinstance(cr.metadata, dict) else {}

    for candidate in (
        recall_debug.get("council_debug"),
        metadata.get("council"),
        metadata.get("council_debug"),
    ):
        compact = _compact_council_debug(candidate if isinstance(candidate, dict) else None)
        if compact:
            return compact

    steps = cr.steps if isinstance(cr.steps, list) else []
    for step in reversed(steps):
        step_result = getattr(step, "result", None)
        if not isinstance(step_result, dict):
            continue
        council_payload = step_result.get("CouncilService")
        if not isinstance(council_payload, dict):
            continue
        compact = _compact_council_debug(council_payload.get("debug_compact") if isinstance(council_payload.get("debug_compact"), dict) else council_payload)
        if compact:
            return compact

    return None

def _schedule_publish(coro: asyncio.Future, label: str) -> None:
    task = asyncio.create_task(coro)

    def _log_result(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception as exc:
            logger.warning("Failed to publish %s: %s", label, exc, exc_info=True)

    task.add_done_callback(_log_result)


def _rec_tape_req(
    *,
    corr_id: str,
    session_id: Optional[str],
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


def _build_prompt_with_history(
    history: List[Dict[str, Any]],
    user_text: str,
    turns: int,
    max_chars: int,
) -> str:
    """Build a single prompt string that includes the last N turns as plain text.

    This is intentionally *Hub-side only* and does not require any schema changes.

    Notes:
      - "turns" means userassistant pairs; we keep up to 2*turns messages.
      - We exclude system messages (the backend already has its own system prompt).
    """
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]

    # "turns" = userassistant pairs -> 2*turns messages
    tail = msgs[-2 * max(0, int(turns)) :] if turns else []

    lines: List[str] = []
    for m in tail:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        speaker = "You" if role == "user" else "Orion"
        # Avoid accidental mega-prompts if something goes sideways
        if len(content) > 6000:
            content = content[:6000].rstrip() + " …"

        lines.append(f"{speaker}: {content}")

    base_user = (user_text or "").strip()
    if not lines:
        return base_user

    header = "Conversation context (most recent last):"

    def _compose(ls: List[str]) -> str:
        ctx = "\n".join(ls).strip()
        return f"{header}\n{ctx}\n\nYou: {base_user}\nOrion:"

    prompt = _compose(lines)

    # Trim oldest context lines until we fit under max_chars
    if max_chars and max_chars > 0:
        while len(prompt) > max_chars and lines:
            lines.pop(0)
            prompt = _compose(lines)

    return prompt



async def _with_biometrics(
    payload: Dict[str, Any],
    *,
    cache: Optional[BiometricsCache],
) -> Dict[str, Any]:
    enriched = dict(payload)
    if cache:
        enriched["biometrics"] = await cache.get_snapshot()
    else:
        enriched["biometrics"] = {
            "status": "NO_SIGNAL",
            "reason": "cache_unavailable",
            "as_of": None,
            "freshness_s": None,
            "constraint": "NONE",
            "cluster": {
                "composite": {"strain": 0.0, "homeostasis": 0.0, "stability": 1.0},
                "trend": {
                    "strain": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                    "homeostasis": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                    "stability": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                },
            },
            "nodes": {},
        }
    return enriched


async def drain_queue(websocket: WebSocket, queue: asyncio.Queue, cache: Optional[BiometricsCache]):
    try:
        while websocket.client_state.name == "CONNECTED":
            msg = await queue.get()
            try:
                await websocket.send_json(await _with_biometrics(msg, cache=cache))
            except WebSocketDisconnect:
                break
            queue.task_done()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)

async def run_tts_remote(text: str, tts_client, queue: asyncio.Queue):
    if not text.strip() or not tts_client:
        return
    try:
        req = TTSRequestPayload(text=text)
        result: TTSResultPayload = await tts_client.speak(req)
        msg = {"audio_response": result.audio_b64, "text": text}
        await queue.put(msg)
    except Exception as e:
        logger.error(f"TTS Remote Failed: {e}")


async def biometrics_heartbeat(
    websocket: WebSocket,
    *,
    cache: Optional[BiometricsCache],
    interval_sec: float,
) -> None:
    try:
        while websocket.client_state.name == "CONNECTED":
            try:
                await websocket.send_json(
                    await _with_biometrics({"biometrics_tick": True}, cache=cache)
                )
            except WebSocketDisconnect:
                break
            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Biometrics heartbeat error: %s", e, exc_info=True)

async def websocket_endpoint(websocket: WebSocket):
    import scripts.main
    bus = scripts.main.bus
    cortex_client = scripts.main.cortex_client
    tts_client = scripts.main.tts_client
    biometrics_cache = scripts.main.biometrics_cache
    notification_cache = scripts.main.notification_cache
    presence_state = scripts.main.presence_state

    await websocket.accept()
    logger.info("WebSocket accepted.")
    if presence_state:
        presence_state.connected()

    client_meta = {
        "user_agent": websocket.headers.get("user-agent"),
        "origin": websocket.headers.get("origin"),
        "x_forwarded_for": websocket.headers.get("x-forwarded-for"),
        "client_host": getattr(websocket.client, "host", None),
        "client_port": getattr(websocket.client, "port", None),
    }

    # Soft warning if services missing, but keep connection alive
    if not bus or not cortex_client:
        logger.warning("OrionBus/CortexClient not ready. Chat will be limited.")
        await websocket.send_json(await _with_biometrics({
            "llm_response": "[SYSTEM WARNING] Bus disconnected. Brain is offline, but UI is active.", 
            "state": "idle"
        }, cache=biometrics_cache))

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": mini_personality_summary()}
    ]

    tts_q: asyncio.Queue = asyncio.Queue()
    if notification_cache is not None:
        notification_cache.register_queue(tts_q)
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q, biometrics_cache))
    biometrics_task = asyncio.create_task(
        biometrics_heartbeat(
            websocket,
            cache=biometrics_cache,
            interval_sec=float(getattr(settings, "BIOMETRICS_PUSH_INTERVAL_SEC", 5.0)),
        )
    )

    try:
        while True:
            raw = await websocket.receive_text()
            if presence_state:
                presence_state.heartbeat()
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mode = data.get("mode") or ("auto" if settings.HUB_AUTO_DEFAULT_ENABLED else "brain")
            disable_tts = data.get("disable_tts", False)
            diagnostic = bool(
                data.get("diagnostic")
                or (isinstance(data.get("options"), dict) and data.get("options", {}).get("diagnostic"))
            )
            session_id = data.get("session_id")
            publish_session_id = session_id or "unknown"
            if not session_id and diagnostic:
                logger.warning("Missing session_id; publishing chat history with session_id=unknown")
            no_write = bool(data.get("no_write", settings.HUB_DEFAULT_NO_WRITE))

            # Trace Verb & Test Stub Logic ---
            # 1. Default to general chat
            trace_verb = "chat_general"

            # 2. Map modes to verbs for the Visualizer
            if mode == "agent":
                trace_verb = "task_execution"
            elif mode == "council":
                trace_verb = "council_deliberation"

            # 3. Force verb for Test/Stub Submissions
            if data.get("test_mode") or data.get("submission_id"):
                trace_verb = "test_submission"
            # -----------------------------------------

            transcript: Optional[str] = None
            is_text_input = False

            # 1. Input Processing
            possible_text = data.get("text_input") or data.get("text") or data.get("content")
            if possible_text:
                transcript = possible_text
                is_text_input = True
            elif data.get("audio"):
                if tts_client:
                    try:
                        await websocket.send_json(
                            await _with_biometrics({"state": "processing"}, cache=biometrics_cache)
                        )
                        stt_req = STTRequestPayload(audio_b64=data.get("audio"))
                        stt_result = await tts_client.transcribe(stt_req)
                        transcript = stt_result.text
                    except Exception as e:
                        logger.error(f"STT Error: {e}")
                        await websocket.send_json(
                            await _with_biometrics({"error": "Transcription failed"}, cache=biometrics_cache)
                        )
                        continue
                else:
                    await websocket.send_json(
                        await _with_biometrics({"error": "STT service unavailable"}, cache=biometrics_cache)
                    )
                    continue

            if not transcript:
                continue

            if not is_text_input:
                await websocket.send_json(
                    await _with_biometrics(
                        {"transcript": transcript, "is_text_input": False},
                        cache=biometrics_cache,
                    )
                )

            # 2. Chat Execution
            if not cortex_client:
                await websocket.send_json(
                    await _with_biometrics({"error": "Cortex disconnected (Bus offline)"}, cache=biometrics_cache)
                )
                continue

            trace_id = str(uuid.uuid4())
            if no_write:
                logger.info("NO_WRITE active (WS) sid=%s", session_id)

            # ----------------------------
            # Hub-side short-term memory
            # ----------------------------
            # N = userassistant pairs; helper keeps up to 2*N messages.
            turns = int(data.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
            prompt_with_ctx = transcript
            # IMPORTANT: store the raw user message for next turn
            history.append({"role": "user", "content": transcript})


            # Build outbound chat request through shared builder to keep WS/HTTP identical
            inactive = validate_single_verb_override(data, node_name=settings.NODE_NAME)
            if inactive:
                await websocket.send_json(await _with_biometrics({"error": inactive.get("message") or inactive.get("error")}, cache=biometrics_cache))
                continue

            continuity_messages = build_continuity_messages(
                history=history,
                latest_user_prompt=transcript,
                turns=turns,
            )
            chat_req, route_debug, use_recall = build_chat_request(
                payload=data,
                session_id=session_id,
                user_id=data.get("user_id"),
                trace_id=trace_id,
                default_mode="brain",
                auto_default_enabled=bool(settings.HUB_AUTO_DEFAULT_ENABLED),
                source_label="hub_ws",
                prompt=prompt_with_ctx,
                messages=continuity_messages,
            )
            workflow_request = chat_req.metadata.get("workflow_request") if isinstance(chat_req.metadata, dict) else None
            execution_policy = workflow_request.get("execution_policy") if isinstance(workflow_request, dict) else None
            logger.info(
                "workflow_resolution_result %s",
                json.dumps(
                    {
                        "correlation_id": trace_id,
                        "matched_workflow_id": (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                        "fallback_route": route_debug.get("fallback_route"),
                        "reason": route_debug.get("workflow_resolution_reason"),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            logger.info(
                "hub_workflow_request corr=%s sid=%s workflow_id=%s invocation_mode=%s schedule_kind=%s source=ws",
                trace_id,
                session_id,
                (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                (execution_policy or {}).get("invocation_mode") if isinstance(execution_policy, dict) else None,
                ((execution_policy or {}).get("schedule") or {}).get("kind") if isinstance(execution_policy, dict) else None,
            )
            chat_req.metadata = dict(chat_req.metadata or {})
            chat_req.metadata["trace_verb"] = trace_verb
            mode = chat_req.mode
            recall_payload = chat_req.recall or {"enabled": use_recall}
            turn_client_meta = dict(client_meta)
            if is_social_room_payload(data):
                turn_client_meta.update(
                    social_room_client_meta(
                        payload=data,
                        route_debug=route_debug,
                        trace_verb=trace_verb,
                        memory_digest=None,
                    )
                )

            logger.info(f"WS Chat Request recall config: {recall_payload} session_id={session_id}")
            logger.info(
                "Routing resolved to mode: %s (verb: %s)",
                mode,
                trace_verb,
            )
            logger.info(
                "WS routing resolved mode=%s route_intent=%s verb=%s allowed_verbs=%s",
                chat_req.mode,
                chat_req.route_intent,
                chat_req.verb,
                len(((chat_req.options or {}).get("allowed_verbs") or [])),
            )
            logger.info(
                "WS Chat Request payload session_id=%s history_len=%s last_user_len=%s last_user_head=%r",
                session_id,
                len(history),
                len(transcript or ""),
                (transcript or "")[:120],
            )
            logger.info(
                "hub_egress corr=%s sid=%s mode=%s verb=%s route_intent=%s allowed_verbs=%s packs=%s",
                trace_id,
                session_id,
                chat_req.mode,
                chat_req.verb,
                (chat_req.options or {}).get("route_intent") or "none",
                len(((chat_req.options or {}).get("allowed_verbs") or [])),
                chat_req.packs or [],
            )
            logger.info(
                "hub_context_messages corr=%s sid=%s mode=%s count=%s roles=%s",
                trace_id,
                session_id,
                chat_req.mode,
                len(chat_req.messages or []),
                [m.role if hasattr(m, "role") else m.get("role") for m in (chat_req.messages or [])][:12],
            )
            _log_hub_route_decision(
                corr_id=trace_id,
                session_id=session_id,
                route_debug=route_debug,
                user_prompt=transcript,
            )
            if diagnostic:
                logger.info("WS outbound CortexChatRequest corr=%s payload=%s", trace_id, chat_req.model_dump(mode="json"))

            _rec_tape_req(
                corr_id=trace_id,
                session_id=session_id,
                mode=mode,
                use_recall=use_recall,
                recall_profile=recall_payload.get("profile"),
                user_head=(transcript or "")[:80],
                no_write=no_write,
            )
            # Publish the inbound user message into chat history
            if bus and not no_write:
                user_env = build_chat_history_envelope(
                    content=transcript,
                    role="user",
                    session_id=publish_session_id,
                    correlation_id=trace_id,
                    speaker=data.get("user_id") or "user",
                    tags=[mode],
                    message_id=f"{trace_id}:user",
                    memory_status="accepted",
                    memory_tier="ephemeral",
                    client_meta=turn_client_meta,
                )
                _schedule_publish(publish_chat_history(bus, [user_env]), "chat.history user")

            orion_response_text = ""
            memory_digest = None
            recall_debug = None
            agent_trace = None
            workflow = None
            metacog_traces: List[Dict[str, Any]] = []
            try:
                resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=trace_id)
                orion_response_text = resp.final_text or ""
                if resp.cortex_result and isinstance(resp.cortex_result.recall_debug, dict):
                    recall_debug = resp.cortex_result.recall_debug
                    memory_digest = recall_debug.get("memory_digest")
                agent_trace = extract_agent_trace_payload(resp.cortex_result)
                raw_traces = getattr(resp.cortex_result, "metacog_traces", None)
                if isinstance(raw_traces, list):
                    metacog_traces = [t for t in raw_traces if isinstance(t, dict)]
                logger.info(
                    "hub_metacog_received corr=%s source=ws traces=%s",
                    trace_id,
                    len(metacog_traces),
                )
                workflow = extract_workflow_payload(resp.cortex_result)
                if isinstance(workflow, dict):
                    logger.info(
                        "hub_workflow_response corr=%s workflow_id=%s status=%s scheduled_count=%s persisted_count=%s rendered_path=%s source=ws",
                        trace_id,
                        workflow.get("workflow_id"),
                        workflow.get("status"),
                        len(workflow.get("scheduled") or []),
                        len(workflow.get("persisted") or []),
                        "scheduled_confirmation" if len(workflow.get("scheduled") or []) else "immediate_or_unscheduled",
                    )
                # If the model echoes "Orion:" due to our prompt format, strip it.
                s = (orion_response_text or "").lstrip()
                if s.startswith("Orion:"):
                    orion_response_text = s[len("Orion:"):].lstrip()
                if hasattr(resp, "cortex_result") and resp.cortex_result:
                    trace_verb = str(
                        ((resp.cortex_result.metadata or {}).get("trace_verb") if isinstance(resp.cortex_result.metadata, dict) else None)
                        or resp.cortex_result.verb
                        or trace_verb
                    )
            except Exception as e:
                logger.error(f"Chat RPC Error: {e}")
                await websocket.send_json(
                    await _with_biometrics({"error": f"Chat failed: {str(e)}"}, cache=biometrics_cache)
                )
                continue

            # 3. Response & Logging
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
                trace_id,
                session_id,
                mode,
                getattr(resp.cortex_result, "status", None),
                len(orion_response_text or ""),
                memory_used,
                recall_count,
            )
            _rec_tape_rsp(
                corr_id=trace_id,
                memory_used=memory_used,
                recall_count=recall_count,
                backend_counts=backend_counts,
                memory_digest=memory_digest,
            )
            ws_payload = {
                "llm_response": orion_response_text,
                "mode": mode,
                "correlation_id": trace_id,
                "memory_digest": memory_digest,
                "memory_used": memory_used,
                "recall_debug": recall_debug,
                "agent_trace": agent_trace,
                "workflow": workflow,
                "no_write": no_write,
                "routing_debug": route_debug,
                "metacog_traces": metacog_traces,
            }
            if mode == "council" or settings.HUB_DEBUG_COUNCIL:
                council_debug = _extract_council_debug_from_result(resp)
                if council_debug:
                    ws_payload["council_debug"] = council_debug
            await websocket.send_json(await _with_biometrics(ws_payload, cache=biometrics_cache))

            # Log to SQL (Best Effort) & Trigger Introspection
            if bus and not no_write:
                enriched_client_meta = dict(turn_client_meta)
                try:
                    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                    if is_social_room_payload(data):
                        enriched_client_meta.update(
                            social_room_client_meta(
                                payload=data,
                                route_debug=route_debug,
                                trace_verb=trace_verb,
                                memory_digest=memory_digest,
                            )
                        )

                    # Extract rich metadata
                    gateway_meta = {}
                    if hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}

                    # Include trace_verb in spark_meta for the Visualizer
                    spark_meta = {
                        "mode": mode,
                        "trace_verb": trace_verb,
                        "use_recall": use_recall,
                        **(gateway_meta if isinstance(gateway_meta, dict) else {}),
                    }

                    chat_row = {
                        "id": trace_id,
                        "correlation_id": trace_id,
                        "source": "hub_ws",
                        "prompt": transcript,
                        "response": orion_response_text,
                        "user_id": data.get("user_id"),
                        "session_id": data.get("session_id"),
                        "spark_meta": spark_meta,
                    }

                    # 1. SQL Log (turn-level row: prompt + response)
                    env_turn = build_chat_turn_envelope(
                        prompt=transcript,
                        response=orion_response_text,
                        session_id=publish_session_id,
                        correlation_id=trace_id,
                        user_id=data.get("user_id"),
                        source_label="hub_ws",
                        spark_meta=spark_meta,
                        turn_id=trace_id,
                        memory_status="accepted",
                        memory_tier="ephemeral",
                        client_meta=enriched_client_meta,
                        reasoning_trace=metacog_traces[0] if metacog_traces else None,
                    )
                    _schedule_publish(publish_chat_turn(bus, env_turn), "chat.history turn")
                    logger.info("Published chat.history turn row -> %s", settings.chat_history_turn_channel)
                    if metacog_traces:
                        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                        for trace in metacog_traces:
                            if _thought_debug_enabled() and isinstance(trace, dict):
                                logger.info(
                                    "THOUGHT_DEBUG_METACOG_PUB stage=hub_ws_prepare corr=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r",
                                    trace_id,
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
                                correlation_id=trace_id,
                                payload=trace,
                            )
                            _schedule_publish(bus.publish("orion:metacog:trace", trace_env), "metacog.trace")
                        if _thought_debug_enabled() and not any(isinstance(t, dict) for t in metacog_traces):
                            logger.info("THOUGHT_DEBUG_METACOG_PUB stage=hub_ws_skipped corr=%s reason=no_valid_trace_dicts", trace_id)
                        logger.info(
                            "hub_metacog_published corr=%s source=ws channel=%s traces=%s",
                            trace_id,
                            "orion:metacog:trace",
                            len(metacog_traces),
                        )
                    if is_social_room_payload(data):
                        _schedule_publish(
                            publish_social_room_turn(
                                bus,
                                prompt=transcript,
                                response=orion_response_text,
                                session_id=publish_session_id,
                                correlation_id=trace_id,
                                user_id=data.get("user_id"),
                                source_label="hub_ws",
                                recall_profile=recall_payload.get("profile"),
                                trace_verb=trace_verb,
                                client_meta=enriched_client_meta,
                                memory_digest=memory_digest,
                            ),
                            "chat.social turn",
                        )
                    # 2. Spark Introspection Candidate
                    candidate_payload = {
                        "trace_id": trace_id,
                        "source": "hub_ws",
                        "prompt": transcript,
                        "response": orion_response_text,
                        "spark_meta": spark_meta
                    }
                    env_spark = BaseEnvelope(
                        kind="spark.candidate",
                        correlation_id=trace_id,
                        source=ServiceRef(name="hub", node=settings.NODE_NAME),
                        payload=candidate_payload
                    )
                    # Kept the literal string to ensure it hits the default introspection channel
                    # and avoids settings attribute errors
                    _schedule_publish(
                        bus.publish("orion:spark:introspect:candidate:log", env_spark),
                        "spark.candidate",
                    )

                except Exception as e:
                    logger.warning(f"Failed to log/introspect chat: {e}")

                # Publish assistant reply into chat history
                try:
                    gateway_meta = {}
                    if hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}
                    assistant_env = build_chat_history_envelope(
                        content=orion_response_text,
                        role="assistant",
                        session_id=publish_session_id,
                        correlation_id=getattr(resp.cortex_result, "correlation_id", None) or trace_id,
                        speaker=gateway_meta.get("speaker") or settings.SERVICE_NAME,
                        model=gateway_meta.get("model"),
                        provider=gateway_meta.get("provider"),
                        tags=[mode, trace_verb],
                        message_id=f"{trace_id}:assistant",
                        memory_status="accepted",
                        memory_tier="ephemeral",
                        client_meta=enriched_client_meta,
                        reasoning_trace=metacog_traces[0] if metacog_traces else None,
                    )
                    _schedule_publish(publish_chat_history(bus, [assistant_env]), "chat.history assistant")
                except Exception as e:
                    logger.warning("Failed to publish assistant chat history: %s", e, exc_info=True)

            # 4. TTS
            if orion_response_text and not disable_tts and tts_client:
                 asyncio.create_task(run_tts_remote(orion_response_text, tts_client, tts_q))

            if orion_response_text:
                history.append({"role": "assistant", "content": orion_response_text})

            # Keep history bounded (system + last 2*turns messages)
            try:
                keep_msgs = 1 + (2 * max(0, int(turns)))
                if len(history) > keep_msgs:
                    history[:] = history[:1] + history[-(2 * turns):]
            except Exception:
                pass
            await websocket.send_json(await _with_biometrics({"state": "idle"}, cache=biometrics_cache))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        biometrics_task.cancel()
        if notification_cache is not None:
            notification_cache.unregister_queue(tts_q)
        if presence_state:
            presence_state.disconnected()
