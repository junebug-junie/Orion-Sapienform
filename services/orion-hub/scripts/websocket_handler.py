# services/orion-hub/scripts/websocket_handler.py
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.biometrics_cache import BiometricsCache
from scripts.chat_history import (
    build_chat_history_envelope,
    publish_chat_history,
    build_chat_turn_envelope,
    publish_chat_turn,
)
from scripts.warm_start import mini_personality_summary
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload

logger = logging.getLogger("orion-hub.ws")


#________________________
# store chat turns
#________________________


def _schedule_publish(coro: asyncio.Future, label: str) -> None:
    task = asyncio.create_task(coro)

    def _log_result(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception as exc:
            logger.warning("Failed to publish %s: %s", label, exc, exc_info=True)

    task.add_done_callback(_log_result)


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
            await websocket.send_json(await _with_biometrics(msg, cache=cache))
            queue.task_done()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
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

async def websocket_endpoint(websocket: WebSocket):
    import scripts.main
    bus = scripts.main.bus
    cortex_client = scripts.main.cortex_client
    tts_client = scripts.main.tts_client
    biometrics_cache = scripts.main.biometrics_cache

    await websocket.accept()
    logger.info("WebSocket accepted.")

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
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q, biometrics_cache))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mode = data.get("mode", "brain")
            disable_tts = data.get("disable_tts", False)
            diagnostic = bool(
                data.get("diagnostic")
                or (isinstance(data.get("options"), dict) and data.get("options", {}).get("diagnostic"))
            )
            session_id = data.get("session_id")
            publish_session_id = session_id or "unknown"
            if not session_id and diagnostic:
                logger.warning("Missing session_id; publishing chat history with session_id=unknown")

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

            logger.info(f"Routing to mode: {mode} (verb: {trace_verb})")
            trace_id = str(uuid.uuid4())

            # ----------------------------
            # Hub-side short-term memory
            # ----------------------------
            # N = userassistant pairs; helper keeps up to 2*N messages.
            turns = int(data.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
            max_chars = int(getattr(settings, "HUB_CONTEXT_MAX_CHARS", 12000))
            prompt_with_ctx = _build_prompt_with_history(
                history=history,
                user_text=transcript,
                turns=turns,
                max_chars=max_chars,
            )
            # IMPORTANT: store the raw user message for next turn
            history.append({"role": "user", "content": transcript})


            # Inject trace_verb into metadata so Cortex might see it too
            chat_req = CortexChatRequest(
                prompt=prompt_with_ctx,
                mode=mode,
                session_id=session_id,
                user_id=data.get("user_id"),
                trace_id=trace_id,
                recall={"enabled": data.get("use_recall", False)},
                packs=data.get("packs"),
                metadata={"source": "hub_ws", "trace_verb": trace_verb} 
            )

            # Handle Verbs selection
            if data.get("verbs"):
                if not chat_req.options: chat_req.options = {}
                chat_req.options["allowed_verbs"] = data.get("verbs")

            # Publish the inbound user message into chat history
            if bus:
                user_env = build_chat_history_envelope(
                    content=transcript,
                    role="user",
                    session_id=publish_session_id,
                    correlation_id=trace_id,
                    speaker=data.get("user_id") or "user",
                    tags=[mode],
                )
                _schedule_publish(publish_chat_history(bus, [user_env]), "chat.history user")

            orion_response_text = ""
            try:
                resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=trace_id)
                orion_response_text = resp.final_text or ""
                # If the model echoes "Orion:" due to our prompt format, strip it.
                s = (orion_response_text or "").lstrip()
                if s.startswith("Orion:"):
                    orion_response_text = s[len("Orion:"):].lstrip()
            except Exception as e:
                logger.error(f"Chat RPC Error: {e}")
                await websocket.send_json(
                    await _with_biometrics({"error": f"Chat failed: {str(e)}"}, cache=biometrics_cache)
                )
                continue

            # 3. Response & Logging
            await websocket.send_json(await _with_biometrics({
                "llm_response": orion_response_text,
                "mode": mode,
            }, cache=biometrics_cache))

            # Log to SQL (Best Effort) & Trigger Introspection
            if bus:
                try:
                    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                    # Extract rich metadata
                    gateway_meta = {}
                    if hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}

                    # Include trace_verb in spark_meta for the Visualizer
                    spark_meta = {
                        "mode": mode,
                        "trace_verb": trace_verb,
                        "use_recall": bool(data.get("use_recall", False)),
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
                    )
                    _schedule_publish(publish_chat_turn(bus, env_turn), "chat.history turn")
                    logger.info("Published chat.history turn row -> %s", settings.chat_history_turn_channel)
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
