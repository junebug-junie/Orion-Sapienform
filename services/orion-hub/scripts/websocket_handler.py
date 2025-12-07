from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.llm_tts_handler import run_tts_only
from scripts.llm_rpc import CouncilRPC
from scripts.warm_start import mini_personality_summary
from scripts.chat_front import run_chat_general

logger = logging.getLogger("orion-hub.ws")


# ---------------------------------------------------------------------
# 🔄 Utility: Drain a queue to the WebSocket
# ---------------------------------------------------------------------
async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    """
    Background task that pulls messages from `queue`
    and forwards them to the WebSocket client.
    """
    try:
        while websocket.client_state.name == "CONNECTED":
            msg = await queue.get()
            await websocket.send_json(msg)
            queue.task_done()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        logger.info("Drain queue task cancelled.")
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)


# ---------------------------------------------------------------------
# 🔁 Shared history normalizer (used in HTTP paths if needed)
# ---------------------------------------------------------------------
def build_llm_history(
    *,
    raw_history: list[dict],
    recall_block: str | None = None,
) -> list[dict]:
    """
    Normalize the conversation history for the LLM:

      - Always start with a single persona system stub (mini_personality_summary).
      - Optionally add one system recall block.
      - Then append ONLY user/assistant messages from stored history.
      - Drop ALL stored system messages (old personas, old memory blocks, etc.).

    This prevents persona duplication and weird “reset” behavior
    from replaying old system prompts each turn.
    """
    history: list[dict] = []

    # 1) Canonical persona stub
    history.append({"role": "system", "content": mini_personality_summary()})

    # 2) Optional recall / memory block
    if recall_block:
        history.append(
            {
                "role": "system",
                "content": recall_block,
            }
        )

    # 3) Only user / assistant messages from prior turns
    for msg in raw_history or []:
        role = (msg.get("role") or "").lower()
        if role in ("user", "assistant"):
            history.append(
                {
                    "role": role,
                    "content": msg.get("content", ""),
                }
            )

    return history


# ---------------------------------------------------------------------
# 🎙️ Main WebSocket endpoint (Gateway / Council + Recall)
# ---------------------------------------------------------------------
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles the main voice/text WebSocket lifecycle for Hub.

    - ASR (audio → text) via `asr` from scripts.main.
    - LLM via LLMGatewayRPC by default (vLLM Mistral behind the scenes).
    - Optional Council mode via CouncilRPC when mode == "council".
    - Optional Recall → memory block injection.
    - Optional TTS stream back to the client.
    """
    from scripts.main import asr, bus  # lazy import shared objects

    await websocket.accept()
    logger.info("WebSocket accepted.")

    if asr is None:
        await websocket.send_json({"error": "ASR model not loaded"})
        await websocket.close()
        return

    # Seed conversation history with Orion's personality stub
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": mini_personality_summary()}
    ]
    has_instructions = False

    # Queue for streaming TTS chunks back to the client
    tts_q: asyncio.Queue = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            raw = await websocket.receive_text()
            data: Dict[str, Any] = json.loads(raw)

            # Mode toggle: "brain" (gateway) | "council"
            mode = data.get("mode", "brain")

            # Flags
            disable_tts = data.get("disable_tts", False)
            instructions = data.get("instructions", "")
            context_len = data.get("context_length", 10)
            temperature = data.get("temperature", 0.7)

            # Session / user identifiers for logging + Collapse Mirror
            user_id = data.get("user_id")
            session_id = data.get("session_id")

            # ---------------------------------------------------------
            # 🗣️ Step 1: Get transcript (text_input or audio + ASR)
            # ---------------------------------------------------------
            transcript: Optional[str] = None
            is_text_input = False

            text_input = data.get("text_input")
            if text_input:
                transcript = text_input
                is_text_input = True
            else:
                audio_b64 = data.get("audio")
                if not audio_b64:
                    logger.warning("No audio or text_input in message.")
                    continue

                await websocket.send_json({"state": "processing"})
                audio_bytes = base64.b64decode(audio_b64)
                transcript = asr.transcribe_bytes(audio_bytes)

            if not transcript:
                await websocket.send_json({"llm_response": "I didn't catch that."})
                await websocket.send_json({"state": "idle"})
                continue

            logger.info("Transcript: %r", transcript)

            if not is_text_input:
                await websocket.send_json(
                    {"transcript": transcript, "is_text_input": is_text_input}
                )

            # ---------------------------------------------------------
            # 🧾 Step 2: Instructions + add user message to history
            # ---------------------------------------------------------
            if instructions and not has_instructions:
                # Insert user-provided instructions just after core personality stub
                history.insert(1, {"role": "system", "content": instructions})
                has_instructions = True

            # Add current user message
            history.append({"role": "user", "content": transcript})

            # Trim non-system messages to context_len
            if context_len and context_len > 0:
                system_messages = [m for m in history if m.get("role") == "system"]
                non_system = [m for m in history if m.get("role") != "system"]

                keep_count = max(context_len - len(system_messages), 0)
                trimmed_non_system = non_system[-keep_count:] if keep_count > 0 else []
                history = system_messages + trimmed_non_system

            logger.info("HISTORY BEFORE LLM CALL (mode=%s): %s", mode, history)

            # ---------------------------------------------------------
            # 🧠 Step 4: LLM via Council or Cortex chat_general
            # ---------------------------------------------------------
            user_prompt = transcript.strip()
            spark_meta = None

            if mode == "council":
                rpc = CouncilRPC(bus)
                reply = await rpc.call_llm(
                    prompt=user_prompt,
                    history=history[:],  # shallow copy
                    temperature=temperature,
                )
                orion_response_text = (
                    reply.get("text") or reply.get("response") or ""
                )
                tokens = len(orion_response_text.split()) if orion_response_text else 0
            else:
                convo = await run_chat_general(
                    bus,
                    session_id=session_id,
                    user_id=user_id,
                    messages=history[:],  # full system+dialogue history
                    chat_mode=mode,
                    temperature=temperature,
                    use_recall=True,  # WS path always uses recall
                )
                orion_response_text = convo.get("text") or ""
                tokens = convo.get("tokens") or 0
                spark_meta = convo.get("spark_meta")

            # ---------------------------------------------------------
            # 💬 Step 5: Send text response to client
            # ---------------------------------------------------------
            await websocket.send_json(
                {
                    "llm_response": orion_response_text,
                    "tokens": tokens,
                    "mode": mode,
                }
            )

            # ---------------------------------------------------------
            # 🔊 Step 6: Kick off TTS in the background (if enabled)
            # ---------------------------------------------------------
            if orion_response_text and not disable_tts:
                asyncio.create_task(
                    run_tts_only(
                        orion_response_text,
                        tts_q,
                        bus=bus,
                        disable_tts=False,
                    )
                )

            # ---------------------------------------------------------
            # 🧠 Step 7: Append to history + publish chat history log
            # ---------------------------------------------------------
            if orion_response_text:
                # Keep Orion's side of the turn in local WS history
                history.append({"role": "assistant", "content": orion_response_text})

                if bus is not None and getattr(bus, "enabled", False):
                    latest_user_prompt = transcript
                    trace_id = str(uuid.uuid4())

                    chat_log_payload = {
                        "id": trace_id,
                        "trace_id": trace_id,
                        "source": settings.SERVICE_NAME,
                        "prompt": latest_user_prompt,
                        "response": orion_response_text,
                        "user_id": user_id,
                        "session_id": session_id,
                        "spark_meta": spark_meta,
                        "created_at": datetime.utcnow().isoformat(),
                        "text": (
                            f"User: {latest_user_prompt}\n"
                            f"Orion: {orion_response_text}"
                        ),
                        "kind": "hub_ws_dialogue",
                    }

                    channel = getattr(settings, "CHANNEL_CHAT_HISTORY_LOG", None)
                    if not channel:
                        logger.warning(
                            "CHANNEL_CHAT_HISTORY_LOG is not configured on settings; "
                            "skipping chat history publish for WS turn."
                        )
                    else:
                        try:
                            bus.publish(channel, chat_log_payload)
                            logger.info(
                                "Published WS dialogue to chat history channel %s id=%s",
                                channel,
                                chat_log_payload["id"],
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to publish WS dialogue to chat history channel: %s",
                                e,
                                exc_info=True,
                            )

            logger.info("HISTORY AFTER LLM CALL (mode=%s): %s", mode, history)

            # Mark the end of this turn's processing phase.
            await websocket.send_json({"state": "idle"})


    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        logger.info("WebSocket closed.")
