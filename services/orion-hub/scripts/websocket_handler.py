# services/orion-hub/scripts/websocket_handler.py
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
from scripts.llm_rpc import BrainRPC, CouncilRPC
from scripts.warm_start import mini_personality_summary
from scripts.recall_rpc import RecallRPC

logger = logging.getLogger("voice-app.ws")


# ---------------------------------------------------------------------
# 🔄 Utility: drain a queue to the WebSocket
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
# 🧠 Memory helpers (Recall → system block)
# ---------------------------------------------------------------------
def _format_fragments_for_ws(fragments: List[Dict[str, Any]], limit: int = 12) -> List[str]:
    """
    Turn recall fragments into short lines suitable for a system
    memory block. We keep it compact but still human-readable if
    we ever inspect the history.
    """
    lines: List[str] = []

    for frag in fragments[:limit]:
        kind = frag.get("kind", "unknown")
        text = (frag.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue

        meta = frag.get("meta") or {}
        observer = meta.get("observer")
        field_resonance = meta.get("field_resonance")

        extras: List[str] = []
        if observer:
            extras.append(f"observer={observer}")
        if field_resonance:
            extras.append(f"field_resonance={field_resonance}")

        suffix = f" [{' | '.join(extras)}]" if extras else ""
        lines.append(f"[{kind}] {text[:260]}{suffix}")

    return lines


async def _build_memory_block_for_ws(
    bus,
    transcript: str,
    session_id: Optional[str],
    max_items: int = 25,
) -> str:
    """
    Call Recall via the bus and build a single system message block
    describing relevant past memories.
    """
    if bus is None or not getattr(bus, "enabled", False):
        return ""

    try:
        recall_client = RecallRPC(bus)
        recall_result = await recall_client.call_recall(
            query=transcript,
            session_id=session_id,
            mode="hybrid",
            time_window_days=14,
            max_items=max_items,
            extras=None,
        )

        fragments = recall_result.get("fragments") or []
        logger.info(
            "WS Recall returned %d fragments: %s",
            len(fragments),
            [
                f"{frag.get('kind')}::{(frag.get('text') or '')[:80]}"
                for frag in fragments[:5]
            ],
        )

        snippet_lines = _format_fragments_for_ws(fragments)
        if not snippet_lines:
            return ""

        memory_block = (
            "Relevant past memories about Juniper, Orion, and recent context.\n"
            "Use ONLY the events listed below as factual memory.\n"
            "If Juniper asks whether you remember something that is not mentioned\n"
            "here or in the recent dialogue history, explicitly say that you do not recall\n"
            "instead of guessing. Do NOT invent specific cities, people, dates, or events.\n"
            + "\n- "
            + "\n- ".join(snippet_lines)
        )
        return memory_block

    except Exception as e:
        logger.warning(
            f"RecallRPC lookup failed in WebSocket handler: {e}",
            exc_info=True,
        )
        return ""


def _strip_old_memory_blocks(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove any previously auto-injected memory blocks so we don’t stack them
    each turn. We detect them by a fixed prefix string.
    """
    marker = "Relevant past memories about Juniper, Orion, and recent context."
    cleaned: List[Dict[str, Any]] = []

    for m in history:
        if (
            m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and marker in m["content"]
        ):
            continue
        cleaned.append(m)

    return cleaned


def _inject_memory_block(
    history: List[Dict[str, Any]],
    memory_block: str,
    has_instructions: bool,
) -> List[Dict[str, Any]]:
    """
    Insert the memory system message after:
      - core persona stub
      - optional user instructions
    """
    if not memory_block:
        return history

    history = _strip_old_memory_blocks(history)

    insert_idx = 1  # after core persona
    if has_instructions:
        insert_idx += 1

    history.insert(
        insert_idx,
        {"role": "system", "content": memory_block},
    )
    return history


# ---------------------------------------------------------------------
# 🎙️ Main WebSocket endpoint (Brain vs Council + Recall)
# ---------------------------------------------------------------------
async def websocket_endpoint(websocket: WebSocket):
    """Handles the main voice and text WebSocket lifecycle."""
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

            # Mode toggle: "brain" (default) | "council"
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

            logger.info(f"Transcript: {transcript!r}")

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

            # ---------------------------------------------------------
            # 🔍 Step 3: Recall → memory block (optional)
            # ---------------------------------------------------------
            memory_block = await _build_memory_block_for_ws(
                bus=bus,
                transcript=transcript,
                session_id=session_id,
                max_items=25,
            )

            if memory_block:
                history = _inject_memory_block(
                    history=history,
                    memory_block=memory_block,
                    has_instructions=has_instructions,
                )

            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY BEFORE LLM CALL (mode={mode}): {history}")

            # ---------------------------------------------------------
            # 🧠 Step 4: LLM via BrainRPC or CouncilRPC (bus-native)
            # ---------------------------------------------------------
            if mode == "council":
                rpc = CouncilRPC(bus)
            else:
                rpc = BrainRPC(bus)

            user_prompt = transcript.strip()

            reply = await rpc.call_llm(
                prompt=user_prompt,
                history=history[:],  # shallow copy
                temperature=temperature,
            )

            orion_response_text = (
                reply.get("text") or reply.get("response") or ""
            )
            tokens = len(orion_response_text.split()) if orion_response_text else 0

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
            # 🧠 Step 7: Append to history + publish to Collapse triage
            # ---------------------------------------------------------
            if orion_response_text:
                history.append({"role": "orion", "content": orion_response_text})

                if bus is not None and getattr(bus, "enabled", False):
                    latest_user_prompt = transcript

                    full_dialogue_payload = {
                        "id": str(uuid.uuid4()),
                        "text": (
                            f"User: {latest_user_prompt}\n"
                            f"Orion: {orion_response_text}"
                        ),
                        "source": settings.SERVICE_NAME,
                        "ts": datetime.utcnow().isoformat(),
                        "prompt": latest_user_prompt,
                        "response": orion_response_text,
                        "user_id": user_id,
                        "session_id": session_id,
                        "mode": mode,
                    }

                    try:
                        bus.publish(
                            settings.CHANNEL_COLLAPSE_TRIAGE,
                            full_dialogue_payload,
                        )
                        logger.info(
                            "Published full dialogue to triage: %s",
                            full_dialogue_payload["id"],
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to publish dialogue to triage: {e}",
                            exc_info=True,
                        )

            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY AFTER LLM CALL (mode={mode}): {history}")

            # Mark the end of this turn's processing phase.
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        logger.info("WebSocket closed.")
