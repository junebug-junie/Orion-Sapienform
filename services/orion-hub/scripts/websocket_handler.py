# scripts/websocket_handler.py
import logging
import asyncio
import base64
import json
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.llm_tts_handler import run_tts_only
from scripts.llm_rpc import BrainRPC
from scripts.warm_start import mini_personality_summary
from scripts.recall_rpc import RecallRPC

logger = logging.getLogger("voice-app.ws")

def render_history_to_prompt(history: list[dict], latest_user: str) -> str:
    """
    Linearize the structured history into a single text prompt.

    This makes sure the LLM always sees:
      - core system persona
      - recall memory block
      - recent dialogue
      - a strong instruction on how to handle missing memories
    """
    lines = []

    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            lines.append(f"[SYSTEM]\n{content}\n")
        elif role in ("assistant", "orion"):
            lines.append(f"Oríon: {content}\n")
        elif role == "user":
            lines.append(f"Juniper: {content}\n")
        else:
            lines.append(f"{role or 'unknown'}: {content}\n")

    lines.append(
        (
            "\n[SYSTEM INSTRUCTION]\n"
            "Use ONLY the information in the SYSTEM blocks and prior dialogue as factual memory.\n"
            "If Juniper asks about something that does NOT appear in those memories, "
            "explicitly say you do not recall, instead of guessing or inventing details.\n"
            "Now respond as Oríon to Juniper's last message above, staying grounded in those memories.\n"
        )
    )

    return "\n".join(lines)


async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    """Drains a queue and sends messages to the WebSocket client."""
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


async def websocket_endpoint(websocket: WebSocket):
    """Handles the main voice and text WebSocket lifecycle."""
    # Pull shared objects from main once per connection
    from scripts.main import asr, bus

    await websocket.accept()
    logger.info("WebSocket accepted.")

    if asr is None:
        await websocket.send_json({"error": "ASR model not loaded"})
        await websocket.close()
        return

    # Seed conversation history with Orion's personality stub
    history = [
        {"role": "system", "content": mini_personality_summary()}
    ]
    has_instructions = False

    tts_q: asyncio.Queue = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            raw = await websocket.receive_text()
            data: Dict[str, Any] = json.loads(raw)

            transcript = None
            is_text_input = False
            disable_tts = data.get("disable_tts", False)

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

            # ─────────────────────────────────────────────────────
            # 📡 Bus: publish transcript + intake
            # ─────────────────────────────────────────────────────
            if bus is not None and getattr(bus, "enabled", False):
                try:
                    bus.publish(
                        settings.CHANNEL_VOICE_TRANSCRIPT,
                        {"type": "transcript", "content": transcript},
                    )
                    bus.publish(
                        settings.CHANNEL_BRAIN_INTAKE,
                        {
                            "source": settings.SERVICE_NAME,
                            "type": "intake",
                            "content": transcript,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to publish transcript/intake to bus: {e}",
                        exc_info=True,
                    )

            instructions = data.get("instructions", "")
            if instructions and not has_instructions:
                # Insert user-provided instructions just after the core personality stub
                history.insert(1, {"role": "system", "content": instructions})
                has_instructions = True

            # Add current user message
            history.append({"role": "user", "content": transcript})

            context_len = data.get("context_length", 10)
            if len(history) > context_len:
                # Preserve leading system messages, trim the rest
                system_messages = [m for m in history if m.get("role") == "system"]
                non_system = [m for m in history if m.get("role") != "system"]
                trimmed_non_system = non_system[-(context_len - len(system_messages)) :]
                history = system_messages + trimmed_non_system

            temperature = data.get("temperature", 0.7)


            # ─────────────────────────────────────────────────────
            # 🔍 Inject Recall memories into history (WS parity with /api/chat)
            # ─────────────────────────────────────────────────────
            memory_snippets = []

            if bus is not None and getattr(bus, "enabled", False):
                try:
                    recall_client = RecallRPC(bus)
                    recall_result = await recall_client.call_recall(
                        query=transcript,
                        session_id=data.get("session_id"),
                        mode="hybrid",
                        time_window_days=14,
                        max_items=12,
                        extras=None,
                    )

                    for frag in (recall_result.get("fragments") or []):
                        kind = frag.get("kind")
                        text = (frag.get("text") or "").strip()
                        if not text:
                            continue
                        # Skip tag-cloud style enrichment if you don't want it in context
                        if kind == "enrichment":
                            continue
                        memory_snippets.append(f"[{kind}] {text[:260]}")

                except Exception as e:
                    logger.warning(
                        f"RecallRPC lookup failed in WebSocket handler: {e}",
                        exc_info=True,
                    )

            if memory_snippets:
                memory_block = (
                    "Relevant past memories about Juniper, Orion, and recent context "
                    "(treat these as ground truth where possible):\n"
                    + "\n".join(f"- {s}" for s in memory_snippets[:6])
                )

                # Remove any previous auto-injected memory blocks so we don’t stack them forever
                history = [
                    m
                    for m in history
                    if not (
                        m.get("role") == "system"
                        and isinstance(m.get("content"), str)
                        and "Relevant past memories about Juniper, Orion, and recent context"
                        in m["content"]
                    )
                ]

                # Insert memory message after:
                #   0: core personality stub
                #   1: optional user instructions (if present)
                insert_idx = 1
                if has_instructions:
                    insert_idx += 1

                history.insert(insert_idx, {"role": "system", "content": memory_block})



            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY BEFORE LLM CALL: {history}")

            # 1) Flatten history + memories into a single prompt
            rpc = BrainRPC(bus)
            rendered_prompt = render_history_to_prompt(history, transcript.strip())

            reply = await rpc.call_llm(
                prompt=rendered_prompt,
                history=[],          # optional: or just omit this parameter
                temperature=temperature,
            )


            orion_response_text = reply.get("text") or ""
            tokens = len(orion_response_text.split())

            # 2) Immediately show the text + tokens to the client
            await websocket.send_json(
                {"llm_response": orion_response_text, "tokens": tokens}
            )

            # 3) Kick off TTS in the background so the UI isn't blocked
            if orion_response_text and not disable_tts:
                asyncio.create_task(
                    run_tts_only(
                        orion_response_text,
                        tts_q,
                        bus=bus,
                        disable_tts=False,
                    )
                )

            if orion_response_text:
                history.append({"role": "orion", "content": orion_response_text})

                if bus is not None and getattr(bus, "enabled", False):
                    latest_user_prompt = transcript

                    user_id = data.get("user_id")
                    session_id = data.get("session_id")

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
                    }

                    try:
                        bus.publish(
                            settings.CHANNEL_COLLAPSE_TRIAGE,
                            full_dialogue_payload,
                        )
                        logger.info(
                            f"Published full dialogue to triage: "
                            f"{full_dialogue_payload['id']}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to publish dialogue to triage: {e}",
                            exc_info=True,
                        )

            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY AFTER LLM CALL: {history}")

            # Mark the end of this turn's processing phase.
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        logger.info("WebSocket closed.")
