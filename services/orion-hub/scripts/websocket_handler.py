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

logger = logging.getLogger("voice-app.ws")


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

    history = []
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
            if not history and instructions:
                history.append({"role": "system", "content": instructions})

            history.append({"role": "user", "content": transcript})

            context_len = data.get("context_length", 10)
            if len(history) > context_len:
                if history and history[0]["role"] == "system":
                    history = [history[0]] + history[-(context_len - 1):]
                else:
                    history = history[-context_len:]

            temperature = data.get("temperature", 0.7)

            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY BEFORE LLM CALL: {history}")

            # 1) LLM first (fast), purely text – bus-aware now
            rpc = BrainRPC(bus)
            reply = await rpc.call_llm(transcript, history[:], temperature)

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
                    latest_user_prompt = transcript  # The user input from this turn

                    # Grab metadata from the original websocket message
                    user_id = data.get("user_id")
                    session_id = data.get("session_id")

                    # Create the full payload for the tagging/triage service
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
            # TTS will independently flip state to "speaking" and then back to "idle".
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        logger.info("WebSocket closed.")
