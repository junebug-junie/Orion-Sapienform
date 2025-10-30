import logging
import asyncio
import base64
import json
import uuid
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.llm_tts_handler import run_llm_tts

logger = logging.getLogger("voice-app.ws")

async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    """Drains a queue and sends messages to the WebSocket client."""
    try:
        while websocket.client_state.name == 'CONNECTED':
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
    from scripts.main import asr, bus

    await websocket.accept()
    logger.info("WebSocket accepted.")
    if asr is None:
        await websocket.send_json({"error": "ASR model not loaded"})
        await websocket.close()
        return

    history = []
    tts_q = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

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
                await websocket.send_json({"transcript": transcript, "is_text_input": is_text_input})

            if bus and bus.enabled:
                bus.publish(settings.CHANNEL_VOICE_TRANSCRIPT, {"type": "transcript", "content": transcript})
                bus.publish(settings.CHANNEL_BRAIN_INTAKE, {
                    "source": settings.PROJECT, "type": "intake", "content": transcript
                })

                bus.publish(settings.CHANNEL_COLLAPSE_TRIAGE, {
                    "id": str(uuid.uuid4()),
                    "text": transcript,
                    "source": settings.SERVICE_NAME,
                    "ts": datetime.utcnow().isoformat()
                })

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

            assistant_response_text, tokens = await run_llm_tts(history[:], temperature, tts_q, disable_tts)

            await websocket.send_json({"llm_response": assistant_response_text, "tokens": tokens})

            if assistant_response_text:
                history.append({"role": "assistant", "content": assistant_response_text})

            # --- DIAGNOSTIC LOGGING ---
            logger.info(f"HISTORY AFTER LLM CALL: {history}")

            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
        logger.info("WebSocket closed.")

