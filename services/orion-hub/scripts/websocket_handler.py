import logging
import asyncio
import base64
import json
from fastapi import WebSocket, WebSocketDisconnect

from .settings import settings
from .llm_tts_handler import run_llm_tts
# The direct import from .main is removed.

logger = logging.getLogger("voice-app.ws")

async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    """Drains a queue and sends messages to the WebSocket client."""
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            queue.task_done()
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)

async def websocket_endpoint(websocket: WebSocket):
    """Handles the main voice WebSocket lifecycle."""
    # Import shared objects locally inside the function.
    from .main import asr, bus
    
    await websocket.accept()
    logger.info("WebSocket accepted.")
    if asr is None:
        await websocket.send_json({"error": "ASR model not loaded"})
        await websocket.close()
        return

    history = []
    llm_q = asyncio.Queue()
    tts_q = asyncio.Queue()
    asyncio.create_task(drain_queue(websocket, llm_q))
    asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            audio_b64 = data.get("audio")
            if not audio_b64:
                logger.warning("No audio in message.")
                continue
            
            # --- ASR Processing ---
            await websocket.send_json({"state": "processing"})
            audio_bytes = base64.b64decode(audio_b64)
            transcript = asr.transcribe_bytes(audio_bytes)

            if not transcript:
                await websocket.send_json({"llm_response": "I didn't catch that."})
                await websocket.send_json({"state": "idle"})
                continue

            logger.info(f"Transcript: {transcript!r}")
            await websocket.send_json({"transcript": transcript})

            # --- Bus Publishing ---
            if bus and bus.enabled:
                bus.publish(settings.CHANNEL_VOICE_TRANSCRIPT, {"type": "transcript", "content": transcript})
                bus.publish(settings.CHANNEL_BRAIN_INTAKE, {
                    "source": settings.PROJECT, "type": "intake", "content": transcript
                })

            # --- Context Management & LLM Task ---
            instructions = data.get("instructions", "")
            if not history and instructions:
                history.append({"role": "system", "content": instructions})
            history.append({"role": "user", "content": transcript})

            context_len = data.get("context_length", 10)
            if len(history) > context_len:
                if history and history[0]["role"] == "system":
                    history = [history[0]] + history[-context_len:]
                else:
                    history = history[-context_len:]

            temperature = data.get("temperature", 0.7)
            asyncio.create_task(run_llm_tts(history[:], temperature, llm_q, tts_q))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket closed.")

