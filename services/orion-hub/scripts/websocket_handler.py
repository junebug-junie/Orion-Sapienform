# services/orion-hub/scripts/websocket_handler.py
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.bus_clients.cortex_client import CortexClient
from scripts.bus_clients.tts_client import TTSClient

logger = logging.getLogger("orion-hub.ws")


async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    try:
        while websocket.client_state.name == "CONNECTED":
            msg = await queue.get()
            await websocket.send_json(msg)
            queue.task_done()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)


async def run_tts_flow(
    text: str,
    queue: asyncio.Queue,
    tts_client: TTSClient,
):
    """
    Helper to synthesize TTS and enqueue the result for the websocket.
    """
    if not text:
        return

    # Notify UI we are working on TTS (optional)
    # await queue.put({"state": "speaking"})

    result = await tts_client.synthesize_speech(text)

    if result.get("ok", True) is False:
        logger.error(f"TTS Error: {result.get('error')}")
        return

    audio_b64 = result.get("audio_b64")
    if audio_b64:
        # Hub UI expects: {"audio_b64": "...", "trace_id": "...", "mime_type": "..."}
        await queue.put({
            "audio_b64": audio_b64,
            "trace_id": result.get("trace_id", "tts-trace"),
            "mime_type": result.get("content_type", "audio/wav"),
        })


async def websocket_endpoint(websocket: WebSocket):
    # Lazy import to ensure bus is initialized
    import scripts.main
    bus = scripts.main.bus

    await websocket.accept()
    logger.info("WebSocket accepted.")

    if not bus:
         logger.error("Bus not available.")
         await websocket.send_json({"error": "Service Unavailable"})
         await websocket.close()
         return

    # Clients
    cortex_client = CortexClient(bus)
    tts_client = TTSClient(bus)

    # Basic History
    history: List[Dict[str, Any]] = []

    # Queue for async messages (TTS chunks, etc)
    tts_q: asyncio.Queue = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            # 1. Receive Raw Data
            raw = await websocket.receive_text()
            
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Extract fields
            mode = data.get("mode", "brain")
            disable_tts = data.get("disable_tts", False)
            session_id = data.get("session_id")
            
            # Transcript / Audio
            # Note: Since we removed native ASR, we expect text from UI (Speech-to-Text done in browser)
            # OR we need to send audio to whisper service.
            # The instructions said: "accept audio or text, forward to tts-whisper over bus RPC".
            # Wait, tts-whisper is for TTS. Does it do ASR?
            # Yes, "orion-whisper-tts". It likely handles ASR too?
            # Let's check `orion-whisper-tts` again. It has `tts.py` and `tts_worker.py`.
            # I don't see `asr.py`.
            # If `orion-whisper-tts` is ONLY TTS, then we have no ASR service?
            # The prompt said "Replace with bus RPC calls to the orion-tts-whisper service".
            # If the user meant ASR too, that service needs to support it.
            # I see `services/orion-hub/scripts/settings.py` had `WHISPER_MODEL_SIZE` etc.
            # And `services/orion-hub/scripts/asr.py`.
            # If I remove native ASR, I must ensure something replaces it.
            # But `orion-whisper-tts` seems to be TTS only based on file names.
            # Let's check `services/orion-whisper-tts/app/main.py` again.
            # It only starts `listener_worker` for TTS.

            # CRITICAL: If there is no ASR service, then Hub cannot support voice input unless the browser sends text.
            # "Hub ... (optionally) send typed speech/tts requests to orion-tts-whisper over the bus".
            # It didn't explicitly say "send ASR requests".
            # But "Remove native ASR/TTS engines from hub... Replace with bus RPC calls to the orion-tts-whisper service".
            # This implies `orion-tts-whisper` handles both or I should assume Text-Input only for now?

            # "For voice endpoints: accept audio or text, forward to tts-whisper over bus RPC, return typed response."
            # This implies tts-whisper does ASR?
            # But I looked at the code of `orion-whisper-tts` and it seems TTS focused.
            # `tts_worker.py` only imports `TTSEngine`.

            # If I am to remove ASR from Hub, and there is no remote ASR, then Hub loses ASR capability.
            # "Hub should no longer contain ... native ASR/TTS engines".
            # Maybe the user assumes `orion-tts-whisper` has it?
            # Or maybe I should assume input is text for now?
            # "accept audio or text".

            # I will implement text path. For audio path, if I cannot find an ASR RPC, I will return an error or log it.
            # Actually, `orion-whisper-tts` name suggests Whisper (ASR) + TTS.
            # But the code I read (`tts_worker.py`) was TTS only.
            # Let me check if there is another worker or file I missed in that dir.

            transcript = data.get("text_input") or data.get("text")

            if not transcript and data.get("audio"):
                 # Placeholder for ASR RPC
                 # If I can't find ASR service, I warn.
                 logger.warning("Audio received but ASR RPC not implemented (orion-whisper-tts appears to be TTS only).")
                 await websocket.send_json({"error": "ASR not available in this refactor."})
                 continue

            if not transcript:
                continue

            # Update History
            history.append({"role": "user", "content": transcript})
            
            # 2. Call Cortex Gateway
            await websocket.send_json({"state": "processing"})

            result = await cortex_client.send_chat_request(
                prompt=transcript,
                mode=mode,
                session_id=session_id,
                # history passed via implicit session context?
                # CortexChatRequest doesn't take history, it takes one prompt.
                # The Gateway builds context.
            )

            response_text = result.get("final_text", "") or ""

            # Reply to UI
            await websocket.send_json({
                "llm_response": response_text,
                "mode": mode,
            })
            
            # 3. TTS
            if response_text and not disable_tts:
                 asyncio.create_task(
                     run_tts_flow(response_text, tts_q, tts_client)
                 )

            history.append({"role": "assistant", "content": response_text})
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
