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
from scripts.warm_start import mini_personality_summary

# Import contract schemas
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload

logger = logging.getLogger("orion-hub.ws")


async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    """
    Consumes TTS audio blobs from the queue and streams them to the websocket client.
    """
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


async def run_tts_remote(text: str, tts_client, queue: asyncio.Queue):
    """
    Sends text to TTS service via bus RPC and puts result on the queue.
    """
    if not text.strip():
        return

    try:
        req = TTSRequestPayload(text=text)
        logger.info(f"Sending TTS request for: {text[:30]}...")

        # Blocking call (async but waits for reply)
        result: TTSResultPayload = await tts_client.speak(req)

        # Prepare message for UI (expects 'audio' key with b64)
        msg = {
            "audio": result.audio_b64,
            "text": text, # Echo text back if UI needs it
        }
        await queue.put(msg)

    except Exception as e:
        logger.error(f"TTS Remote Failed: {e}")
        # Optionally notify UI of failure, or just silence


async def websocket_endpoint(websocket: WebSocket):
    # Lazy import to ensure we get initialized clients
    import scripts.main
    bus = scripts.main.bus
    cortex_client = scripts.main.cortex_client
    tts_client = scripts.main.tts_client

    await websocket.accept()
    logger.info("WebSocket accepted. Waiting for messages...")

    # Check for Bus/Clients availability
    if not bus or not cortex_client or not tts_client:
        logger.error("Bus or Clients not initialized. Refusing WS connection.")
        await websocket.send_json({"error": "Service unavailable (Bus/RPC missing)"})
        await websocket.close()
        return

    # Seed conversation history
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": mini_personality_summary()}
    ]
    has_instructions = False

    tts_q: asyncio.Queue = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            # 1. Receive Raw Data
            raw = await websocket.receive_text()
            
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from client")
                continue

            # Mode & Settings
            mode = data.get("mode", "brain")
            disable_tts = data.get("disable_tts", False)
            instructions = data.get("instructions", "")
            context_len = data.get("context_length", 10)
            user_id = data.get("user_id")
            session_id = data.get("session_id")

            # ---------------------------------------------------------
            # 🗣️ Step 1: Input Handling (Text or Audio)
            # ---------------------------------------------------------
            transcript: Optional[str] = None
            is_text_input = False

            possible_text = data.get("text_input") or data.get("text") or data.get("content")
            
            if possible_text:
                transcript = possible_text
                is_text_input = True

            # If audio is present, try to transcribe via Bus RPC
            elif data.get("audio"):
                audio_b64 = data.get("audio")
                if audio_b64:
                    await websocket.send_json({"state": "processing"})
                    try:
                        logger.info(f"Processing audio input via Bus RPC...")
                        stt_req = STTRequestPayload(audio_b64=audio_b64)
                        stt_result: STTResultPayload = await tts_client.transcribe(stt_req)
                        transcript = stt_result.text
                        logger.info(f"Transcribed: {transcript}")
                    except Exception as e:
                        logger.error(f"STT RPC Failed: {e}")
                        await websocket.send_json({"error": "Voice transcription failed"})
                        continue

            if not transcript:
                logger.warning("Empty transcript derived (no text or audio).")
                continue

            logger.info(f"Transcript: {transcript}")

            if not is_text_input:
                await websocket.send_json({"transcript": transcript, "is_text_input": False})

            # ---------------------------------------------------------
            # 🧾 Step 2: Update History
            # ---------------------------------------------------------
            if instructions and not has_instructions:
                history.insert(1, {"role": "system", "content": instructions})
                has_instructions = True

            history.append({"role": "user", "content": transcript})

            # Trim History
            if context_len and context_len > 0:
                system_msgs = [m for m in history if m.get("role") == "system"]
                other_msgs = [m for m in history if m.get("role") != "system"]
                keep = max(context_len - len(system_msgs), 0)
                trimmed = other_msgs[-keep:] if keep > 0 else []
                history = system_msgs + trimmed

            # ---------------------------------------------------------
            # 🧠 Step 3: Call Cortex Gateway via Bus
            # ---------------------------------------------------------
            logger.info(f"Routing to mode: {mode}")

            use_recall = data.get("use_recall", False)

            chat_req = CortexChatRequest(
                prompt=transcript,
                mode=mode,
                session_id=session_id,
                user_id=user_id,
                recall={"enabled": use_recall},
                metadata={"source": "hub_ws"}
            )

            orion_response_text = ""

            try:
                resp: CortexChatResult = await cortex_client.chat(chat_req)
                orion_response_text = resp.final_text or ""
            except Exception as e:
                logger.error(f"Chat RPC Error: {e}")
                await websocket.send_json({"error": f"Chat failed: {str(e)}"})
                continue

            # ---------------------------------------------------------
            # 💬 Step 4: Reply
            # ---------------------------------------------------------
            logger.info(f"Orion Response: {orion_response_text[:100]}...")
            await websocket.send_json({
                "llm_response": orion_response_text,
                "mode": mode,
            })

            # ---------------------------------------------------------
            # 🗄️ Step 4.5: Persist chat turn to SQL writer
            # ---------------------------------------------------------
            try:
                from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

                corr_id = (
                    getattr(resp.cortex_result, "correlation_id", None)
                    or str(uuid.uuid4())
                )

                # Use the corr_id as the row id so we never end up
                # inserting a NULL primary key (and so writes are idempotent).
                row_id = str(corr_id)

                chat_row = {
                    "id": row_id,
                    "correlation_id": str(corr_id),
                    "source": "hub_ws",
                    "prompt": transcript,
                    "response": orion_response_text,
                    "user_id": user_id,
                    "session_id": session_id,
                    "spark_meta": {
                        "mode": mode,
                        "use_recall": bool(use_recall),
                    },
                }

                env = BaseEnvelope(
                    kind="chat.history"
                  , correlation_id=str(corr_id)
                  , source=ServiceRef(name="hub", node=settings.NODE_NAME)
                  , payload=chat_row
                )

                await bus.publish(settings.CHANNEL_CHAT_HISTORY_LOG, env)
            except Exception as e:
                logger.error(f"Failed to publish chat history: {e}", exc_info=True)

            # ---------------------------------------------------------
            # 🔊 Step 5: TTS (Remote)
            # ---------------------------------------------------------
            if orion_response_text and not disable_tts:
                 asyncio.create_task(
                    run_tts_remote(orion_response_text, tts_client, tts_q)
                )

            # ---------------------------------------------------------
            # 📝 Step 6: Log History
            # ---------------------------------------------------------
            if orion_response_text:
                history.append({"role": "assistant", "content": orion_response_text})
            
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
