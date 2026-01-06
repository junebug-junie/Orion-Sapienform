# services/orion-hub/scripts/websocket_handler.py
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from scripts.settings import settings
from scripts.warm_start import mini_personality_summary
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload

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

async def run_tts_remote(text: str, tts_client, queue: asyncio.Queue):
    if not text.strip() or not tts_client:
        return
    try:
        req = TTSRequestPayload(text=text)
        result: TTSResultPayload = await tts_client.speak(req)
        msg = {"audio": result.audio_b64, "text": text}
        await queue.put(msg)
    except Exception as e:
        logger.error(f"TTS Remote Failed: {e}")

async def websocket_endpoint(websocket: WebSocket):
    import scripts.main
    bus = scripts.main.bus
    cortex_client = scripts.main.cortex_client
    tts_client = scripts.main.tts_client

    await websocket.accept()
    logger.info("WebSocket accepted.")

    # PATCH: Soft warning if services missing, but keep connection alive
    if not bus or not cortex_client:
        logger.warning("OrionBus/CortexClient not ready. Chat will be limited.")
        await websocket.send_json({
            "llm_response": "[SYSTEM WARNING] Bus disconnected. Brain is offline, but UI is active.", 
            "state": "idle"
        })

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": mini_personality_summary()}
    ]
    
    tts_q: asyncio.Queue = asyncio.Queue()
    drain_task = asyncio.create_task(drain_queue(websocket, tts_q))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mode = data.get("mode", "brain")
            disable_tts = data.get("disable_tts", False)
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
                        await websocket.send_json({"state": "processing"})
                        stt_req = STTRequestPayload(audio_b64=data.get("audio"))
                        stt_result = await tts_client.transcribe(stt_req)
                        transcript = stt_result.text
                    except Exception as e:
                        logger.error(f"STT Error: {e}")
                        await websocket.send_json({"error": "Transcription failed"})
                        continue
                else:
                    await websocket.send_json({"error": "STT service unavailable"})
                    continue

            if not transcript:
                continue

            if not is_text_input:
                await websocket.send_json({"transcript": transcript, "is_text_input": False})

            # 2. Chat Execution
            if not cortex_client:
                await websocket.send_json({"error": "Cortex disconnected (Bus offline)"})
                continue

            logger.info(f"Routing to mode: {mode}")
            trace_id = str(uuid.uuid4())
            
            chat_req = CortexChatRequest(
                prompt=transcript,
                mode=mode,
                session_id=data.get("session_id"),
                user_id=data.get("user_id"),
                trace_id=trace_id,
                recall={"enabled": data.get("use_recall", False)},
                packs=data.get("packs"),
                metadata={"source": "hub_ws"}
            )

            # Handle Verbs selection
            if data.get("verbs"):
                if not chat_req.options: chat_req.options = {}
                chat_req.options["allowed_verbs"] = data.get("verbs")

            orion_response_text = ""
            try:
                resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=trace_id)
                orion_response_text = resp.final_text or ""
            except Exception as e:
                logger.error(f"Chat RPC Error: {e}")
                await websocket.send_json({"error": f"Chat failed: {str(e)}"})
                continue

            # 3. Response & Logging
            await websocket.send_json({
                "llm_response": orion_response_text,
                "mode": mode,
            })

            # Log to SQL (Best Effort) & Trigger Introspection
            if bus:
                try:
                    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
                    
                    # Extract rich metadata
                    gateway_meta = {}
                    if hasattr(resp, "cortex_result") and resp.cortex_result:
                        gateway_meta = resp.cortex_result.metadata or {}
                    
                    spark_meta = {
                        "mode": mode,
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

                    # 1. SQL Log
                    env_sql = BaseEnvelope(
                        kind="chat.history",
                        correlation_id=trace_id,
                        source=ServiceRef(name="hub", node=settings.NODE_NAME),
                        payload=chat_row
                    )
                    await bus.publish(settings.CHANNEL_CHAT_HISTORY_LOG, env_sql)

                    # 2. Spark Introspection Candidate (PATCH: This triggers the thought stream)
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
                    # We use the literal string to ensure it hits the default introspection channel
                    await bus.publish("orion.spark.candidate", env_spark)

                except Exception as e:
                    logger.warning(f"Failed to log/introspect chat: {e}")

            # 4. TTS
            if orion_response_text and not disable_tts and tts_client:
                 asyncio.create_task(run_tts_remote(orion_response_text, tts_client, tts_q))

            if orion_response_text:
                history.append({"role": "assistant", "content": orion_response_text})
            
            await websocket.send_json({"state": "idle"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        drain_task.cancel()
