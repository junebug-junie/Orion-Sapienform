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
from scripts.llm_rpc import CouncilRPC
from scripts.warm_start import mini_personality_summary
from scripts.chat_front import run_chat_general, run_chat_agentic

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


async def websocket_endpoint(websocket: WebSocket):
    # [CRITICAL FIX] Lazy import inside the function.
    # We must access scripts.main.bus NOW, not at module load time.
    # This ensures we get the connected bus, not the startup 'None'.
    import scripts.main
    bus = scripts.main.bus
    asr = scripts.main.asr

    await websocket.accept()
    logger.info("WebSocket accepted. Waiting for messages...")

    # [FIX] Do NOT close connection if ASR is missing. Text chat should still work.
    if asr is None:
        logger.warning("ASR model is NOT loaded. Voice input will fail, but text chat is active.")
        await websocket.send_json({"warning": "ASR not loaded; voice disabled."})

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
            
            # [DEBUG] LOG RAW INPUT
            logger.info(f"WS RECEIVED RAW: {raw[:200]}...") 

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
            temperature = data.get("temperature", 0.7)
            user_id = data.get("user_id")
            session_id = data.get("session_id")

            # ---------------------------------------------------------
            # 🗣️ Step 1: Get transcript (Handle multiple keys)
            # ---------------------------------------------------------
            transcript: Optional[str] = None
            is_text_input = False

            # Check ALL common text keys
            possible_text = data.get("text_input") or data.get("text") or data.get("content")
            
            if possible_text:
                transcript = possible_text
                is_text_input = True
            else:
                audio_b64 = data.get("audio")
                if audio_b64:
                    if asr:
                        await websocket.send_json({"state": "processing"})
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                            transcript = asr.transcribe_bytes(audio_bytes)
                        except Exception as e:
                            logger.error(f"ASR Error: {e}")
                            await websocket.send_json({"error": "ASR processing failed"})
                    else:
                        logger.error("Received audio but ASR is not loaded.")
                        await websocket.send_json({"error": "ASR not available"})
                else:
                    logger.warning(f"Message contained no text or audio. Keys found: {list(data.keys())}")
                    continue

            if not transcript:
                logger.info("Empty transcript derived.")
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

            # Trim history
            if context_len and context_len > 0:
                system_msgs = [m for m in history if m.get("role") == "system"]
                other_msgs = [m for m in history if m.get("role") != "system"]
                keep = max(context_len - len(system_msgs), 0)
                trimmed = other_msgs[-keep:] if keep > 0 else []
                history = system_msgs + trimmed

            # ---------------------------------------------------------
            # 🧠 Step 3: Call LLM / Bus
            # ---------------------------------------------------------
            logger.info(f"Routing to mode: {mode}")
            
            # [CHECK] Ensure bus is alive before calling
            if bus is None:
                # Last ditch effort to grab it if startup was slow
                import scripts.main
                bus = scripts.main.bus
                if bus is None:
                    logger.error("CRITICAL: Bus is still None during request handling.")
                    await websocket.send_json({"error": "Orion Bus unavailable"})
                    continue

            orion_response_text = ""
            tokens = 0
            spark_meta = None

            if mode == "council":
                rpc = CouncilRPC(bus)
                reply = await rpc.call_chat(
                    prompt=transcript,
                    history=history[:-1], 
                    temperature=temperature
                )
                orion_response_text = reply.get("text") or reply.get("response") or ""

            elif mode == "agentic":
                convo = await run_chat_agentic(
                    bus,
                    session_id=session_id,
                    user_id=user_id,
                    messages=history[:],
                    chat_mode=mode,
                    temperature=temperature,
                    use_recall=True,
                )
                orion_response_text = convo.get("text") or ""
                tokens = convo.get("tokens") or 0

            else:
                # Default: Chat General (Brain)
                convo = await run_chat_general(
                    bus,
                    session_id=session_id,
                    user_id=user_id,
                    messages=history[:],
                    chat_mode=mode,
                    temperature=temperature,
                    use_recall=True,
                )
                orion_response_text = convo.get("text") or ""
                tokens = convo.get("tokens") or 0
                spark_meta = convo.get("spark_meta")

            # ---------------------------------------------------------
            # 💬 Step 4: Reply
            # ---------------------------------------------------------
            logger.info(f"Orion Response: {orion_response_text[:100]}...")
            await websocket.send_json({
                "llm_response": orion_response_text,
                "tokens": tokens,
                "mode": mode,
            })

            # ---------------------------------------------------------
            # 🔊 Step 5: TTS
            # ---------------------------------------------------------
            if orion_response_text and not disable_tts:
                asyncio.create_task(
                    run_tts_only(orion_response_text, tts_q, bus=bus, disable_tts=False)
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
