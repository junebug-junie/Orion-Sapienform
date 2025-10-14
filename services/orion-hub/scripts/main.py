import os
import logging
import asyncio
import base64
import json
import aiohttp
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from scripts.asr import ASR
from scripts.tts import TTS
from orion.core.bus.service import OrionBus
from orion.schemas.collapse_mirror import CollapseMirrorEntry


# ───────────────────────────────────────────────────────────────
# 🪵 Logging
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice-app")


# ───────────────────────────────────────────────────────────────
# ⚙️  Configuration (env-based)
# ───────────────────────────────────────────────────────────────
PROJECT = os.getenv("PROJECT", "orion-janus")
ORION_BUS_URL = os.getenv("ORION_BUS_URL", f"redis://{PROJECT}-bus-core:6379/0")
bus = OrionBus(url=ORION_BUS_URL)

# --- Voice / LLM Channels ---
CHANNEL_VOICE_TRANSCRIPT = os.getenv("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
CHANNEL_VOICE_LLM = os.getenv("CHANNEL_VOICE_LLM", "orion:voice:llm")
CHANNEL_VOICE_TTS = os.getenv("CHANNEL_VOICE_TTS", "orion:voice:tts")

# --- Collapse Event Channels ---
CHANNEL_COLLAPSE_INTAKE = os.getenv("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
CHANNEL_COLLAPSE_TRIAGE = os.getenv("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

# --- Brain Channels ---
CHANNEL_BRAIN_INTAKE = os.getenv("CHANNEL_BRAIN_INTAKE", "orion:brain:intake")
CHANNEL_BRAIN_OUT = os.getenv("CHANNEL_BRAIN_OUT", "orion:brain:out")

# --- Whisper Model Configuration ---
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# --- Cognitive Backend (Brain) ---
BRAIN_URL = os.getenv("BRAIN_URL", "http://orion-brain:8088")
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "60"))


# ───────────────────────────────────────────────────────────────
# 🚀 FastAPI setup
# ───────────────────────────────────────────────────────────────
app = FastAPI()
asr = None
tts = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global asr, tts
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}")
    asr = ASR(WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    tts = TTS()
    logger.info("Startup complete.")


# ───────────────────────────────────────────────────────────────
# 📄 Static Files + UI
# ───────────────────────────────────────────────────────────────
templates_dir = "templates"
html_content = "<html><body><h1>Error: templates/index.html not found</h1></body></html>"
try:
    with open(os.path.join(templates_dir, "index.html"), "r") as f:
        html_content = f.read()
except FileNotFoundError:
    logger.error(f"CRITICAL: Could not read 'index.html' from '{templates_dir}'.")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse(content=html_content, status_code=200)


# ───────────────────────────────────────────────────────────────
# 🧠 WebSocket Logic (Voice → Brain → Speech)
# ───────────────────────────────────────────────────────────────
async def drain_queue(websocket: WebSocket, queue: asyncio.Queue):
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            queue.task_done()
    except Exception as e:
        logger.error(f"drain_queue error: {e}", exc_info=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket accepted.")
    if asr is None:
        await websocket.send_json({"error": "Whisper not loaded"})
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
            temperature = data.get("temperature", 0.7)
            context_len = data.get("context_length", 10)
            instructions = data.get("instructions", "")

            if not audio_b64:
                logger.warning("No audio in message.")
                continue

            await websocket.send_json({"state": "processing"})
            audio_bytes = base64.b64decode(audio_b64)
            transcript = asr.transcribe_bytes(audio_bytes)

            if not transcript:
                await websocket.send_json({"llm_response": "I didn't catch that."})
                await websocket.send_json({"state": "idle"})
                continue

            logger.info(f"Transcript: {transcript!r}")
            await websocket.send_json({"transcript": transcript})

            # 🧠 Publish voice + brain intake events
            bus.publish(CHANNEL_VOICE_TRANSCRIPT, {"type": "transcript", "content": transcript})
            bus.publish(CHANNEL_BRAIN_INTAKE, {
                "source": PROJECT,
                "type": "intake",
                "content": transcript,
            })

            # Manage rolling context
            if not history and instructions:
                history.append({"role": "system", "content": instructions})
            history.append({"role": "user", "content": transcript})

            if len(history) > context_len:
                if history[0]["role"] == "system":
                    history = [history[0]] + history[-context_len:]
                else:
                    history = history[-context_len:]

            asyncio.create_task(run_llm_tts(history[:], temperature, llm_q, tts_q))
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket closed.")


# ───────────────────────────────────────────────────────────────
# 🧩 Brain + TTS Handling
# ───────────────────────────────────────────────────────────────
async def run_llm_tts(history, temperature, llm_q: asyncio.Queue, tts_q: asyncio.Queue):
    try:
        url = BRAIN_URL.rstrip("/") + "/chat"
        payload = {
            "model": os.getenv("LLM_MODEL", "mistral:instruct"),
            "messages": history,
            "temperature": temperature,
            "stream": False
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=LLM_TIMEOUT_S) as r:
                r.raise_for_status()
                data = await r.json()
                logger.info(f"Brain raw response: {data}")

        text = (
            data.get("response")
            or data.get("text")
            or (data.get("message", {}).get("content"))
            or ""
        ).strip()

        tokens = (
            data.get("tokens")
            or data.get("eval_count")
            or len(text.split())
        )

        await llm_q.put({"llm_response": text, "tokens": tokens})
        bus.publish(CHANNEL_VOICE_LLM, {"type": "llm_response", "content": text, "tokens": tokens})
        bus.publish(CHANNEL_BRAIN_OUT, {"type": "brain_response", "content": text, "tokens": tokens})

        if not text:
            await llm_q.put({"state": "idle"})
            return

        await llm_q.put({"state": "speaking"})
        for chunk in tts.synthesize_chunks(text):
            await tts_q.put({"audio_response": chunk})
            bus.publish(CHANNEL_VOICE_TTS, {"type": "audio_response", "size": len(chunk)})
        await llm_q.put({"state": "idle"})

    except Exception as e:
        logger.error(f"run_llm_tts error: {e}", exc_info=True)
        await llm_q.put({"error": "LLM or TTS failed."})
        await llm_q.put({"state": "idle"})


# ───────────────────────────────────────────────────────────────
# 📜 Collapse Mirror Integration
# ───────────────────────────────────────────────────────────────
@app.get("/schema/collapse")
def get_collapse_schema():
    """Expose CollapseMirrorEntry schema for UI templating."""
    logger.info("Fetching CollapseMirrorEntry schema")
    return JSONResponse(CollapseMirrorEntry.schema())


@app.post("/submit-collapse")
async def submit_collapse(data: dict):
    logger.info(f"🔥 /submit-collapse called with: {data}")

    if not bus.enabled:
        logger.error("OrionBus is not connected")
        return {"success": False, "error": "OrionBus disabled or not connected"}

    # ⚠️ Validate channel mapping at runtime
    channel_name = os.getenv("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
    if not channel_name.startswith("orion:"):
        logger.warning(f"⚠️ CHANNEL_COLLAPSE_INTAKE not namespaced correctly → {channel_name}")
    else:
        logger.info(f"✅ Using bus channel: {channel_name}")

    try:
        entry = CollapseMirrorEntry(**data).with_defaults()
        bus.publish(channel_name, entry.dict())
        logger.info(f"📡 Hub published collapse → {channel_name}: {entry.dict()}")
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
