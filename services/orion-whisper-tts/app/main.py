# services/orion-whisper-tts/app/main.py

import logging
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .settings import settings
from .tts_worker import listener_worker

logging.basicConfig(
    level=logging.INFO,
    format="[WHISPER-TTS] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-whisper-tts")

app = FastAPI(title="Orion Whisper/TTS Service")


@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "Starting Whisper/TTS service %s v%s",
        settings.service_name,
        settings.service_version,
    )
    # Start the bus listener in a background thread
    threading.Thread(target=listener_worker, daemon=True).start()


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.service_version,
        }
    )
