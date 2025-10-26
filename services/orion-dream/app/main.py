# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import uvicorn
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.settings import settings
from app.dream_api import app as dream_api
from app.dream_cycle import run_dream
from app.context import initialize_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dream-app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌙 Orion Dream module starting up…")

    initialize_bus()

    yield

    logger.info("💤 Orion Dream module shutting down…")

app = FastAPI(
    title="Orion Dream Module",
    description="Generates nightly dreams from Orion’s stored memories.",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

# Mount dream_api routes (future: /dreams/trigger, /dreams/logs, etc.)
app.mount("/dreams", dream_api)

@app.post("/dreams/run", summary="Manually run the dream cycle")
async def run_dream_endpoint():
    """
    Triggers the dream cycle via the event bus.
    This is now a fire-and-forget operation.
    """
    dream_status_msg = await run_dream()
    return {"status": "triggered", "message": dream_status_msg}


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
