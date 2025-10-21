# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.settings import settings
from app.dream_api import app as dream_api
from app.dream_cycle import run_dream

# --------------------------------------------------
# Optional lifespan hook (can preload models or warm connections)
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🌙 Orion Dream module starting up…")
    yield
    print("💤 Orion Dream module shutting down…")

# --------------------------------------------------
# Main FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Orion Dream Module",
    description="Generates nightly dreams from Orion’s stored memories.",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

# Mount dream_api routes (future: /dreams/trigger, /dreams/logs, etc.)
app.mount("/dreams", dream_api)

# Simple manual trigger endpoint
@app.post("/dreams/run", summary="Manually run the dream cycle")
async def run_dream_endpoint():
    dream_text = await run_dream()
    return {"status": "complete", "dream_text": dream_text}


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
