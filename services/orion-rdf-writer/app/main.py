import threading
import logging
import sys
from fastapi import FastAPI
from app.settings import settings
from app.router import router as rdf_router
# Import the new listener_worker function from service.py
from app.service import listener_worker

# Ensure root logger is configured
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

app = FastAPI(title=settings.SERVICE_NAME, version="1.0.0")
app.include_router(rdf_router)

@app.on_event("startup")
def on_startup():
    """
    Starts the single, unified listener worker in a background thread.
    """
    if settings.ORION_BUS_ENABLED:
        print("🚀 Starting RDF-Writer listener thread...", file=sys.stderr)
        threading.Thread(target=listener_worker, daemon=True).start()
    else:
        print("⚠️ Bus is disabled; RDF-Writer will be idle.", file=sys.stderr)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "listening_on": settings.get_all_subscribe_channels(),
        "graphdb_url": settings.GRAPHDB_URL,
        "bus_url": settings.ORION_BUS_URL,
    }

