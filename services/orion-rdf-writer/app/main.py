import threading, logging, sys
from fastapi import FastAPI
from app.service import OrionRDFWriterService
from app.settings import settings


# --- ADD THIS DEBUG BLOCK ---
print("--- VERIFYING RUNTIME SETTINGS ---", flush=True)
print(f"Loaded BATCH_SIZE: {settings.BATCH_SIZE}", flush=True)
print(f"Loaded LOG_LEVEL: {settings.LOG_LEVEL}", flush=True)
print("----------------------------------", flush=True)
# --- END DEBUG BLOCK ---

# Ensure root logger is configured
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

app = FastAPI(title=settings.SERVICE_NAME, version="1.0.0")
rdf_service = OrionRDFWriterService()

@app.on_event("startup")
def on_startup():
    print(">>> FastAPI startup hook executing...", file=sys.stderr)
    threading.Thread(target=rdf_service.start, daemon=True).start()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_len": len(rdf_service.queue),
        "service": settings.SERVICE_NAME,
        "graphdb_url": settings.GRAPHDB_URL,
        "bus_url": settings.ORION_BUS_URL,
    }

