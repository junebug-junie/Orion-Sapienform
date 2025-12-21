from fastapi import FastAPI
from dotenv import load_dotenv
import asyncio
import traceback
import json
import redis.asyncio as redis
from uuid import uuid4
from datetime import datetime

from app import routes
from app.settings import settings
from orion.core.bus.service import OrionBus
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.exec_worker import start_collapse_mirror_exec_worker

# ───────────────────────────────────────────────
# 🪞 Orion Collapse Mirror — Event Transformer
# ───────────────────────────────────────────────

load_dotenv()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

# Register routes (e.g. /api/log/collapse)
app.include_router(routes.router, prefix="/api")

# Global synchronous bus (used for publishing + exec worker)
bus: OrionBus | None = None


# ───────────────────────────────────────────────
# Health Endpoints
# ───────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick healthcheck endpoint."""
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/")
def read_root():
    return {"message": f"{settings.SERVICE_NAME} is alive"}


# ───────────────────────────────────────────────
# Async Intake Listener
# ───────────────────────────────────────────────

async def listen_for_intake():
    """
    Listens to raw collapse intake messages,
    enriches with ID, timestamp, and provenance,
    then republishes to the triage channel.
    """
    listen_channel = settings.CHANNEL_COLLAPSE_INTAKE
    publish_channel = settings.CHANNEL_COLLAPSE_TRIAGE
    print(f"📡 Connecting to Redis bus → {settings.ORION_BUS_URL}")
    print(f"👂 Listening on {listen_channel} → publishing to {publish_channel}")

    try:
        client = redis.from_url(settings.ORION_BUS_URL, decode_responses=True)
        async with client.pubsub() as pubsub:
            await pubsub.subscribe(listen_channel)
            print(f"✅ Subscribed to {listen_channel}. Awaiting messages...")

            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue

                try:
                    raw = message["data"]
                    print(f"👂 Received intake payload: {raw}")
                    data = json.loads(raw)
                    entry = CollapseMirrorEntry.model_validate(data)

                    # Enrich event with ID, timestamp, provenance
                    collapse_id = f"collapse_{uuid4().hex}"
                    enriched = {
                        "id": collapse_id,
                        "service_name": settings.SERVICE_NAME,
                        "timestamp": datetime.utcnow().isoformat(),
                        **entry.model_dump(),
                    }

                    if bus:
                        bus.publish(publish_channel, enriched)
                        print(f"📤 Published enriched collapse {collapse_id} → {publish_channel}")
                    else:
                        print("⚠️ Bus not initialized; skipping publish.")

                except Exception as e:
                    print(f"❌ Error processing intake message: {e}")
                    traceback.print_exc()

    except asyncio.CancelledError:
        print("🛑 Intake listener task cancelled. Shutting down gracefully.")
    except Exception as e:
        print(f"💥 Fatal intake listener error: {e}")
        traceback.print_exc()


# ───────────────────────────────────────────────
# Lifecycle Events
# ───────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global bus
    if settings.ORION_BUS_ENABLED:
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
        print(f"🚀 {settings.SERVICE_NAME} starting up (v{settings.SERVICE_VERSION})")

        # ✅ Start the exec-step worker (threaded, blocking subscribe)
        start_collapse_mirror_exec_worker(bus)

        # ✅ Start the async intake listener
        asyncio.create_task(listen_for_intake())
    else:
        print("⚠️ OrionBus disabled — intake listener not started.")


@app.on_event("shutdown")
async def shutdown_event():
    print(f"👋 {settings.SERVICE_NAME} shutting down.")


# ───────────────────────────────────────────────
# Local Run (development only)
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
