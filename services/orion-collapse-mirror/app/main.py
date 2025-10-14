from fastapi import FastAPI
from dotenv import load_dotenv
import asyncio
import traceback
import json
import redis.asyncio as redis

from app import routes
from app.db import init_db, get_db
from app.chroma_db import embedder
from app.settings import settings
from orion.core.bus.service import OrionBus
from app.services.collapse_service import log_and_persist
from orion.schemas.collapse_mirror import CollapseMirrorEntry

load_dotenv()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

app.include_router(routes.router, prefix="/api")

# The synchronous bus is now only used for publishing.
# It's created on-demand to ensure thread-safety.
bus: OrionBus | None = None

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
    }

@app.get("/")
def read_root():
    return {"message": "Conjourney Memory API is alive"}


# 👂 Intake listener (Refactored to handle publishing)
async def listen_for_intake():
    """
    Creates an async connection to listen for intake messages, persists them,
    and then publishes the result to the downstream triage channel.
    """
    listen_channel = settings.CHANNEL_COLLAPSE_INTAKE
    publish_channel = settings.CHANNEL_COLLAPSE_TRIAGE
    
    print(f"📡 Async intake listener connecting to {settings.ORION_BUS_URL}...")

    try:
        # Use the async Redis client for subscribing
        client = redis.from_url(settings.ORION_BUS_URL, decode_responses=True)
        async with client.pubsub() as pubsub:
            await pubsub.subscribe(listen_channel)
            print(f"✅ Subscribed to {listen_channel}. Waiting for messages...")

            async for message in pubsub.listen():
                if message['type'] != 'message':
                    continue

                print("👂 Collapse intake message received:", message['data'])
                try:
                    data = json.loads(message['data'])
                    entry = CollapseMirrorEntry.model_validate(data)

                    db = next(get_db())
                    try:
                        # The service function now returns the payload to be published
                        result = log_and_persist(entry=entry, db=db)
                        payload_to_publish = result.get("payload")
                        
                        if payload_to_publish and bus:
                            bus.publish(publish_channel, payload_to_publish)
                            print(f"📡 Published collapse {result['id']} → {publish_channel}")
                        
                        print(f"✅ Collapse persisted: {entry.summary[:20]}...")
                    finally:
                        db.close()

                except Exception as e:
                    print(f"❌ Intake processing error: {e}")
                    traceback.print_exc()

    except Exception as e:
        print(f"💥 Intake listener fatal error: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    global bus
    init_db()
    try:
        _ = embedder.encode("warmup").tolist()
        print("✅ Embedding model warmed up")
    except Exception as e:
        print("⚠️ Embedding warmup failed:", e)

    # Initialize the synchronous bus for publishing
    if settings.ORION_BUS_ENABLED:
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
        print("📡 Launching async intake listener task...")
        asyncio.create_task(listen_for_intake())
    else:
        print("⚠️ OrionBus disabled, intake listener not started")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8087, reload=True)

