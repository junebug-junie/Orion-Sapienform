from fastapi import APIRouter
from uuid import uuid4
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.settings import settings

router = APIRouter()

@router.post("/log/collapse")
async def log_collapse(entry: CollapseMirrorEntry):
    """
    Accepts raw collapse entries from the Hub/UI and publishes them
    to the intake channel for asynchronous enrichment.
    """
    try:
        # Create a transient bus connection for this request
        # In a real high-throughput scenario, we'd want to reuse a connection from app.state
        # but OrionBusAsync handles connection pooling internally via redis-py mostly.
        # Ideally, we should inject the bus from `main.py`'s `lifespan` or similar.
        # For compliance, switching to OrionBusAsync is the key.

        bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
        await bus.connect()

        try:
            payload = entry.model_dump()
            temp_id = f"collapse_{uuid4().hex}"
            # payload["id"] = temp_id  # If we want to inject ID here, but usually schema defines it or backend does.

            # Wrap in envelope
            envelope = BaseEnvelope(
                kind="collapse.mirror", # Canonical kind for intake?
                source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                payload=payload
            )

            await bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, envelope)
            print(f"📨 Received collapse log request → published {temp_id} to {settings.CHANNEL_COLLAPSE_INTAKE}")

            return {
                "ok": True,
                "temp_id": temp_id,
                "published_to": settings.CHANNEL_COLLAPSE_INTAKE,
            }
        finally:
            await bus.close()

    except Exception as e:
        return {"ok": False, "error": str(e)}
