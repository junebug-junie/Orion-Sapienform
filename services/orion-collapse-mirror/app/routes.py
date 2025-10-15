from fastapi import APIRouter
from uuid import uuid4
from orion.core.bus.service import OrionBus
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.settings import settings

router = APIRouter()

# Create a direct OrionBus instance for publishing
bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)

@router.post("/log/collapse")
def log_collapse(entry: CollapseMirrorEntry):
    """
    Accepts raw collapse entries from the Hub/UI and publishes them
    to the intake channel for asynchronous enrichment.
    """
    try:
        payload = entry.model_dump()
        # Give it a temporary client-side ID so you can track before enrichment
        temp_id = f"collapse_{uuid4().hex}"
        payload["id"] = temp_id
        bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, payload)
        print(f"📨 Received collapse log request → published {temp_id} to {settings.CHANNEL_COLLAPSE_INTAKE}")
        return {
            "ok": True,
            "temp_id": temp_id,
            "published_to": settings.CHANNEL_COLLAPSE_INTAKE,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
