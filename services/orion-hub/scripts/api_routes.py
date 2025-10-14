import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from orion.schemas.collapse_mirror import CollapseMirrorEntry
# The direct import from .main is removed to prevent circular dependency.

logger = logging.getLogger("voice-app.api")
router = APIRouter()

@router.get("/")
async def root():
    """Serves the main HTML page."""
    # Import locally to ensure the main module is fully loaded first.
    from .main import html_content
    return HTMLResponse(content=html_content, status_code=200)

@router.get("/health")
def health():
    """Provides a simple health check endpoint."""
    return {"status": "ok", "service": settings.SERVICE_NAME}

@router.get("/schema/collapse")
def get_collapse_schema():
    """Exposes the CollapseMirrorEntry schema for UI templating."""
    logger.info("Fetching CollapseMirrorEntry schema")
    return JSONResponse(CollapseMirrorEntry.schema())

@router.post("/submit-collapse")
async def submit_collapse(data: dict):
    """Receives Collapse Mirror data and publishes it to the bus."""
    # Import locally to ensure the main module is fully loaded first.
    from .main import bus
    logger.info(f"🔥 /submit-collapse called with: {data}")

    if not bus or not bus.enabled:
        logger.error("Submission failed: OrionBus is disabled or not connected.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "OrionBus disabled or not connected"}
        )

    logger.info(f"✅ Using bus channel: {settings.CHANNEL_COLLAPSE_INTAKE}")
    try:
        entry = CollapseMirrorEntry(**data).with_defaults()
        bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode='json'))
        logger.info(f"📡 Hub published collapse → {settings.CHANNEL_COLLAPSE_INTAKE}")
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

