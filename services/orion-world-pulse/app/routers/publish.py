from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.state import RUN_RESULTS
from app.services.publish_hub import publish_hub_message
from app.services.renderers import render_hub_digest
from app.settings import settings

router = APIRouter()


@router.post("/api/world-pulse/runs/{run_id}/publish-hub-message")
def publish_hub(run_id: str):
    result = RUN_RESULTS.get(run_id)
    if result is None or result.digest is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not settings.world_pulse_hub_messages_enabled:
        result.run.hub_publish_status = "skipped"
        return {"ok": False, "run_id": run_id, "status": "hub_messages_disabled"}
    msg = render_hub_digest(result.digest)
    publish_result = publish_hub_message(message=msg, dry_run=result.run.dry_run)
    result.run.hub_publish_status = str(publish_result.get("status", "failed"))
    return {"run_id": run_id, **publish_result}

