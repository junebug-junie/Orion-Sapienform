from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.state import RUN_RESULTS
from app.services.publish_email import publish_email_preview
from app.services.publish_hub import publish_hub_message
from app.services.renderers import render_email_digest, render_hub_digest
from app.settings import settings
from app.wiring import notify_client

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


@router.post("/api/world-pulse/runs/{run_id}/publish-email")
def publish_email(run_id: str):
    result = RUN_RESULTS.get(run_id)
    if result is None or result.digest is None:
        raise HTTPException(status_code=404, detail="Run not found")
    email = render_email_digest(
        result.digest,
        subject_prefix="Orion Daily World Pulse",
        to=[],
        from_email=None,
        dry_run=settings.world_pulse_email_dry_run,
    )
    ok = publish_email_preview(notify_client=notify_client(), email=email, enabled=settings.world_pulse_email_enabled)
    if not settings.world_pulse_email_enabled:
        result.run.email_status = "skipped"
    else:
        result.run.email_status = "published" if ok else "failed"
    return {"ok": ok, "run_id": run_id, "status": result.run.email_status}
