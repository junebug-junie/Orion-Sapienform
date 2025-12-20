from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import AlertPayload

router = APIRouter()


def model_to_json_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return dict(obj)


@router.post("/security/test-alert")
async def test_alert():
    from ..context import ctx

    state = ctx.state_store.load()
    now = datetime.utcnow()

    alert = AlertPayload(
        ts=now,
        service=ctx.settings.SERVICE_NAME,
        version=ctx.settings.SERVICE_VERSION,
        alert_id="test-" + now.strftime("%Y%m%dT%H%M%SZ"),
        visit_id="test-visit",
        camera_id="office-cam",
        armed=state.armed,
        mode=state.mode,
        humans_present=True,
        best_identity="unknown",
        best_identity_conf=0.0,
        identity_votes={"unknown": 1.0},
        reason="test_alert",
        severity="low",
        snapshots=[],
        rate_limit={
            "global_blocked": False,
            "identity_blocked": False,
            "global_cooldown_sec": ctx.settings.SECURITY_GLOBAL_COOLDOWN_SEC,
            "identity_cooldown_sec": ctx.settings.SECURITY_IDENTITY_COOLDOWN_SEC,
        },
    )

    snapshots = ctx.notifier.capture_snapshots(alert)
    alert.snapshots = snapshots

    if ctx.bus.enabled:
        try:
            ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_ALERTS, model_to_json_dict(alert))
        except Exception:
            pass

    if ctx.settings.NOTIFY_MODE == "inline":
        ctx.notifier.send_email(alert, snapshots)

    return JSONResponse({"ok": True})


@router.post("/test-alert")
async def test_alert_root():
    return await test_alert()
