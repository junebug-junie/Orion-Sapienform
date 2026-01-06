from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


def build_state_response(ctx) -> Dict[str, Any]:
    state = ctx.state_store.load()
    
    # FIX: Use guard instead of removed visit_manager
    # guard.last_alert_ts is Dict[str, float] (epoch time)
    all_alerts = ctx.guard.last_alert_ts.values()
    latest_ts_float = max(all_alerts) if all_alerts else None
    
    last_alert_ts = None
    if latest_ts_float:
        last_alert_ts = datetime.fromtimestamp(latest_ts_float, tz=timezone.utc).isoformat()

    return {
        "enabled": ctx.settings.SECURITY_ENABLED,
        "armed": state.armed,
        "mode": state.mode,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "updated_by": state.updated_by,
        "last_alert": {
            "ts": last_alert_ts,
            "reason": "person_presence" if last_alert_ts else None,
        },
    }


@router.get("/security/state")
async def get_security_state():
    from ..context import ctx
    return JSONResponse(build_state_response(ctx))


@router.post("/security/state")
async def set_security_state(payload: Dict[str, Any]):
    from ..context import ctx

    armed = payload.get("armed")
    mode = payload.get("mode")

    if mode is not None:
        mode = str(mode)
        if mode not in ("vacation_strict", "off"):
            mode = ctx.settings.SECURITY_MODE

    ctx.state_store.save(
        armed=bool(armed) if armed is not None else None,
        mode=mode,
        updated_by="ui:tailscale",
    )

    return JSONResponse(build_state_response(ctx))


# UI-compat shortcuts for the Tailwind panel JS
@router.get("/state")
async def get_security_state_root():
    return await get_security_state()


@router.post("/state")
async def set_security_state_root(payload: Dict[str, Any]):
    return await set_security_state(payload)
