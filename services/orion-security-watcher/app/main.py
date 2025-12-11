# services/orion-security-watcher/app/main.py
import asyncio
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from orion.core.bus.service import OrionBus

from .models import AlertPayload, VisionEvent
from .notifications import Notifier
from .settings import get_settings
from .state_store import SecurityStateStore
from .visits import VisitManager

settings = get_settings()

app = FastAPI(
    title="Orion Security Watcher",
    version=settings.SERVICE_VERSION,
)

templates = Jinja2Templates(directory="app/templates")

bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
state_store = SecurityStateStore(settings)
visit_manager = VisitManager(settings)
notifier = Notifier(settings)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _model_to_json_dict(obj: Any) -> Dict[str, Any]:
    """
    Safely convert a pydantic model (or plain dict) into a JSON-serializable dict.
    Uses mode='json' so datetimes become ISO strings.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _build_state_response() -> Dict[str, Any]:
    state = state_store.load()
    last_alert_ts = visit_manager.last_alert_ts.isoformat() if visit_manager.last_alert_ts else None

    return {
        "enabled": settings.SECURITY_ENABLED,
        "armed": state.armed,
        "mode": state.mode,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "updated_by": state.updated_by,
        "last_alert": {
            "ts": last_alert_ts,
            "reason": "rate_limited_or_unknown" if last_alert_ts else None,
        },
    }


# ─────────────────────────────────────────────
# Background bus loop
# ─────────────────────────────────────────────

def _handle_bus_message(msg: Dict[str, Any]) -> None:
    data = msg.get("data")
    if not isinstance(data, dict):
        return

    try:
        event = VisionEvent.model_validate(data)
    except Exception:
        return

    state = state_store.load()

    visit_summary, alert_payload = visit_manager.process_event(event, state)

    # Publish visit summary if we got one
    if visit_summary is not None and bus.enabled:
        try:
            bus.publish(
                settings.CHANNEL_SECURITY_VISITS,
                _model_to_json_dict(visit_summary),
            )
        except Exception:
            pass

    # Handle alert (rate-limited decision already made)
    if alert_payload is not None:
        # Capture snapshots
        snapshots = notifier.capture_snapshots(alert_payload)
        alert_payload.snapshots = snapshots

        # Publish alert on bus (JSON-safe)
        if bus.enabled:
            try:
                bus.publish(
                    settings.CHANNEL_SECURITY_ALERTS,
                    _model_to_json_dict(alert_payload),
                )
            except Exception:
                pass

        # Inline notification
        if settings.NOTIFY_MODE == "inline":
            notifier.send_email(alert_payload, snapshots)


def bus_worker() -> None:
    if not (settings.SECURITY_ENABLED and bus.enabled):
        return

    for msg in bus.subscribe(settings.VISION_EVENTS_SUBSCRIBE_RAW):
        try:
            _handle_bus_message(msg)
        except Exception:
            # Don't kill loop on bad message
            continue


@app.on_event("startup")
async def on_startup():
    if settings.SECURITY_ENABLED and bus.enabled:
        asyncio.create_task(asyncio.to_thread(bus_worker))


# ─────────────────────────────────────────────
# UI + API routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = state_store.load()
    context = {
        "request": request,
        "armed": state.armed,
        "mode": state.mode,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/health")
async def health():
    state = state_store.load()
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "enabled": settings.SECURITY_ENABLED,
        "armed": state.armed,
        "mode": state.mode,
        "bus_enabled": bus.enabled,
        "vision_channel": settings.VISION_EVENTS_SUBSCRIBE_RAW,
    }


# --- Core state endpoints (programmatic paths) ---

@app.get("/security/state")
async def get_security_state():
    return JSONResponse(_build_state_response())


@app.post("/security/state")
async def set_security_state(payload: Dict[str, Any]):
    armed = payload.get("armed")
    mode = payload.get("mode")

    # Normalize mode
    if mode is not None:
        mode = str(mode)
        if mode not in ("vacation_strict", "off"):
            mode = settings.SECURITY_MODE

    state_store.save(
        armed=bool(armed) if armed is not None else None,
        mode=mode,
        updated_by="ui:tailscale",
    )

    return JSONResponse(_build_state_response())


# --- UI-compat shortcuts (/state used by the Tailwind panel JS) ---

@app.get("/state")
async def get_security_state_root():
    return JSONResponse(_build_state_response())


@app.post("/state")
async def set_security_state_root(payload: Dict[str, Any]):
    return await set_security_state(payload)


# --- Test alert routes ---

@app.post("/security/test-alert")
async def test_alert():
    """
    Synthetic alert for pipeline testing (no real vision event).
    """
    state = state_store.load()
    now = datetime.utcnow()

    alert = AlertPayload(
        ts=now,
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
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
            "global_cooldown_sec": settings.SECURITY_GLOBAL_COOLDOWN_SEC,
            "identity_cooldown_sec": settings.SECURITY_IDENTITY_COOLDOWN_SEC,
        },
    )

    snapshots = notifier.capture_snapshots(alert)
    alert.snapshots = snapshots

    if bus.enabled:
        try:
            bus.publish(
                settings.CHANNEL_SECURITY_ALERTS,
                _model_to_json_dict(alert),
            )
        except Exception:
            pass

    if settings.NOTIFY_MODE == "inline":
        notifier.send_email(alert, snapshots)

    return JSONResponse({"ok": True})


@app.post("/test-alert")
async def test_alert_root():
    """
    UI-compat: /test-alert (what the button currently calls).
    Delegates to /security/test-alert logic.
    """
    return await test_alert()
