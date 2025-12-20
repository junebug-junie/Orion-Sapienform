from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models import VisionEvent

router = APIRouter()


class SimulatePersonRequest(BaseModel):
    stream_id: str = "office-cam"
    camera_id: str = "office-cam"
    bbox: tuple[int, int, int, int] = (100, 80, 80, 120)
    score: float = 0.40
    frame_index: int = 1
    frame_step: int = 5
    repeats: int = 3


def _make_event(req: SimulatePersonRequest, frame_index: int) -> dict:
    ts = datetime.now(timezone.utc)
    return {
        "ts": ts.isoformat(),
        "stream_id": req.stream_id,
        "frame_index": int(frame_index),
        "service": "orion-vision-edge",
        "service_version": "debug",
        "detections": [
            {
                "kind": "yolo",
                "bbox": list(req.bbox),
                "score": float(req.score),
                "label": "person",
                "track_id": None,
                "meta": {},
            }
        ],
        "note": "debug.simulate-person",
        "meta": {"camera": req.camera_id},
    }


def _dump(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


@router.post("/debug/simulate-person")
async def simulate_person(
    payload: SimulatePersonRequest,
    armed: Optional[int] = 0,
    ignore_cooldown: Optional[int] = 0,
    reset_streak: Optional[int] = 0,
    send: Optional[int] = 0,
):
    """
    send=1 will:
      - capture snapshots
      - publish alert to bus
      - send inline email (if NOTIFY_MODE=inline)
    """
    from ..context import ctx

    if armed:
        ctx.state_store.save(armed=True, mode="vacation_strict", updated_by="debug:simulate")

    if reset_streak:
        ctx.visit_manager._human_streak = 0
        ctx.visit_manager._last_human_frame_index = None

    if ignore_cooldown:
        ctx.visit_manager.last_alert_ts = None

    state = ctx.state_store.load()

    visit_summary_out = None
    alert_out = None

    frame_index = payload.frame_index
    for _ in range(max(1, int(payload.repeats))):
        event_dict = _make_event(payload, frame_index)
        event = VisionEvent.model_validate(event_dict)
        visit_summary_out, alert_out = ctx.visit_manager.process_event(event, state)
        frame_index += int(payload.frame_step)

    # Optionally perform the full downstream actions
    if send and alert_out is not None:
        snapshots = ctx.notifier.capture_snapshots(alert_out)
        alert_out.snapshots = snapshots

        if ctx.bus.enabled:
            try:
                ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_ALERTS, _dump(alert_out))
            except Exception:
                pass

        if ctx.settings.NOTIFY_MODE == "inline":
            try:
                ctx.notifier.send_email(alert_out, snapshots)
            except Exception:
                pass

    return JSONResponse(
        {
            "ok": True,
            "state": {"armed": state.armed, "mode": state.mode},
            "thresholds": {
                "min_face_area": getattr(ctx.visit_manager, "min_face_area", None),
                "min_person_area": getattr(ctx.visit_manager, "min_person_area", None),
                "min_yolo_score": getattr(ctx.visit_manager, "min_yolo_score", None),
                "min_human_streak": getattr(ctx.visit_manager, "min_human_streak", None),
                "streak_max_gap": getattr(ctx.visit_manager, "streak_max_gap", None),
                "global_cooldown_sec": getattr(ctx.visit_manager, "global_cooldown", None),
            },
            "debug": {
                "streak": getattr(ctx.visit_manager, "_human_streak", None),
                "last_human_frame_index": getattr(ctx.visit_manager, "_last_human_frame_index", None),
                "last_alert_ts": ctx.visit_manager.last_alert_ts.isoformat() if ctx.visit_manager.last_alert_ts else None,
            },
            "visit_summary": _dump(visit_summary_out),
            "alert_payload": _dump(alert_out),
            "sent": bool(send and alert_out is not None),
        }
    )
