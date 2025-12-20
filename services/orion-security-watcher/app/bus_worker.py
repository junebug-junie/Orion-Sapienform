from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .models import VisionEvent

logger = logging.getLogger("orion-security-watcher.bus_worker")


def model_to_json_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _coerce_payload_to_dict(payload: Any) -> Optional[Dict[str, Any]]:
    """
    OrionBus may yield msg["data"] as:
      - dict (already decoded)
      - JSON string
      - bytes of JSON
    We normalize to dict or return None.
    """
    if payload is None:
        return None

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="ignore")
        except Exception:
            return None

    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return None
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def handle_bus_message(ctx, msg: Dict[str, Any]) -> None:
    payload = _coerce_payload_to_dict(msg.get("data"))
    if payload is None:
        # Uncomment if you want to see what's being dropped:
        # logger.debug(f"[SECURITY] Dropped non-dict payload type={type(msg.get('data'))}")
        return

    try:
        event = VisionEvent.model_validate(payload)
    except Exception:
        # Not a VisionEvent we recognize
        return

    state = ctx.state_store.load()
    visit_summary, alert_payload = ctx.visit_manager.process_event(event, state)

    # Publish visit summary
    if visit_summary is not None and ctx.bus.enabled:
        try:
            ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_VISITS, model_to_json_dict(visit_summary))
        except Exception:
            logger.exception("[SECURITY] Failed to publish visit summary")

    if alert_payload is None:
        return

    # Snapshots
    snapshots = ctx.notifier.capture_snapshots(alert_payload)
    alert_payload.snapshots = snapshots

    # Publish alert payload
    if ctx.bus.enabled:
        try:
            ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_ALERTS, model_to_json_dict(alert_payload))
        except Exception:
            logger.exception("[SECURITY] Failed to publish alert payload")

    # Inline email
    if ctx.settings.NOTIFY_MODE == "inline":
        try:
            ctx.notifier.send_email(alert_payload, snapshots)
        except Exception:
            logger.exception("[SECURITY] Inline email send raised exception")


def bus_worker(ctx) -> None:
    """
    Subscribe to plausible vision channels to be robust against older configs.
    """
    if not (ctx.settings.SECURITY_ENABLED and ctx.bus.enabled):
        logger.info("[SECURITY] Bus worker not started (disabled or bus not enabled)")
        return

    channels = [
        ctx.settings.VISION_EVENTS_SUBSCRIBE_RAW,
        "vision.events",
        "orion:vision:raw",
    ]
    channels = list({c for c in channels if c})

    logger.info(f"[SECURITY] Subscribing to vision channels: {channels}")

    for msg in ctx.bus.subscribe(*channels):
        try:
            handle_bus_message(ctx, msg)
        except Exception:
            logger.exception("[SECURITY] Error handling bus message")
