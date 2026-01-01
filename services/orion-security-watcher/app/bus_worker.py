from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from .models import VisionEvent

logger = logging.getLogger("orion-security-watcher.bus_worker")


def _model_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _source(ctx) -> ServiceRef:
    return ServiceRef(
        name=ctx.settings.SERVICE_NAME,
        version=ctx.settings.SERVICE_VERSION,
        node=getattr(ctx.settings, "NODE_NAME", None),
    )


async def _handle_envelope(ctx, env: BaseEnvelope) -> None:
    payload: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message" and isinstance(payload.get("payload"), dict):
        payload = payload["payload"]

    try:
        event = VisionEvent.model_validate(payload)
    except Exception:
        # Not a VisionEvent we recognize
        return

    state = ctx.state_store.load()
    visit_summary, alert_payload = ctx.visit_manager.process_event(event, state)

    # Publish visit summary
    if visit_summary is not None and ctx.bus.enabled:
        visit_env = env.derive_child(
            kind="security.visit",
            source=_source(ctx),
            payload=_model_to_dict(visit_summary),
        )
        try:
            await ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_VISITS, visit_env)
        except Exception:
            logger.exception("[SECURITY] Failed to publish visit summary")

    if alert_payload is None:
        return

    # Snapshots
    snapshots = ctx.notifier.capture_snapshots(alert_payload)
    alert_payload.snapshots = snapshots

    # Publish alert payload
    if ctx.bus.enabled:
        alert_env = env.derive_child(
            kind="security.alert",
            source=_source(ctx),
            payload=_model_to_dict(alert_payload),
        )
        try:
            await ctx.bus.publish(ctx.settings.CHANNEL_SECURITY_ALERTS, alert_env)
        except Exception:
            logger.exception("[SECURITY] Failed to publish alert payload")

    # Inline email
    if ctx.settings.NOTIFY_MODE == "inline":
        try:
            ctx.notifier.send_email(alert_payload, snapshots)
        except Exception:
            logger.exception("[SECURITY] Inline email send raised exception")


async def bus_worker(ctx) -> None:
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

    await ctx.bus.connect()

    async with ctx.bus.subscribe(*channels) as pubsub:
        async for msg in ctx.bus.iter_messages(pubsub):
            data = msg.get("data")
            if data is None:
                continue

            decoded = ctx.bus.codec.decode(data)
            if not decoded.ok or decoded.envelope is None:
                continue

            try:
                await _handle_envelope(ctx, decoded.envelope)
            except Exception:
                logger.exception("[SECURITY] Error handling bus message")
