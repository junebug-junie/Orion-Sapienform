# services/orion-security-watcher/app/bus_worker.py
from __future__ import annotations

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionEdgeArtifact

# Import local notification model
from .models import AlertPayload 
from .context import ctx

logger = logging.getLogger("orion-security-watcher.bus_worker")

VISION_GUARD_ALERT_KIND = "vision.guard.alert.v1"
VISION_GUARD_SIGNAL_KIND = "vision.guard.signal.v1"


def _source(ctx) -> ServiceRef:
    return ServiceRef(
        name=ctx.settings.SERVICE_NAME,
        version=ctx.settings.SERVICE_VERSION,
        node=getattr(ctx.settings, "NODE_NAME", None),
    )


async def _handle_envelope(ctx, env: BaseEnvelope) -> None:
    # 1. Decode Payload
    try:
        # We expect VisionEdgeArtifact
        artifact = VisionEdgeArtifact.model_validate(env.payload)
    except Exception:
        # Ignore other messages
        return

    # 2. Process Artifact -> Alert
    alert = ctx.guard.process_artifact(artifact)

    # 3. Handle Alert
    if alert:
        # Enrich with snapshot if possible
        img_path = artifact.inputs.get("image_path")
        if img_path:
            alert.snapshot_path = img_path

        logger.info(f"[GUARD] Alert: {alert.summary}")

        # Publish Alert (New Schema)
        if ctx.bus.enabled:
            alert_env = env.derive_child(
                kind=VISION_GUARD_ALERT_KIND,
                source=_source(ctx),
                payload=alert
            )
            try:
                await ctx.bus.publish(ctx.settings.CHANNEL_VISION_GUARD_ALERT, alert_env)
            except Exception as e:
                logger.error(f"Failed to publish alert: {e}")

        # Notify (Legacy Email) - ADAPTER LAYER
        if ctx.settings.NOTIFY_MODE == "inline":
            try:
                # Load current security state for context
                state = ctx.state_store.load()
                
                # Convert timestamp (float epoch to datetime)
                ts_dt = datetime.fromtimestamp(alert.ts, tz=timezone.utc)

                # Construct Legacy AlertPayload
                # We map the new VisionGuardAlert fields to the old schema the notifier expects
                legacy_payload = AlertPayload(
                    ts=ts_dt,
                    service=ctx.settings.SERVICE_NAME,
                    version=ctx.settings.SERVICE_VERSION,
                    alert_id=f"guard-{int(alert.ts)}",
                    visit_id="guard-session", # Placeholder
                    camera_id=alert.camera_id,
                    armed=state.armed,
                    mode=state.mode,
                    humans_present=True, # Implied by alert
                    best_identity="unknown",
                    best_identity_conf=0.0,
                    identity_votes={},
                    reason=alert.summary,
                    severity=alert.severity, # "high" matches
                    snapshots=[], 
                    rate_limit={}
                )

                # Capture Snapshots (using legacy notifier logic)
                snapshots = ctx.notifier.capture_snapshots(legacy_payload)
                
                # Send Email
                ctx.notifier.send_email(legacy_payload, snapshots)
                
            except Exception as ex:
                logger.error(f"Failed to send notification email: {ex}", exc_info=True)

    # 4. Generate Signals (Periodic)
    signals = ctx.guard.generate_signals()
    for sig in signals:
        if ctx.bus.enabled:
            sig_env = env.derive_child(
                kind=VISION_GUARD_SIGNAL_KIND,
                source=_source(ctx),
                payload=sig
            )
            try:
                await ctx.bus.publish(ctx.settings.CHANNEL_VISION_GUARD_SIGNAL, sig_env)
            except Exception as e:
                logger.error(f"Failed to publish signal: {e}")


async def bus_worker(ctx) -> None:
    """
    Subscribe to vision artifacts.
    """
    if not (ctx.settings.SECURITY_ENABLED and ctx.bus.enabled):
        logger.info("[SECURITY] Bus worker not started (disabled)")
        return

    channels = [ctx.settings.CHANNEL_VISION_ARTIFACTS]
    logger.info(f"[SECURITY] Subscribing to: {channels}")

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
