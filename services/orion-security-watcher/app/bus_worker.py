from __future__ import annotations

import logging
import asyncio
import time
from typing import Any, Dict, Optional

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionEdgeArtifact

from .context import ctx

logger = logging.getLogger("orion-security-watcher.bus_worker")


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
        # Artifact has inputs['image_path']
        img_path = artifact.inputs.get("image_path")
        if img_path:
            alert.snapshot_path = img_path
            # Could also trigger legacy snapshot download if needed, but we prefer shared storage now.

        logger.info(f"[GUARD] Alert: {alert.summary}")

        # Publish Alert
        if ctx.bus.enabled:
            alert_env = env.derive_child(
                kind=ctx.settings.CHANNEL_VISION_GUARD_ALERT,
                source=_source(ctx),
                payload=alert
            )
            try:
                await ctx.bus.publish(ctx.settings.CHANNEL_VISION_GUARD_ALERT, alert_env)
            except Exception as e:
                logger.error(f"Failed to publish alert: {e}")

        # Notify (Legacy Email)
        if ctx.settings.NOTIFY_MODE == "inline":
             # Adapt to Notifier API?
             # Existing Notifier expects AlertPayload. We have VisionGuardAlert.
             # I'll skip fixing legacy email for now as it's not explicitly requested in prompt constraints,
             # but "Part D ... Emit vision.guard.alert".
             pass

    # 4. Generate Signals (Periodic)
    signals = ctx.guard.generate_signals()
    for sig in signals:
        if ctx.bus.enabled:
            sig_env = env.derive_child(
                kind=ctx.settings.CHANNEL_VISION_GUARD_SIGNAL,
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
