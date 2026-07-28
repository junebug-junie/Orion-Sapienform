"""
Bus synaptic transport trigger notifier.

Subscribes to orion:equilibrium:metacog:trigger and emits in-app notifications
when the bus_synaptic transport trigger fires, indicating a real inter-service
RPC anomaly detected by the FalkorDB-based polling system.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.notify import HubNotificationEvent

logger = logging.getLogger("orion-hub.bus_synaptic_notifier")


class BusSynapticTriggerNotifier:
    """
    Watches orion:equilibrium:metacog:trigger for bus_synaptic transport triggers
    and publishes in-app notifications (HubNotificationEvent) when they fire.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        metacog_trigger_channel: str = "orion:equilibrium:metacog:trigger",
        notify_channel: str = "orion:notify:in_app",
    ) -> None:
        self.enabled = enabled
        self.metacog_trigger_channel = metacog_trigger_channel
        self.notify_channel = notify_channel
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self, bus: OrionBusAsync) -> None:
        """Start listening for metacog triggers."""
        if not self.enabled:
            logger.info("bus_synaptic_trigger_notifier disabled")
            return

        if self._task and not self._task.done():
            return

        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="bus-synaptic-trigger-notifier")
        logger.info(
            "bus_synaptic_trigger_notifier started: %s → %s",
            self.metacog_trigger_channel,
            self.notify_channel,
        )

    async def stop(self) -> None:
        """Stop listening."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        """Main listen loop."""
        if not self._bus:
            return

        logger.info("Subscribing to metacog triggers: %s", self.metacog_trigger_channel)
        try:
            async with self._bus.subscribe(self.metacog_trigger_channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    try:
                        await self._handle_metacog_trigger(msg)
                    except Exception as exc:
                        logger.exception("bus_synaptic_notifier_handle_failed: %s", exc)

        except asyncio.CancelledError:
            logger.info("bus_synaptic_trigger_notifier task cancelled")
        except Exception as exc:
            logger.error("bus_synaptic_trigger_notifier loop failed: %s", exc, exc_info=True)

    async def _handle_metacog_trigger(self, msg: Dict[str, Any]) -> None:
        """Process an incoming metacog trigger."""
        if not self._bus:
            return

        decoded = self._bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.debug("Failed to decode metacog trigger message")
            return

        payload = decoded.envelope.payload
        if not payload:
            return

        # Check if this is a transport trigger
        trigger_kind = payload.get("trigger_kind")
        if trigger_kind != "transport":
            return

        # Check if it's specifically the bus_synaptic evidence source
        upstream = payload.get("upstream", {})
        evidence_source = upstream.get("evidence_source")
        if evidence_source != "bus_synaptic_prediction_error":
            return

        # This is a bus_synaptic transport trigger — emit notification
        error_value = upstream.get("error", "unknown")
        reason = payload.get("reason", "")

        notification = HubNotificationEvent(
            kind="info",
            title="Bus Anomaly Detected",
            summary=f"Inter-service RPC prediction error: {error_value}",
            body=(
                f"The bus synaptic transport trigger has fired.\n"
                f"Prediction error: {error_value}\n"
                f"This indicates real inter-service RPC anomalies detected via FalkorDB graph analysis.\n"
                f"Reason: {reason}\n"
                f"Check orion-equilibrium-service logs for details."
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            dismissible=True,
            priority="normal",
        )

        try:
            await self._bus.publish(
                self.notify_channel,
                notification.model_dump(mode="json"),
            )
            logger.info(
                "bus_synaptic_trigger_notification published: error=%s",
                error_value,
            )
        except Exception as exc:
            logger.error("Failed to publish bus_synaptic notification: %s", exc)
