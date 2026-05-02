"""Gateway bus chassis — subscription management using Hunter."""
import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter

from .normalization_state import NormalizationStateRegistry
from .processor import SignalProcessor
from .settings import settings
from .signal_window import SignalWindow

logger = logging.getLogger(__name__)


def build_chassis_config() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=settings.SHUTDOWN_GRACE_SEC,
    )


class GatewayService:
    def __init__(self):
        self._cfg = build_chassis_config()
        self._window = SignalWindow(window_sec=settings.SIGNAL_WINDOW_SEC)
        self._norm_state = NormalizationStateRegistry()
        self._hunter: Hunter | None = None

    def _make_processor(self, bus: OrionBusAsync) -> SignalProcessor:
        return SignalProcessor(
            bus=bus,
            signal_window=self._window,
            norm_state=self._norm_state,
            output_channel_prefix=settings.SIGNALS_OUTPUT_CHANNEL,
            passthrough_pattern=settings.SIGNALS_PASSTHROUGH_PATTERN,
            service_ref=ServiceRef(
                name=settings.SERVICE_NAME,
                version=settings.SERVICE_VERSION,
                node=settings.NODE_NAME,
            ),
        )

    async def start(self) -> None:
        patterns = settings.ORGAN_CHANNELS
        logger.info(f"Gateway subscribing to {len(patterns)} channel patterns")

        processor_holder: list[SignalProcessor] = []

        async def handle(env: BaseEnvelope) -> None:
            if not processor_holder:
                return
            await processor_holder[0].handle_envelope(env)

        self._hunter = Hunter(
            self._cfg,
            patterns=patterns,
            handler=handle,
            concurrent_handlers=True,
        )

        await self._hunter.start_background()
        proc = self._make_processor(self._hunter.bus)
        processor_holder.append(proc)
        logger.info("Gateway service started")

    async def stop(self) -> None:
        if self._hunter:
            await self._hunter.stop()
        logger.info("Gateway service stopped")

    def get_signal_window(self) -> SignalWindow:
        return self._window
