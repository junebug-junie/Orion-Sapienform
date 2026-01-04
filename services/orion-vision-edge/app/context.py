# app/context.py
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import OrionCodec

from .settings import get_settings
from .capture import CameraSource
from .detectors import build_detectors

# Singletons for this service
settings = get_settings()

bus = OrionBusAsync(
    url=settings.ORION_BUS_URL,
    enabled=settings.ORION_BUS_ENABLED,
    codec=OrionCodec(),
)

camera = CameraSource(settings.SOURCE)

# (name, mode, instance) as defined in build_detectors()
detectors = build_detectors()
