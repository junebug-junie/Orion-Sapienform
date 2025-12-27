"""Orion Bus core.

This package contains:
  - Legacy sync bus wrapper: :mod:`orion.core.bus.service`
  - Async bus wrapper: :mod:`orion.core.bus.service_async`
  - Titanium contracts: :mod:`orion.core.bus.bus_schemas`
  - Invisible chassis patterns: :mod:`orion.core.bus.bus_service_chassis`

Keep this module tiny: only re-export stable entrypoints.
"""

from .service import OrionBus  # legacy sync wrapper
from .service_async import OrionBusAsync
from .bus_schemas import BaseEnvelope
from .bus_service_chassis import Rabbit, Hunter, Clock, ChassisConfig

__all__ = [
    "OrionBus",
    "OrionBusAsync",
    "BaseEnvelope",
    "Rabbit",
    "Hunter",
    "Clock",
    "ChassisConfig",
]
