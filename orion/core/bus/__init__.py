from __future__ import annotations

from typing import Any

__all__ = ["OrionBusAsync"]


def __getattr__(name: str) -> Any:
    # Avoid import-time circular dependencies by lazily importing the bus.
    if name == "OrionBusAsync":
        from .async_service import OrionBusAsync  # local import

        return OrionBusAsync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
