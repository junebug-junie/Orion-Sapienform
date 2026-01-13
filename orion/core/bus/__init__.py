from __future__ import annotations

from typing import Any

__all__ = ["OrionBusAsync"]


def __getattr__(name: str) -> Any:
    # Lazy import to avoid circular imports during schema/registry initialization.
    if name == "OrionBusAsync":
        from .async_service import OrionBusAsync  # noqa: WPS433 (import inside function)

        return OrionBusAsync
    raise AttributeError(name)
