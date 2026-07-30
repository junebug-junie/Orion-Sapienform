from __future__ import annotations

from typing import Any


class WorldContextCapsuleCache:
    """Latest orion:world_context:daily_capsule payload, updated in-process by the bus Hunter.

    Single-writer (the Hunter's asyncio task), single-process: no lock needed --
    both the write (main.py's handle_world_context_capsule) and the read
    (executor.py's call_step_services) run on the same event loop thread.
    """

    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None

    def set(self, capsule: dict[str, Any]) -> None:
        self._latest = capsule

    def get(self) -> dict[str, Any] | None:
        return self._latest


_CACHE = WorldContextCapsuleCache()


def get_world_context_capsule_cache() -> WorldContextCapsuleCache:
    return _CACHE
