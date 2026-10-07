"""Pool-state source for the situation brief's "which model am I running on" line.

GPU pool stage 6.3: ``_fetch_runtime_context`` used to read orion-llm-gateway's ``GET /routes``
(a compatibility view the gateway generated from pool state, being retired in 6.5). It now reads
pool state itself -- one ``orion:gpu_pool:state`` RPC with the pool's config -- and builds the same
per-route view with ``orion.gpu_pool.route_view``. Same bind-once-per-process shape as the other
Redis/bus-backed situation stores; ``state_buses.bind_situation_state_buses`` binds it.

Unbound (a process that never bound a bus) or an unreachable pool both give the all-``unknown``
view, which the runtime provider reports as unavailable -- never a guessed model.
"""
from __future__ import annotations

from typing import Any

from orion.gpu_pool.route_view import fetch_route_view

_BUS: Any = None

SOURCE_NAME = "orion-situational"


def bind_runtime_route_view_bus(bus: Any) -> None:
    global _BUS
    _BUS = bus


async def read_route_view(*, timeout_sec: float) -> dict[str, Any]:
    return await fetch_route_view(_BUS, source=SOURCE_NAME, timeout_sec=timeout_sec)
