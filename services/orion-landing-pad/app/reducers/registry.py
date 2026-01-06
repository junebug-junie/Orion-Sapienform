from __future__ import annotations

from typing import Awaitable, Callable, Dict

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.pad import PadEventV1

from .fallback import fallback_reducer
from .stubs import metric_reducer, snapshot_reducer

ReducerFn = Callable[[BaseEnvelope, str], Awaitable[PadEventV1 | None]]


class ReducerRegistry:
    def __init__(self) -> None:
        self._reducers: Dict[str, ReducerFn] = {}
        self._install_defaults()

    def _install_defaults(self) -> None:
        self.register("telemetry.metric.v1", metric_reducer)
        self.register("spark.state.snapshot.v1", snapshot_reducer)

    def register(self, kind: str, reducer: ReducerFn) -> None:
        self._reducers[kind] = reducer

    def get(self, kind: str) -> ReducerFn:
        return self._reducers.get(kind, fallback_reducer)
