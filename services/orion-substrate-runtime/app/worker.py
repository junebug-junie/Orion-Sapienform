from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.substrate.biometrics_loop.constants import (
    ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    NODE_BIOMETRICS_PROJECTION_ID,
)
from orion.substrate.biometrics_loop.pipeline import (
    _empty_node_bio,
    _empty_pressure,
    process_biometrics_grammar_events,
)
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.pipeline import process_execution_grammar_events
from orion.substrate.execution_loop.projection import empty_execution_projection

from .publish import publish_accepted_events
from .settings import get_settings
from .store import BiometricsSubstrateStore

logger = logging.getLogger("orion.substrate.runtime")


class BiometricsSubstrateWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = BiometricsSubstrateStore(self._settings.postgres_uri)
        self._catalog = NodeCatalog.load(self._settings.node_catalog_path)
        self._stop = asyncio.Event()
        self._bus = None

    async def start(self) -> None:
        s = self._settings
        if s.orion_bus_enabled and s.publish_accepted_pressure_grammar:
            from orion.core.bus.async_service import OrionBusAsync

            self._bus = OrionBusAsync(url=s.orion_bus_url)
            await self._bus.connect()
        asyncio.create_task(self._poll_loop(), name="biometrics-substrate-poll")

    async def stop(self) -> None:
        self._stop.set()
        if self._bus is not None:
            await self._bus.close()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                last_event_id, to_publish = await asyncio.to_thread(self._tick)
                if self._bus and to_publish:
                    await publish_accepted_events(self._bus, to_publish)
                if last_event_id:
                    created_at = self._store.grammar_event_created_at(last_event_id)
                    if created_at:
                        self._store.advance_cursor(
                            event_id=last_event_id,
                            created_at=created_at,
                        )
                if self._settings.enable_execution_trajectory_reducer:
                    last_exec_id = await asyncio.to_thread(self._execution_tick)
                    if last_exec_id:
                        created_at = self._store.grammar_event_created_at(last_exec_id)
                        if created_at:
                            self._store.advance_execution_cursor(
                                event_id=last_exec_id,
                                created_at=created_at,
                            )
            except Exception:
                logger.exception("biometrics_substrate_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.grammar_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> tuple[str | None, list[GrammarEventV1]]:
        events = self._store.fetch_biometrics_grammar_events(limit=50)
        if not events:
            return None, []

        now = datetime.now(timezone.utc)
        published: list[GrammarEventV1] = []

        def load_node_bio() -> NodeBiometricsProjectionV1:
            loaded = self._store.load_node_biometrics(NODE_BIOMETRICS_PROJECTION_ID)
            return loaded or _empty_node_bio(now)

        def load_pressure() -> ActiveNodePressureProjectionV1:
            loaded = self._store.load_active_pressure(ACTIVE_NODE_PRESSURE_PROJECTION_ID)
            return loaded or _empty_pressure(now)

        def publish_hook(accepted: list[GrammarEventV1]) -> None:
            published.extend(accepted)

        process_biometrics_grammar_events(
            events=events,
            catalog=self._catalog,
            load_node_bio=load_node_bio,
            save_node_bio=self._store.save_node_biometrics,
            load_pressure=load_pressure,
            save_pressure=self._store.save_active_pressure,
            save_receipt=self._store.save_receipt,
            save_emission=self._store.save_emission,
            publish_accepted=publish_hook,
            enable_node_reducer=self._settings.enable_biometrics_node_reducer,
            enable_organ=self._settings.enable_biometrics_pressure_organ,
            enable_pressure_reducer=self._settings.enable_node_pressure_reducer,
            stale_after_sec=self._settings.biometrics_node_stale_after_sec,
            min_confidence=self._settings.biometrics_pressure_min_confidence,
            now=now,
        )

        return events[-1].event_id, published

    def _execution_tick(self) -> str | None:
        events = self._store.fetch_execution_grammar_events(limit=50)
        if not events:
            return None

        now = datetime.now(timezone.utc)

        def load_projection():
            loaded = self._store.load_execution_trajectory(EXECUTION_TRAJECTORY_PROJECTION_ID)
            return loaded or empty_execution_projection(now=now)

        process_execution_grammar_events(
            events=events,
            load_projection=load_projection,
            save_projection=self._store.save_execution_trajectory,
            save_receipt=self._store.save_receipt,
            now=now,
        )

        return events[-1].event_id
