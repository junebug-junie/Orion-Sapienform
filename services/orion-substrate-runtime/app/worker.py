from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.substrate.biometrics_loop.constants import (
    ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    GRAMMAR_CURSOR_NAME,
    NODE_BIOMETRICS_PROJECTION_ID,
)
from orion.substrate.biometrics_loop.pipeline import (
    _empty_node_bio,
    _empty_pressure,
    process_biometrics_grammar_events,
)
from orion.substrate.execution_loop.constants import (
    EXECUTION_GRAMMAR_CURSOR_NAME,
    EXECUTION_TRAJECTORY_PROJECTION_ID,
)
from orion.substrate.execution_loop.pipeline import process_execution_grammar_events
from orion.substrate.execution_loop.projection import empty_execution_projection
from orion.substrate.transport_loop.constants import (
    TRANSPORT_BUS_PROJECTION_ID,
    TRANSPORT_GRAMMAR_CURSOR_NAME,
)
from orion.substrate.prediction_error import (
    execution_prediction_error,
    transport_prediction_error,
)
from orion.substrate.transport_loop.pipeline import (
    empty_transport_projection,
    process_transport_grammar_events,
)
from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.substrate.chat_loop.constants import (
    CHAT_GRAMMAR_CURSOR_NAME,
    CHAT_SESSION_PROJECTION_ID,
)
from orion.substrate.chat_loop.pipeline import process_chat_grammar_events

from .publish import publish_accepted_events
from .reducer_health import (
    health_snapshots,
    record_cursor_advance,
    record_error,
    record_quarantine,
    record_success,
    record_tick,
)
from .settings import Settings, get_settings
from .store import BiometricsSubstrateStore

logger = logging.getLogger("orion.substrate.runtime")

_PREDICTION_ERROR_NODE_FLAG = "SUBSTRATE_WRITE_PREDICTION_ERROR_NODES"
_TRUTHY = {"1", "true", "yes", "on"}


def _prediction_error_nodes_enabled() -> bool:
    return os.getenv(_PREDICTION_ERROR_NODE_FLAG, "false").strip().lower() in _TRUTHY


def _prediction_error_receipt(
    *,
    reducer_key: str,
    node_id: str,
    prediction_error: float,
    now: datetime,
) -> Any:
    from orion.schemas.reduction_receipt import ReductionReceiptV1
    from orion.schemas.state_delta import StateDeltaV1
    import uuid

    delta_id = f"prediction_error:{reducer_key}:{now.isoformat()}"
    receipt_id = f"receipt:prediction_error:{reducer_key}:{uuid.uuid4().hex[:8]}"
    return ReductionReceiptV1(
        receipt_id=receipt_id,
        state_deltas=[
            StateDeltaV1(
                delta_id=delta_id,
                target_projection=f"substrate.{reducer_key}.projection",
                target_kind="prediction_signal",
                target_id=node_id,
                operation="update",
                after={
                    "node_id": node_id,
                    "pressure_hints": {"prediction_error": round(prediction_error, 4)},
                },
                caused_by_event_ids=[],
                reducer_id=f"substrate.{reducer_key}",
            )
        ],
        created_at=now,
    )


@dataclass(frozen=True)
class ReducerSpec:
    reducer_key: str
    cursor_name: str
    source_service: str
    enabled: Callable[[Settings], bool]
    batch_limit: Callable[[Settings], int]


REDUCER_SPECS: tuple[ReducerSpec, ...] = (
    ReducerSpec(
        reducer_key="biometrics",
        cursor_name=GRAMMAR_CURSOR_NAME,
        source_service="orion-biometrics",
        enabled=lambda s: True,
        batch_limit=lambda s: s.biometrics_grammar_batch_limit,
    ),
    ReducerSpec(
        reducer_key="execution_trajectory",
        cursor_name=EXECUTION_GRAMMAR_CURSOR_NAME,
        source_service="orion-cortex-exec",
        enabled=lambda s: s.enable_execution_trajectory_reducer,
        batch_limit=lambda s: s.execution_grammar_batch_limit,
    ),
    ReducerSpec(
        reducer_key="transport_bus",
        cursor_name=TRANSPORT_GRAMMAR_CURSOR_NAME,
        source_service="orion-bus",
        enabled=lambda s: s.enable_transport_bus_reducer,
        batch_limit=lambda s: s.transport_grammar_batch_limit,
    ),
    ReducerSpec(
        reducer_key="chat_grammar",
        cursor_name=CHAT_GRAMMAR_CURSOR_NAME,
        source_service="orion-hub",
        enabled=lambda s: s.enable_chat_grammar_reducer,
        batch_limit=lambda s: s.chat_grammar_batch_limit,
    ),
)


class BiometricsSubstrateWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = BiometricsSubstrateStore(self._settings.postgres_uri)
        self._catalog = NodeCatalog.load(self._settings.node_catalog_path)
        self._stop = asyncio.Event()
        self._bus = None
        self._tasks: list[asyncio.Task[None]] = []
        self._substrate_graph_store: Any = None

    async def start(self) -> None:
        s = self._settings
        if s.orion_bus_enabled and s.publish_accepted_pressure_grammar:
            from orion.core.bus.async_service import OrionBusAsync

            self._bus = OrionBusAsync(url=s.orion_bus_url)
            await self._bus.connect()
        self._tasks = [
            asyncio.create_task(self._biometrics_poll_loop(), name="biometrics-substrate-poll"),
            asyncio.create_task(self._execution_poll_loop(), name="execution-substrate-poll"),
            asyncio.create_task(self._transport_poll_loop(), name="transport-substrate-poll"),
            asyncio.create_task(self._chat_poll_loop(), name="chat-substrate-poll"),
            asyncio.create_task(self._prune_loop(), name="substrate-receipt-pruner"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._bus is not None:
            await self._bus.close()

    async def _prune_loop(self) -> None:
        interval = float(self._settings.receipt_prune_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("substrate_receipt_prune_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _prune_tick(self) -> None:
        from app.receipt_pruner import (
            log_receipt_pressure,
            maybe_run_emergency_prune,
            refresh_pressure_cache,
            run_safe_prune,
        )

        refresh_pressure_cache(self._store._engine, self._settings)
        run_safe_prune(
            self._store._engine,
            batch_size=int(self._settings.receipt_prune_batch_size),
        )
        maybe_run_emergency_prune(self._store._engine, self._settings)
        log_receipt_pressure(self._store._engine, self._settings)

    async def _biometrics_poll_loop(self) -> None:
        spec = REDUCER_SPECS[0]
        interval = float(self._settings.grammar_poll_interval_sec)
        while not self._stop.is_set():
            enabled = spec.enabled(self._settings)
            record_tick(spec.reducer_key, cursor_name=spec.cursor_name, enabled=enabled)
            try:
                last_event_id, to_publish = await asyncio.to_thread(self._tick)
                if self._bus and to_publish:
                    await publish_accepted_events(
                        self._bus,
                        to_publish,
                        channel=self._settings.accepted_pressure_grammar_channel,
                    )
                if last_event_id:
                    await asyncio.to_thread(
                        self._advance_cursor,
                        spec,
                        last_event_id,
                        self._store.advance_cursor,
                    )
            except Exception:
                logger.exception("biometrics_substrate_tick_failed")
                record_error(
                    spec.reducer_key,
                    cursor_name=spec.cursor_name,
                    enabled=enabled,
                    event_id=None,
                    reason="biometrics_substrate_tick_failed",
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _execution_poll_loop(self) -> None:
        await self._grammar_reducer_poll_loop(REDUCER_SPECS[1], self._execution_tick)

    async def _transport_poll_loop(self) -> None:
        await self._grammar_reducer_poll_loop(REDUCER_SPECS[2], self._transport_tick)

    async def _chat_poll_loop(self) -> None:
        await self._grammar_reducer_poll_loop(REDUCER_SPECS[3], self._chat_tick)

    async def _grammar_reducer_poll_loop(
        self,
        spec: ReducerSpec,
        tick_fn: Callable[[], str | None],
    ) -> None:
        interval = float(self._settings.grammar_poll_interval_sec)
        while not self._stop.is_set():
            enabled = spec.enabled(self._settings)
            if not enabled:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                continue

            record_tick(spec.reducer_key, cursor_name=spec.cursor_name, enabled=True)
            try:
                last_event_id = await asyncio.to_thread(tick_fn)
                if last_event_id:
                    if spec.cursor_name == EXECUTION_GRAMMAR_CURSOR_NAME:
                        advance_fn = self._store.advance_execution_cursor
                    elif spec.cursor_name == CHAT_GRAMMAR_CURSOR_NAME:
                        advance_fn = self._store.advance_chat_cursor
                    else:
                        advance_fn = self._store.advance_transport_cursor
                    await asyncio.to_thread(
                        self._advance_cursor,
                        spec,
                        last_event_id,
                        advance_fn,
                    )
            except Exception:
                logger.exception("%s_substrate_tick_failed", spec.reducer_key)
                record_error(
                    spec.reducer_key,
                    cursor_name=spec.cursor_name,
                    enabled=True,
                    event_id=None,
                    reason=f"{spec.reducer_key}_substrate_tick_failed",
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _advance_cursor(
        self,
        spec: ReducerSpec,
        event_id: str,
        advance_fn: Callable[..., None],
    ) -> None:
        created_at = self._store.grammar_event_created_at(event_id)
        if not created_at:
            reason = "grammar_event_created_at_missing"
            logger.error(
                "substrate_cursor_commit_failed reducer=%s cursor=%s event_id=%s reason=%s",
                spec.reducer_key,
                spec.cursor_name,
                event_id,
                reason,
            )
            record_error(
                spec.reducer_key,
                cursor_name=spec.cursor_name,
                enabled=spec.enabled(self._settings),
                event_id=event_id,
                reason=reason,
            )
            return
        try:
            advance_fn(event_id=event_id, created_at=created_at)
            record_cursor_advance(
                spec.reducer_key,
                cursor_name=spec.cursor_name,
                enabled=spec.enabled(self._settings),
            )
        except Exception as exc:
            reason = f"cursor_advance_failed:{exc}"
            logger.exception(
                "substrate_cursor_commit_failed reducer=%s cursor=%s event_id=%s reason=%s",
                spec.reducer_key,
                spec.cursor_name,
                event_id,
                reason,
            )
            record_error(
                spec.reducer_key,
                cursor_name=spec.cursor_name,
                enabled=spec.enabled(self._settings),
                event_id=event_id,
                reason=reason,
            )
            raise

    def _process_events_with_poison_isolation(
        self,
        *,
        spec: ReducerSpec,
        events: list[GrammarEventV1],
        process_batch: Callable[[list[GrammarEventV1]], None],
    ) -> str | None:
        if not events:
            return None

        try:
            process_batch(events)
            record_success(
                spec.reducer_key,
                cursor_name=spec.cursor_name,
                enabled=spec.enabled(self._settings),
                batch_events=len(events),
            )
            return events[-1].event_id
        except Exception as batch_exc:
            if len(events) == 1:
                event = events[0]
                if self._should_quarantine_poison(spec, event.event_id):
                    return self._quarantine_poison_event(
                        spec=spec,
                        event=event,
                        reason=str(batch_exc),
                    )
                record_error(
                    spec.reducer_key,
                    cursor_name=spec.cursor_name,
                    enabled=spec.enabled(self._settings),
                    event_id=event.event_id,
                    reason=str(batch_exc),
                )
                raise

            logger.warning(
                "substrate_batch_failed_isolating_poison reducer=%s cursor=%s "
                "batch_size=%d reason=%s",
                spec.reducer_key,
                spec.cursor_name,
                len(events),
                batch_exc,
            )
            last_good: str | None = None
            for event in events:
                try:
                    process_batch([event])
                    last_good = event.event_id
                except Exception as exc:
                    if self._should_quarantine_poison(spec, event.event_id):
                        last_good = self._quarantine_poison_event(
                            spec=spec,
                            event=event,
                            reason=str(exc),
                        )
                    else:
                        record_error(
                            spec.reducer_key,
                            cursor_name=spec.cursor_name,
                            enabled=spec.enabled(self._settings),
                            event_id=event.event_id,
                            reason=str(exc),
                        )
                        raise
            if last_good:
                record_success(
                    spec.reducer_key,
                    cursor_name=spec.cursor_name,
                    enabled=spec.enabled(self._settings),
                    batch_events=1,
                )
            return last_good

    def _should_quarantine_poison(self, spec: ReducerSpec, event_id: str) -> bool:
        snap = health_snapshots().get(spec.reducer_key)
        if snap is None:
            return False
        if snap.blocked_event_id != event_id:
            return False
        return snap.blocked_failures >= int(self._settings.reducer_poison_max_retries)

    def _quarantine_poison_event(
        self,
        *,
        spec: ReducerSpec,
        event: GrammarEventV1,
        reason: str,
    ) -> str:
        logger.error(
            "substrate_poison_event_quarantined reducer=%s cursor=%s stream=%s "
            "event_id=%s trace_id=%s reason=%s",
            spec.reducer_key,
            spec.cursor_name,
            spec.source_service,
            event.event_id,
            event.trace_id,
            reason,
        )
        record_error(
            spec.reducer_key,
            cursor_name=spec.cursor_name,
            enabled=spec.enabled(self._settings),
            event_id=event.event_id,
            reason=reason,
        )
        record_quarantine(
            spec.reducer_key,
            cursor_name=spec.cursor_name,
            enabled=spec.enabled(self._settings),
            event_id=event.event_id,
        )
        self._store.save_quarantine(
            reducer_key=spec.reducer_key,
            cursor_name=spec.cursor_name,
            event_id=event.event_id,
            trace_id=event.trace_id,
            reason=reason,
        )
        self._store.save_receipt(
            ReductionReceiptV1(
                receipt_id=f"quarantine:{spec.reducer_key}:{event.event_id}",
                rejected_event_ids=[event.event_id],
                warnings=[f"quarantined:{reason}"],
                created_at=datetime.now(timezone.utc),
            )
        )
        return event.event_id

    def _tick(self) -> tuple[str | None, list[GrammarEventV1]]:
        spec = REDUCER_SPECS[0]
        events = self._store.fetch_biometrics_grammar_events(
            limit=spec.batch_limit(self._settings),
        )
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

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_biometrics_grammar_events(
                events=batch,
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

        last_id = self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )
        return last_id, published

    def _write_prediction_error_node(
        self,
        *,
        node_id: str,
        error: float,
        now: datetime,
        reducer_key: str = "",
    ) -> None:
        """Upsert a durable substrate node carrying the surprise (Rung 1 bridge).

        Default-off (env flag), fail-open: never raises out of a tick. When
        enabled it writes a single node under a fixed identity_key so re-writes
        collapse (no unbounded node growth), letting the dynamics engine seed
        pressure from ``metadata['prediction_error']``.
        """
        if not _prediction_error_nodes_enabled():
            return

        try:
            store = self._substrate_graph_store
            if store is None:
                from orion.substrate.graphdb_store import build_substrate_store_from_env

                store = build_substrate_store_from_env()
                self._substrate_graph_store = store
        except Exception:
            logger.exception("substrate_prediction_error_store_init_failed")
            return

        try:
            from orion.core.schemas.cognitive_substrate import (
                ConceptNodeV1,
                SubstrateProvenanceV1,
                SubstrateSignalBundleV1,
            )
            from orion.substrate.adapters._common import make_temporal

            salience = max(0.0, min(1.0, error))
            node = ConceptNodeV1(
                node_id=node_id,
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"substrate:{node_id}",
                temporal=make_temporal(observed_at=now),
                provenance=SubstrateProvenanceV1(
                    authority="local_inferred",
                    source_kind="substrate_prediction_error",
                    source_channel="substrate.runtime",
                    producer="substrate_runtime_worker",
                    tier_rank=2,
                ),
                signals=SubstrateSignalBundleV1(confidence=1.0, salience=salience),
                metadata={
                    "source_kind": "substrate_prediction_error",
                    "prediction_error": round(salience, 6),
                    "reducer_key": reducer_key,
                },
            )
            store.upsert_node(
                identity_key=f"substrate_prediction_error|{node_id}",
                node=node,
            )
        except Exception:
            logger.warning(
                "substrate_prediction_error_node_upsert_failed node_id=%s",
                node_id,
                exc_info=True,
            )

    def _execution_tick(self) -> str | None:
        spec = REDUCER_SPECS[1]
        events = self._store.fetch_execution_grammar_events(
            limit=spec.batch_limit(self._settings),
        )
        if not events:
            return None

        now = datetime.now(timezone.utc)

        def load_projection():
            loaded = self._store.load_execution_trajectory(EXECUTION_TRAJECTORY_PROJECTION_ID)
            return loaded or empty_execution_projection(now=now)

        prev_projection = load_projection()

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_execution_grammar_events(
                events=batch,
                load_projection=load_projection,
                save_projection=self._store.save_execution_trajectory,
                save_receipt=self._store.save_receipt,
                now=now,
            )

        last_id = self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )

        if last_id is not None:
            curr_projection = load_projection()
            error = execution_prediction_error(prev_projection, curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="execution_trajectory",
                        node_id="node:substrate.execution",
                        prediction_error=error,
                        now=now,
                    )
                )
                self._write_prediction_error_node(
                    node_id="node:substrate.execution",
                    error=error,
                    now=now,
                    reducer_key="execution_trajectory",
                )

        return last_id

    def _chat_tick(self) -> str | None:
        spec = REDUCER_SPECS[3]
        events = self._store.fetch_chat_grammar_events(
            limit=spec.batch_limit(self._settings),
        )
        if not events:
            return None

        now = datetime.now(timezone.utc)

        def _load_chat_projection() -> ChatSessionProjectionV1:
            return self._store.load_chat_session_projection(
                CHAT_SESSION_PROJECTION_ID
            ) or ChatSessionProjectionV1(
                projection_id=CHAT_SESSION_PROJECTION_ID,
                generated_at=datetime.now(timezone.utc),
            )

        def _save_chat_projection(p: ChatSessionProjectionV1) -> None:
            self._store.save_chat_session_projection(p)

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_chat_grammar_events(
                events=batch,
                load_projection=_load_chat_projection,
                save_projection=_save_chat_projection,
                save_receipt=self._store.save_receipt,
                now=now,
            )

        return self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )

    def _transport_tick(self) -> str | None:
        spec = REDUCER_SPECS[2]
        events = self._store.fetch_transport_grammar_events(
            limit=spec.batch_limit(self._settings),
        )
        if not events:
            return None

        now = datetime.now(timezone.utc)

        def load_projection():
            loaded = self._store.load_transport_bus_projection(TRANSPORT_BUS_PROJECTION_ID)
            return loaded or empty_transport_projection(now=now)

        prev_projection = load_projection()

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_transport_grammar_events(
                events=batch,
                load_projection=load_projection,
                save_projection=self._store.save_transport_bus_projection,
                save_receipt=self._store.save_receipt,
                now=now,
                stream_depth_critical=self._settings.bus_stream_depth_critical,
            )

        last_id = self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )

        if last_id is not None:
            curr_projection = load_projection()
            error = transport_prediction_error(prev_projection, curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="transport_bus",
                        node_id="node:substrate.transport",
                        prediction_error=error,
                        now=now,
                    )
                )
                self._write_prediction_error_node(
                    node_id="node:substrate.transport",
                    error=error,
                    now=now,
                    reducer_key="transport_bus",
                )

        return last_id
