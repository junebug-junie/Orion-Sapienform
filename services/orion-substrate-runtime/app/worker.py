from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.biometrics.node_catalog import NodeCatalog
from orion.core.schemas.drives import DriveStateV1
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.harness_finalize import HarnessPostTurnClosureV1
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
# Staleness gate for the cached drive state: a stalled drive publisher must not
# keep forcing involuntary movement forever off a frozen snapshot. Fail-open
# toward *not* moving.
_DRIVE_STATE_MAX_AGE_SEC = 300.0
# Fixed signal until HarnessPostTurnClosureV1 carries surprise_level_at_draft.
_HARNESS_CLOSURE_UNRESOLVED_ERROR = 0.65


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
        self._sql_engine: Any = None
        # Orion embodiment (C producer): latest drive state cached off the bus,
        # mapped to one involuntary intent per dynamics tick. Default-off.
        self._latest_drive_state: DriveStateV1 | None = None
        self._latest_drive_state_at: datetime | None = None

    @property
    def bus(self):
        return self._bus

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop

    async def start(self) -> None:
        s = self._settings
        if s.orion_bus_enabled:
            from orion.core.bus.async_service import OrionBusAsync

            self._bus = OrionBusAsync(url=s.orion_bus_url)
            await self._bus.connect()
        self._tasks = [
            asyncio.create_task(self._biometrics_poll_loop(), name="biometrics-substrate-poll"),
            asyncio.create_task(self._execution_poll_loop(), name="execution-substrate-poll"),
            asyncio.create_task(self._transport_poll_loop(), name="transport-substrate-poll"),
            asyncio.create_task(self._chat_poll_loop(), name="chat-substrate-poll"),
            asyncio.create_task(self._prune_loop(), name="substrate-receipt-pruner"),
            asyncio.create_task(self._dynamics_tick_loop(), name="substrate-dynamics-tick"),
            asyncio.create_task(self._episodic_tick_loop(), name="substrate-episodic-tick"),
            asyncio.create_task(
                self._attention_broadcast_loop(), name="substrate-attention-broadcast"
            ),
            asyncio.create_task(
                self._endogenous_curiosity_loop(), name="substrate-endogenous-curiosity"
            ),
        ]
        # Orion embodiment C producer: cache DriveStateV1 off the bus so the
        # dynamics tick can map it to an involuntary intent. Gated + fail-open.
        if self._bus is not None and s.embodiment_c_tick_enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._drive_state_listener_loop(), name="substrate-embodiment-drive-listener"
                )
            )
        # Orion embodiment perception ingest: fold town perception into the
        # substrate graph. Gated + fail-open; perception cadence is slower than
        # the drive tick and contributions are bounded, so this does not form a
        # runaway feedback loop with the C producer (see embodiment CD spec).
        if self._bus is not None and s.embodiment_perception_substrate_enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._perception_ingest_loop(),
                    name="substrate-embodiment-perception-ingest",
                )
            )

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

    def _get_substrate_graph_store(self, *, log_label: str):
        """Return the cached shared substrate graph store, building it on first use.

        Shared by ``_write_prediction_error_node`` and ``_dynamics_tick`` — both
        need the same durable store and cache it on ``self._substrate_graph_store``
        so repeated ticks don't rebuild it. Fail-open: returns None on init error
        (caller decides whether that means "skip this tick").
        """
        try:
            store = self._substrate_graph_store
            if store is None:
                from orion.substrate.graphdb_store import build_substrate_store_from_env

                store = build_substrate_store_from_env()
                self._substrate_graph_store = store
            return store
        except Exception:
            logger.exception(log_label)
            return None

    def _get_sql_engine(self):
        """Return a cached SQLAlchemy engine for direct Postgres writes.

        Graph substrate store does not expose ``.engine``; turn referent persistence
        uses ``settings.postgres_uri`` like the orion-thought store pattern.
        Fail-open: returns None on init error.
        """
        try:
            engine = self._sql_engine
            if engine is None:
                from sqlalchemy import create_engine

                engine = create_engine(self._settings.postgres_uri, pool_pre_ping=True)
                self._sql_engine = engine
            return engine
        except Exception:
            logger.exception("turn_referent_sql_engine_init_failed")
            return None

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

        store = self._get_substrate_graph_store(
            log_label="substrate_prediction_error_store_init_failed"
        )
        if store is None:
            logger.warning(
                "substrate_prediction_error_node_skipped_no_store node_id=%s reducer_key=%s",
                node_id,
                reducer_key,
            )
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
            logger.info(
                "substrate_prediction_error_node_written node_id=%s error=%.3f reducer_key=%s",
                node_id,
                error,
                reducer_key,
            )
        except Exception:
            logger.warning(
                "substrate_prediction_error_node_upsert_failed node_id=%s",
                node_id,
                exc_info=True,
            )

    def handle_post_turn_closure(self, closure: HarnessPostTurnClosureV1) -> None:
        """Bridge unresolved harness surprise into durable prediction_error nodes."""
        if not closure.surprise_unresolved:
            return
        if _prediction_error_nodes_enabled():
            logger.info(
                "post_turn_closure_prediction_error_write corr=%s node_id=%s error=%.2f",
                closure.correlation_id,
                f"harness_closure:{closure.correlation_id}",
                _HARNESS_CLOSURE_UNRESOLVED_ERROR,
            )
            self._write_prediction_error_node(
                node_id=f"harness_closure:{closure.correlation_id}",
                error=_HARNESS_CLOSURE_UNRESOLVED_ERROR,
                now=datetime.now(timezone.utc),
                reducer_key="post_turn_closure",
            )
        else:
            logger.info(
                "post_turn_closure_prediction_error_skipped corr=%s reason=SUBSTRATE_WRITE_PREDICTION_ERROR_NODES_disabled",
                closure.correlation_id,
            )

        try:
            from .turn_referent_store import persist_turn_referent

            store = self._get_substrate_graph_store(log_label="turn_referent_store_init_failed")
            engine = getattr(store, "engine", None) if store is not None else None
            if engine is None:
                engine = self._get_sql_engine()
            if engine is not None:
                persist_turn_referent(closure, engine=engine)
        except Exception:
            logger.warning(
                "turn_referent_persist_failed corr=%s",
                closure.correlation_id,
                exc_info=True,
            )

    def _dynamics_tick(self) -> None:
        """Periodic pacemaker for the graph substrate (closes PR #766 rung-1 gap).

        Runs SubstrateDynamicsEngine.tick() against the same durable store
        _write_prediction_error_node writes to, so pressure seeded from
        metadata['prediction_error'] actually propagates instead of sitting
        inert. Default-off, fail-open: never raises out of a tick.
        """
        if not self._settings.enable_dynamics_tick:
            return

        store = self._get_substrate_graph_store(
            log_label="substrate_dynamics_store_init_failed"
        )
        if store is None:
            return

        try:
            from orion.substrate.dynamics import SubstrateDynamicsEngine

            engine = SubstrateDynamicsEngine(store=store)
            result = engine.tick(now=datetime.now(timezone.utc))
            logger.info(
                "substrate_dynamics_tick_completed activation_updates=%d "
                "pressure_updates=%d dormancy_transitions=%d",
                len(result.activation_updates),
                len(result.pressure_updates),
                len(result.dormancy_transitions),
            )
        except Exception:
            logger.exception("substrate_dynamics_tick_failed")

    async def _dynamics_tick_loop(self) -> None:
        interval = float(self._settings.dynamics_tick_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._dynamics_tick)
            except Exception:
                logger.exception("substrate_dynamics_tick_loop_failed")
            # Orion embodiment C producer reuses the dynamics cadence (no new
            # timer, real drive source). Gated + fail-open inside _emit_c_intent.
            try:
                await self._emit_c_intent()
            except Exception:
                logger.exception("substrate_embodiment_c_tick_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _service_ref(self):
        from orion.core.bus.bus_schemas import ServiceRef

        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    def _build_c_intent(self) -> Any:
        """Sync core of the C producer: map the cached drive state to one intent.

        Returns an ``EmbodimentIntentV1`` (or ``None``). Fail-open: gated by
        ``embodiment_c_tick_enabled``; returns ``None`` when disabled, when no
        drive state has been observed yet, or when the cached drive state is
        stale — so neither a missing nor a stalled drive source forces movement.
        """
        if not self._settings.embodiment_c_tick_enabled:
            return None
        drive = self._latest_drive_state
        if drive is None:
            return None
        observed_at = self._latest_drive_state_at
        if observed_at is not None:
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - observed_at).total_seconds()
            if age > _DRIVE_STATE_MAX_AGE_SEC:
                logger.info(
                    "substrate_embodiment_c_intent_skipped_stale age_sec=%.1f", age
                )
                return None
        from orion.embodiment.drive_map import map_drive_state_to_intent

        return map_drive_state_to_intent(
            drive,
            correlation_id=f"substrate-c-tick:{uuid4().hex[:12]}",
            in_conversation=False,
        )

    async def _emit_c_intent(self) -> None:
        """Publish at most one involuntary embodiment intent per dynamics tick.

        Gated + fail-open: builds the intent from the latest cached drive state
        and publishes it as an ``embodiment.intent.v1`` envelope. Never raises.
        """
        if self._bus is None:
            return
        try:
            intent = self._build_c_intent()
        except Exception:
            logger.exception("substrate_embodiment_c_intent_build_failed")
            return
        if intent is None:
            return
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope
            from orion.core.bus.resilience import publish_with_reconnect
            from orion.schemas.embodiment import EMBODIMENT_INTENT_KIND

            env = BaseEnvelope(
                kind=EMBODIMENT_INTENT_KIND,
                source=self._service_ref(),
                correlation_id=uuid4(),
                payload=intent.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus,
                self._settings.embodiment_channel_intent,
                env,
                log_label="substrate_embodiment_intent",
            )
            logger.info(
                "substrate_embodiment_c_intent_published kind=%s source=%s reason=%s",
                intent.kind,
                intent.source,
                intent.reason,
            )
        except Exception:
            logger.exception("substrate_embodiment_c_intent_publish_failed")

    async def _drive_state_listener_loop(self) -> None:
        """Subscribe to the drive-state channel and cache the latest DriveStateV1.

        Mirrors the post-turn-closure decode pattern. Gated at task-creation by
        ``embodiment_c_tick_enabled``; fail-open on decode/validate errors.
        """
        channel = self._settings.drives_state_channel
        logger.info("substrate_embodiment_drive_listener subscribing channel=%s", channel)
        try:
            async with self._bus.subscribe(channel) as pubsub:
                while not self._stop.is_set():
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    if not msg or msg.get("type") not in ("message", "pmessage"):
                        continue
                    try:
                        self._cache_drive_state_message(msg)
                    except Exception:
                        logger.exception("substrate_embodiment_drive_state_handle_failed")
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("substrate_embodiment_drive_listener stopped channel=%s", channel)

    def _cache_drive_state_message(self, raw_msg: dict[str, Any]) -> None:
        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("substrate_embodiment_drive_state_decode_failed: %s", decoded.error)
            return
        try:
            drive = DriveStateV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("substrate_embodiment_drive_state_invalid err=%s", exc)
            return
        self._latest_drive_state = drive
        self._latest_drive_state_at = datetime.now(timezone.utc)

    async def _perception_ingest_loop(self) -> None:
        """Subscribe to town perception and fold it into the substrate graph.

        Gated at task-creation by ``embodiment_perception_substrate_enabled``;
        fail-open on decode/validate/store errors.
        """
        channel = self._settings.embodiment_channel_perception
        logger.info("substrate_embodiment_perception_ingest subscribing channel=%s", channel)
        try:
            async with self._bus.subscribe(channel) as pubsub:
                while not self._stop.is_set():
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    if not msg or msg.get("type") not in ("message", "pmessage"):
                        continue
                    try:
                        await asyncio.to_thread(self._ingest_perception_message, msg)
                    except Exception:
                        logger.exception("substrate_embodiment_perception_ingest_handle_failed")
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("substrate_embodiment_perception_ingest stopped channel=%s", channel)

    def _ingest_perception_message(self, raw_msg: dict[str, Any]) -> None:
        # Defense-in-depth gate (the loop is only spawned when enabled, but keep
        # the flag check here too, mirroring the C-hook and self-state consumers).
        if not self._settings.embodiment_perception_substrate_enabled:
            return
        from orion.schemas.embodiment import WorldPerceptionV1
        from orion.substrate.relational.adapters import map_town_perception_to_substrate

        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("substrate_embodiment_perception_decode_failed: %s", decoded.error)
            return
        try:
            perception = WorldPerceptionV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("substrate_embodiment_perception_invalid err=%s", exc)
            return

        record = map_town_perception_to_substrate({"perception": perception})
        if record is None or not record.nodes:
            return

        store = self._get_substrate_graph_store(
            log_label="substrate_embodiment_perception_store_init_failed"
        )
        if store is None:
            logger.warning("substrate_embodiment_perception_skipped_no_store")
            return

        written = 0
        for node in record.nodes:
            player_id = str(node.metadata.get("player_id") or node.label)
            try:
                store.upsert_node(
                    identity_key=f"town_perception|{player_id}",
                    node=node,
                )
                written += 1
            except Exception:
                logger.warning(
                    "substrate_embodiment_perception_node_upsert_failed player=%s",
                    player_id,
                    exc_info=True,
                )
        logger.info(
            "substrate_embodiment_perception_ingested nodes=%d player_id=%s",
            written,
            perception.player_id,
        )

    def _episodic_tick(self) -> None:
        """Rung 4: roll the last *completed* receipt window into an episode.

        Windows are clock-aligned to episodic_window_seconds, so every tick
        inside the same period sees the same completed window and the same
        receipt set — the derived episode_id is stable and the insert is
        idempotent (ON CONFLICT DO NOTHING). Output is proposal-marked and
        never mutates accepted truth. Default-off, fail-open.
        """
        s = self._settings
        if not s.enable_episodic_tick:
            return
        try:
            from datetime import timedelta

            from orion.substrate.episodic_consolidation import (
                EpisodicConsolidationEvaluator,
            )

            window = max(1, int(s.episodic_window_seconds))
            now = datetime.now(timezone.utc)
            end = datetime.fromtimestamp(
                (int(now.timestamp()) // window) * window, tz=timezone.utc
            )
            start = end - timedelta(seconds=window)
            cap = max(1, int(s.episodic_max_receipts))
            # Fetch past the cap so the evaluator can flag truncation honestly.
            receipts = self._store.fetch_receipts_between(
                start=start, end=end, limit=max(cap * 4, 256)
            )
            evaluator = EpisodicConsolidationEvaluator(
                window_seconds=window, max_receipts_per_episode=cap
            )
            episode = evaluator.consolidate(receipts=receipts, window_end=end)
            if episode is None:
                return
            inserted = self._store.save_episode_summary(episode)
            pruned = self._store.prune_episode_summaries(
                older_than=now - timedelta(days=float(s.episodic_retention_days))
            )
            logger.info(
                "substrate_episodic_tick_completed episode_id=%s receipts=%d "
                "inserted=%s pruned=%d",
                episode.episode_id,
                episode.receipt_count_total,
                inserted,
                pruned,
            )
        except Exception:
            logger.exception("substrate_episodic_tick_failed")

    def _attention_broadcast_tick(self) -> None:
        """Rung 3: run the workspace competition over the substrate graph.

        High-pressure nodes (dynamic_pressure from rung 1, prediction_error,
        and the belief-derived nodes the rung-2 lanes materialize) compete via
        the same select_actions policy the chat frame uses; the winning
        coalition is persisted as a single-row projection other organs can
        query. No action is taken from the broadcast. Default-off, fail-open.
        """
        s = self._settings
        if not s.enable_attention_broadcast:
            return

        store = self._get_substrate_graph_store(
            log_label="substrate_attention_broadcast_store_init_failed"
        )
        if store is None:
            return

        try:
            from orion.substrate.attention_broadcast import (
                broadcast_projection_from_frame,
                build_substrate_attention_frame,
            )

            state = store.snapshot()
            frame = build_substrate_attention_frame(
                nodes=list(state.nodes.values()),
                min_salience=float(s.attention_broadcast_min_salience),
                now=datetime.now(timezone.utc),
            )
            projection = broadcast_projection_from_frame(frame)
            self._store.save_attention_broadcast(projection)
            try:
                self._store.save_coalition_dwell(projection)
            except Exception:
                logger.exception("substrate_coalition_dwell_persist_failed")
            logger.info(
                "substrate_attention_broadcast_completed selected=%s loop=%s "
                "open_loops=%d",
                projection.selected_action_type,
                projection.selected_open_loop_id,
                len(frame.open_loops),
            )
        except Exception:
            logger.exception("substrate_attention_broadcast_failed")

    async def _attention_broadcast_loop(self) -> None:
        interval = float(self._settings.attention_broadcast_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._attention_broadcast_tick)
            except Exception:
                logger.exception("substrate_attention_broadcast_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _repair_appraisal_from_chat(self) -> Any | None:
        """Best-effort repair-pressure input for endogenous curiosity seeds."""
        try:
            projection = self._store.load_chat_session_projection(CHAT_SESSION_PROJECTION_ID)
            if projection is None or not projection.turns:
                return None
            best_level = 0.0
            best_conf = 0.0
            evidence_ids: list[str] = []
            for turn in projection.turns.values():
                level = float(getattr(turn, "repair_pressure_level", 0.0) or 0.0)
                if level <= best_level:
                    continue
                best_level = level
                best_conf = float(getattr(turn, "repair_pressure_confidence", 0.0) or 0.0)
                evidence_ids = list(getattr(turn, "evidence_event_ids", None) or [])[:8]
            if best_level <= 0.0:
                return None
            from types import SimpleNamespace

            return SimpleNamespace(
                dimensions={"level": best_level},
                causal_molecule_ids=evidence_ids,
                summary=f"chat repair pressure level={best_level:.2f}",
                confidence=best_conf or 0.6,
            )
        except Exception:
            logger.exception("substrate_endogenous_curiosity_repair_load_failed")
            return None

    @staticmethod
    def _neutral_frontier_metacog_inputs() -> tuple[Any, Any]:
        """Low-tension metacog inputs so endogenous seeds drive the decision path."""
        from orion.graph_cognition.brief import MetacogPerceptionBriefV1
        from orion.graph_cognition.evidence import SignalEvidenceBundleV1
        from orion.graph_cognition.interpreters import (
            CoherenceAssessmentV1,
            ConceptDriftSignalV1,
            ContradictionCandidateSetV1,
            GoalPressureStateV1,
            GraphCognitionReportV1,
            IdentityConflictSignalV1,
            SocialContinuityAssessmentV1,
        )

        evidence = SignalEvidenceBundleV1(
            spans=(),
            truncated=False,
            degraded=True,
            notes=("endogenous_curiosity_tick",),
        )
        report = GraphCognitionReportV1(
            coherence=CoherenceAssessmentV1(
                score=0.85, confidence=0.5, evidence=evidence, notes=()
            ),
            identity_conflict=IdentityConflictSignalV1(
                conflict_score=0.1, active=False, confidence=0.5, evidence=evidence
            ),
            goal_pressure=GoalPressureStateV1(
                pressure_score=0.1,
                stalled_goal_count=0,
                competing_goal_density=0.0,
                confidence=0.5,
                evidence=evidence,
            ),
            social_continuity=SocialContinuityAssessmentV1(
                continuity_score=0.8, confidence=0.5, degraded=False, evidence=evidence
            ),
            concept_drift=ConceptDriftSignalV1(
                drift_score=0.1, active=False, confidence=0.5, evidence=evidence
            ),
            contradiction_candidates=ContradictionCandidateSetV1(
                candidates=(), confidence=0.5, evidence=evidence
            ),
        )
        brief = MetacogPerceptionBriefV1(
            top_tensions=(),
            top_stabilizers=("coherence",),
            overall_priority="advance",
            recommended_verbs=("observe",),
            confidence=0.5,
            degraded=True,
            supporting_evidence=evidence,
            notes_for_router=("endogenous_curiosity_tick",),
        )
        return report, brief

    def _endogenous_curiosity_tick(self) -> None:
        """Rung 5: seed curiosity candidates from intrinsic substrate signals.

        Reads prediction-error nodes (rung 1), the latest attention broadcast
        (rung 3), and chat repair pressure, then routes bounded
        ``curiosity_candidate`` signals through ``FrontierCuriosityEvaluator``
        without operator_requested. Output is decision/plan only — no expansion,
        landing, or auto-apply. Default-off; kill switch beats enable.
        """
        s = self._settings
        if not s.enable_endogenous_curiosity or s.endogenous_curiosity_kill_switch:
            return

        from orion.substrate.endogenous_curiosity import (
            EndogenousCuriosityConfig,
            endogenous_curiosity_candidates,
        )

        config = EndogenousCuriosityConfig(
            enabled=True,
            kill_switch=False,
            budget=max(1, int(s.endogenous_curiosity_budget)),
            min_repair_level=float(s.endogenous_curiosity_min_repair_level),
        )

        nodes: list[Any] = []
        store = self._get_substrate_graph_store(
            log_label="substrate_endogenous_curiosity_store_init_failed"
        )
        if store is not None:
            try:
                nodes = list(store.snapshot().nodes.values())
            except Exception:
                logger.exception("substrate_endogenous_curiosity_snapshot_failed")

        attention_frame = None
        try:
            broadcast = self._store.load_attention_broadcast()
            if broadcast is not None:
                attention_frame = broadcast.frame
        except Exception:
            logger.exception("substrate_endogenous_curiosity_broadcast_load_failed")

        seeds = endogenous_curiosity_candidates(
            nodes=nodes,
            repair_appraisal=self._repair_appraisal_from_chat(),
            attention_frame=attention_frame,
            config=config,
        )
        if not seeds:
            try:
                self._store.save_endogenous_curiosity_candidates([])
            except Exception:
                logger.exception("substrate_endogenous_curiosity_persist_failed")
            logger.info("substrate_endogenous_curiosity_tick_completed seeds=0 outcome=noop")
            return

        if store is None:
            logger.info(
                "substrate_endogenous_curiosity_tick_completed seeds=%d outcome=skipped_no_store",
                len(seeds),
            )
            return

        try:
            from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator

            cognition_report, perception_brief = self._neutral_frontier_metacog_inputs()
            evaluator = FrontierCuriosityEvaluator(store=store)
            result = evaluator.evaluate(
                anchor_scope="orion",
                subject_ref="entity:orion",
                cognition_report=cognition_report,
                perception_brief=perception_brief,
                operator_requested=False,
                endogenous_signals=seeds,
            )
            endogenous_count = sum(
                1 for sig in result.signals if "endogenous_seed" in (sig.notes or [])
            )
            # Persist a bounded candidate set for the felt-state curiosity lane;
            # endogenous seeds first, then the rest, capped at 8. Best-effort:
            # the tick must never fail because persistence failed.
            try:
                preferred = [
                    sig for sig in result.signals if "endogenous_seed" in (sig.notes or [])
                ]
                rest = [
                    sig for sig in result.signals if "endogenous_seed" not in (sig.notes or [])
                ]
                persisted = (preferred + rest)[:8]
                if persisted:
                    self._store.save_endogenous_curiosity_candidates(persisted)
            except Exception:
                logger.exception("substrate_endogenous_curiosity_persist_failed")
            logger.info(
                "substrate_endogenous_curiosity_tick_completed seeds=%d outcome=%s "
                "task=%s endogenous_signals=%d",
                len(seeds),
                result.decision.outcome,
                result.decision.chosen_task_type,
                endogenous_count,
            )
        except Exception:
            logger.exception("substrate_endogenous_curiosity_tick_failed")

    async def _endogenous_curiosity_loop(self) -> None:
        interval = float(self._settings.endogenous_curiosity_tick_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._endogenous_curiosity_tick)
            except Exception:
                logger.exception("substrate_endogenous_curiosity_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _episodic_tick_loop(self) -> None:
        interval = float(self._settings.episodic_tick_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._episodic_tick)
            except Exception:
                logger.exception("substrate_episodic_tick_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

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
