from __future__ import annotations

import asyncio
import logging
import math
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import requests

from orion.biometrics.node_catalog import NodeCatalog
from orion.core.schemas.drives import DriveStateV1
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.codebase_delta import CodebaseDeltaV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.harness_finalize import HarnessPostTurnClosureV1
from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1
from orion.structural_mass.git_delta import GitChurnDelta
from orion.structural_mass.graph_delta import GraphStructuralDelta
from orion.structural_mass.pr_lifecycle import PrLifecycleDelta
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
    CodebaseMassBaseline,
    biometrics_prediction_error,
    bus_synaptic_prediction_error,
    chat_prediction_error,
    codebase_prediction_error,
    execution_prediction_error,
    route_prediction_error,
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
from orion.substrate.route_loop.constants import (
    ROUTE_ARBITRATION_PROJECTION_ID,
    ROUTE_GRAMMAR_CURSOR_NAME,
)
from orion.substrate.route_loop.pipeline import process_route_grammar_events
from orion.substrate.route_loop.projection import empty_route_projection

from .health_monitor import HealthMonitor
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
# node_id -> domain key, matching the fixed identities _write_prediction_error_node()
# upserts to and the domain names orion.substrate.attention_self_model expects in
# prediction_error_by_domain. node:substrate.harness_closure is intentionally
# excluded -- it is not one of that reducer's recognized domains.
# node:substrate.codebase (codebase_prediction_error(), 2026-07-31) is ALSO
# intentionally excluded here, for a different reason: the design spec
# (docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, Phase 3)
# explicitly defers wiring any downstream consumer -- including this reducer's
# own prediction_error_by_domain -- until Phase 1's replay reads MET. Confirmed
# live by code review 2026-07-31: without this comment, a future "complete the
# domain parity" pass could add this entry without realizing that decision was
# already made deliberately, not an oversight. Revisit only alongside a real
# Phase 3 patch, not as a drive-by fix.
_PREDICTION_ERROR_DOMAIN_NODE_IDS = {
    "node:substrate.execution": "execution",
    "node:substrate.transport": "transport",
    "node:substrate.biometrics": "biometrics",
    "node:substrate.chat": "chat",
    "node:substrate.route": "route",
    "node:substrate.bus_synaptic": "bus_synaptic",
}


def _prediction_error_nodes_enabled() -> bool:
    return os.getenv(_PREDICTION_ERROR_NODE_FLAG, "false").strip().lower() in _TRUTHY


# docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md
# item 1: five of the six real transport-bus pressure signals have read as
# exactly zero in every window checked so far -- honestly, not from a bug (see
# that spec's "Current architecture" section) -- with the practical effect that
# a real incident could occur and pass unnoticed, since nothing currently logs
# it. `stream_depth_pressure` (the sixth) is deliberately excluded from this
# trigger set for two compounding reasons, not just one: (a) it is structurally
# nonzero on almost every real tick (any nonzero queue depth divides through
# DEFAULT_STREAM_DEPTH_CRITICAL=100_000 to a small positive number), so an
# ">0" check on it would fire constantly rather than flagging a genuine
# incident; and (b) any depth spike large enough to be worth flagging already
# trips `backpressure_count > 0` first -- services/orion-bus's own observer
# (bus_observer.py) marks a stream "backpressure" once its length crosses
# BUS_STREAM_DEPTH_WARNING (default 25,000 -- 25% of the critical threshold
# above), and `backpressure` IS in this trigger set. `max_stream_depth` (its
# raw input) is still surfaced as context below whenever one of the other
# five signals fires.
_TRANSPORT_INCIDENT_FIELDS = (
    "backpressure",
    "catalog_drift_pressure",
    "contract_pressure",
    "observer_failure_pressure",
    "reliability_pressure",
)


def _log_transport_incident_signals(projection: Any) -> None:
    """Log (INFO) any bus whose pressure/count signals are non-trivially nonzero
    this tick -- see docs/superpowers/specs/2026-07-22-transport-bus-signal-
    quality-measurement-design.md item 1. Read-only w.r.t. the projection;
    never raises, so a malformed row can't take down the transport tick."""
    try:
        for bus_id, bus in (projection.buses or {}).items():
            nonzero = {
                field: value
                for field in _TRANSPORT_INCIDENT_FIELDS
                if (value := getattr(bus, field, 0.0)) > 0.0
            }
            if nonzero:
                logger.info(
                    "transport_incident_signal bus=%s nonzero=%s max_stream_depth=%s",
                    bus_id,
                    nonzero,
                    getattr(bus, "max_stream_depth", None),
                )
    except Exception:
        logger.warning("failed to check transport incident signals", exc_info=True)


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
    ReducerSpec(
        reducer_key="route_grammar",
        cursor_name=ROUTE_GRAMMAR_CURSOR_NAME,
        source_service="orion-cortex-orch",
        enabled=lambda s: s.enable_route_grammar_reducer,
        batch_limit=lambda s: s.route_grammar_batch_limit,
    ),
)


class BiometricsSubstrateWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = BiometricsSubstrateStore(self._settings.postgres_uri)
        self._catalog = NodeCatalog.load(self._settings.node_catalog_path)
        self._health_monitor = HealthMonitor(self._store, self._settings)
        self._stop = asyncio.Event()
        self._bus = None
        self._tasks: list[asyncio.Task[None]] = []
        self._substrate_graph_store: Any = None
        self._sql_engine: Any = None
        self._bus_synaptic_client: Any = None
        # Orion embodiment (C producer): latest drive state cached off the bus,
        # mapped to one involuntary intent per dynamics tick. Default-off.
        self._latest_drive_state: DriveStateV1 | None = None
        self._latest_drive_state_at: datetime | None = None
        # FieldChannelAnomalyScoreV1 cached off the bus for the brain-frame
        # field_anomaly dimension -- mood-arc encoder reconstruction error over
        # live field-channel pressures, independent of the honesty_metrics
        # (active-inference) and spotlight (coalition_stability) signals.
        self._latest_field_anomaly: FieldChannelAnomalyScoreV1 | None = None
        self._latest_field_anomaly_at: datetime | None = None
        self._brain_frame_seq: int = 0
        # AST/HOT self-model tick's in-process trend buffer (docs/superpowers/
        # specs/2026-07-29-ast-hot-reducer-live-ticking-design.md). Private,
        # per-process state -- never persisted mid-computation, only the
        # reducer's own final prediction_error_trend_by_domain output is
        # written anywhere (to substrate_attention_self_model, this tick's
        # sole owned table). Restart-loses-window by design (see settings.py's
        # attention_self_model_trend_window_ticks docstring).
        self._attention_self_model_trend_buffer: deque[dict[str, float]] = deque(
            maxlen=max(2, int(self._settings.attention_self_model_trend_window_ticks))
        )

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
            asyncio.create_task(self._route_poll_loop(), name="route-substrate-poll"),
            asyncio.create_task(self._prune_loop(), name="substrate-receipt-pruner"),
            asyncio.create_task(self._health_loop(), name="substrate-health-monitor"),
            asyncio.create_task(self._dynamics_tick_loop(), name="substrate-dynamics-tick"),
            asyncio.create_task(self._episodic_tick_loop(), name="substrate-episodic-tick"),
            asyncio.create_task(
                self._bus_synaptic_tick_loop(), name="substrate-bus-synaptic-tick"
            ),
            asyncio.create_task(
                self._attention_broadcast_loop(), name="substrate-attention-broadcast"
            ),
            asyncio.create_task(
                self._endogenous_curiosity_loop(), name="substrate-endogenous-curiosity"
            ),
            asyncio.create_task(self._brain_frame_loop(), name="substrate-brain-frame"),
        ]
        # Drive-state listener: feeds the embodiment C producer
        # (embodiment_c_tick_enabled, default off). Chat stance / Mind measurement
        # SoR is Postgres drive_audits, not this listener. The optional legacy
        # substrate-graph materialization that used to also gate this task (plus
        # its own drive-audit listener) was removed 2026-07-30: DriveStateV1/
        # DriveAuditV1 are write-never now that DriveEngine (the only producer)
        # is deleted, so that path could never fire again regardless of flag.
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
        # Field-channel anomaly listener: caches the latest mood-arc encoder
        # reconstruction-error reading for the brain-frame field_anomaly
        # dimension. Read-only display signal, no feature flag beyond bus
        # availability -- rides on the same s.brain_frame_enabled gate that
        # already governs whether anything consumes the cached value.
        if self._bus is not None:
            self._tasks.append(
                asyncio.create_task(
                    self._field_channel_anomaly_listener_loop(),
                    name="substrate-field-channel-anomaly-listener",
                )
            )
        # codebase_prediction_error consumer (docs/superpowers/specs/2026-07-
        # 30-codebase-mass-signal-design.md). Own dedicated flag, default off
        # -- see SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE's own comment
        # in settings.py for why this doesn't reuse
        # SUBSTRATE_WRITE_PREDICTION_ERROR_NODES.
        if self._bus is not None and s.enable_codebase_prediction_error_node:
            self._tasks.append(
                asyncio.create_task(
                    self._codebase_delta_listener_loop(),
                    name="substrate-codebase-delta-listener",
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

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._health_monitor.run_tick)
            except Exception:
                logger.exception("substrate_runtime_health_check_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.health_check_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

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

    async def _route_poll_loop(self) -> None:
        await self._grammar_reducer_poll_loop(REDUCER_SPECS[4], self._route_tick)

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
                    elif spec.cursor_name == ROUTE_GRAMMAR_CURSOR_NAME:
                        advance_fn = self._store.advance_route_cursor
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

        prev_projection = load_node_bio()

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

        if last_id is not None:
            curr_projection = load_node_bio()
            error = biometrics_prediction_error(prev_projection, curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="node_biometrics",
                        node_id="node:substrate.biometrics",
                        prediction_error=error,
                        now=now,
                    )
                )
            # Write the node on every tick, not just when error > 0.0 -- the
            # same fix already applied to bus_synaptic (2026-07-30), extended
            # here after `route` was found frozen at a stale 0.00025 while its
            # tick kept running: a categorical/quiet domain returns error == 0.0
            # on most ticks, so gating the node write left the node holding its
            # last NON-ZERO value indefinitely. Consumers that poll this node's
            # raw current value (orion-equilibrium-service, AST/HOT's
            # prediction_error_by_domain) then read a stale high-water mark as
            # if it were a current reading -- a value that can rise but never
            # come back down to a genuine calm 0.0. The receipt above stays
            # gated: it is an audit trail of notable events, not a polled
            # "current state" read, so skipping it on a calm tick is correct.
            self._write_prediction_error_node(
                node_id="node:substrate.biometrics",
                error=error,
                now=now,
                reducer_key="node_biometrics",
                contributing_id=last_id,
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

    def _get_bus_synaptic_client(self):
        """Return the cached read-only FalkorDB client for the bus synaptic graph
        (``orion_bus_synapse``, written by ``orion-bus-mirror``), building it on
        first use.

        Deliberately a separate client from ``_get_substrate_graph_store`` --
        that one is a ``ConceptNodeV1``-aware store bound to this service's own
        durable substrate graph name; this is a plain read-only Cypher client
        pointed at a *different* graph name on the same FalkorDB instance
        (``FALKORDB_URI`` already used by ``_get_substrate_graph_store``, same
        pattern ``orion.substrate.graphdb_store.build_substrate_store_from_env``
        reads it -- ``os.environ`` directly, not a pydantic ``Settings`` field).
        Fail-open: returns None on init error (caller decides whether that means
        "skip this tick").
        """
        try:
            client = self._bus_synaptic_client
            if client is None:
                from orion.graph.falkor_client import RedisGraphQueryClient

                client = RedisGraphQueryClient(
                    uri=os.getenv("FALKORDB_URI", "redis://localhost:6379"),
                    graph_name=self._settings.falkordb_bus_graph,
                )
                self._bus_synaptic_client = client
            return client
        except Exception:
            logger.exception("substrate_bus_synaptic_client_init_failed")
            return None

    _BUS_SYNAPTIC_EDGE_QUERIES = (
        "MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel) "
        "WHERE e.gap_zscore IS NOT NULL AND e.count > $min_count "
        "AND e.last_seen_epoch > $stale_cutoff_epoch "
        "RETURN e.gap_zscore AS zscore",
        "MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ) "
        "WHERE e.latency_zscore IS NOT NULL AND e.count > $min_count "
        "AND e.last_seen_epoch > $stale_cutoff_epoch "
        "RETURN e.latency_zscore AS zscore",
    )

    def _bus_synaptic_tick(self) -> None:
        """Periodic read of the live bus synaptic graph, feeding
        ``bus_synaptic_prediction_error`` (mirrors the shared writer the other
        five Active-Inference domains use). Default-off, fail-open: never raises
        out of a tick.

        No grammar-event batch, no persisted ``prev`` projection -- unlike
        every other ``_*_tick`` in this module, this reads FalkorDB fresh each
        call. The z-score baseline this reads is already maintained
        continuously by ``orion-bus-mirror``'s own EWMA writer -- see
        ``bus_synaptic_prediction_error``'s docstring for why that means no
        prev/curr diff is needed here.

        ``e.last_seen_epoch > stale_cutoff`` (review finding, 2026-07-25):
        neither this query nor the Hub debug routes it mirrors filter on edge
        recency -- an edge from a decommissioned organ or abandoned channel
        keeps its frozen last-computed z-score forever once ``count`` clears
        the cold-start floor, which would silently drag this "right now"
        aggregate with stale history. Bounded by
        ``bus_synaptic_max_edge_age_sec`` (default 1h) so a long-dead edge
        ages out of the aggregate instead of contributing forever.
        """
        if not self._settings.enable_bus_synaptic_tick:
            return

        client = self._get_bus_synaptic_client()
        if client is None:
            return

        try:
            min_count = self._settings.bus_synaptic_min_edge_count
            stale_cutoff_epoch = (
                datetime.now(timezone.utc).timestamp()
                - self._settings.bus_synaptic_max_edge_age_sec
            )
            params = {"min_count": min_count, "stale_cutoff_epoch": stale_cutoff_epoch}
            zscores: list[float] = []
            for cypher in self._BUS_SYNAPTIC_EDGE_QUERIES:
                rows = client.graph_query(cypher, params)
                for row in rows:
                    value = row.get("zscore") if isinstance(row, dict) else None
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        zscores.append(float(value))

            error = bus_synaptic_prediction_error(zscores)
            now = datetime.now(timezone.utc)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="bus_synaptic",
                        node_id="node:substrate.bus_synaptic",
                        prediction_error=error,
                        now=now,
                    )
                )
            # Write the node on every tick, not just when error > 0.0 (confirmed
            # live 2026-07-30: gating this the same way as the receipt above left
            # `node:substrate.bus_synaptic` frozen at a stale prediction_error=1.0
            # for hours while this tick kept logging real, varying, non-saturated
            # values every 30s the whole time -- orion-equilibrium-service polls
            # this node's raw current value with no staleness check of its own
            # (transport_metacog_gate.py), so a value that can only ever go up
            # and never refresh back down to a genuine calm reading produces
            # permanent false "Bus Anomaly Detected" alerts once any real spike
            # ever occurs. The receipt above stays gated -- it's an audit trail
            # of notable prediction-error events, not a polled "current state"
            # read, so skipping a receipt for a calm (error == 0.0) tick is
            # correct and not the bug.
            self._write_prediction_error_node(
                node_id="node:substrate.bus_synaptic",
                error=error,
                now=now,
                reducer_key="bus_synaptic",
            )
            logger.info(
                "substrate_bus_synaptic_tick_completed edge_count=%d error=%.3f",
                len(zscores),
                error,
            )
        except Exception:
            logger.exception("substrate_bus_synaptic_tick_failed")

    async def _bus_synaptic_tick_loop(self) -> None:
        interval = float(self._settings.bus_synaptic_tick_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._bus_synaptic_tick)
            except Exception:
                logger.exception("substrate_bus_synaptic_tick_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

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

    # Cap on how many per-turn attribution ids a shared prediction-error node
    # accumulates in metadata['contributing_turn_ids']. This is attribution
    # (who last touched this node), not functional evidence like
    # pressure_reducer.py's evidence_event_ids -- so a smaller cap is enough.
    _CONTRIBUTING_ID_CAP = 20

    def _write_prediction_error_node(
        self,
        *,
        node_id: str,
        error: float,
        now: datetime,
        reducer_key: str = "",
        contributing_id: str | None = None,
    ) -> None:
        """Upsert a durable substrate node carrying the surprise (Rung 1 bridge).

        Default-off (env flag), fail-open: never raises out of a tick. When
        enabled it writes a single node under a fixed identity_key so re-writes
        collapse (no unbounded node growth), letting the dynamics engine seed
        pressure from ``metadata['prediction_error']``.

        This builds a fresh ``ConceptNodeV1`` on every call rather than
        round-tripping the existing one, so any dynamics-engine-owned metadata
        already durably set on this same ``node_id`` (``dynamic_pressure``,
        ``dynamic_pressure_reason``, ``dormant``, ``dormancy_updated_at`` --
        now native Cypher properties, see falkor_codec.py) must be carried
        forward explicitly. Without this, a repeat prediction-error event for
        the same fixed node_id -- the documented, intended "re-writes collapse"
        path -- would durably clobber SubstrateDynamicsEngine.tick()'s computed
        state back to 0.0/False on every re-write, since encode_node_properties()
        now always includes those keys in the upsert's SET clause.

        ``contributing_id`` (optional) is a per-call identifier -- e.g. a
        harness turn's ``correlation_id`` -- appended to
        ``metadata['contributing_turn_ids']`` so repeat writes to a shared
        node_id keep bounded per-incident attribution instead of only the
        latest error magnitude. Appended in order (oldest first), deduped,
        and capped to the most recent ``_CONTRIBUTING_ID_CAP`` entries. This
        method must stay fail-open, so any malformed carried-forward value is
        tolerated rather than raised.
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
            from orion.substrate.falkor_codec import DYNAMICS_ENGINE_OWNED_METADATA_KEYS

            salience = max(0.0, min(1.0, error))
            metadata: dict[str, Any] = {
                "source_kind": "substrate_prediction_error",
                "prediction_error": round(salience, 6),
                "reducer_key": reducer_key,
            }
            # Best-effort: not every injected store double implements
            # get_node_by_id (e.g. unit-test fakes that only cover
            # upsert_node), and a lookup failure must not block the write
            # itself -- this stays fail-open like the rest of this method.
            try:
                existing = store.get_node_by_id(node_id)
            except Exception:
                existing = None
            contributing_ids: list[str] = []
            if existing is not None:
                for key in DYNAMICS_ENGINE_OWNED_METADATA_KEYS:
                    if key in existing.metadata:
                        metadata[key] = existing.metadata[key]
                # Carry forward prior per-turn attribution. Tolerate a
                # missing/malformed stored value (not a list, or a list of
                # non-strings) rather than raising -- this write must stay
                # fail-open.
                try:
                    stored = existing.metadata.get("contributing_turn_ids")
                    if isinstance(stored, list):
                        contributing_ids = [str(item) for item in stored]
                except Exception:
                    contributing_ids = []
            if contributing_id is not None and contributing_id not in contributing_ids:
                contributing_ids.append(contributing_id)
            if contributing_ids:
                metadata["contributing_turn_ids"] = contributing_ids[-self._CONTRIBUTING_ID_CAP :]
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
                metadata=metadata,
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
                "node:substrate.harness_closure",
                _HARNESS_CLOSURE_UNRESOLVED_ERROR,
            )
            self._write_prediction_error_node(
                node_id="node:substrate.harness_closure",
                error=_HARNESS_CLOSURE_UNRESOLVED_ERROR,
                now=datetime.now(timezone.utc),
                reducer_key="post_turn_closure",
                contributing_id=closure.correlation_id,
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

    async def _field_channel_anomaly_listener_loop(self) -> None:
        """Subscribe to the field-channel anomaly-score channel and cache the
        latest FieldChannelAnomalyScoreV1.

        Mirrors _drive_state_listener_loop. Producer: orion-field-digester's
        mood-arc reconstruction-error scorer (app/anomaly_scorer.py), an
        independent third signal alongside honesty_metrics (active-inference
        prediction error) and spotlight.coalition_stability (GWT-dispatch
        lane) -- different model, different inputs (field-channel pressures,
        not domain prediction-error or attention-coalition tracking).
        """
        channel = self._settings.field_channel_anomaly_score_channel
        logger.info("substrate_field_channel_anomaly_listener subscribing channel=%s", channel)
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
                        self._cache_field_channel_anomaly_message(msg)
                    except Exception:
                        logger.exception("substrate_field_channel_anomaly_handle_failed")
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("substrate_field_channel_anomaly_listener stopped channel=%s", channel)

    def _cache_field_channel_anomaly_message(self, raw_msg: dict[str, Any]) -> None:
        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("substrate_field_channel_anomaly_decode_failed: %s", decoded.error)
            return
        try:
            score = FieldChannelAnomalyScoreV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("substrate_field_channel_anomaly_invalid err=%s", exc)
            return
        self._latest_field_anomaly = score
        self._latest_field_anomaly_at = datetime.now(timezone.utc)

    async def _codebase_delta_listener_loop(self) -> None:
        """Subscribe to the codebase-mass domain's bus channel and score+write
        each real event (docs/superpowers/specs/2026-07-30-codebase-mass-
        signal-design.md, "Producer + consumer patch design").

        Mirrors `_field_channel_anomaly_listener_loop`'s subscribe shape, but
        does real work per message rather than caching a value: reads the
        persisted `CodebaseMassBaseline` (own dedicated table, see
        `store.get_latest_codebase_mass_baseline()`'s own docstring for why
        this domain can't reuse `_write_prediction_error_node()`'s node
        metadata the way it was first proposed), reconstructs whichever one
        of git/pr_lifecycle/graph the event carries, scores it via
        `codebase_prediction_error()`, writes the score to
        `node:substrate.codebase` via the same shared
        `_write_prediction_error_node()` writer every other domain uses, and
        persists the updated baseline for the next tick. This service still
        does zero external I/O of its own for this domain -- all git/GitHub/
        graphify access lives in `orion-cocreation-signals`, the producer of
        this channel.
        """
        channel = self._settings.codebase_delta_channel
        logger.info("substrate_codebase_delta_listener subscribing channel=%s", channel)
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
                        await asyncio.to_thread(self._handle_codebase_delta_message, msg)
                    except Exception:
                        logger.exception("substrate_codebase_delta_handle_failed")
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("substrate_codebase_delta_listener stopped channel=%s", channel)

    def _handle_codebase_delta_message(self, raw_msg: dict[str, Any]) -> None:
        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("substrate_codebase_delta_decode_failed: %s", decoded.error)
            return
        try:
            event = CodebaseDeltaV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("substrate_codebase_delta_invalid err=%s", exc)
            return

        git_delta = pr_delta = graph_delta = None
        if event.domain == "git" and event.git is not None:
            g = event.git
            git_delta = GitChurnDelta(
                prev_sha=g.prev_sha, head_sha=g.head_sha, commit_count=g.commit_count,
                files_added=g.files_added, files_deleted=g.files_deleted,
                files_modified=g.files_modified, lines_added=g.lines_added,
                lines_removed=g.lines_removed,
            )
        elif event.domain == "pr_lifecycle" and event.pr_lifecycle is not None:
            p = event.pr_lifecycle
            pr_delta = PrLifecycleDelta(
                since=p.since, until=p.until, submitted_count=p.submitted_count,
                merged_count=p.merged_count, closed_without_merge_count=p.closed_without_merge_count,
                submitted_numbers=tuple(p.submitted_numbers), merged_numbers=tuple(p.merged_numbers),
                closed_without_merge_numbers=tuple(p.closed_without_merge_numbers),
                possibly_truncated=p.possibly_truncated,
            )
        elif event.domain == "graph" and event.graph is not None:
            gr = event.graph
            graph_delta = GraphStructuralDelta(
                node_count_delta=gr.node_count_delta, edge_count_delta=gr.edge_count_delta,
                community_count_delta=gr.community_count_delta,
                god_node_jaccard_similarity=gr.god_node_jaccard_similarity,
                god_nodes_entered=tuple(gr.god_nodes_entered) if gr.god_nodes_entered is not None else None,
                god_nodes_exited=tuple(gr.god_nodes_exited) if gr.god_nodes_exited is not None else None,
            )
        else:
            # Schema-illegal given CodebaseDeltaV1's own model_validator (the
            # producer can't construct/publish this shape), but a live
            # consumer must not trust a wire payload just because the
            # producer's own validator would have rejected it -- fail open,
            # not silently proceed with three None deltas (a real
            # codebase_prediction_error() call with nothing populated is a
            # legitimate "no-op tick" elsewhere; here it would misrepresent a
            # decode-time data problem as one).
            logger.error(
                "substrate_codebase_delta_domain_field_mismatch domain=%s", event.domain
            )
            return

        baseline = self._store.get_latest_codebase_mass_baseline()
        result = codebase_prediction_error(
            git_delta=git_delta, pr_delta=pr_delta, graph_delta=graph_delta, baseline=baseline,
        )
        now = datetime.now(timezone.utc)
        self._write_prediction_error_node(
            node_id="node:substrate.codebase",
            error=result.score,
            now=now,
            reducer_key="codebase",
        )
        self._store.save_codebase_mass_baseline(
            result.baseline, retention_days=self._settings.codebase_mass_baseline_retention_days,
        )
        logger.info(
            "substrate_codebase_delta_scored domain=%s score=%.3f",
            event.domain, result.score,
        )

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
            try:
                self._store.save_attention_broadcast_history(
                    projection, retention_hours=s.attention_broadcast_log_retention_hours
                )
            except Exception:
                logger.exception("substrate_attention_broadcast_history_persist_failed")
            logger.info(
                "substrate_attention_broadcast_completed selected=%s loop=%s "
                "open_loops=%d",
                projection.selected_action_type,
                projection.selected_open_loop_id,
                len(frame.open_loops),
            )
            if s.enable_attention_self_model_tick:
                try:
                    self._attention_self_model_tick(state=state, broadcast=projection)
                except Exception:
                    logger.exception("substrate_attention_self_model_tick_failed")
        except Exception:
            logger.exception("substrate_attention_broadcast_failed")

    def _fetch_heartbeat_h1(self) -> dict | None:
        """One synchronous GET to orion-heartbeat's own `/h1` endpoint, same
        blocking-call-inside-an-async-tick pattern this file's other synchronous
        SQL reads already use (e.g. `_brain_frame_self_state()` above) --
        this tick already does a cross-service Postgres read for the field
        lane, so one more short-timeout cross-service call is not a new kind
        of I/O for it.

        Returns the `/h1` response's `"h1"` dict verbatim (already the shape
        `reduce_attention_self_model()`'s `heartbeat_h1` param expects), or
        `None` if disabled (`heartbeat_h1_url` unset -- the default), the
        service hasn't computed its first tick yet (`{"ok": false, ...}`),
        or the request fails for any reason. Fail-open: orion-heartbeat is
        an additive, independently-owned service; its unavailability must
        never fail or delay this tick.
        """
        url = self._settings.heartbeat_h1_url
        if not url:
            return None
        try:
            response = requests.get(
                url, timeout=self._settings.heartbeat_h1_fetch_timeout_sec
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.exception("substrate_heartbeat_h1_fetch_failed")
            return None
        if not isinstance(payload, dict) or not payload.get("ok"):
            return None
        h1 = payload.get("h1")
        return h1 if isinstance(h1, dict) else None

    def _attention_self_model_tick(self, *, state: Any, broadcast: Any) -> None:
        """AST/HOT self-model live tick (docs/superpowers/specs/2026-07-29-
        ast-hot-reducer-live-ticking-design.md).

        Read-only against every existing live input: `state` is the same
        FalkorDB graph snapshot `_attention_broadcast_tick()` already fetched
        for its own broadcast computation (no extra store round-trip),
        `broadcast` is the projection that same tick just built, and the
        field lane is one cross-service Postgres read
        (`get_latest_field_attention_frame()`) -- this method never calls
        `upsert_node`/`save_attention_broadcast`/any existing writer. Its only
        write is `save_attention_self_model()`, into a table nothing else in
        the repo touches. Called from a context that already wraps this in
        its own try/except (see caller) -- this method itself does not need
        one, but keeps its own for a clean, itself-named log line
        (`substrate_attention_self_model_tick_failed`) distinguishable from
        the broadcast tick's own failure log if something upstream changes.

        **NOT this codebase's first live caller of `reduce_attention_self_model()`
        -- review finding 2026-07-29, correcting this patch's own original
        premise.** `_brain_frame_tick()` (below, `SUBSTRATE_BRAIN_FRAME_ENABLED`
        default `True`) already calls it live, every `brain_frame_interval_sec`
        (~5s default), with `field_frame=None` and no
        `prediction_error_trend_by_domain` -- a narrower computation whose
        result is embedded inline into one brain-frame UI dimension
        (`assemble_brain_frame(attention_payload=...)`'s `honesty_metrics`),
        never persisted as its own durable, queryable artifact. This tick is
        the first to compute the *complete* self-model (real field lane +
        real trend) and the first to persist it durably
        (`substrate_attention_self_model`) for future consumers -- e.g. the
        CollapseMirror "insight" trigger design doc this patch exists to
        unblock. The two computations are intentionally left un-unified here
        (different purposes, different tables, no write-path collision --
        confirmed via code review) rather than refactored to share one call,
        which was judged out of scope for this patch; revisit if having two
        independently-computed self-models live at once ever causes real
        confusion for a consumer.
        """
        from orion.substrate.attention_self_model import reduce_attention_self_model
        from orion.substrate.prediction_error_trend import compute_prediction_error_trend

        pe_by_domain = self._brain_frame_prediction_error_by_domain(state.nodes.values())
        if pe_by_domain:
            self._attention_self_model_trend_buffer.append(pe_by_domain)
        pe_trend_by_domain = compute_prediction_error_trend(
            list(self._attention_self_model_trend_buffer)
        )
        field_frame = self._store.get_latest_field_attention_frame()
        heartbeat_h1 = self._fetch_heartbeat_h1()
        # now= must be passed explicitly (matching _brain_frame_tick()'s own
        # existing reduce_attention_self_model() call, worker.py's other live
        # caller -- see this method's docstring). Without it, reduce_
        # attention_self_model() falls back to field_frame.generated_at for
        # this tick's own generated_at, which is an upstream, independently-
        # owned poll-cadence timestamp, not this tick's own. Review finding
        # 2026-07-29: save_attention_self_model()'s dedup key is derived from
        # generated_at -- if the field lane ever stalls, every subsequent
        # tick would silently no-op via ON CONFLICT DO NOTHING while this
        # method's own completion log kept firing every tick looking
        # healthy, exactly the "ticks alive, table frozen" failure shape
        # this whole design was meant to be immune to.
        self_model = reduce_attention_self_model(
            broadcast=broadcast,
            field_frame=field_frame,
            now=datetime.now(timezone.utc),
            prediction_error_by_domain=pe_by_domain or None,
            prediction_error_trend_by_domain=pe_trend_by_domain or None,
            heartbeat_h1=heartbeat_h1,
        )
        self._store.save_attention_self_model(
            self_model,
            retention_hours=self._settings.attention_self_model_log_retention_hours,
        )
        logger.info(
            "substrate_attention_self_model_tick_completed attention_reason=%s "
            "confidence=%s prediction_error_confidence=%s predicted_shift=%s",
            self_model.attention_reason,
            self_model.confidence,
            self_model.prediction_error_confidence,
            self_model.predicted_shift,
        )

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

    def _brain_frame_lane_health(self) -> dict:
        """Fetch reducer lane health for lane regions, remapped to friendly lane keys.

        ``build_substrate_grammar_truth`` keys its lag/backlog/quarantine dicts by
        cursor_name (e.g. ``execution_grammar_reducer``); the brain-frame producer
        and UI expect the friendly reducer keys (e.g. ``execution_trajectory``).
        Remap here so no phantom/mislabeled lanes reach the frame. Fail-open to {}.
        """
        try:
            from app.grammar_truth import (
                REDUCER_KEY_BY_CURSOR,
                build_substrate_grammar_truth,
            )

            truth = build_substrate_grammar_truth(self._store)

            def _remap(d):
                out = {}
                for k, v in (d or {}).items():
                    out[REDUCER_KEY_BY_CURSOR.get(k, k)] = v
                return out

            return {
                "cursor_lag_by_reducer": _remap(truth.get("cursor_lag_by_reducer")),
                "pending_backlog_by_reducer": _remap(truth.get("pending_backlog_by_reducer")),
                "quarantine_by_reducer": _remap(truth.get("quarantine_by_reducer")),
            }
        except Exception:
            logger.exception("brain_frame_lane_health_failed")
            return {}

    def _brain_frame_self_state(self) -> dict | None:
        """Latest self-state row payload (dict) or None. Fail-open."""
        try:
            from sqlalchemy import text

            engine = self._get_sql_engine()
            if engine is None:
                return None
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
                        ORDER BY generated_at DESC LIMIT 1
                        """
                    )
                ).mappings().first()
            if not row:
                return None
            payload = row["self_state_json"]
            if isinstance(payload, str):
                import json as _json

                payload = _json.loads(payload)
            return payload if isinstance(payload, dict) else None
        except Exception:
            logger.exception("brain_frame_self_state_load_failed")
            return None

    def _brain_frame_prediction_error_by_domain(self, nodes: Iterable[Any]) -> dict:
        """Read each Active-Inference domain's latest prediction_error off the
        already-fetched node snapshot -- zero extra I/O.

        ``_write_prediction_error_node()`` durably upserts one fixed-identity
        ``ConceptNodeV1`` per domain (``node:substrate.<domain>``,
        ``metadata['prediction_error']``) every tick it fires
        (``SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true``). Those nodes are
        already part of the same graph snapshot ``assemble_brain_frame()``
        uses for node_kind regions, so this just reads them back rather than
        making a second store round-trip.
        """
        out: dict[str, float] = {}
        for node in nodes:
            node_id = str(getattr(node, "node_id", "") or "")
            domain = _PREDICTION_ERROR_DOMAIN_NODE_IDS.get(node_id)
            if domain is None:
                continue
            md = getattr(node, "metadata", None) or {}
            value = md.get("prediction_error")
            if value is not None:
                try:
                    out[domain] = float(value)
                except (TypeError, ValueError):
                    continue
        return out

    def _brain_frame_tick(self):
        """Assemble + persist one brain frame. Returns the frame or None."""
        s = self._settings
        if not s.brain_frame_enabled:
            return None
        try:
            from app.brain_frame_producer import assemble_brain_frame

            store = self._get_substrate_graph_store(
                log_label="brain_frame_graph_store_init_failed"
            )
            nodes: list[Any] = []
            edges: list[Any] = []
            if store is not None:
                try:
                    state = store.snapshot()
                    nodes = list(state.nodes.values())
                    edges = list(getattr(state, "edges", []) or [])
                except Exception:
                    logger.exception("brain_frame_snapshot_failed")

            attention = None
            try:
                attention = self._store.load_attention_broadcast()
            except Exception:
                logger.exception("brain_frame_attention_load_failed")

            now = datetime.now(timezone.utc)

            # AttentionSelfModelV1 (carries the unconditional
            # prediction_error_confidence field the honesty_metrics brain-frame
            # dimension reads) has no live producer anywhere in this codebase --
            # reduce_attention_self_model() was, until this call, only ever
            # invoked by offline analysis scripts (scripts/analysis/
            # measure_ast_hot_reducer.py). Compute it live here from real,
            # already-available inputs rather than passing the narrower
            # AttentionBroadcastProjectionV1 (`attention` above), which has no
            # prediction_error_confidence field at all.
            attention_self_model = None
            try:
                from orion.substrate.attention_self_model import (
                    reduce_attention_self_model,
                )

                attention_self_model = reduce_attention_self_model(
                    broadcast=attention,
                    field_frame=None,
                    now=now,
                    prediction_error_by_domain=self._brain_frame_prediction_error_by_domain(nodes)
                    or None,
                )
            except Exception:
                logger.exception("brain_frame_attention_self_model_failed")

            frame = assemble_brain_frame(
                nodes=nodes,
                edges=edges,
                lane_health=self._brain_frame_lane_health(),
                self_state=self._brain_frame_self_state(),
                attention=attention,
                attention_payload=attention_self_model,
                field_anomaly=self._latest_field_anomaly,
                settings=s,
                now=now,
                tick_seq=self._brain_frame_seq,
            )
            self._brain_frame_seq += 1
            try:
                self._store.save_brain_frame(
                    frame, retention_hours=int(s.brain_frame_retention_hours)
                )
            except Exception:
                logger.exception("brain_frame_persist_failed")
            logger.info(
                "brain_frame_tick_completed phase=%s regions=%d nodes=%d frame_id=%s",
                frame.phase,
                len(frame.regions),
                len(frame.nodes),
                frame.frame_id,
            )
            return frame
        except Exception:
            logger.exception("brain_frame_tick_failed")
            return None

    async def _publish_brain_frame(self, frame) -> None:
        if self._bus is None or frame is None:
            return
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope
            from orion.core.bus.resilience import publish_with_reconnect
            from orion.schemas.brain_frame import SUBSTRATE_BRAIN_FRAME_KIND

            env = BaseEnvelope(
                kind=SUBSTRATE_BRAIN_FRAME_KIND,
                source=self._service_ref(),
                correlation_id=uuid4(),
                payload=frame.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus,
                "orion:substrate:brain_frame",
                env,
                log_label="substrate_brain_frame",
            )
        except Exception:
            logger.exception("brain_frame_publish_failed")

    async def _brain_frame_loop(self) -> None:
        interval = float(self._settings.brain_frame_interval_sec)
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self._brain_frame_tick)
                await self._publish_brain_frame(frame)
            except Exception:
                logger.exception("substrate_brain_frame_loop_failed")
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
        if seeds:
            # Visibility, 2026-07-26: which source/node actually won a budget
            # slot this tick. _prediction_error_candidates() (orion/substrate/
            # endogenous_curiosity.py) iterates every substrate node generically
            # -- a new node with a `prediction_error` metadata field joins this
            # consumer automatically, with no code change and no announcement
            # here otherwise (this is exactly how bus_synaptic started feeding
            # this tick the moment PR #1377 wrote to its node, and exactly how
            # the old broken transport domain went unnoticed winning slots for
            # weeks). Cheap, best-effort, never raises.
            try:
                sources = [
                    f"{next((n.split(':', 1)[1] for n in (sig.notes or []) if n.startswith('source:')), 'unknown')}"
                    f"={sig.focal_node_refs[0] if sig.focal_node_refs else '-'}"
                    for sig in seeds
                ]
                logger.info(
                    "substrate_endogenous_curiosity_seed_sources sources=%s",
                    ",".join(sources)[:500],
                )
            except Exception:
                logger.exception("substrate_endogenous_curiosity_seed_source_log_failed")
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
                max_runs=self._settings.execution_trajectory_max_runs,
                max_age_sec=self._settings.execution_trajectory_max_age_sec,
            )

        last_id = self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )

        if last_id is not None:
            curr_projection = load_projection()
            error = execution_prediction_error(prev_projection, curr_projection)
            # execution_prediction_error mutates curr_projection's EWMA baseline
            # fields in place (prediction_error_baseline_ewma/_var/_n) -- persist
            # that regardless of whether error > 0.0 this tick, or the baseline
            # never survives past this process's lifetime and every tick after a
            # restart re-cold-starts at n=0. Cheap: save_execution_trajectory is
            # the same upsert process_batch already calls above.
            self._store.save_execution_trajectory(curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="execution_trajectory",
                        node_id="node:substrate.execution",
                        prediction_error=error,
                        now=now,
                    )
                )
            # Write the node on every tick, not just when error > 0.0 -- the
            # same fix already applied to bus_synaptic (2026-07-30), extended
            # here after `route` was found frozen at a stale 0.00025 while its
            # tick kept running: a categorical/quiet domain returns error == 0.0
            # on most ticks, so gating the node write left the node holding its
            # last NON-ZERO value indefinitely. Consumers that poll this node's
            # raw current value (orion-equilibrium-service, AST/HOT's
            # prediction_error_by_domain) then read a stale high-water mark as
            # if it were a current reading -- a value that can rise but never
            # come back down to a genuine calm 0.0. The receipt above stays
            # gated: it is an audit trail of notable events, not a polled
            # "current state" read, so skipping it on a calm tick is correct.
            self._write_prediction_error_node(
                node_id="node:substrate.execution",
                error=error,
                now=now,
                reducer_key="execution_trajectory",
                contributing_id=last_id,
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

        prev_projection = _load_chat_projection()

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_chat_grammar_events(
                events=batch,
                load_projection=_load_chat_projection,
                save_projection=_save_chat_projection,
                save_receipt=self._store.save_receipt,
                now=now,
            )

        last_id = self._process_events_with_poison_isolation(
            spec=spec,
            events=events,
            process_batch=process_batch,
        )

        if last_id is not None:
            curr_projection = _load_chat_projection()
            error = chat_prediction_error(prev_projection, curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="chat_session",
                        node_id="node:substrate.chat",
                        prediction_error=error,
                        now=now,
                    )
                )
            # Write the node on every tick, not just when error > 0.0 -- the
            # same fix already applied to bus_synaptic (2026-07-30), extended
            # here after `route` was found frozen at a stale 0.00025 while its
            # tick kept running: a categorical/quiet domain returns error == 0.0
            # on most ticks, so gating the node write left the node holding its
            # last NON-ZERO value indefinitely. Consumers that poll this node's
            # raw current value (orion-equilibrium-service, AST/HOT's
            # prediction_error_by_domain) then read a stale high-water mark as
            # if it were a current reading -- a value that can rise but never
            # come back down to a genuine calm 0.0. The receipt above stays
            # gated: it is an audit trail of notable events, not a polled
            # "current state" read, so skipping it on a calm tick is correct.
            self._write_prediction_error_node(
                node_id="node:substrate.chat",
                error=error,
                now=now,
                reducer_key="chat_session",
                contributing_id=last_id,
            )

        return last_id

    def _route_tick(self) -> str | None:
        spec = REDUCER_SPECS[4]
        events = self._store.fetch_route_grammar_events(
            limit=spec.batch_limit(self._settings),
        )
        if not events:
            return None

        now = datetime.now(timezone.utc)

        def load_projection():
            loaded = self._store.load_route_arbitration(ROUTE_ARBITRATION_PROJECTION_ID)
            return loaded or empty_route_projection(now=now)

        prev_projection = load_projection()

        def process_batch(batch: list[GrammarEventV1]) -> None:
            process_route_grammar_events(
                events=batch,
                load_projection=load_projection,
                save_projection=self._store.save_route_arbitration,
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
            error = route_prediction_error(prev_projection, curr_projection)
            if error > 0.0:
                self._store.save_receipt(
                    _prediction_error_receipt(
                        reducer_key="route_arbitration",
                        node_id="node:substrate.route",
                        prediction_error=error,
                        now=now,
                    )
                )
            # Write the node on every tick, not just when error > 0.0 -- the
            # same fix already applied to bus_synaptic (2026-07-30), extended
            # here after `route` was found frozen at a stale 0.00025 while its
            # tick kept running: a categorical/quiet domain returns error == 0.0
            # on most ticks, so gating the node write left the node holding its
            # last NON-ZERO value indefinitely. Consumers that poll this node's
            # raw current value (orion-equilibrium-service, AST/HOT's
            # prediction_error_by_domain) then read a stale high-water mark as
            # if it were a current reading -- a value that can rise but never
            # come back down to a genuine calm 0.0. The receipt above stays
            # gated: it is an audit trail of notable events, not a polled
            # "current state" read, so skipping it on a calm tick is correct.
            self._write_prediction_error_node(
                node_id="node:substrate.route",
                error=error,
                now=now,
                reducer_key="route_arbitration",
                contributing_id=last_id,
            )

        return last_id

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

        # 2026-07-26: prev_projection removed here -- it was only ever read by
        # the now-removed transport_prediction_error(prev_projection,
        # curr_projection) call below. Keeping it would be a wasted Postgres
        # read every tick for a value nobody consumes (review caught this).

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
            _log_transport_incident_signals(curr_projection)
            # 2026-07-26: node:substrate.transport's prediction_error write REMOVED
            # (docs/superpowers/specs/2026-07-26-transport-domain-retirement-bus-
            # synaptic-successor-design.md). transport_prediction_error() itself
            # (orion/substrate/prediction_error.py) is a narrow 2-Redis-Stream
            # "world_pulse" census, not real bus traffic, already excluded from
            # attention_self_model.py's ACTIVE_INFERENCE_DOMAINS -- but the write
            # here kept running regardless, and orion/substrate/endogenous_
            # curiosity.py's generic node iteration kept reading it, once pinned
            # at signal_strength=1.0 for a full 24h (2026-07-16, see that module's
            # comment) and winning real curiosity-candidate budget slots on a fake
            # signal. bus_synaptic_prediction_error() (node:substrate.bus_synaptic)
            # already flows into that same consumer automatically -- no new wiring
            # needed there, killing this write is the entire change. Everything
            # else in this reducer (event processing, projection state, incident
            # logging above, catalog_drift_pressure/observer_failure_pressure via
            # config/field/orion_field_topology.v1.yaml's still-live edge) is
            # unaffected -- confirmed via _grammar_reducer_poll_loop that reducer-
            # health/cursor tracking is driven by last_id, not by this block.
            # transport_prediction_error() itself is kept, not deleted -- review
            # confirmed it has zero callers anywhere in the repo now, live or
            # offline (the two analysis scripts this comment originally claimed
            # used it, measure_transport_bus_signal_history.py and measure_
            # transport_biometrics_prediction_error_correlation.py, only mention
            # its name in prose docstrings -- both read persisted values
            # straight out of Postgres, no import). Kept anyway as the cheapest
            # option (deleting it buys nothing and risks a future script
            # wanting it for genuine historical replay).

        return last_id
