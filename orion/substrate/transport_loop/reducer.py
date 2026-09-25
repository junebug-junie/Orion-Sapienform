from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    TRANSPORT_BUS_PROJECTION_ID,
    TRANSPORT_REDUCER_ID,
    TRANSPORT_SOURCE_SERVICE,
)
from .extract import ATOM_ROLES, extract_transport_bus_state_from_events, parse_bus_transport_trace_id

logger = logging.getLogger(__name__)

# Loads every stored grammar event of one trace (grammar_events, ordered like
# the reducer cursor: created_at, event_id).
TraceEventsLoader = Callable[[str], list[GrammarEventV1]]

# A bus observer tick is whole only when both ends are present. Every live
# observer trace carries both (24,634/24,634 bus.transport:athena traces over
# 2026-09-18..25; services/orion-bus/app/bus_observer.py::run_observer_tick
# emits tick_started + tick_completed, or tick_started + tick_failed).
_TICK_STARTED_ROLE = "bus_observer_tick_started"
_TICK_TERMINAL_ROLES = frozenset({"bus_observer_tick_completed", "bus_observer_tick_failed"})


def _roles(events: list[GrammarEventV1]) -> set[str]:
    return {(e.atom.semantic_role or "").strip() for e in events if e.atom}


def _is_whole_tick(events: list[GrammarEventV1]) -> bool:
    roles = _roles(events)
    return _TICK_STARTED_ROLE in roles and bool(roles & _TICK_TERMINAL_ROLES)


def _union_events(stored: list[GrammarEventV1], batch: list[GrammarEventV1], trace_id: str) -> list[GrammarEventV1]:
    """Stored trace order first (same order the cursor reads), then any batch
    event the loader did not return. Deduped by event_id."""
    out: list[GrammarEventV1] = []
    seen: set[str] = set()
    for event in [*stored, *batch]:
        if event.trace_id != trace_id or event.event_id in seen:
            continue
        seen.add(event.event_id)
        out.append(event)
    return out


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _noop_receipt(
    events: list[GrammarEventV1],
    *,
    reducer_id: str,
    clock: datetime,
    warnings: list[str] | None = None,
) -> ReductionReceiptV1:
    noop_ids = [e.event_id for e in events]
    return ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=noop_ids,
        ),
        noop_event_ids=noop_ids,
        warnings=list(warnings or []),
        created_at=clock,
    )


def reduce_transport_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: TransportBusProjectionV1,
    now: datetime | None = None,
    reducer_id: str = TRANSPORT_REDUCER_ID,
    load_trace_events: TraceEventsLoader | None = None,
) -> tuple[TransportBusProjectionV1, ReductionReceiptV1]:
    """Reduce one trace group into `buses[bus:<node>]`, which it REPLACES.

    Only a whole observer tick (tick_started .. tick_completed/tick_failed) is
    ever written. The reducer cursor pages grammar events by (created_at,
    event_id) with a row limit, so one observer window can be cut across two
    batches. Reducing a piece used to fabricate values: a tail without
    bus_health_observed read redis_ping_ok=None -> 0.5 backlog health /
    delivery confidence / reliability pressure over the real bus:athena, and a
    head without bus_census_computed read "no catalog drift". Now:
      * a piece that is not a whole tick is rebuilt from the stored trace via
        `load_trace_events` (the head was already persisted, since the cursor
        passed it), and
      * if the rebuilt trace is still not whole (in-flight window, no loader,
        loader failure), nothing is written and the prior reading stands.
    """
    clock = _utc_now(now)
    if not events:
        receipt = ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=[],
            ),
            noop_event_ids=[],
            created_at=clock,
        )
        return projection, receipt

    trace_id = events[0].trace_id or ""
    parsed = parse_bus_transport_trace_id(trace_id)
    if not parsed:
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock)

    if any(e.provenance.source_service != TRANSPORT_SOURCE_SERVICE for e in events):
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock)

    updated = deepcopy(projection)
    updated.updated_at = clock
    if updated.projection_id != TRANSPORT_BUS_PROJECTION_ID:
        updated.projection_id = TRANSPORT_BUS_PROJECTION_ID

    warnings: list[str] = []
    # A piece carrying no observer atom (trace_started/ended, edges, the
    # zscore atom after tick_completed) has nothing to add; skip it before any
    # trace reload.
    if not (_roles(events) & ATOM_ROLES):
        warnings.append(f"no bus observer evidence in trace {trace_id}")
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)

    # Only whole ticks are ever written, so a bus already holding this exact
    # trace has its final reading. sql-writer commits a trace atomically, so a
    # head piece usually reloads the whole tick and writes it; the tail piece
    # (which carries tick_completed) must not write it a second time. A piece
    # of a window OLDER than the stored one must never replace it either
    # (sample_window_id is YYYYMMDDTHHMMSSZ, so string order is time order).
    node_id, sample_window_id = parsed
    stored = projection.buses.get(f"bus:{node_id}")
    if stored is not None and stored.source_trace_id == trace_id:
        warnings.append(f"observer window already applied: {trace_id}")
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)
    if stored is not None and stored.sample_window_id > sample_window_id:
        warnings.append(f"older observer window skipped: {trace_id} < {stored.sample_window_id}")
        logger.info("transport_older_window_skipped trace_id=%s stored_window=%s", trace_id, stored.sample_window_id)
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)

    tick_events = events
    if not _is_whole_tick(tick_events) and load_trace_events is not None:
        try:
            tick_events = _union_events(load_trace_events(trace_id), events, trace_id)
        except Exception as exc:  # a failed reload must not write a partial window
            # Broad on purpose: any reload failure holds the piece. If the
            # reload itself is what fails (grant, bad stored row), every split
            # window is held and bus:<node> goes stale -- hence the WARNING.
            warnings.append(f"trace reload failed for {trace_id}: {type(exc).__name__}: {exc}")
            logger.warning("transport_trace_reload_failed trace_id=%s err=%r", trace_id, exc)
            tick_events = events
    if not _is_whole_tick(tick_events):
        warnings.append(f"incomplete observer window held: {trace_id}")
        logger.info(
            "transport_incomplete_window_held trace_id=%s events=%d reloaded=%s",
            trace_id,
            len(tick_events),
            load_trace_events is not None,
        )
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)

    try:
        incoming = extract_transport_bus_state_from_events(
            tick_events,
            now=clock,
        )
    except ValueError as exc:
        warnings.append(str(exc))
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)

    # A bus state with zero bus-observer evidence is not a reading, it is the
    # extractor's defaults (redis_ping_ok=None -> 0.5 "half health"). Writing
    # it would overwrite/mint a bus entry with fabricated pressures -- exactly
    # how bus:rpc_timeout was born (2026-09-22 audit). Splits BETWEEN observer
    # atoms are handled by the whole-tick rule above.
    if not incoming.evidence_event_ids:
        warnings.append(f"no bus observer evidence in trace {incoming.source_trace_id}")
        return projection, _noop_receipt(events, reducer_id=reducer_id, clock=clock, warnings=warnings)

    existing = updated.buses.get(incoming.target_id)
    operation = "create" if existing is None else "update"
    updated.buses[incoming.target_id] = incoming

    event_ids = [
        e.event_id
        for e in tick_events
        if e.atom and (e.atom.semantic_role or "").strip() not in {"", "trace_started", "trace_ended", "edge_emitted"}
    ]
    after_payload = incoming.model_dump(mode="json")
    after_payload["pressure_hints"] = {
        k: after_payload[k]
        for k in (
            "catalog_drift_pressure",
            "observer_failure_pressure",
            "contract_pressure",
            "reliability_pressure",
        )
    }

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=event_ids,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=event_ids,
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=TRANSPORT_BUS_PROJECTION_ID,
                    target_kind="transport_bus",
                    target_id=incoming.target_id,
                    operation=operation,
                    caused_by_event_ids=event_ids,
                ),
                target_projection=TRANSPORT_BUS_PROJECTION_ID,
                target_kind="transport_bus",
                target_id=incoming.target_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=after_payload,
                caused_by_event_ids=event_ids,
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="transport_bus",
                projection_id=TRANSPORT_BUS_PROJECTION_ID,
                node_id=incoming.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
