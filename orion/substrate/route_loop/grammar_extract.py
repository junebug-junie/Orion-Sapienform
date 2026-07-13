from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.route_projection import RouteArbitrationRunStateV1

from .constants import ROUTE_SOURCE_SERVICE
from .ids import parse_route_trace_id

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _parse_summary_kv(summary: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in _KV_RE.findall(summary or ""):
        out[key.lower()] = val.strip()
    return out


def _boolish(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"true", "1", "yes", "on"}


def extract_route_state_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
) -> RouteArbitrationRunStateV1:
    clock = _utc_now(now)
    if not events:
        raise ValueError("events must not be empty")

    trace_id = events[0].trace_id
    parsed = parse_route_trace_id(trace_id or "")
    node_id = parsed[0] if parsed else "unknown"
    correlation_id = parsed[1] if parsed else (events[0].correlation_id or "unknown")

    run = RouteArbitrationRunStateV1(
        trace_id=trace_id or "",
        correlation_id=correlation_id,
        session_id=events[0].session_id,
        turn_id=events[0].turn_id,
        node_id=node_id,
        last_updated_at=clock,
    )

    for event in events:
        if event.provenance.source_service != ROUTE_SOURCE_SERVICE:
            continue
        atom = event.atom
        if not atom:
            continue
        role = atom.semantic_role or ""
        kv = _parse_summary_kv(atom.summary or "")
        run.evidence_event_ids.append(event.event_id)

        if role == "route_arbitration_decided":
            run.lane = kv.get("lane", run.lane)
            run.lane_reason = kv.get("lane_reason", run.lane_reason)
            run.mind_requested = _boolish(kv.get("mind_requested"))
            skip_reason = kv.get("mind_skip_reason")
            if skip_reason and skip_reason.lower() != "none":
                run.mind_skip_reason = skip_reason
            run.output_mode = kv.get("output_mode", run.output_mode)

    return run
