from __future__ import annotations

from copy import deepcopy

from orion.schemas.route_projection import RouteArbitrationRunStateV1


def merge_route_run_state(
    existing: RouteArbitrationRunStateV1 | None,
    incoming: RouteArbitrationRunStateV1,
) -> RouteArbitrationRunStateV1:
    if existing is None:
        return incoming

    merged = deepcopy(existing)
    if incoming.lane != "unknown":
        merged.lane = incoming.lane
    if incoming.lane_reason != "unknown":
        merged.lane_reason = incoming.lane_reason
    if incoming.mind_requested:
        merged.mind_requested = incoming.mind_requested
    if incoming.mind_skip_reason:
        merged.mind_skip_reason = incoming.mind_skip_reason
    if incoming.output_mode != "unknown":
        merged.output_mode = incoming.output_mode
    if incoming.session_id:
        merged.session_id = incoming.session_id
    if incoming.turn_id:
        merged.turn_id = incoming.turn_id

    merged.evidence_event_ids = sorted(
        set(existing.evidence_event_ids) | set(incoming.evidence_event_ids)
    )[-200:]
    merged.last_updated_at = incoming.last_updated_at
    return merged
