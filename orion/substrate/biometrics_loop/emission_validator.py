from __future__ import annotations

from collections import defaultdict

from orion.schemas.organ_emission import OrganEmissionV1

from orion.substrate.biometrics_loop.pressure_organ import ALLOWED_PRESSURE_ROLES


def group_candidate_events_by_trace(events: list) -> list[list]:
    grouped: dict[str, list] = defaultdict(list)
    order: list[str] = []
    for event in events:
        trace_id = event.trace_id
        if trace_id not in grouped:
            order.append(trace_id)
        grouped[trace_id].append(event)
    return [grouped[trace_id] for trace_id in order]


def validate_organ_emission(emission: OrganEmissionV1, *, max_events: int = 8) -> OrganEmissionV1:
    traces = group_candidate_events_by_trace(emission.candidate_events)
    if len(traces) > max_events:
        raise ValueError(f"organ emission exceeds max_events ({max_events})")

    for trace in traces:
        kinds = [event.event_kind for event in trace]
        assert kinds[0] == "trace_started", kinds
        assert "atom_emitted" in kinds, kinds
        assert kinds[-1] == "trace_ended", kinds

        for event in trace:
            if event.atom:
                assert event.atom.semantic_role in ALLOWED_PRESSURE_ROLES
            assert event.debug_trace is None if hasattr(event, "debug_trace") else True

    if emission.debug_trace is not None:
        raise ValueError("organ emission must not include debug_trace")
    return emission
