from __future__ import annotations

from orion.autonomy.models import AutonomyStateV1
from orion.autonomy.repository import (
    AutonomyLookupV1,
    GraphAutonomyRepository,
    select_preferred_autonomy_lookup,
)


class _CaptureQueryClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def select(self, sparql: str):
        self.queries.append(sparql)
        if "GRAPH <http://conjourney.net/graph/autonomy/drives>" in sparql:
            return [
                {
                    "artifact_id": {"value": "drive-1"},
                    "created_at": {"value": "2026-03-31T12:00:01Z"},
                    "dominant_drive": {"value": "coherence"},
                    "drive_name": {"value": "coherence"},
                    "drive_pressure": {"value": "0.9"},
                }
            ]
        return []


def test_drive_audit_query_pins_latest_artifact_before_optional_joins() -> None:
    client = _CaptureQueryClient()
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=client,
        drives_query_limit=20,
    )
    repo.get_latest("orion")
    drives_queries = [q for q in client.queries if "autonomy/drives" in q]
    assert drives_queries
    assert "SELECT ?artifact ?artifact_id ?created_at" in drives_queries[0]
    assert "LIMIT 1" in drives_queries[0]
    assert "LIMIT 20" in drives_queries[0]


def test_select_preferred_autonomy_lookup_falls_back_to_relationship_drives() -> None:
    orion_state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        source="graph",
    )
    relationship_state = AutonomyStateV1(
        subject="relationship",
        model_layer="relationship-model",
        entity_id="relationship:orion|juniper",
        dominant_drive="relational",
        drive_pressures={"relational": 0.8},
        source="graph",
    )
    by_subject = {
        "orion": AutonomyLookupV1(
            subject="orion",
            state=orion_state,
            availability="degraded",
            unavailable_reason="timeout",
            subquery_diagnostics={"drives": {"status": "timeout", "row_count": 0}},
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=relationship_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 12}},
        ),
    }
    selected = select_preferred_autonomy_lookup(by_subject)
    assert selected.selected_subject == "relationship"
    assert selected.contextual_fallback is True
    assert selected.lookup is not None
    assert selected.lookup.state is not None
    assert selected.lookup.state.dominant_drive == "relational"
