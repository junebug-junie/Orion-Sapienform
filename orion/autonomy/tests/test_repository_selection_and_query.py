from __future__ import annotations

from unittest.mock import MagicMock

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


def test_select_preferred_skips_orion_identity_only_when_drives_timeout() -> None:
    """Partial Orion state (identity/goals) must not win over relationship when Orion drives timed out."""
    orion_state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        goal_headlines=[],
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
            subquery_diagnostics={
                "identity": {"status": "ok", "row_count": 1},
                "drives": {"status": "timeout", "row_count": 0},
                "goals": {"status": "empty", "row_count": 0},
            },
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=relationship_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 5}},
        ),
    }
    selected = select_preferred_autonomy_lookup(by_subject)
    assert selected.selected_subject == "relationship"
    assert selected.contextual_fallback is True


def test_defer_orion_drives_by_default_without_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES", raising=False)
    client = MagicMock()
    client.select.return_value = [
        {
            "artifact_id": {"value": "id-1"},
            "summary": {"value": "steady"},
            "created_at": {"value": "2026-05-20T12:00:00Z"},
        }
    ]
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=20.0,
        query_client=client,
        subquery_max_workers=1,
    )
    lookup = repo.get_latest("orion", observer={"consumer": "chat_stance"})
    drives_diag = (lookup.subquery_diagnostics or {}).get("drives") or {}
    assert drives_diag.get("status") == "deferred"
    assert not any("DriveAudit" in str(c[0][0]) for c in client.select.call_args_list)
