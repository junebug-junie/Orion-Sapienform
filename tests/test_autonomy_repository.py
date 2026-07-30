from __future__ import annotations

from orion.autonomy.repository import GraphAutonomyRepository


class FakeQueryClient:
    def __init__(self, rows_by_graph):
        self.rows_by_graph = rows_by_graph

    def select(self, sparql: str):
        if "graph/autonomy/identity" in sparql:
            return self.rows_by_graph.get("identity", [])
        if "graph/autonomy/drives" in sparql:
            return self.rows_by_graph.get("drives", [])
        if "graph/autonomy/goals" in sparql:
            return self.rows_by_graph.get("goals", [])
        return []


def _lit(value: str):
    return {"type": "literal", "value": value}


def test_graph_repository_returns_empty_when_no_rows() -> None:
    repo = GraphAutonomyRepository(endpoint="http://graph/repositories/collapse", timeout_sec=1.0, query_client=FakeQueryClient({}))
    result = repo.get_latest("orion")
    assert result.availability == "empty"
    assert result.state is None


def test_graph_repository_maps_latest_state_bundle() -> None:
    rows = {
        "identity": [
            {
                "artifact_id": _lit("identity-1"),
                "summary": _lit("Preserve continuity"),
                "anchor_strategy": _lit("continuity_anchor"),
                "created_at": _lit("2026-03-29T12:00:00+00:00"),
            }
        ],
        "drives": [
            {
                "artifact_id": _lit("audit-1"),
                "dominant_drive": _lit("continuity"),
                "created_at": _lit("2026-03-29T12:01:00+00:00"),
                "active_drive": _lit("continuity"),
                "drive_name": _lit("continuity"),
                "drive_pressure": _lit("0.91"),
                "tension_kind": _lit("identity_drift"),
            },
            {
                "artifact_id": _lit("audit-1"),
                "dominant_drive": _lit("continuity"),
                "created_at": _lit("2026-03-29T12:01:00+00:00"),
                "active_drive": _lit("coherence"),
                "drive_name": _lit("coherence"),
                "drive_pressure": _lit("0.71"),
                "tension_kind": _lit("scope_sprawl"),
            },
            {
                "artifact_id": _lit("audit-0"),
                "dominant_drive": _lit("coherence"),
                "created_at": _lit("2026-03-29T11:59:00+00:00"),
            },
        ],
        "goals": [
            {
                "artifact_id": _lit("goal-2"),
                "goal_statement": _lit("Converge on bounded runtime stance loop."),
                "drive_origin": _lit("continuity"),
                "priority": _lit("0.8"),
                "proposal_signature": _lit("sig-2"),
                "cooldown_until": _lit("2026-03-30T00:00:00+00:00"),
                "created_at": _lit("2026-03-29T12:03:00+00:00"),
            }
        ],
    }
    repo = GraphAutonomyRepository(endpoint="http://graph/repositories/collapse", timeout_sec=1.0, query_client=FakeQueryClient(rows))
    result = repo.get_latest("orion")

    # latest_drive_audit_id/dominant_drive/active_drives/drive_pressures/
    # tension_kinds removed from AutonomyStateV1 2026-07-30 (chore/delete-
    # orion-drives Wave 2a) -- GraphAutonomyRepository still parses the
    # "drives" SPARQL rows (kept for the fixture's not-empty/generated_at
    # fallback checks below), but no longer constructs AutonomyStateV1 with
    # them. See GraphAutonomyRepository._query_subject's own comment.
    assert result.availability == "available"
    assert result.state is not None
    assert result.state.latest_identity_snapshot_id == "identity-1"
    assert result.state.latest_goal_ids == ["goal-2"]
