from __future__ import annotations

from orion.autonomy.repository import GraphAutonomyRepository


class _StubClient:
    def __init__(self, rows):
        self.rows = rows
        self.last_sparql = ""

    def select(self, sparql: str):
        self.last_sparql = sparql
        return self.rows


def test_fetch_active_goals_filters_superseded_and_dedupes_drive_origin():
    rows = [
        {"artifact_id": {"value": "goal-a"}, "goal_statement": {"value": "A"}, "drive_origin": {"value": "autonomy"},
         "priority": {"value": "0.9"}, "proposal_signature": {"value": "s1"}, "proposal_status": {"value": "active"},
         "created_at": {"value": "2026-05-21T10:00:00Z"}},
        {"artifact_id": {"value": "goal-b"}, "goal_statement": {"value": "B"}, "drive_origin": {"value": "autonomy"},
         "priority": {"value": "0.5"}, "proposal_signature": {"value": "s2"}, "proposal_status": {"value": "proposed"},
         "created_at": {"value": "2026-05-21T09:00:00Z"}},
        {"artifact_id": {"value": "goal-c"}, "goal_statement": {"value": "C"}, "drive_origin": {"value": "relational"},
         "priority": {"value": "0.7"}, "proposal_signature": {"value": "s3"}, "proposal_status": {"value": "proposed"},
         "created_at": {"value": "2026-05-21T08:00:00Z"}},
        {"artifact_id": {"value": "goal-d"}, "goal_statement": {"value": "D"}, "drive_origin": {"value": "coherence"},
         "priority": {"value": "0.2"}, "proposal_signature": {"value": "s4"}, "proposal_status": {"value": "archived"},
         "created_at": {"value": "2026-05-21T07:00:00Z"}},
    ]
    repo = GraphAutonomyRepository(endpoint="http://fake", timeout_sec=1.0, query_client=_StubClient(rows), goals_limit=3)
    goals, row_count = repo._fetch_active_goals(subject="orion", model_layer="self-model", entity_id="self:orion")
    assert row_count == 4
    assert [g.drive_origin for g in goals] == ["autonomy", "relational"]
    assert goals[0].artifact_id == "goal-a"
    assert "proposalStatus" in repo._query_client.last_sparql or "proposal_status" in repo._query_client.last_sparql.lower()


def test_fetch_active_goals_reads_planned_and_executing_lifecycle_fields():
    rows = [
        {
            "artifact_id": {"value": "goal-planned"},
            "goal_statement": {"value": "Plan the coherence review."},
            "drive_origin": {"value": "coherence"},
            "priority": {"value": "0.85"},
            "proposal_signature": {"value": "sig-planned"},
            "proposal_status": {"value": "planned"},
            "planned_task_id": {"value": "task-abc-123"},
            "created_at": {"value": "2026-05-21T12:00:00Z"},
        },
        {
            "artifact_id": {"value": "goal-executing"},
            "goal_statement": {"value": "Execute the coherence review."},
            "drive_origin": {"value": "predictive"},
            "priority": {"value": "0.8"},
            "proposal_signature": {"value": "sig-executing"},
            "proposal_status": {"value": "executing"},
            "planned_task_id": {"value": "task-def-456"},
            "created_at": {"value": "2026-05-21T11:00:00Z"},
        },
        {
            "artifact_id": {"value": "goal-done"},
            "goal_statement": {"value": "Finished goal."},
            "drive_origin": {"value": "autonomy"},
            "priority": {"value": "0.99"},
            "proposal_signature": {"value": "sig-done"},
            "proposal_status": {"value": "completed"},
            "planned_task_id": {"value": "task-old"},
            "completed_at": {"value": "2026-05-21T09:00:00Z"},
            "created_at": {"value": "2026-05-21T08:00:00Z"},
        },
    ]
    repo = GraphAutonomyRepository(endpoint="http://fake", timeout_sec=1.0, query_client=_StubClient(rows), goals_limit=3)
    goals, row_count = repo._fetch_active_goals(subject="orion", model_layer="self-model", entity_id="self:orion")

    assert row_count == 3
    assert "plannedTaskId" in repo._query_client.last_sparql
    assert "completedAt" in repo._query_client.last_sparql
    assert [g.artifact_id for g in goals] == ["goal-planned", "goal-executing"]
    assert goals[0].proposal_status == "planned"
    assert goals[0].planned_task_id == "task-abc-123"
    assert goals[0].completed_at is None
    assert goals[1].proposal_status == "executing"
    assert goals[1].planned_task_id == "task-def-456"
