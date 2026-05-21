from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orion.autonomy.repository import GraphAutonomyRepository


def _rows_for_sparql(sparql: str) -> list[dict]:
    if "IdentitySnapshot" in sparql:
        return [
            {
                "artifact_id": {"value": "id-1"},
                "summary": {"value": "steady"},
                "anchor_strategy": {"value": "anchor"},
                "created_at": {"value": "2026-05-20T12:00:00Z"},
            }
        ]
    if "ProposedGoal" in sparql:
        return [
            {
                "artifact_id": {"value": "goal-1"},
                "drive_origin": {"value": "autonomy"},
                "goal_statement": {"value": "Focus coherence"},
                "priority": {"value": "0.8"},
                "proposal_status": {"value": "proposed"},
                "created_at": {"value": "2026-05-20T12:00:00Z"},
            }
        ]
    if "DriveAudit" in sparql:
        pytest.fail("Orion drives query should be deferred for chat_stance")
    return []


def test_defer_orion_drives_for_chat_stance(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES", "true")
    client = MagicMock()
    client.select.side_effect = lambda sparql: _rows_for_sparql(sparql)

    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=20.0,
        query_client=client,
        subquery_max_workers=1,
    )
    lookup = repo.get_latest("orion", observer={"consumer": "chat_stance", "correlation_id": "c1"})
    drives_diag = (lookup.subquery_diagnostics or {}).get("drives") or {}
    assert drives_diag.get("status") == "deferred"
    assert lookup.availability in {"available", "degraded", "empty"}
    drive_queries = [c[0][0] for c in client.select.call_args_list if "DriveAudit" in c[0][0]]
    assert drive_queries == []


def test_orion_drives_still_queried_for_non_chat_stance(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES", "true")
    client = MagicMock()
    client.select.side_effect = lambda sparql: (
        [
            {
                "artifact_id": {"value": "drive-1"},
                "created_at": {"value": "2026-05-20T12:00:00Z"},
                "dominant_drive": {"value": "coherence"},
            }
        ]
        if "DriveAudit" in sparql
        else _rows_for_sparql(sparql)
    )

    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=20.0,
        query_client=client,
        subquery_max_workers=1,
    )
    lookup = repo.get_latest("orion", observer={"consumer": "hub_probe"})
    drives_diag = (lookup.subquery_diagnostics or {}).get("drives") or {}
    assert drives_diag.get("status") != "deferred"
