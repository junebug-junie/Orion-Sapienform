from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orion.autonomy.repository import GraphAutonomyRepository
from orion.autonomy.summary import summarize_autonomy_lookup


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


def test_deferred_drives_degraded_reason_says_deferred_not_failed(monkeypatch) -> None:
    """Deferred drives should produce 'drives facet deferred', not 'drives facet failed (deferred)'."""
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES", "true")
    client = MagicMock()
    client.select.side_effect = lambda sparql: _rows_for_sparql(sparql)

    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=20.0,
        query_client=client,
        subquery_max_workers=1,
    )
    lookup = repo.get_latest("orion", observer={"consumer": "chat_stance", "correlation_id": "c2"})
    summary = summarize_autonomy_lookup(
        lookup.state,
        selected_subject="orion",
        availability=lookup.availability,
        subquery_diagnostics=lookup.subquery_diagnostics,
    )
    assert summary.degraded_reason is not None
    assert "failed" not in summary.degraded_reason
    assert "deferred" in summary.degraded_reason


def test_orion_drives_queried_for_chat_stance_when_defer_disabled(monkeypatch) -> None:
    """Orion drives are queried on chat_stance only when AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=false."""
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES", "false")
    drive_rows = [
        {
            "artifact_id": {"value": "drive-1"},
            "created_at": {"value": "2026-05-20T12:00:00Z"},
            "dominant_drive": {"value": "coherence"},
        }
    ]
    client = MagicMock()
    client.select.side_effect = lambda sparql: drive_rows if "DriveAudit" in sparql else _rows_for_sparql(sparql)

    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=20.0,
        query_client=client,
        subquery_max_workers=1,
    )
    repo._drives_query_client = client  # drives use a separate client; patch it too
    lookup = repo.get_latest("orion", observer={"consumer": "chat_stance", "correlation_id": "c3"})
    drives_diag = (lookup.subquery_diagnostics or {}).get("drives") or {}
    assert drives_diag.get("status") != "deferred"
    assert drives_diag.get("status") == "ok"
    drive_queries = [c[0][0] for c in client.select.call_args_list if "DriveAudit" in c[0][0]]
    assert drive_queries, "DriveAudit query was not issued — drives were incorrectly deferred by default"


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
