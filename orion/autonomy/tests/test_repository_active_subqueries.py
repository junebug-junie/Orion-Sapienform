from __future__ import annotations

from unittest.mock import MagicMock

from orion.autonomy.repository import GraphAutonomyRepository


def test_active_subqueries_identity_only_issues_single_sparql_family() -> None:
    client = MagicMock()
    client.select.return_value = []

    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.test:7200/repositories/collapse",
        timeout_sec=1.0,
        query_client=client,
        active_subqueries=("identity",),
    )
    out = repo.get_latest("orion", observer={"consumer": "chat_stance", "correlation_id": "t1"})
    assert out.availability in {"empty", "unavailable"}
    assert client.select.call_count == 1
    sparql = client.select.call_args[0][0]
    assert "IdentitySnapshot" in sparql
    assert "DriveAudit" not in sparql
    assert "ProposedGoal" not in sparql
