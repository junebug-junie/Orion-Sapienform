"""Graph autonomy list_latest: short-circuit requires consumer allow-list + autonomy_subject_fanout=bounded."""

from __future__ import annotations

import pytest

from orion.autonomy.repository import AutonomyLookupV1, GraphAutonomyRepository


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> GraphAutonomyRepository:
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_SHORT_CIRCUIT", "true")
    monkeypatch.setenv("AUTONOMY_SHORT_CIRCUIT_CONSUMERS", "chat_stance,autonomy_ctx_adapter")
    return GraphAutonomyRepository(
        endpoint="http://fake-graphdb/repositories/test",
        timeout_sec=1.0,
        subject_max_workers=1,
    )


def test_list_latest_short_circuits_for_autonomy_ctx_adapter(
    monkeypatch: pytest.MonkeyPatch, repo: GraphAutonomyRepository
) -> None:
    called: list[str] = []

    def fake_query(self: GraphAutonomyRepository, subject: str, *, observer: dict | None = None) -> AutonomyLookupV1:
        del self, observer
        called.append(subject)
        return AutonomyLookupV1(subject=subject, state=None, availability="available")

    monkeypatch.setattr(GraphAutonomyRepository, "_query_subject", fake_query)

    repo.list_latest(
        ["orion", "relationship", "juniper"],
        observer={
            "consumer": "autonomy_ctx_adapter",
            "correlation_id": "t1",
            "autonomy_subject_fanout": "bounded",
        },
    )
    assert called == ["orion"]


def test_list_latest_does_not_short_circuit_on_partial_orion(
    monkeypatch: pytest.MonkeyPatch, repo: GraphAutonomyRepository
) -> None:
    called: list[str] = []

    def fake_query(self: GraphAutonomyRepository, subject: str, *, observer: dict | None = None) -> AutonomyLookupV1:
        del self, observer
        called.append(subject)
        if subject == "orion":
            return AutonomyLookupV1(
                subject=subject,
                state=None,
                availability="available",
                unavailable_reason="timeout",
                subquery_diagnostics={"drives": {"status": "timeout", "row_count": 0}},
            )
        return AutonomyLookupV1(subject=subject, state=None, availability="available")

    monkeypatch.setattr(GraphAutonomyRepository, "_query_subject", fake_query)

    repo.list_latest(
        ["orion", "relationship", "juniper"],
        observer={
            "consumer": "autonomy_ctx_adapter",
            "correlation_id": "t-partial",
            "autonomy_subject_fanout": "bounded",
        },
    )
    assert called == ["orion", "relationship", "juniper"]


def test_list_latest_full_fanout_when_consumer_not_in_short_circuit_set(
    monkeypatch: pytest.MonkeyPatch, repo: GraphAutonomyRepository
) -> None:
    called: list[str] = []

    def fake_query(self: GraphAutonomyRepository, subject: str, *, observer: dict | None = None) -> AutonomyLookupV1:
        del self, observer
        called.append(subject)
        return AutonomyLookupV1(subject=subject, state=None, availability="available")

    monkeypatch.setattr(GraphAutonomyRepository, "_query_subject", fake_query)

    repo.list_latest(
        ["orion", "relationship", "juniper"],
        observer={"consumer": "autonomy_graph_probe", "correlation_id": "t2", "autonomy_subject_fanout": "full"},
    )
    assert called == ["orion", "relationship", "juniper"]


def test_list_latest_respects_autonomy_short_circuit_consumers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTONOMY_CHAT_STANCE_SHORT_CIRCUIT", "true")
    monkeypatch.setenv("AUTONOMY_SHORT_CIRCUIT_CONSUMERS", "chat_stance")
    repo = GraphAutonomyRepository(
        endpoint="http://fake-graphdb/repositories/test",
        timeout_sec=1.0,
        subject_max_workers=1,
    )
    called: list[str] = []

    def fake_query(self: GraphAutonomyRepository, subject: str, *, observer: dict | None = None) -> AutonomyLookupV1:
        del self, observer
        called.append(subject)
        return AutonomyLookupV1(subject=subject, state=None, availability="available")

    monkeypatch.setattr(GraphAutonomyRepository, "_query_subject", fake_query)

    repo.list_latest(
        ["orion", "relationship", "juniper"],
        observer={
            "consumer": "autonomy_ctx_adapter",
            "correlation_id": "t3",
            "autonomy_subject_fanout": "bounded",
        },
    )
    assert called == ["orion", "relationship", "juniper"]


def test_list_latest_full_fanout_when_fanout_full_even_for_eligible_consumer(
    monkeypatch: pytest.MonkeyPatch, repo: GraphAutonomyRepository
) -> None:
    called: list[str] = []

    def fake_query(self: GraphAutonomyRepository, subject: str, *, observer: dict | None = None) -> AutonomyLookupV1:
        del self, observer
        called.append(subject)
        return AutonomyLookupV1(subject=subject, state=None, availability="available")

    monkeypatch.setattr(GraphAutonomyRepository, "_query_subject", fake_query)

    repo.list_latest(
        ["orion", "relationship", "juniper"],
        observer={
            "consumer": "autonomy_ctx_adapter",
            "correlation_id": "t4",
            "autonomy_subject_fanout": "full",
        },
    )
    assert called == ["orion", "relationship", "juniper"]
