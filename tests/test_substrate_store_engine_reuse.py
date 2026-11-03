"""Regression: substrate stores must reuse one engine, not build a pool per call.

Live 2026-11-02: GraphReviewTelemetryRecorder, GraphReviewQueue and
SubstratePolicyProfileStore called create_engine() in every load/persist. The
hub's autonomy loop reads telemetry every 30 s, so each tick left one idle
Postgres connection open until garbage collection -- 149 held, headroom alarm.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest


class _FakeConn:
    def execute(self, *_a, **_k):
        return self

    def fetchall(self):
        return []


class _FakeEngine:
    @contextmanager
    def begin(self):
        yield _FakeConn()


@pytest.fixture
def engines(monkeypatch):
    import sqlalchemy

    built: list[_FakeEngine] = []

    def _fake(*_a, **_k):
        built.append(_FakeEngine())
        return built[-1]

    monkeypatch.setattr(sqlalchemy, "create_engine", _fake)
    return built


def test_review_telemetry_reuses_one_engine(engines) -> None:
    from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryQueryV1
    from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder

    store = GraphReviewTelemetryRecorder(postgres_url="postgresql://x/y")
    for _ in range(5):
        store.query(GraphReviewTelemetryQueryV1(limit=1))
    assert store.source_kind() == "postgres"
    assert len(engines) == 1


def test_review_queue_reuses_one_engine(engines) -> None:
    from orion.substrate.review_queue import GraphReviewQueue

    store = GraphReviewQueue(postgres_url="postgresql://x/y")
    for _ in range(5):
        store._load_from_postgres()
        store._persist_to_postgres()
    assert store.source_kind() == "postgres"
    assert len(engines) == 1


def test_policy_profiles_reuses_one_engine(engines) -> None:
    from orion.substrate.policy_profiles import SubstratePolicyProfileStore

    store = SubstratePolicyProfileStore(postgres_url="postgresql://x/y")
    for _ in range(5):
        store._load_from_postgres()
        store._persist_to_postgres()
    assert store.source_kind() == "postgres"
    assert len(engines) == 1
