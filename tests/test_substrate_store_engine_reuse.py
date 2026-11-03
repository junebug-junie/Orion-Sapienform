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


def test_felt_state_readers_share_one_engine_per_url(engines, monkeypatch) -> None:
    """orion/hub/association.py builds a new reader on every hub chat turn."""
    import sqlalchemy

    from orion.substrate import felt_state_reader

    monkeypatch.setattr(felt_state_reader, "create_engine", sqlalchemy.create_engine)
    felt_state_reader._shared_engine.cache_clear()
    try:
        readers = [
            felt_state_reader.SubstrateFeltStateReader(enabled=True, database_url="postgresql://x/y", max_age_sec=60)
            for _ in range(5)
        ]
        other = felt_state_reader.SubstrateFeltStateReader(enabled=True, database_url="postgresql://x/z", max_age_sec=60)
        assert all(r._engine is readers[0]._engine for r in readers)
        assert other._engine is not readers[0]._engine
        assert len(engines) == 2
    finally:
        felt_state_reader._shared_engine.cache_clear()
