"""Check recording observes the existing decision, never changes it."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

NOW = datetime(2026, 10, 9, 8, tzinfo=timezone.utc)


def deps(*, count=4, idle=120, last_end=None):
    from app.cycle import CycleDeps
    rows = {"metacog": [dict(id=str(i), summary="observed", severity="critical", dedupe_key=str(i))
                        for i in range(count)]}
    async def complete(prompt):
        return '{"link": false}'
    return CycleDeps(load_source_rows=lambda since, limit, until=None: rows if until is None else {},
                     load_idle_minutes=lambda: idle, load_last_window_start=lambda: NOW-timedelta(hours=7),
                     load_last_attempt_end=lambda: last_end, persist_cycle=lambda c: True, complete=complete)


@pytest.mark.parametrize("case", ["refractory", "busy", "low", "due", "forced"])
def test_every_check_is_saved_and_observer_failure_does_not_change_gates(monkeypatch, case):
    from app import cycle
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(cycle, "datetime", Clock)
    kwargs = {"count": 1} if case == "low" else {"idle": 1} if case in {"busy", "forced"} else (
        {"last_end": NOW-timedelta(hours=1)} if case == "refractory" else {})
    outcomes = []
    for observer in (None, "success", "raises", "false"):
        d, saved = deps(**kwargs), []
        def record(observation):
            saved.append(observation)
            if observer == "raises":
                raise RuntimeError("store unavailable")
            return observer != "false"
        if observer:
            d.persist_pressure_observation = record
        result = asyncio.run(cycle.run_cycle_once(d, force=case == "forced"))
        outcomes.append(None if result is None else (result.status, result.no_link_count, len(result.replay)))
        assert len(saved) == bool(observer)
        if saved:
            assert saved[0].observed_at == NOW
            assert saved[0].reading.pressure == (1 if case == "low" else 4)
            assert saved[0].forced == (case == "forced")
            assert saved[0].min_interval_hours == cycle.settings.DREAM_MIN_INTERVAL_HOURS
    assert outcomes.count(outcomes[0]) == 4
    assert (outcomes[0] is not None) == (case in {"due", "forced"})


def test_failed_source_read_is_saved_as_invalid_not_as_confirmed_calm():
    from app.cycle import run_cycle_once
    from app.cycle_store import SourceRows
    d, saved = deps(count=0), []
    def failed(since, limit, until=None):
        rows = SourceRows()
        rows.read_errors.append("metacog")
        return rows
    d.load_source_rows = failed
    d.persist_pressure_observation = lambda observation: saved.append(observation) or True
    asyncio.run(run_cycle_once(d))
    assert saved[0].reading.pressure == 0
    assert saved[0].source_errors == ["current:metacog", "prior:metacog"]


def test_real_loader_keeps_failure_provenance_without_changing_empty_fallback(monkeypatch):
    from app import cycle_store
    class Broken:
        def connect(self):
            raise RuntimeError("source unavailable")
    monkeypatch.setattr(cycle_store, "_get_engine", lambda: Broken())
    rows = cycle_store.load_source_rows(NOW, 5000)
    assert set(rows.read_errors) == set(cycle_store.SOURCE_QUERIES)
    assert all(v == [] for v in rows.values())


def test_clock_query_failures_survive_real_cache_wrappers_and_reach_history(monkeypatch):
    from app import cycle_store, main, cycle
    class Broken:
        def connect(self):
            raise RuntimeError("clock unavailable")
    monkeypatch.setattr(cycle_store, "_get_engine", lambda: Broken())
    monkeypatch.setattr(main, "_CYCLE_STATE", {"window_start": NOW-timedelta(hours=7), "attempt_end": NOW})
    d = main.build_cycle_deps()
    # Existing in-process floors are preserved even while the SQL reads fail.
    assert d.load_last_window_start() == NOW-timedelta(hours=7)
    assert d.load_last_attempt_end() == NOW
    saved = []
    d.load_source_rows = lambda *a, **kw: {}
    d.load_idle_minutes = lambda: 60
    d.persist_pressure_observation = lambda observation: saved.append(observation) or True
    asyncio.run(cycle.run_cycle_once(d))
    assert saved[0].source_errors == ["last_window_start", "last_attempt_end"]
    assert saved[0].last_attempt_end == NOW


def test_pressure_history_postgres_round_trip_idempotency_and_retention(monkeypatch):
    uri = os.environ.get("REGULATION_HISTORY_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("requires isolated regulation_history_test Postgres")
    assert urlparse(uri).path == "/regulation_history_test", "refuse any other database"
    from sqlalchemy import create_engine, text
    from app import cycle_store
    from orion.schemas.dream_cycle import DreamPressureObservationV1, SleepPressureV1
    engine = create_engine(uri)
    migration = Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_regulation_history.sql"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(migration.read_text())
    monkeypatch.setattr(cycle_store, "_history_engine", engine)
    observation = DreamPressureObservationV1(check_id="test-dream-check", observed_at=NOW,
        reading=SleepPressureV1(since=NOW-timedelta(hours=1), computed_at=NOW,
            pressure=0, threshold=3, idle_required_minutes=45, idle_minutes=60),
        trigger="pressure", forced=False, min_interval_hours=6, check_interval_sec=600, lookback_hours=48)
    assert cycle_store.persist_pressure_observation(observation)
    assert cycle_store.persist_pressure_observation(observation)
    with engine.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM dream_pressure_observation WHERE check_id='test-dream-check'")).scalar() == 1
        payload = conn.execute(text("SELECT observation_json FROM dream_pressure_observation WHERE check_id='test-dream-check'")).scalar()
        assert payload["reading"]["pressure"] == 0 and payload["source_errors"] == []
        conn.execute(text("UPDATE dream_pressure_observation SET created_at=now()-interval '31 days' WHERE check_id='test-dream-check'"))
    assert cycle_store.persist_pressure_observation(observation.model_copy(update={"check_id": "test-fresh-check"}))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM dream_pressure_observation WHERE check_id='test-dream-check'")).scalar() == 0
    engine.dispose()
