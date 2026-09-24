"""The expect key is refreshed on its own cheap clock (review finding 4):
a window that opens between two 15-minute rhythm ticks must steer the camera
within a minute, not up to 15 minutes late. No Postgres, no Redis."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app import vision_rhythm as vr
from app.settings import Settings

T0 = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)


class _Conn:
    def __init__(self, rows):
        self.rows = rows
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        self.sql.append(" ".join(str(stmt).split()))
        rows = [r for r in self.rows if r.window_start <= params["t"] < r.window_end]
        return SimpleNamespace(fetchall=lambda: rows)


class _Engine:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return self._conn


class _FakeRedis:
    def __init__(self):
        self.sets, self.deletes = {}, []

    def set(self, key, value, ex=None):
        self.sets[key] = ex

    def delete(self, key):
        self.deletes.append(key)

    def close(self):
        pass


def _fake_redis_module(monkeypatch):
    client = _FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=SimpleNamespace(from_url=lambda *a, **k: client)))
    return client


def _window(eid, start, end, stream="walkway", subject="label:vehicle"):
    return SimpleNamespace(expectation_id=eid, stream_id=stream, subject_key=subject,
                           window_start=start, window_end=end)


def test_window_opening_between_rhythm_ticks_sets_the_key(monkeypatch) -> None:
    client = _fake_redis_module(monkeypatch)
    monkeypatch.setattr(vr, "_occurred", lambda *a, **k: None)
    conn = _Conn([_window("e1", T0 + timedelta(minutes=1), T0 + timedelta(minutes=31))])
    engine = _Engine(conn)
    # Just before the window opens: nothing.
    assert vr.run_one_expect_refresh(engine=engine, redis_url="redis://x", now=T0)["expect_keys"] == 0
    assert client.sets == {}
    # One refresh later (60 s): the key is set, TTL to window end.
    out = vr.run_one_expect_refresh(engine=engine, redis_url="redis://x", now=T0 + timedelta(minutes=2))
    assert out == {"open_streams": 1, "expect_keys": 1}
    assert client.sets == {"orion:vision:expect:walkway": 29 * 60}
    # Read-only: the refresh never writes Postgres.
    assert all(s.startswith("SELECT") for s in conn.sql)


def test_met_window_deletes_the_key(monkeypatch) -> None:
    client = _fake_redis_module(monkeypatch)
    monkeypatch.setattr(vr, "_occurred", lambda *a, **k: T0)
    conn = _Conn([_window("e1", T0 - timedelta(minutes=5), T0 + timedelta(minutes=25))])
    vr.run_one_expect_refresh(engine=_Engine(conn), redis_url="redis://x", now=T0)
    assert client.deletes == ["orion:vision:expect:walkway"] and client.sets == {}


def test_missing_table_is_migration_missing(monkeypatch) -> None:
    from app.vision_individuals import MigrationMissing

    class _Boom(_Conn):
        def execute(self, stmt, params=None):
            raise RuntimeError('relation "vision_percept_expectation" does not exist')

    with pytest.raises(MigrationMissing):
        vr.run_one_expect_refresh(engine=_Engine(_Boom([])), redis_url="redis://x", now=T0)


def test_refresh_interval_defaults_to_a_minute_and_is_wired_at_boot() -> None:
    assert Settings.model_fields["vision_expect_refresh_interval_sec"].default == 60.0
    from pathlib import Path

    main = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    assert "vision_expect_refresh_loop(settings)" in main
