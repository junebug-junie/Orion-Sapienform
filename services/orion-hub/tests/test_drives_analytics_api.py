from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import api_routes


@pytest.fixture
def hub_main():
    import scripts.main as _hub_main

    return _hub_main


class _FakeConn:
    """Dispatch by SQL keyword so multi-query window/series helpers work."""

    def __init__(self, *, handlers=None, raise_exc: BaseException | None = None):
        self._handlers = handlers or {}
        self._raise_exc = raise_exc

    def _dispatch(self, query: str, args):
        if self._raise_exc is not None:
            raise self._raise_exc
        q = " ".join(str(query).lower().split())
        for key, fn in self._handlers.items():
            if key in q:
                return fn(query, args)
        return None

    async def fetchrow(self, query, *args, **_kwargs):
        result = self._dispatch(query, args)
        if isinstance(result, list):
            return result[0] if result else None
        return result

    async def fetch(self, query, *args, **_kwargs):
        result = self._dispatch(query, args)
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_exc_info):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


def _snapshot_row(*, subject: str = "orion") -> dict:
    now = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "artifact_id": "audit-1",
        "subject": subject,
        "active_count": 2,
        "active_drives": ["predictive", "relational"],
        "dominant_drive": "predictive",
        "summary": "ok",
        "drive_pressures": {
            "coherence": 0.1,
            "continuity": 0.1,
            "capability": 0.1,
            "relational": 0.4,
            "predictive": 0.7,
            "autonomy": 0.1,
        },
        "tick_attribution": {"predictive": 0.5, "relational": 0.2},
        "tension_kinds": ["substrate.world_coverage_gap"],
        "correlation_id": "corr-1",
        "observed_at": now,
        "created_at": now,
    }


@pytest.mark.asyncio
async def test_snapshot_degrades_when_pool_missing(monkeypatch, hub_main) -> None:
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    payload = await api_routes.api_drives_analytics_snapshot(subject="orion")
    assert payload["degraded"] is True
    assert "error" in payload["source"] or "error" in payload


@pytest.mark.asyncio
async def test_snapshot_returns_pressures_from_fake_row(monkeypatch, hub_main) -> None:
    import scripts.drives_analytics as da_live

    now = datetime.now(timezone.utc)
    row = _snapshot_row()
    row["observed_at"] = now
    row["created_at"] = now
    pool = _FakePool(
        _FakeConn(
            handlers={
                "order by coalesce(observed_at, created_at) desc": lambda _q, _a: row,
            }
        )
    )
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", pool, raising=False)
    payload = await api_routes.api_drives_analytics_snapshot(subject="orion")
    assert payload["degraded"] is False
    assert "predictive" in payload["drive_pressures"]
    assert payload["drive_pressures"]["predictive"] == 0.7
    assert payload["tick_attribution"]["predictive"] == 0.5
    assert payload["stale"] is False
    assert da_live.STALE_AFTER_SEC == 300


@pytest.mark.asyncio
async def test_subjects_include_allowlist_and_discovered(monkeypatch, hub_main) -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
    pool = _FakePool(
        _FakeConn(
            handlers={
                "group by subject": lambda _q, _a: [
                    {
                        "subject": "orion",
                        "row_count": 10,
                        "oldest_ts": now - timedelta(hours=2),
                        "newest_ts": now,
                    },
                    {
                        "subject": "custom-subject",
                        "row_count": 3,
                        "oldest_ts": now - timedelta(hours=1),
                        "newest_ts": now,
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", pool, raising=False)
    payload = await api_routes.api_drives_analytics_subjects()
    assert payload["degraded"] is False
    names = [s["subject"] for s in payload["subjects"]]
    assert names[0:3] == ["orion", "relationship", "juniper"]
    assert "custom-subject" in names


@pytest.mark.asyncio
async def test_window_builds_kpis(monkeypatch, hub_main) -> None:
    import scripts.drives_analytics_queries as daq

    now = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)

    def _handler(query, _args):
        q = " ".join(str(query).lower().split())
        if "group by active_count" in q:
            return [{"active_count": 2, "n": 8}, {"active_count": 3, "n": 2}]
        if "group by dominant_drive" in q:
            return [{"dominant_drive": "predictive", "n": 6}, {"dominant_drive": "relational", "n": 4}]
        if "count(*)::bigint as row_count" in q:
            return {
                "row_count": 10,
                "oldest_ts": now - timedelta(hours=3),
                "newest_ts": now,
            }
        if "avg(nullif(drive_pressures" in q:
            return {
                "coherence": 0.2,
                "continuity": 0.2,
                "capability": 0.2,
                "relational": 0.4,
                "predictive": 0.6,
                "autonomy": 0.1,
            }
        if "select tick_attribution" in q:
            return [
                {"tick_attribution": {"predictive": 0.5}, "observed_at": now},
                {"tick_attribution": None, "observed_at": now - timedelta(hours=1)},
            ]
        return []

    pool = _FakePool(
        _FakeConn(
            handlers={
                "from drive_audits": _handler,
            }
        )
    )
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", pool, raising=False)
    monkeypatch.setattr(daq, "_now_utc", lambda: now)
    payload = await api_routes.api_drives_analytics_window(subject="orion", hours=24)
    assert payload["degraded"] is False
    assert payload["mean_pressures"]["predictive"] == 0.6
    assert payload["kpis"]["gate_verdict_drive_only"] in {
        "GO_DRIVE_ONLY",
        "SATURATED",
        "NO-GO",
        "UNMEASURABLE",
    }
    assert payload["attribution"]["attributed_row_count"] == 1
    assert payload["coverage"]["row_count"] == 10


@pytest.mark.asyncio
async def test_series_returns_tick_rate_and_pressures(monkeypatch, hub_main) -> None:
    import scripts.drives_analytics as da_live
    import scripts.drives_analytics_queries as daq

    now = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
    pressures = {k: 0.3 for k in da_live.DRIVE_KEYS}
    pressures["predictive"] = 0.8

    def _handler(query, _args):
        q = " ".join(str(query).lower().split())
        if "select coalesce(observed_at, created_at) as ts" in q:
            return [
                {"ts": now - timedelta(minutes=2), "drive_pressures": pressures},
                {"ts": now - timedelta(minutes=1), "drive_pressures": pressures},
            ]
        if "count(*)::bigint as row_count" in q:
            return {
                "row_count": 2,
                "oldest_ts": now - timedelta(minutes=2),
                "newest_ts": now - timedelta(minutes=1),
            }
        return []

    pool = _FakePool(_FakeConn(handlers={"from drive_audits": _handler}))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", pool, raising=False)
    monkeypatch.setattr(daq, "_now_utc", lambda: now)
    payload = await api_routes.api_drives_analytics_series(subject="orion", hours=1)
    assert payload["degraded"] is False
    assert isinstance(payload["tick_rate"], list)
    assert "predictive" in payload["pressures"]
    assert payload["pressures"]["predictive"][0]["v"] == 0.8


def test_no_mutation_routes_under_drives_analytics_prefix() -> None:
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    found = False
    for route in api_routes.router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if path.startswith("/api/drives-analytics"):
            found = True
            assert methods.isdisjoint(mutating)
    assert found


@pytest.mark.asyncio
async def test_goal_alignment_degrades_when_goals_unavailable(monkeypatch, hub_main) -> None:
    import scripts.drives_analytics as da_live
    import scripts.drives_analytics_queries as daq

    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    monkeypatch.setattr(
        daq,
        "fetch_goal_alignment_sync",
        lambda **kwargs: {
            "degraded": True,
            "goals_available": False,
            "source": {"subject": "orion", "degraded": True, "error": "goals unavailable"},
            "subject": "orion",
            "active_goals": [],
            "per_drive": {
                k: {"pressure": 0.0, "has_matching_goal": False, "color_align": "neutral"}
                for k in da_live.DRIVE_KEYS
            },
            "funnel": {
                "proposed": 0,
                "active": 0,
                "planned": 0,
                "executing": 0,
                "completed": 0,
                "archived": 0,
            },
            "funnel_scope": "active_headlines_only",
            "notes": ["goals unavailable"],
        },
    )
    payload = await api_routes.api_drives_analytics_goal_alignment(subject="orion")
    assert payload["degraded"] is True
    assert payload["goals_available"] is False
    assert "goals unavailable" in payload["notes"]


def test_divergence_fallback_banner_flag(monkeypatch) -> None:
    import scripts.drives_analytics as da_live
    import scripts.drives_analytics_queries as daq

    monkeypatch.delenv("CONCEPT_STORE_PATH", raising=False)

    class _Mod:
        @staticmethod
        def load_drive_state_v1(store_path, subject):
            return None, "missing"

    monkeypatch.setitem(sys.modules, "_orion_drive_state_divergence_audit_for_hub", _Mod())
    payload = daq.fetch_divergence_sync(
        subject="orion", audit_pressures={k: 0.1 for k in da_live.DRIVE_KEYS}
    )
    assert payload["store_path_is_fallback_default"] is True
    assert payload["degraded"] is True
    assert payload["autonomy_state_v2_note"]


def test_goal_alignment_sync_exposes_funnel_scope() -> None:
    import scripts.drives_analytics_queries as daq

    payload = daq.fetch_goal_alignment_sync(
        subject="orion",
        pressures={"predictive": 0.8},
        saturated=False,
        stale=False,
    )
    assert payload["funnel_scope"] == "active_headlines_only"
