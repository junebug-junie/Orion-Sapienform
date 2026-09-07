"""World-pulse-read schedule route + Hub settings defaults.

Wallet A keys must be imported from `orion.world_pulse_read.wallet_a`, never
retyped as string literals. A dashboard that copies `orion:wp_read:...` would
keep rendering a confident 0 the day the prefix changes.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read import wallet_b as wb

HUB_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PY = HUB_ROOT / "app" / "settings.py"
ENV_EXAMPLE = HUB_ROOT / ".env_example"
MAIN_PY = HUB_ROOT / "scripts" / "main.py"
ROUTES_PY = HUB_ROOT / "scripts" / "world_pulse_read_routes.py"

_EXPECTED_DEFAULTS = {
    "HUB_WORLD_PULSE_READ_ENABLED": ("bool", "True"),
    "HUB_WORLD_PULSE_READ_TICK_SEC": ("float", "300"),
    "HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC": ("float", "1800"),
    "HUB_WORLD_PULSE_READ_DAILY_CAP": ("int", "6"),
    "HUB_WORLD_PULSE_READ_WINDOW_START_HOUR": ("int", "8"),
    "HUB_WORLD_PULSE_READ_WINDOW_END_HOUR": ("int", "22"),
    "HUB_WORLD_PULSE_READ_TIMEOUT_SEC": ("float", "3500"),
    "HUB_WORLD_PULSE_READ_SESSION_ID": ("str", '"orion_world_pulse_read"'),
    "HUB_WORLD_PULSE_READ_LLM_ROUTE": ("str", '"agent"'),
    "HUB_WORLD_PULSE_READ_STAGE2_ENABLED": ("bool", "True"),
    "HUB_WORLD_PULSE_READ_STAGE2_TICK_SEC": ("float", "300"),
    "HUB_WORLD_PULSE_READ_STAGE2_MIN_COOLDOWN_SEC": ("float", "1800"),
    "HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP": ("int", "6"),
    "HUB_WORLD_PULSE_READ_STAGE2_WINDOW_START_HOUR": ("int", "8"),
    "HUB_WORLD_PULSE_READ_STAGE2_WINDOW_END_HOUR": ("int", "22"),
    "HUB_WORLD_PULSE_READ_STAGE2_TIMEOUT_SEC": ("float", "3500"),
    "HUB_WORLD_PULSE_READ_STAGE2_SESSION_ID": ("str", '"orion_world_pulse_read_stage2"'),
    "HUB_WORLD_PULSE_READ_STAGE2_LLM_ROUTE": ("str", '"agent"'),
    "HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS": ("int", "5"),
}


def _field_default(src: str, name: str, typ: str) -> str:
    match = re.search(
        rf"{name}:\s*{re.escape(typ)}\s*=\s*Field\(\s*default=([^,\n]+)",
        src,
    )
    assert match, f"{name} Field default not found in settings.py"
    return match.group(1).strip()


def test_settings_defaults_match_wallet_a_live_contract() -> None:
    src = SETTINGS_PY.read_text(encoding="utf-8")
    for name, (typ, expected) in _EXPECTED_DEFAULTS.items():
        got = _field_default(src, name, typ)
        if typ in {"float", "int"}:
            assert float(got) == float(expected), f"{name}: {got!r} != {expected!r}"
        else:
            assert got == expected, f"{name}: {got!r} != {expected!r}"


def test_env_example_ships_keys_and_keeps_wallet_a_independent() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for name in _EXPECTED_DEFAULTS:
        assert re.search(rf"^{re.escape(name)}=", text, re.M), f"{name} missing from .env_example"
    enabled = re.search(r"^HUB_WORLD_PULSE_READ_ENABLED=(.+)$", text, re.M)
    assert enabled, "HUB_WORLD_PULSE_READ_ENABLED missing"
    assert enabled.group(1).strip().lower() in {"true", "1", "yes"}, (
        "deploy default is on after migration; .env_example enabled=%r" % enabled.group(1)
    )
    cap = re.search(r"^HUB_WORLD_PULSE_READ_DAILY_CAP=(.+)$", text, re.M)
    assert cap and int(float(cap.group(1).strip())) == 6
    stage2 = re.search(r"^HUB_WORLD_PULSE_READ_STAGE2_ENABLED=(.+)$", text, re.M)
    assert stage2, "HUB_WORLD_PULSE_READ_STAGE2_ENABLED missing"
    assert stage2.group(1).strip().lower() in {"true", "1", "yes"}
    wallet_b = re.search(r"^HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP=(.+)$", text, re.M)
    assert wallet_b and int(float(wallet_b.group(1).strip())) == 6
    trips = re.search(r"^HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS=(.+)$", text, re.M)
    assert trips and int(float(trips.group(1).strip())) == 5
    assert "HUB_CURIOSITY_INVESTIGATION_DAILY_CAP" in text
    # The independence note must sit near the world-pulse-read keys, not only
    # in the curiosity block further up the file.
    wp_block = text[text.index("HUB_WORLD_PULSE_READ_ENABLED") :]
    assert "HUB_CURIOSITY_INVESTIGATION_DAILY_CAP" in wp_block[:2000]
    sync_src = HUB_ROOT.parents[1] / "scripts" / "sync_local_env_from_example.py"
    assert '"HUB_WORLD_PULSE_READ_"' in sync_src.read_text(encoding="utf-8")


def test_schedule_route_imports_wallet_a_keys_never_retyped() -> None:
    src = ROUTES_PY.read_text(encoding="utf-8")
    assert "from orion.world_pulse_read.wallet_a import" in src or (
        "from orion.world_pulse_read import wallet_a" in src
    )
    assert "WALLET_A_COOLDOWN_KEY" in src
    assert "WALLET_A_COUNT_KEY_PREFIX" in src
    assert "orion:wp_read:wallet_a:last_at" not in src
    assert "orion:wp_read:wallet_a:count:" not in src


def test_main_wires_pipeline_from_settings_opt_in() -> None:
    src = MAIN_PY.read_text(encoding="utf-8")
    assert "WorldPulseReadPipeline" in src
    assert "enabled=settings.HUB_WORLD_PULSE_READ_ENABLED" in src
    assert "timezone_name=settings.HUB_ENDOGENOUS_OUTREACH_TZ" in src
    assert "pool_provider=lambda: getattr(app.state, \"memory_pg_pool\", None)" in src
    assert "step_relay_provider=lambda: harness_step_relay" in src
    ctor = src.split("world_pulse_read_pipeline = WorldPulseReadPipeline(", 1)[1].split(
        "await world_pulse_read_pipeline.start", 1
    )[0]
    assert "store_provider=" in ctor
    assert "_get_substrate_store" in ctor
    assert "harness_rpc_bus=rpc_bus" in src
    assert "world_pulse_read_router" in src or "world_pulse_read_routes" in src
    # Module-global must be declared in startup/shutdown global lists (UnboundLocalError otherwise).
    startup_block = src.split("async def startup_event", 1)[1].split("\nasync def ", 1)[0]
    shutdown_block = src.split("async def shutdown_event", 1)[1].split("\nasync def ", 1)[0]
    startup_global = re.search(r"^\s*global (.+)$", startup_block, re.MULTILINE)
    shutdown_global = re.search(r"^\s*global (.+)$", shutdown_block, re.MULTILINE)
    assert startup_global is not None
    assert shutdown_global is not None
    assert "world_pulse_read_pipeline" in startup_global.group(1)
    assert "world_pulse_read_pipeline" in shutdown_global.group(1)
    assert "WorldPulseReadStage2Pipeline" in src
    assert "enabled=settings.HUB_WORLD_PULSE_READ_STAGE2_ENABLED" in src
    assert "daily_cap=settings.HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP" in src
    assert "max_round_trips=settings.HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS" in src
    assert "world_pulse_read_stage2" in startup_global.group(1)
    assert "world_pulse_read_stage2" in shutdown_global.group(1)
    assert "store_provider=concept_atlas_routes_runtime._get_substrate_store" in src


class _FakeRedis:
    def __init__(self, store: dict[str, str] | None = None) -> None:
        self.store = store or {}

    async def get(self, key):
        return self.store.get(key)


def _schedule_app() -> FastAPI:
    from scripts.world_pulse_read_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_schedule_app())


def test_schedule_payload_includes_daily_cap_and_wallet_a_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import world_pulse_read_routes as routes

    cfg = SimpleNamespace(
        HUB_WORLD_PULSE_READ_ENABLED=False,
        HUB_WORLD_PULSE_READ_DAILY_CAP=6,
        HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC=1800.0,
        HUB_ENDOGENOUS_OUTREACH_TZ="UTC",
    )
    monkeypatch.setattr(routes, "_settings", lambda: cfg)
    monkeypatch.setattr(routes, "_redis", lambda: _FakeRedis())

    response = client.get("/world-pulse-read/api/schedule")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["daily_cap"] == 6
    assert payload["done_today"] == 0
    assert payload["cooldown_key"] == wa.WALLET_A_COOLDOWN_KEY
    assert payload["count_key_prefix"] == wa.WALLET_A_COUNT_KEY_PREFIX
    assert payload["cooldown_key"] == "orion:wp_read:wallet_a:last_at"
    assert payload["count_key_prefix"] == "orion:wp_read:wallet_a:count:"


def test_schedule_payload_reads_done_today_from_wallet_a_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import world_pulse_read_routes as routes

    cfg = SimpleNamespace(
        HUB_WORLD_PULSE_READ_ENABLED=True,
        HUB_WORLD_PULSE_READ_DAILY_CAP=6,
        HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC=1800.0,
        HUB_ENDOGENOUS_OUTREACH_TZ="UTC",
    )
    store = {f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-06": "3"}
    monkeypatch.setattr(routes, "_settings", lambda: cfg)
    monkeypatch.setattr(routes, "_redis", lambda: _FakeRedis(store))
    monkeypatch.setattr(routes, "_local_date", lambda _tz: "2026-09-06")

    response = client.get("/world-pulse-read/api/schedule")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["done_today"] == 3
    assert payload["daily_cap"] == 6
    assert payload["cooldown_key"] is wa.WALLET_A_COOLDOWN_KEY or (
        payload["cooldown_key"] == wa.WALLET_A_COOLDOWN_KEY
    )


def test_schedule_handler_source_uses_imported_wallet_constants() -> None:
    from scripts.world_pulse_read_routes import world_pulse_read_schedule

    src = inspect.getsource(world_pulse_read_schedule)
    assert "WALLET_A_COOLDOWN_KEY" in src
    assert "WALLET_A_COUNT_KEY_PREFIX" in src
    assert "orion:wp_read:wallet_a:last_at" not in src
    assert "orion:wp_read:wallet_a:count:" not in src


def test_status_route_imports_wallet_keys_never_retyped() -> None:
    src = ROUTES_PY.read_text(encoding="utf-8")
    assert "WALLET_B_COOLDOWN_KEY" in src
    assert "WALLET_B_COUNT_KEY_PREFIX" in src
    assert "orion:wp_read:wallet_b:last_at" not in src
    assert "orion:wp_read:wallet_b:count:" not in src


class _PoolCtx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def acquire(self):
        return _PoolCtx()


def test_status_payload_includes_both_wallets_and_queue_counts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import world_pulse_read_routes as routes

    cfg = SimpleNamespace(
        HUB_WORLD_PULSE_READ_ENABLED=True,
        HUB_WORLD_PULSE_READ_DAILY_CAP=6,
        HUB_WORLD_PULSE_READ_STAGE2_ENABLED=True,
        HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP=6,
        HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS=5,
        HUB_ENDOGENOUS_OUTREACH_TZ="UTC",
    )
    today = "2026-09-06"
    store = {
        f"{wa.WALLET_A_COUNT_KEY_PREFIX}{today}": "3",
        wa.WALLET_A_COOLDOWN_KEY: "2026-09-06T15:00:00+00:00",
        f"{wb.WALLET_B_COUNT_KEY_PREFIX}{today}": "1",
        wb.WALLET_B_COOLDOWN_KEY: "2026-09-06T16:00:00+00:00",
    }

    async def _q(_conn):
        return {"pending": 2, "claimed": 0, "done": 10, "failed": 1, "skipped": 0}

    async def _s2(_conn):
        return {"pending": 4, "claimed": 0, "done": 6, "failed": 0, "skipped": 0}

    async def _ts(_conn):
        return {
            "last_stage1_at": datetime(2026, 9, 6, 15, tzinfo=timezone.utc),
            "last_stage2_at": datetime(2026, 9, 6, 16, tzinfo=timezone.utc),
        }

    monkeypatch.setattr(routes, "_settings", lambda: cfg)
    monkeypatch.setattr(routes, "_redis", lambda: _FakeRedis(store))
    monkeypatch.setattr(routes, "_local_date", lambda _tz: today)
    monkeypatch.setattr(routes, "_pool", lambda: _FakePool())
    monkeypatch.setattr(routes, "count_seeds_by_status", _q)
    monkeypatch.setattr(routes, "count_stage2_by_status", _s2)
    monkeypatch.setattr(routes, "last_stage_timestamps", _ts)

    response = client.get("/world-pulse-read/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["wallet_a"]["enabled"] is True
    assert payload["wallet_a"]["done_today"] == 3
    assert payload["wallet_a"]["daily_cap"] == 6
    assert payload["wallet_a"]["cooldown_key"] == wa.WALLET_A_COOLDOWN_KEY
    assert payload["wallet_a"]["count_key_prefix"] == wa.WALLET_A_COUNT_KEY_PREFIX
    assert payload["wallet_b"]["enabled"] is True
    assert payload["wallet_b"]["done_today"] == 1
    assert payload["wallet_b"]["daily_cap"] == 6
    assert payload["wallet_b"]["cooldown_key"] == wb.WALLET_B_COOLDOWN_KEY
    assert payload["wallet_b"]["count_key_prefix"] == wb.WALLET_B_COUNT_KEY_PREFIX
    assert payload["queue"]["pending"] == 2
    assert payload["queue"]["done"] == 10
    assert payload["stage2_queue"]["pending"] == 4
    assert payload["stage2_queue"]["done"] == 6
    assert payload["last_stage1_at"]
    assert payload["last_stage2_at"]
    assert payload["stage2_max_round_trips"] == 5


def test_schedule_still_works_alongside_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import world_pulse_read_routes as routes

    cfg = SimpleNamespace(
        HUB_WORLD_PULSE_READ_ENABLED=True,
        HUB_WORLD_PULSE_READ_DAILY_CAP=6,
        HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC=1800.0,
        HUB_ENDOGENOUS_OUTREACH_TZ="UTC",
    )
    monkeypatch.setattr(routes, "_settings", lambda: cfg)
    monkeypatch.setattr(routes, "_redis", lambda: _FakeRedis())
    response = client.get("/world-pulse-read/api/schedule")
    assert response.status_code == 200
    assert response.json()["cooldown_key"] == wa.WALLET_A_COOLDOWN_KEY
