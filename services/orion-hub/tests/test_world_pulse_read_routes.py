"""World-pulse-read schedule route + Hub settings defaults.

Wallet A keys must be imported from `orion.world_pulse_read.wallet_a`, never
retyped as string literals. A dashboard that copies `orion:wp_read:...` would
keep rendering a confident 0 the day the prefix changes.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.world_pulse_read import wallet_a as wa

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
