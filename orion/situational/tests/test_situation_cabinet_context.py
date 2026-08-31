"""2026-08-31: Orion's own physical cabinet sensors in the situation brief.

`LabContextV1` (`orion/schemas/situation.py`) has always been a stub -- a
hand-waved GPU-cluster thermal/power framing that never wired to a real
producer (`_build_lab_context` unconditionally returns `available=False`).
Juniper: "we had stubbed these as lab, but it is much richer!" -- Orion
already has a real, live physical sensor array on its own cabinet (Athena
Nano ESP32 node: BME680 climate, LTR390 UV/ALS, magnetometer, PMSA003I
particulate, VL53L1X lidar, BNO085 IMU/vibration), already read today by
`services/orion-hub/scripts/cabinet_sensors_routes.py`'s own
`/api/cabinet/sensors/latest` route. This section reuses that exact same
shared `orion.telemetry.cabinet_sensors`/`cabinet_snapshot_merge` machinery
for the chat-turn situation brief -- a new `CabinetContextV1`, not an
extension of `LabContextV1` (which stays its own distinct, still-unwired
concept).

Mirrors `test_situation_curiosity_reverie_context.py`'s framing: the
properties that matter are (1) ON by default in orion-hub, the only process
with the `/run/orion-sensors` bind mount, (2) every failure mode (disabled /
unconfigured / missing file / stale frame / empty measurements / read
exception) degrades to an honest "unavailable" state rather than a guess or
an exception, and (3) the rendered prompt line only calls out an
`*_activity` pressure as "notable" when it clears
`_CABINET_ACTIVITY_NOTABLE_THRESHOLD`, not on every non-zero volatility
reading.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.schemas.situation import CabinetContextV1, SituationBriefV1, SituationDiagnosticsV1
from orion.situational import context as situation_mod
from orion.situational.context import (
    SituationSettings,
    _build_cabinet_context,
    _build_prompt_fragment,
    _fetch_cabinet_context,
    settings_from_runtime,
)

CFG = situation_mod  # alias kept for readability below


@pytest.fixture(autouse=True)
def _clear_cabinet_cache():
    # Same leak risk _CURIOSITY_CACHE/_REVERIE_CACHE's own fixtures guard
    # against -- a cached result from one test would otherwise bleed into
    # the next test's assertions.
    situation_mod._CABINET_CACHE.clear()
    yield
    situation_mod._CABINET_CACHE.clear()


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _diag() -> SituationDiagnosticsV1:
    return SituationDiagnosticsV1()


def _frame(**overrides):
    base = {
        "schema": "orion.sensor_frame.v1",
        "seq": 1,
        "uptime_ms": 1000,
        "environment": {
            "temp_c": 24.6,
            "humidity_pct": 45.0,
            "pressure_hpa": 900.0,
            "gas_resistance_ohm": 12000.0,
        },
        "uv": {"raw": 17.0, "als_raw": 1292.0},
        "magnetic": {"x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0, "magnitude_ut": 53.0},
        "particulate": {"pm1_ug_m3": 2.0, "pm25_ug_m3": 4.0, "pm10_ug_m3": 5.0},
        "lidar": {"distance_mm": 438.0, "status": 0},
        "imu": {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.80665,
            "yaw_deg": 1.0,
            "pitch_deg": 2.0,
            "roll_deg": 3.0,
        },
    }
    base.update(overrides)
    return base


def _write_snapshot(path: Path, *, received_at: str | None = None, status: str = "ok", frame=None) -> None:
    payload = {
        "status": status,
        "received_at": received_at or datetime.now(timezone.utc).isoformat(),
        "device": "athena-nano-a",
        "frame": _frame() if frame is None else frame,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- defaults ----------------------------------------------------------------


def test_cabinet_disabled_by_default_generically() -> None:
    """Conservative generic default (like lab/perception) -- the file read
    only works where the sensor-file mount actually exists, which is
    orion-hub specifically, wired via hub_settings_to_runtime_namespace."""
    assert settings_from_runtime(SimpleNamespace()).cabinet_enabled is False


def test_hub_adapter_enables_cabinet_by_default() -> None:
    cfg = settings_from_runtime(situation_mod.hub_settings_to_runtime_namespace(SimpleNamespace()))
    assert cfg.cabinet_enabled is True
    assert cfg.cabinet_ttl_seconds == 30
    assert cfg.cabinet_sensors_path == "/run/orion-sensors/latest.json"


def test_hub_adapter_reuses_existing_cabinet_sensor_path_keys() -> None:
    """No new sensor-path keys -- reuses Hub's EXISTING CABINET_SENSORS_PATH/
    CABINET_SENSORS_B_PATH/CABINET_SENSORS_STALE_AFTER_SEC (already wired for
    cabinet_sensors_routes.py's own routes)."""
    hub_settings = SimpleNamespace(
        CABINET_SENSORS_PATH="/run/orion-sensors/latest.json",
        CABINET_SENSORS_B_PATH="/run/orion-sensors/latest_b.json",
        CABINET_SENSORS_STALE_AFTER_SEC=15.0,
        ORION_SITUATION_CABINET_ENABLED=False,
        ORION_SITUATION_CABINET_TTL_SECONDS=90,
    )
    cfg = settings_from_runtime(situation_mod.hub_settings_to_runtime_namespace(hub_settings))
    assert cfg.cabinet_enabled is False
    assert cfg.cabinet_ttl_seconds == 90
    assert cfg.cabinet_sensors_path == "/run/orion-sensors/latest.json"
    assert cfg.cabinet_sensors_b_path == "/run/orion-sensors/latest_b.json"
    assert cfg.cabinet_stale_after_sec == 15.0


# --- provider states -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cabinet_disabled_yields_unavailable_not_an_error() -> None:
    diag = _diag()
    ctx = await _build_cabinet_context(_cfg(cabinet_enabled=False), diag)
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["cabinet"] == "disabled"


@pytest.mark.asyncio
async def test_cabinet_no_sensors_path_is_unconfigured_not_an_error() -> None:
    """cortex-exec has no /run/orion-sensors bind mount today (only Hub does)
    -- this is a real, distinct, non-error state."""
    diag = _diag()
    ctx = await _build_cabinet_context(
        _cfg(cabinet_enabled=True, cabinet_sensors_path=""), diag
    )
    assert ctx.available is False
    assert ctx.source == "unconfigured"


def test_fetch_missing_snapshot_file_is_unavailable(tmp_path: Path) -> None:
    cfg = _cfg(
        cabinet_enabled=True,
        cabinet_sensors_path=str(tmp_path / "does_not_exist.json"),
        cabinet_stale_after_sec=10.0,
    )
    ctx = _fetch_cabinet_context(cfg)
    assert ctx.available is False
    assert ctx.source == "unavailable"


def test_fetch_stale_snapshot_is_stale_not_available(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    old = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    _write_snapshot(path, received_at=old)
    cfg = _cfg(cabinet_enabled=True, cabinet_sensors_path=str(path), cabinet_stale_after_sec=10.0)
    ctx = _fetch_cabinet_context(cfg)
    assert ctx.available is False
    assert ctx.source == "stale"
    assert ctx.age_seconds is not None and ctx.age_seconds >= 100


def test_fetch_empty_frame_is_empty_not_available(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(
        path,
        frame={"schema": "orion.sensor_frame.v1", "seq": 1, "uptime_ms": 1, "environment": {}},
    )
    cfg = _cfg(cabinet_enabled=True, cabinet_sensors_path=str(path), cabinet_stale_after_sec=10.0)
    ctx = _fetch_cabinet_context(cfg)
    assert ctx.available is False
    assert ctx.source == "empty"


def test_fetch_live_snapshot_populates_measurements_and_pressures(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path)
    cfg = _cfg(cabinet_enabled=True, cabinet_sensors_path=str(path), cabinet_stale_after_sec=10.0)
    ctx = _fetch_cabinet_context(cfg)
    assert ctx.available is True
    assert ctx.source == "cabinet_sensors"
    assert ctx.device == "athena-nano-a"
    assert ctx.temp_c == 24.6
    assert ctx.humidity_pct == 45.0
    assert ctx.gas_resistance_ohm == 12000.0
    assert ctx.magnetic_ut == 53.0
    assert ctx.lidar_mm == 438.0
    # Vibration derived from |accel| - 1g -- accel is exactly 1g here (still
    # air), so this should read ~0, not None (a real measurement was taken).
    assert ctx.vibration_g is not None
    assert ctx.vibration_g == pytest.approx(0.0, abs=1e-6)
    # Pressures ARE computed on the very first read (EwmaBand/InductionTracker
    # do not require history to return a value -- see
    # orion.telemetry.cabinet_sensors's HAND-VERIFIED REST POINT note).
    assert ctx.climate_activity is not None
    assert ctx.em_activity is not None
    assert ctx.vibration_activity is not None
    assert ctx.proximity_activity is not None


@pytest.mark.asyncio
async def test_build_cabinet_context_fail_open_on_read_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    diag = _diag()

    def _boom(cfg):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(situation_mod, "_fetch_cabinet_context", _boom)
    ctx = await _build_cabinet_context(_cfg(cabinet_enabled=True, cabinet_sensors_path="/x"), diag)
    assert ctx.available is False
    assert ctx.source == "error"
    assert "disk gone" in diag.provider_errors["cabinet"]


@pytest.mark.asyncio
async def test_build_cabinet_context_caches_within_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path)
    cfg = _cfg(cabinet_enabled=True, cabinet_sensors_path=str(path), cabinet_ttl_seconds=300)
    diag = _diag()

    calls = {"n": 0}
    real_fetch = situation_mod._fetch_cabinet_context

    def _counting_fetch(c):
        calls["n"] += 1
        return real_fetch(c)

    monkeypatch.setattr(situation_mod, "_fetch_cabinet_context", _counting_fetch)
    first = await _build_cabinet_context(cfg, diag)
    second = await _build_cabinet_context(cfg, diag)
    assert first.available is True
    assert second is first
    assert calls["n"] == 1


# --- prompt rendering ----------------------------------------------------------


def _brief(cabinet: CabinetContextV1) -> SituationBriefV1:
    """Same technique test_situation_curiosity_reverie_context.py's own
    _brief() uses: build via the production helpers, swap in the
    sub-context under test."""
    cfg = _cfg(cabinet_enabled=False)
    diag = _diag()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    now = datetime.now(timezone.utc)
    return SituationBriefV1(
        generated_at=now,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, now)),
        place=situation_mod._build_place_context(cfg),
        cabinet=cabinet,
    )


def test_available_cabinet_renders_measurements() -> None:
    brief = _brief(
        CabinetContextV1(
            available=True,
            source="cabinet_sensors",
            age_seconds=5.0,
            temp_c=24.6,
            humidity_pct=45.0,
            gas_resistance_ohm=12000.0,
            magnetic_ut=53.0,
            vibration_g=0.001,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Your cabinet sensors" in text
    assert "temp=24.6C" in text
    assert "humidity=45%" in text


def test_unavailable_cabinet_renders_no_line_at_all() -> None:
    """Same omission behavior as curiosity/reverie (not weather/lab/
    perception's always-on placeholder) -- see _build_prompt_fragment's own
    comment for why: unlike perception/affect, an absent cabinet read
    carries no confabulation risk, and an always-on line here already
    reproduced the exact 2026-08-26 budget regression this file has on
    record (caught live while writing this feature)."""
    text = _build_prompt_fragment(_brief(CabinetContextV1()), 4000).compact_text
    assert "cabinet" not in text.lower()


def test_notable_activity_is_called_out() -> None:
    brief = _brief(
        CabinetContextV1(
            available=True,
            source="cabinet_sensors",
            temp_c=24.6,
            vibration_g=0.05,
            vibration_activity=0.95,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Notably: elevated vibration" in text


def test_sub_threshold_activity_is_not_called_out() -> None:
    """Continuous EWMA volatility reads a little above zero constantly --
    only a genuinely elevated reading should earn a sentence."""
    brief = _brief(
        CabinetContextV1(
            available=True,
            source="cabinet_sensors",
            temp_c=24.6,
            vibration_activity=0.1,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Notably:" not in text
