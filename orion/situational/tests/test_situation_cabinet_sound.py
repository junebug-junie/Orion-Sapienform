"""2026-10-09: cabinet mic loudness in the situation brief.

Orion told Juniper their cabinet was "quiet as a basement" while the server
fans roar. The cabinet USB mic was already captured (host reader ->
/run/orion-audio/latest.json -> biometrics -> orion_biometrics_summary) but
the chat situation never saw it. These tests pin: the live level becomes
dBFS, it is compared against the last 24 h of stored readings, it survives
a stale Nano frame, it degrades honestly, and the prompt line carries a
scale so the number can't be read as "quiet".
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.schemas.situation import CabinetContextV1
from orion.situational import cabinet_sound_reader
from orion.situational import context as situation_mod
from orion.situational.cabinet_sound_reader import CabinetSoundHistory, biometrics_summary_cutoff
from orion.situational.tests.test_situation_cabinet_context import _brief
from orion.situational.context import (
    _build_prompt_fragment,
    _fetch_cabinet_context,
    hub_settings_to_runtime_namespace,
    settings_from_runtime,
)
from orion.telemetry.ambient_audio import DBFS_FLOOR, pcm16_to_dbfs


@pytest.fixture(autouse=True)
def _clear_cabinet_cache():
    situation_mod._CABINET_CACHE.clear()
    yield
    situation_mod._CABINET_CACHE.clear()


def _write_mic(path: Path, *, rms: float = 5092.7, peak: int = 16855, status: str = "ok",
               received_at: str | None = None) -> None:
    path.write_text(json.dumps({
        "schema": "orion.ambient_audio.v1",
        "status": status,
        "received_at": received_at or datetime.now(timezone.utc).isoformat(),
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
        "sample_rate": 16000,
        "channels": 1,
        "rms": rms,
        "peak": peak,
    }), encoding="utf-8")


def _cfg(mic_path: Path, **overrides):
    cfg = settings_from_runtime(SimpleNamespace())
    cfg.cabinet_enabled = True
    cfg.cabinet_sensors_path = str(mic_path.parent / "no-nano.json")  # Nano missing
    cfg.ambient_audio_path = str(mic_path)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# Real 24 h band measured live 2026-10-09 (p10/p50/p90 of cabinet_ambient_rms).
_HISTORY = CabinetSoundHistory(rows=2767, p10_rms=4828.1, median_rms=7734.4,
                               p90_rms=8879.1, recent_median_rms=5030.5)


# --- dBFS -----------------------------------------------------------------


def test_dbfs_conversion_anchors() -> None:
    assert pcm16_to_dbfs(32768) == 0.0
    assert pcm16_to_dbfs(16384) == pytest.approx(-6.02, abs=0.01)
    assert pcm16_to_dbfs(0) == DBFS_FLOOR == pytest.approx(-90.3, abs=0.01)
    assert pcm16_to_dbfs(5092.7) == pytest.approx(-16.17, abs=0.01)


# --- fetch ----------------------------------------------------------------


def test_fresh_mic_survives_missing_nano(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mic = tmp_path / "latest.json"
    _write_mic(mic)
    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", lambda **_: _HISTORY)
    ctx = _fetch_cabinet_context(_cfg(mic))
    assert ctx.available is False  # Nano side
    assert ctx.sound_available is True
    assert ctx.sound_dbfs == pytest.approx(-16.2)
    assert ctx.sound_peak_dbfs == pytest.approx(-5.8)
    assert ctx.sound_usual_dbfs == pytest.approx(-12.5)
    assert ctx.sound_recent_dbfs == pytest.approx(-16.3)
    assert ctx.sound_vs_usual == "usual"


@pytest.mark.parametrize("recent,expected", [(4000.0, "quieter"), (6000.0, "usual"), (9500.0, "louder")])
def test_vs_usual_judges_the_10min_median_not_one_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recent: float, expected: str
) -> None:
    mic = tmp_path / "latest.json"
    _write_mic(mic, rms=30000.0)  # one loud clank in the live window
    history = _HISTORY._replace(recent_median_rms=recent)
    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", lambda **_: history)
    assert _fetch_cabinet_context(_cfg(mic)).sound_vs_usual == expected


@pytest.mark.parametrize("rms,expected", [(4000.0, "quieter"), (6000.0, "usual"), (9500.0, "louder")])
def test_vs_usual_falls_back_to_live_level_without_recent_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rms: float, expected: str
) -> None:
    mic = tmp_path / "latest.json"
    _write_mic(mic, rms=rms)
    history = _HISTORY._replace(recent_median_rms=None)
    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", lambda **_: history)
    assert _fetch_cabinet_context(_cfg(mic)).sound_vs_usual == expected


def test_mic_failure_does_not_wipe_nano_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mic = tmp_path / "latest.json"
    _write_mic(mic)
    nano = CabinetContextV1(available=True, source="cabinet_sensors", temp_c=28.7)
    monkeypatch.setattr(situation_mod, "_fetch_cabinet_sensor_context", lambda cfg: nano)

    def boom(**_):
        raise RuntimeError("bad DSN")

    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", boom)
    ctx = _fetch_cabinet_context(_cfg(mic))
    assert ctx.available is True and ctx.temp_c == 28.7
    assert ctx.sound_available is False


def test_brief_cache_hit_expires_once_mic_reading_is_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "latest.json")
    brief = _brief(CabinetContextV1(sound_available=True, sound_age_seconds=1.0, sound_dbfs=-16.0))
    gate = situation_mod._cached_percept_outlived_gate
    assert gate(brief, 2.0, cfg) is False
    assert gate(brief, cfg.ambient_audio_stale_after_sec + 1.0, cfg) is True
    assert gate(_brief(CabinetContextV1()), 1000.0, cfg) is False


@pytest.mark.parametrize("status,age", [("error", 0), ("ok", 60)])
def test_stale_or_errored_mic_reports_no_sound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               status: str, age: int) -> None:
    mic = tmp_path / "latest.json"
    received = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - age, timezone.utc)
    _write_mic(mic, status=status, received_at=received.isoformat())
    called = []
    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", lambda **_: called.append(1))
    ctx = _fetch_cabinet_context(_cfg(mic))
    assert ctx.sound_available is False and ctx.sound_dbfs is None
    assert called == []  # no DB read for a reading we won't report


def test_no_history_keeps_live_level_without_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mic = tmp_path / "latest.json"
    _write_mic(mic)
    monkeypatch.setattr(situation_mod, "fetch_cabinet_sound_history", lambda **_: None)
    ctx = _fetch_cabinet_context(_cfg(mic))
    assert ctx.sound_available is True and ctx.sound_dbfs is not None
    assert ctx.sound_usual_dbfs is None and ctx.sound_vs_usual is None


def test_mic_off_when_no_path_configured() -> None:
    assert settings_from_runtime(SimpleNamespace()).ambient_audio_path == ""


def test_hub_adapter_reuses_existing_ambient_audio_keys() -> None:
    ns = hub_settings_to_runtime_namespace(SimpleNamespace(
        AMBIENT_AUDIO_PATH="/x/latest.json",
        AMBIENT_AUDIO_STALE_AFTER_SEC=7.0,
        CABINET_AMBIENT_HISTORY_NODE="athena",
    ))
    cfg = settings_from_runtime(ns)
    assert cfg.ambient_audio_path == "/x/latest.json"
    assert cfg.ambient_audio_stale_after_sec == 7.0
    assert cfg.ambient_audio_history_node == "athena"


# --- render ---------------------------------------------------------------


def test_sound_line_states_level_band_and_scale() -> None:
    text = _build_prompt_fragment(_brief(CabinetContextV1(
        sound_available=True, sound_age_seconds=1.0, sound_dbfs=-10.2, sound_peak_dbfs=-4.0,
        sound_recent_dbfs=-11.0, sound_usual_low_dbfs=-16.6, sound_usual_dbfs=-12.6,
        sound_usual_high_dbfs=-11.3, sound_vs_usual="louder",
    )), 7200).compact_text
    assert "Your cabinet's sound (mic" in text
    assert "-10 dBFS, louder than usual (last 24h ranged -17 to -11, median -13; last 10 min -11)" in text
    assert "not calibrated" in text and "own range" in text
    assert "-50" not in text  # no unverified absolute anchors
    assert "Your cabinet sensors" not in text  # Nano unavailable, still no Nano line


def test_sound_line_without_history_has_no_band() -> None:
    text = _build_prompt_fragment(_brief(CabinetContextV1(
        sound_available=True, sound_age_seconds=1.0, sound_dbfs=-16.0,
    )), 7200).compact_text
    assert "-16 dBFS." in text and "usual" not in text


def test_no_sound_line_when_mic_unavailable() -> None:
    text = _build_prompt_fragment(_brief(CabinetContextV1()), 7200).compact_text
    assert "cabinet's sound" not in text


# --- history reader -------------------------------------------------------


def test_cutoff_matches_stored_varchar_format() -> None:
    dt = datetime(2026, 10, 9, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert biometrics_summary_cutoff(dt) == "2026-10-09 03:04:05.000006+00"


class _FakeEngine:
    def __init__(self, row):
        self.row = row

    @contextmanager
    def connect(self):
        row = self.row
        yield SimpleNamespace(execute=lambda *a, **k: SimpleNamespace(one=lambda: row))


def test_history_none_without_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cabinet_sound_reader, "_get_engine", lambda: None)
    assert cabinet_sound_reader.fetch_cabinet_sound_history() is None


def test_history_none_when_too_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cabinet_sound_reader, "_get_engine",
                        lambda: _FakeEngine((10, 1.0, 2.0, 3.0, 2.0)))
    assert cabinet_sound_reader.fetch_cabinet_sound_history() is None


def test_history_parses_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cabinet_sound_reader, "_get_engine",
                        lambda: _FakeEngine((2767, 4828.1, 7734.4, 8879.1, None)))
    h = cabinet_sound_reader.fetch_cabinet_sound_history()
    assert h == CabinetSoundHistory(2767, 4828.1, 7734.4, 8879.1, None)


def test_history_fails_open_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def connect(self):
            raise RuntimeError("db down")
    monkeypatch.setattr(cabinet_sound_reader, "_get_engine", lambda: Boom())
    assert cabinet_sound_reader.fetch_cabinet_sound_history() is None
