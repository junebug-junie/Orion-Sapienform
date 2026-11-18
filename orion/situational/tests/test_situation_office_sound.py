"""2026-11-07: office sound meter (dBA) in the situation brief.

The office Nano's sound meter is a real dBA meter that hangs in Juniper's
office. Before this, Orion's brief only had the cabinet mic (uncalibrated,
"louder/quieter than usual"). These tests pin: the dBA number reaches the
prompt and is named as the office (never the cabinet or the closet), a stale
reading says nothing, the 24 h band judges on the 10-min median, Hub turns it
on with its existing OFFICE_ENV_* keys, and a cached brief expires with the
reading.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.schemas.situation import OfficeSoundContextV1, SituationDiagnosticsV1
from orion.situational import context as situation_mod
from orion.situational import office_sound_reader
from orion.situational.context import (
    _build_prompt_fragment,
    _fetch_office_sound_context,
    hub_settings_to_runtime_namespace,
    settings_from_runtime,
)
from orion.situational.office_sound_reader import OfficeSound
from orion.situational.tests.test_situation_cabinet_context import _brief


def _cfg(**overrides):
    cfg = settings_from_runtime(SimpleNamespace())
    cfg.office_sound_node = "pan-1"
    cfg.office_sound_stale_after_sec = 90.0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _reading(*, age_sec: float = 10.0, dba: float = 46.2, recent: float | None = 45.0,
             band: tuple[float, float, float] | None = (38.0, 44.0, 52.0)) -> OfficeSound:
    p10, p50, p90 = band if band else (None, None, None)
    return OfficeSound(
        dba=dba,
        received_at=datetime.now(timezone.utc) - timedelta(seconds=age_sec),
        p10_dba=p10, median_dba=p50, p90_dba=p90, recent_median_dba=recent,
    )


def _fragment(office: OfficeSoundContextV1) -> str:
    brief = _brief(situation_mod.CabinetContextV1()).model_copy(update={"office_sound": office})
    return _build_prompt_fragment(brief, 7200).compact_text


def test_fresh_reading_with_band_reaches_prompt_as_office(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_office_sound", lambda **_: _reading())
    ctx = _fetch_office_sound_context(_cfg())
    assert ctx.available and ctx.dba == 46.2 and ctx.vs_usual == "usual"
    text = _fragment(ctx)
    line = next(l for l in text.splitlines() if "office's sound" in l)
    assert "46 dBA" in line
    assert "not in your cabinet or the closet" in line
    assert "about as loud as usual for the office" in line
    assert "38-52 dBA" in line


def test_band_judges_on_recent_median_not_one_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    # One loud reading (a door slam) inside a quiet 10 minutes stays "quieter".
    monkeypatch.setattr(situation_mod, "fetch_office_sound",
                        lambda **_: _reading(dba=80.0, recent=35.0))
    assert _fetch_office_sound_context(_cfg()).vs_usual == "quieter"
    monkeypatch.setattr(situation_mod, "fetch_office_sound",
                        lambda **_: _reading(dba=40.0, recent=70.0))
    assert _fetch_office_sound_context(_cfg()).vs_usual == "louder"


def test_thin_history_still_reports_level_without_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_office_sound", lambda **_: _reading(band=None))
    ctx = _fetch_office_sound_context(_cfg())
    assert ctx.available and ctx.vs_usual is None and ctx.usual_dba is None
    line = next(l for l in _fragment(ctx).splitlines() if "office's sound" in l)
    assert "46 dBA" in line and "usual" not in line


def test_stale_or_missing_reading_says_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_office_sound", lambda **_: _reading(age_sec=300))
    ctx = _fetch_office_sound_context(_cfg())
    assert not ctx.available
    monkeypatch.setattr(situation_mod, "fetch_office_sound", lambda **_: None)
    assert not _fetch_office_sound_context(_cfg()).available
    assert "office's sound" not in _fragment(OfficeSoundContextV1())


def test_unset_node_is_disabled_and_reader_never_called(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(situation_mod, "fetch_office_sound", lambda **_: called.append(1))
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_office_sound_context(_cfg(office_sound_node=""), diag))
    assert not ctx.available and not called
    assert diag.provider_status["office_sound"] == "disabled"


def test_reader_error_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_):
        raise RuntimeError("db down")
    monkeypatch.setattr(situation_mod, "fetch_office_sound", boom)
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_office_sound_context(_cfg(), diag))
    assert not ctx.available and diag.provider_status["office_sound"] == "error"


def test_generic_default_off_hub_turns_it_on_with_existing_keys() -> None:
    assert settings_from_runtime(SimpleNamespace()).office_sound_node == ""
    hub = SimpleNamespace(OFFICE_ENV_NODE="pan-1", OFFICE_ENV_STALE_AFTER_SEC=90.0)
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub))
    assert cfg.office_sound_node == "pan-1"
    assert cfg.office_sound_stale_after_sec == 90.0


def test_cached_brief_expires_once_office_reading_is_stale() -> None:
    cfg = _cfg()
    brief = _brief(situation_mod.CabinetContextV1()).model_copy(
        update={"office_sound": OfficeSoundContextV1(available=True, age_seconds=60.0, dba=45.0)}
    )
    gate = situation_mod._cached_percept_outlived_gate
    assert gate(brief, 20.0, cfg) is False
    assert gate(brief, 40.0, cfg) is True


def test_reader_without_dsn_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert office_sound_reader.fetch_office_sound(node="pan-1") is None
