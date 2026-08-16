from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import app.situation as situation
from app.situation import build_situation_for_ctx


def _settings(**overrides):
    base = {
        "orion_situation_enabled": True,
        "orion_situation_ttl_seconds": 300,
        "orion_situation_prompt_max_chars": 400,
        "orion_situation_timezone": "America/Denver",
        "orion_situation_location_label": "Utah",
        "orion_situation_locality": "Vernal",
        "orion_situation_region": "Utah",
        "orion_situation_country": "US",
        "orion_situation_location_precision": "city",
        "orion_situation_weather_enabled": False,
        "orion_situation_weather_provider": "stub",
        "orion_situation_weather_lat": None,
        "orion_situation_weather_lon": None,
        "orion_situation_weather_ttl_seconds": 600,
        "orion_situation_umbrella_precip_prob_threshold": 40,
        "orion_situation_jacket_temp_f_threshold": 55,
        "orion_situation_high_wind_mph_threshold": 25,
        "orion_situation_hot_car_temp_f_threshold": 80,
        "orion_situation_agenda_enabled": False,
        "orion_situation_lab_context_enabled": True,
        "orion_situation_lab_provider": "stub",
        # False here (unlike the True production default) for the same
        # reason weather_enabled is False above: avoid a real network call
        # to orion-llm-gateway from every unrelated situation test.
        "orion_situation_runtime_enabled": False,
        "orion_situation_runtime_route": "chat",
        "orion_situation_runtime_ttl_seconds": 120,
        "orion_situation_runtime_probe_timeout_sec": 2.0,
        "cortex_exec_llm_gateway_url": "http://llm-gateway:8210",
        "orion_presence_default_requestor": "Juniper",
        "orion_presence_persist_allowed": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_situation_marks_temporal_resume_for_long_gap():
    ctx = {"session_id": "sid-temporal", "raw_user_text": "yeah do that"}
    settings = _settings()
    build_situation_for_ctx(ctx, settings)
    brief, fragment = build_situation_for_ctx(ctx, settings)
    assert brief["kind"] == "situation.brief.v1"
    assert "conversation_phase" in brief
    assert fragment["kind"] == "situation.prompt_fragment.v1"


def test_situation_presence_child_affordance():
    from app import situation as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    ctx = {
        "session_id": "sid-kid",
        "raw_user_text": "can you explain this for my kid",
        "presence_context": {
            "audience_mode": "kid_present",
            "companions": [{"display_name": "Kid", "relationship": "child", "role": "asker", "age_band": "child"}],
            "requestor": {"display_name": "Juniper"},
        },
    }
    brief, fragment = build_situation_for_ctx(ctx, _settings())
    kinds = {item["kind"] for item in brief["affordances"]}
    assert brief["presence"]["audience_mode"] == "kid_present"
    assert "kid_friendly_explanation" in kinds
    assert "kid_present" in fragment["compact_text"]


def test_situation_cache_refreshes_when_presence_changes():
    from app import situation as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    session = "sid-cache-presence"
    brief_solo, _ = build_situation_for_ctx(
        {"session_id": session, "presence_context": {"audience_mode": "solo"}},
        _settings(orion_situation_ttl_seconds=300),
    )
    brief_kid, fragment_kid = build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "kid_present",
                "companions": [{"display_name": "Kid", "relationship": "child", "role": "listener", "age_band": "child"}],
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    assert brief_solo["presence"]["audience_mode"] == "solo"
    assert brief_kid["presence"]["audience_mode"] == "kid_present"
    assert "kid_present" in fragment_kid["compact_text"]


def test_situation_cache_refreshes_when_requestor_changes():
    from app import situation as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    session = "sid-cache-requestor"
    build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "solo",
                "requestor": {"display_name": "Juniper"},
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    brief_guest, _ = build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "solo",
                "requestor": {"display_name": "Guest"},
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    assert brief_guest["presence"]["requestor"]["display_name"] == "Guest"


def test_situation_outdoor_departure_affordance():
    ctx = {"session_id": "sid-outdoor", "raw_user_text": "I am heading out the door soon"}
    brief, fragment = build_situation_for_ctx(ctx, _settings())
    kinds = {item["kind"] for item in brief["affordances"]}
    assert "outdoor_departure" in kinds
    assert len(fragment["compact_text"]) <= 400


def test_situation_disabled_returns_no_brief_or_fragment():
    ctx = {"session_id": "sid-disabled", "raw_user_text": "hello"}
    brief, fragment = build_situation_for_ctx(ctx, _settings(orion_situation_enabled=False))
    assert brief == {}
    assert fragment == {}


class _FakeUrlopenResponse:
    """Mimics the `with urlopen(...) as resp: resp.read()` shape `_fetch_weather`
    and `_fetch_runtime_context` both use."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    # _RUNTIME_CACHE keys only by route name ("chat" in every test below),
    # not by session -- unlike _SITUATION_CACHE/_WEATHER_CACHE it would
    # otherwise leak a mocked result across unrelated test cases.
    situation._RUNTIME_CACHE.clear()
    yield
    situation._RUNTIME_CACHE.clear()


def test_runtime_context_reports_live_model_when_route_is_up(monkeypatch):
    routes_payload = {
        "default_route": "chat",
        "routes": [
            {
                "id": "chat",
                "served_by": "circe-worker-1",
                "backend": "llamacpp",
                "status": "up",
                "latency_ms": 12,
                "last_checked_at": "2026-08-14T00:00:00+00:00",
                "model": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
            }
        ],
    }
    monkeypatch.setattr(
        situation, "urlopen", lambda url, timeout=None: _FakeUrlopenResponse(routes_payload)
    )
    ctx = {"session_id": "sid-runtime-up", "raw_user_text": "hello"}
    brief, fragment = build_situation_for_ctx(ctx, _settings(orion_situation_runtime_enabled=True))
    assert brief["runtime"]["available"] is True
    assert brief["runtime"]["model_id"] == "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    assert brief["runtime"]["served_by"] == "circe-worker-1"
    assert "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" in fragment["compact_text"]


def test_runtime_context_degrades_when_gateway_unreachable(monkeypatch):
    def _raise(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(situation, "urlopen", _raise)
    ctx = {"session_id": "sid-runtime-down", "raw_user_text": "hello"}
    brief, fragment = build_situation_for_ctx(ctx, _settings(orion_situation_runtime_enabled=True))
    assert brief["runtime"]["available"] is False
    assert brief["runtime"]["model_id"] is None
    assert brief["diagnostics"]["provider_status"]["runtime"] == "error"
    assert "unavailable" in fragment["compact_text"].lower()
    assert "Qwen" not in fragment["compact_text"]


def test_runtime_context_degrades_when_route_missing_from_response(monkeypatch):
    monkeypatch.setattr(
        situation,
        "urlopen",
        lambda url, timeout=None: _FakeUrlopenResponse({"default_route": "chat", "routes": []}),
    )
    ctx = {"session_id": "sid-runtime-missing-route", "raw_user_text": "hello"}
    brief, _ = build_situation_for_ctx(ctx, _settings(orion_situation_runtime_enabled=True))
    assert brief["runtime"]["available"] is False
    assert brief["diagnostics"]["provider_status"]["runtime"] == "error"


def test_runtime_context_disabled_by_default_in_shared_fixture(monkeypatch):
    # Sanity check on the shared _settings() default itself: it must be
    # False, or every unrelated situation test would attempt a real network
    # call to orion-llm-gateway.
    called = {"n": 0}

    def _raise(url, timeout=None):
        called["n"] += 1
        raise AssertionError("urlopen should not be called when runtime context is disabled")

    monkeypatch.setattr(situation, "urlopen", _raise)
    ctx = {"session_id": "sid-runtime-disabled", "raw_user_text": "hello"}
    brief, fragment = build_situation_for_ctx(ctx, _settings())
    assert called["n"] == 0
    assert brief["runtime"]["available"] is False
    assert brief["runtime"]["source"] == "disabled"
    assert "unavailable" in fragment["compact_text"].lower()


def test_runtime_context_caches_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _urlopen(url, timeout=None):
        calls["n"] += 1
        return _FakeUrlopenResponse(
            {
                "routes": [
                    {"id": "chat", "served_by": "circe-worker-1", "backend": "llamacpp", "status": "up", "model": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"}
                ]
            }
        )

    monkeypatch.setattr(situation, "urlopen", _urlopen)
    cfg = situation.settings_from_runtime(_settings(orion_situation_runtime_enabled=True))
    diagnostics = situation.SituationDiagnosticsV1()
    situation._build_runtime_context(cfg, diagnostics)
    situation._build_runtime_context(cfg, diagnostics)
    assert calls["n"] == 1
