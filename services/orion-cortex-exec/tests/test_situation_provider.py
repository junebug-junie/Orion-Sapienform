from __future__ import annotations

from types import SimpleNamespace

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
