from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.main import (
    _daily_metacog_dedupe_key,
    _daily_pulse_dedupe_key,
    _extract_plan_final_text,
    _json_loads_strict,
    build_daily_window,
    should_run_daily,
)
from orion.schemas.actions.daily import DailyMetacogV1, DailyPulseV1


def test_daily_dedupe_keys_stable():
    window = build_daily_window(
        now_utc=datetime(2026, 2, 15, 14, 0, tzinfo=timezone.utc),
        tz_name="America/Denver",
        override_date="2026-02-14",
    )
    assert _daily_pulse_dedupe_key(window).startswith("actions:daily_pulse:2026-02-14:")
    assert _daily_metacog_dedupe_key(window).startswith("actions:daily_metacog:2026-02-14:")


def test_should_run_daily_decision():
    now = datetime(2026, 2, 15, 16, 0, tzinfo=timezone.utc)  # 09:00 Denver winter
    should, local_date = should_run_daily(
        now_utc=now,
        tz_name="America/Denver",
        hour_local=8,
        minute_local=30,
        last_ran_date=None,
    )
    assert should is True
    should_again, _ = should_run_daily(
        now_utc=now,
        tz_name="America/Denver",
        hour_local=8,
        minute_local=30,
        last_ran_date=local_date,
    )
    assert should_again is False


def test_extract_plan_final_text_from_steps():
    payload = {
        "result": {
            "steps": [
                {"result": {"LLMGatewayService": {"text": "hello world"}}},
            ]
        }
    }
    assert _extract_plan_final_text(payload) == "hello world"


def test_daily_schema_validation_forbid_extra_keys():
    ok = DailyPulseV1.model_validate(
        {
            "date": "2026-02-14",
            "timezone": "America/Denver",
            "yesterday_theme": "Theme",
            "today_focus": "Focus",
            "gentle_challenge": "Challenge",
            "confidence": 0.7,
        }
    )
    assert ok.confidence == 0.7

    with pytest.raises(Exception):
        DailyPulseV1.model_validate(
            {
                "date": "2026-02-14",
                "timezone": "America/Denver",
                "yesterday_theme": "Theme",
                "today_focus": "Focus",
                "gentle_challenge": "Challenge",
                "confidence": 0.7,
                "unexpected": "nope",
            }
        )


def test_metacog_schema_and_json_parser():
    raw = """{
      \"date\": \"2026-02-14\",
      \"timezone\": \"America/Denver\",
      \"thinking_patterns\": [\"pattern\"],
      \"blindspots\": [],
      \"course_correction\": \"tighten retrieval grounding\",
      \"tomorrow_experiment\": \"ask one clarifying question first\",
      \"confidence\": 0.6
    }"""
    parsed = _json_loads_strict(raw)
    model = DailyMetacogV1.model_validate(parsed)
    assert model.thinking_patterns == ["pattern"]

    with pytest.raises(Exception):
        _json_loads_strict("[]")
