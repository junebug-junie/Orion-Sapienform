from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.main import (
    _rpc_request_with_retry,
    _daily_metacog_dedupe_key,
    _daily_pulse_dedupe_key,
    _clamp_daily_metacog_payload,
    _extract_daily_llm_diagnostics,
    _extract_plan_final_text,
    _is_likely_incomplete_json,
    _json_loads_strict,
    _log_daily_json_parse_failure,
    _normalize_daily_skill_selection,
    _should_retry_for_truncated_generation,
    build_daily_window,
    should_run_daily,
)
from orion.cognition.skills_manifest import load_skill_manifest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
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


def test_daily_schema_accepts_skill_selection_fields():
    pulse = DailyPulseV1.model_validate(
        {
            "date": "2026-02-14",
            "timezone": "America/Denver",
            "yesterday_theme": "Theme",
            "today_focus": "Focus",
            "focus_skill_id": "skills.system.time_now.v1",
            "gentle_challenge": "Challenge",
            "confidence": 0.7,
        }
    )
    assert pulse.focus_skill_id == "skills.system.time_now.v1"

    metacog = DailyMetacogV1.model_validate(
        {
            "date": "2026-02-14",
            "timezone": "America/Denver",
            "thinking_patterns": ["pattern"],
            "blindspots": [],
            "course_correction": "tighten retrieval grounding",
            "tomorrow_experiment": "run one bounded check",
            "tomorrow_experiment_skill_id": "skills.system.time_now.v1",
            "confidence": 0.6,
        }
    )
    assert metacog.tomorrow_experiment_skill_id == "skills.system.time_now.v1"


def test_normalize_daily_skill_selection_rejects_unknown_and_mutating_ids():
    unknown_pulse, unknown_invalid = _normalize_daily_skill_selection(
        {"focus_skill_id": "skills.system.fake.v1"},
        action_name="daily_pulse_v1",
    )
    assert unknown_invalid == ["focus_skill_id"]
    assert unknown_pulse["focus_skill_id"] is None

    mutating_pulse, mutating_invalid = _normalize_daily_skill_selection(
        {"focus_skill_id": "skills.runtime.docker_prune_stopped_containers.v1"},
        action_name="daily_pulse_v1",
    )
    assert mutating_invalid == ["focus_skill_id"]
    assert mutating_pulse["focus_skill_id"] is None

    valid_metacog, valid_invalid = _normalize_daily_skill_selection(
        {"tomorrow_experiment_skill_id": "skills.system.time_now.v1"},
        action_name="daily_metacog_v1",
    )
    assert valid_invalid == []
    assert valid_metacog["tomorrow_experiment_skill_id"] == "skills.system.time_now.v1"


def test_shared_skill_manifest_loads_entries():
    entries = load_skill_manifest()
    ids = {item.skill_id for item in entries}
    assert "skills.system.time_now.v1" in ids
    assert "skills.runtime.docker_prune_stopped_containers.v1" in ids


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


def test_clamp_daily_metacog_payload_truncates_over_limit_fields():
    parsed = {
        "date": "2026-02-14",
        "timezone": "America/Denver",
        "thinking_patterns": ["pattern"],
        "blindspots": [],
        "course_correction": "c" * 350,
        "tomorrow_experiment": "t" * 260,
        "confidence": 0.6,
    }
    clamped = _clamp_daily_metacog_payload(parsed, action_name="daily_metacog_v1")
    assert len(clamped["course_correction"]) == 300
    assert len(clamped["tomorrow_experiment"]) == 240
    DailyMetacogV1.model_validate(clamped)


def test_json_loads_strict_accepts_wrapped_json():
    wrapped = """Here you go:
```json
{
  "date": "2026-02-14",
  "timezone": "America/Denver",
  "thinking_patterns": ["pattern"],
  "blindspots": [],
  "course_correction": "tighten retrieval grounding",
  "tomorrow_experiment": "ask one clarifying question first",
  "confidence": 0.6,
}
```
Thanks."""
    parsed = _json_loads_strict(wrapped)
    model = DailyMetacogV1.model_validate(parsed)
    assert model.timezone == "America/Denver"


def test_extract_daily_llm_diagnostics_reads_finish_reason():
    payload = {
        "result": {
            "steps": [
                {
                    "result": {
                        "LLMGatewayService": {
                            "model_used": "llama3.1",
                            "usage": {"total_tokens": 700},
                            "raw": {
                                "choices": [{"finish_reason": "length"}],
                            },
                        }
                    }
                }
            ]
        }
    }
    diag = _extract_daily_llm_diagnostics(payload)
    assert diag["model"] == "llama3.1"
    assert diag["finish_reason"] == "length"
    assert diag["usage"]["total_tokens"] == 700


def test_retry_classification_for_truncated_json():
    partial = """{
      "date": "2026-03-28",
      "timezone": "America/Denver",
      "thinking_patterns": ["a"]
    """
    assert _is_likely_incomplete_json(partial) is True
    assert _should_retry_for_truncated_generation(text=partial, diagnostics={}) is True
    assert _should_retry_for_truncated_generation(
        text="{\"ok\": true}",
        diagnostics={"finish_reason": "length"},
    ) is True


def test_log_daily_json_parse_failure_includes_tail_and_finish_reason(caplog: pytest.LogCaptureFixture):
    caplog.set_level("ERROR")
    text = "{\"date\":\"2026-03-28\",\"timezone\":\"America/Denver\",\"thinking_patterns\":[\"x\"]"
    _log_daily_json_parse_failure(
        action_name="daily_metacog_v1",
        final_text=text,
        diagnostics={"model": "llama3", "finish_reason": "length", "stop_reason": None},
        timeout_sec=240.0,
        requested_max_tokens=1024,
        parse_error=ValueError("missing closing brace"),
    )
    msg = caplog.records[-1].getMessage()
    assert "finish_reason=length" in msg
    assert "tail='" in msg


def test_rpc_request_with_retry_retries_once_after_timeout():
    class FakeBus:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):  # noqa: ANN001
            self.calls.append(
                {
                    "request_channel": request_channel,
                    "reply_channel": reply_channel,
                    "attempt": envelope.payload["attempt"],
                    "timeout_sec": timeout_sec,
                }
            )
            if len(self.calls) == 1:
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}")
            return {"data": b"ok"}

    bus = FakeBus()
    source = ServiceRef(name="tests", version="0.1.0", node="athena")

    result = asyncio.run(
        _rpc_request_with_retry(
            bus=bus,
            request_channel="orion:cortex:exec:request",
            reply_prefix="orion:exec:result",
            timeout_sec=12.5,
            operation_name="daily plan daily_pulse_v1",
            envelope_factory=lambda reply_channel, attempt: BaseEnvelope(
                kind="test.request",
                source=source,
                reply_to=reply_channel,
                payload={"attempt": attempt},
            ),
        )
    )

    assert result == {"data": b"ok"}
    assert [call["attempt"] for call in bus.calls] == [1, 2]
    assert all(str(call["reply_channel"]).startswith("orion:exec:result:") for call in bus.calls)
    assert bus.calls[0]["reply_channel"] != bus.calls[1]["reply_channel"]


def test_rpc_request_with_retry_raises_after_last_timeout():
    class FakeBus:
        async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):  # noqa: ANN001
            raise TimeoutError(f"RPC timeout waiting on {reply_channel}")

    source = ServiceRef(name="tests", version="0.1.0", node="athena")

    with pytest.raises(TimeoutError):
        asyncio.run(
            _rpc_request_with_retry(
                bus=FakeBus(),
                request_channel="orion:cortex:exec:request",
                reply_prefix="orion:exec:result",
                timeout_sec=5.0,
                operation_name="daily plan daily_pulse_v1",
                envelope_factory=lambda reply_channel, attempt: BaseEnvelope(
                    kind="test.request",
                    source=source,
                    reply_to=reply_channel,
                    payload={"attempt": attempt},
                ),
            )
        )
