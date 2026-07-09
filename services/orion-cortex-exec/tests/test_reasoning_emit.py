from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT))
for module_name in [name for name in list(sys.modules) if name == "app" or name.startswith("app.")]:
    del sys.modules[module_name]

from app.reasoning_emit import build_reasoning_call, publish_reasoning_call
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.reasoning import ReasoningCallV1


SOURCE = ServiceRef(name="cortex-exec", version="0", node="n")
CHANNEL = "orion:cognition:reasoning_call"
# BaseEnvelope.correlation_id is a UUID (matches the real run_plan correlation_id).
CORR = "11111111-1111-1111-1111-111111111456"

# Fields that could ever hold trace/thinking text — must NOT appear in the payload.
_FORBIDDEN_TEXT_FIELDS = {
    "reasoning_content",
    "reasoning_trace",
    "inline_think_content",
    "thinking_text",
    "trace",
    "reasoning",
}


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        self.published.append((channel, env))


class _RaisingBus:
    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        raise RuntimeError("bus down")


def _base_kwargs(**overrides):
    kwargs = dict(
        correlation_id=CORR,
        verb="chat_general",
        mode="brain",
        node_id="athena",
        turn_id="turn-9",
        thinking_enabled=False,
        diagnostics={},
        completion_tokens=None,
        prompt_tokens=None,
    )
    kwargs.update(overrides)
    return kwargs


# --- build_reasoning_call: reasoning_present mapping ---------------------------


def test_reasoning_present_true_when_provider_has_reasoning_content():
    call = build_reasoning_call(**_base_kwargs(diagnostics={"provider_has_reasoning_content": True}))
    assert call.reasoning_present is True


def test_reasoning_present_true_when_provider_reasoning_available():
    call = build_reasoning_call(**_base_kwargs(diagnostics={"provider_reasoning_available": True}))
    assert call.reasoning_present is True


def test_reasoning_present_true_when_think_tags_detected():
    call = build_reasoning_call(**_base_kwargs(diagnostics={"think_tags_detected": True}))
    assert call.reasoning_present is True


def test_reasoning_present_false_when_no_reasoning_flags():
    call = build_reasoning_call(
        **_base_kwargs(
            diagnostics={
                "provider_has_reasoning_content": False,
                "provider_reasoning_available": None,
                "think_tags_detected": False,
            }
        )
    )
    assert call.reasoning_present is False


# --- build_reasoning_call: trace-present mapping ------------------------------


def test_reasoning_trace_present_maps_from_diagnostics():
    on = build_reasoning_call(**_base_kwargs(diagnostics={"provider_has_reasoning_trace": True}))
    off = build_reasoning_call(**_base_kwargs(diagnostics={"provider_has_reasoning_trace": False}))
    assert on.reasoning_trace_present is True
    assert off.reasoning_trace_present is False


# --- build_reasoning_call: token coercion ------------------------------------


def test_tokens_coerced_from_strings_and_floats():
    call = build_reasoning_call(
        **_base_kwargs(completion_tokens="42", prompt_tokens=17.0, thinking_tokens="8")
    )
    assert call.completion_tokens == 42
    assert call.prompt_tokens == 17
    assert call.thinking_tokens == 8


def test_bad_and_negative_tokens_become_none():
    call = build_reasoning_call(
        **_base_kwargs(completion_tokens="garbage", prompt_tokens=-5, thinking_tokens=None)
    )
    assert call.completion_tokens is None
    assert call.prompt_tokens is None
    assert call.thinking_tokens is None


def test_thinking_tokens_default_none():
    call = build_reasoning_call(**_base_kwargs())
    assert call.thinking_tokens is None


# --- build_reasoning_call: robustness (never raises) --------------------------


def test_never_raises_on_none_diagnostics():
    call = build_reasoning_call(**_base_kwargs(diagnostics=None))
    assert isinstance(call, ReasoningCallV1)
    assert call.reasoning_present is False


def test_never_raises_on_garbage_diagnostics():
    call = build_reasoning_call(**_base_kwargs(diagnostics="not-a-dict", completion_tokens=object()))
    assert isinstance(call, ReasoningCallV1)
    assert call.reasoning_present is False
    assert call.completion_tokens is None


def test_empty_turn_id_becomes_none():
    call = build_reasoning_call(**_base_kwargs(turn_id="   "))
    assert call.turn_id is None


def test_fields_passthrough():
    call = build_reasoning_call(
        **_base_kwargs(verb="reflect", mode="reverie", node_id="orion", thinking_enabled=True)
    )
    assert call.verb == "reflect"
    assert call.mode == "reverie"
    assert call.node_id == "orion"
    assert call.thinking_enabled is True


# --- publish_reasoning_call ---------------------------------------------------


def test_publish_emits_exactly_one_valid_envelope():
    bus = _Bus()
    call = build_reasoning_call(**_base_kwargs(diagnostics={"provider_has_reasoning_content": True}))
    asyncio.run(publish_reasoning_call(bus, source=SOURCE, channel=CHANNEL, call=call))

    assert len(bus.published) == 1
    channel, env = bus.published[0]
    assert channel == CHANNEL
    assert isinstance(env, BaseEnvelope)
    # Payload round-trips into a valid ReasoningCallV1.
    reparsed = ReasoningCallV1.model_validate(env.payload)
    assert reparsed.reasoning_present is True
    assert reparsed.correlation_id == CORR


def test_publish_payload_has_no_trace_text_field():
    bus = _Bus()
    call = build_reasoning_call(**_base_kwargs(diagnostics={"provider_has_reasoning_trace": True}))
    asyncio.run(publish_reasoning_call(bus, source=SOURCE, channel=CHANNEL, call=call))

    _, env = bus.published[0]
    payload = env.payload
    assert isinstance(payload, dict)
    for forbidden in _FORBIDDEN_TEXT_FIELDS:
        assert forbidden not in payload, f"payload leaked trace field: {forbidden}"
    # Trace presence is a bool only.
    assert payload["reasoning_trace_present"] is True
    assert isinstance(payload["reasoning_trace_present"], bool)


def test_publish_swallows_bus_failure():
    call = build_reasoning_call(**_base_kwargs())
    # Must not raise even though bus.publish raises.
    asyncio.run(publish_reasoning_call(_RaisingBus(), source=SOURCE, channel=CHANNEL, call=call))
