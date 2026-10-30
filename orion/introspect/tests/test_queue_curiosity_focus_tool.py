"""The `queue_curiosity_focus` introspect tool: argument contract, transport, registry, brief line."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.introspect.brief import CURIOSITY, QUEUE_CURIOSITY_FOCUS, introspect_brief_lines
from orion.introspect.tests.test_introspect_tools import BINDING, DreamBus
from orion.introspect.tools import RPC_TIMEOUT_SEC, IntrospectTools, IntrospectUnknownError
from orion.introspect.transport import QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL, REQUEST_KIND, RESULT_PREFIX
from orion.schemas.introspect import (
    FOCUS_TEXT_CAP,
    IntrospectRequestV1,
    IntrospectResultV1,
    QueueCuriosityFocusArguments,
)

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
VALID = {"question": "How much of the bus traffic is heartbeats?", "why": "I keep guessing at it"}


def _ok():
    return IntrospectResultV1(ok=True, operation="queue_curiosity_focus", as_of=NOW,
                              total_available=0).model_dump(mode="json")


def _invoke(bus, args):
    return asyncio.run(IntrospectTools(bus, BINDING).invoke("queue_curiosity_focus", args))


@pytest.mark.parametrize("fields", [{}, {"origin": "juniper"}, {"question": "  padded  "}])
def test_valid_focus_args(fields):
    args = QueueCuriosityFocusArguments.model_validate({**VALID, **fields})
    assert args.question == args.question.strip() and args.origin in ("self", "juniper")


@pytest.mark.parametrize("fields", [
    {"question": "   "},
    {"why": ""},
    {"question": "q" * (FOCUS_TEXT_CAP + 1)},
    {"origin": "hub"},
    {"priority": "urgent"},  # extra fields are rejected, never ignored
])
def test_invalid_focus_args_are_rejected(fields):
    with pytest.raises(ValidationError):
        QueueCuriosityFocusArguments.model_validate({**VALID, **fields})


def test_registry_channel_operation_and_skew_writer_know_the_tool():
    from orion.schema_skew_discovery import DECLARED_WRITERS
    from orion.schemas.registry import resolve

    assert resolve("QueueCuriosityFocusArguments") is QueueCuriosityFocusArguments
    assert QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL == "orion:introspect:queue_curiosity_focus:request"
    assert IntrospectRequestV1(operation="queue_curiosity_focus", binding=BINDING, args={}).operation
    assert "orion.schemas.introspect:QueueCuriosityFocusArguments" in DECLARED_WRITERS
    channels = {c["name"]: c for c in yaml.safe_load((REPO / "orion/bus/channels.yaml").read_text())["channels"]}
    req = channels[QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL]
    assert req["schema_id"] == "IntrospectRequestV1" and req["consumer_services"] == ["orion-hub"]
    # orion-mind (2026-10-30): Orion Prime's resident loop queues its own between-conversations question.
    assert req["producer_services"] == ["orion-harness-governor", "orion-mind"]


def test_the_tool_uses_its_channel_with_binding_and_clean_args():
    bus = DreamBus(_ok())
    out = _invoke(bus, {**VALID, "question": f" {VALID['question']} "})
    assert out["ok"] is True and out["operation"] == "queue_curiosity_focus"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL and timeout == RPC_TIMEOUT_SEC
    assert envelope.kind == REQUEST_KIND and reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    request = IntrospectRequestV1.model_validate(envelope.payload)
    assert request.binding == BINDING and request.args == {**VALID, "origin": "self"}


def test_bad_args_never_reach_the_bus():
    bus = DreamBus(_ok())
    with pytest.raises(ValidationError):
        _invoke(bus, {**VALID, "question": ""})
    assert bus.sent == []


def test_a_refusal_says_not_queued_and_no_reply_says_unknown():
    refused = IntrospectResultV1(ok=False, operation="queue_curiosity_focus", as_of=NOW,
                                 error="Juniper asked for this focus; it stays until a run takes it or it expires")
    with pytest.raises(IntrospectUnknownError, match=r"queue_curiosity_focus: not queued \(Juniper asked"):
        _invoke(DreamBus(refused.model_dump(mode="json")), VALID)
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        _invoke(DreamBus(_ok(), raise_exc=TimeoutError()), VALID)
    outage = IntrospectResultV1(ok=False, operation="queue_curiosity_focus", as_of=NOW,
                                error="queue_curiosity_focus_unavailable; answer unknown")
    with pytest.raises(IntrospectUnknownError, match=r"queue_curiosity_focus: answer unknown \("):
        _invoke(DreamBus(outage.model_dump(mode="json")), VALID)


def test_description_is_honest_that_it_starts_nothing_and_is_not_forced():
    spec = next(s for s in IntrospectTools(DreamBus(_ok()), BINDING).tool_specs() if s.name == "queue_curiosity_focus")
    for phrase in ("starts nothing now", "one slot", "cannot replace a focus Juniper asked for",
                   "expires after 48 hours", "decides itself whether to explore it or set it", "focus_id",
                   "Only from a conversation with Juniper", "curiosity(kind=focus)"):
        assert phrase in spec.description, phrase


def test_brief_names_the_tool_and_says_no_id_means_not_queued():
    text = " ".join(introspect_brief_lines(BINDING))
    assert QUEUE_CURIOSITY_FOCUS == "mcp__orion-introspect__queue_curiosity_focus"
    assert f"{QUEUE_CURIOSITY_FOCUS}," in text
    assert "Say it was queued only if the tool returned a focus_id" in text
    assert "do not search for another" in text
    assert f"call {CURIOSITY} with kind=focus" in text
