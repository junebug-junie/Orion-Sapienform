from __future__ import annotations

from uuid import uuid4

from app.logic import (
    ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
    ActionDedupe,
    build_llm_envelope,
    build_notify_request,
    extract_message_sections,
    should_trigger,
)

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2


def _entry(*, observer: str = "juniper") -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        observer=observer,
        trigger="t",
        observer_state=["a"],
        type="reflect",
        emergent_entity="x",
        summary="s",
        mantra="m",
    )


def _env() -> BaseEnvelope:
    return BaseEnvelope(
        kind="collapse.mirror.entry",
        source=ServiceRef(name="test"),
        correlation_id=str(uuid4()),
        payload={},
    )


def test_rule_filters_juniper_casefold():
    assert should_trigger(_entry(observer="juniper")) is True
    assert should_trigger(_entry(observer="Juniper")) is True
    assert should_trigger(_entry(observer="JUNIPER")) is True
    assert should_trigger(_entry(observer="orion")) is False


def test_dedupe_blocks_second_processing():
    d = ActionDedupe(ttl_seconds=60)
    key = "collapse_123"
    assert d.try_acquire(key) is True
    # inflight blocks
    assert d.try_acquire(key) is False
    d.mark_done(key)
    # done blocks
    assert d.try_acquire(key) is False


def test_llm_prompt_contains_relevant_memory_marker():
    parent = _env()
    src = ServiceRef(name="orion-actions")
    entry = _entry(observer="Juniper")
    env = build_llm_envelope(
        parent,
        source=src,
        entry=entry,
        memory_rendered="hello memory",
        reply_to="orion:exec:result:LLMGatewayService:1",
        route="chat",
    )
    messages = env.payload.get("messages")
    assert isinstance(messages, list)
    user_msg = messages[1]["content"]
    assert "RELEVANT MEMORY" in user_msg
    assert "hello memory" in user_msg


def test_extract_message_delimiters():
    text = """
    [INTROSPECT]
    hidden
    [/INTROSPECT]

    [MESSAGE]
    hi Juniper
    [/MESSAGE]
    """
    intro, msg = extract_message_sections(text)
    assert intro.strip() == "hidden"
    assert msg.strip() == "hi Juniper"


def test_notify_request_is_chat_message_event_kind_and_has_dedupe_key():
    parent = _env()
    entry = _entry(observer="juniper")
    req = build_notify_request(
        source_service="orion-actions",
        recipient_group="juniper_primary",
        session_id="collapse_mirror",
        correlation_id=str(parent.correlation_id),
        dedupe_key=entry.event_id,
        dedupe_window_seconds=86400,
        entry=entry,
        action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
        introspect_text="int",
        message_text="message to Juniper",
    )
    assert req.event_kind == "orion.chat.message"
    assert req.dedupe_key is not None
    assert entry.event_id in req.dedupe_key
    assert req.session_id == "collapse_mirror"
    assert req.recipient_group == "juniper_primary"
