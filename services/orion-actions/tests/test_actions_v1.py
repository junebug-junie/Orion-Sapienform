import asyncio
import os
import sys
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import (  # noqa: E402
    ActionDedupe,
    build_collapse_mirror_chat_reply_envelope,
    publish_collapse_mirror_chat_reply,
    should_trigger,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402
from orion.schemas.collapse_mirror_chat_reply import (  # noqa: E402
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
)


class _FakeBus:
    def __init__(self) -> None:
        self.published = []
        self.rpc_calls = 0

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))

    async def rpc_request(self, *args, **kwargs):
        self.rpc_calls += 1
        raise AssertionError("rpc_request should not be used for collapse chat reply")


def _entry(observer: str = "juniper", event_id: str = "evt-1") -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        event_id=event_id,
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


def test_actions_publishes_hub_chat_reply_not_cortex_verb():
    bus = _FakeBus()
    parent = _env()
    env = build_collapse_mirror_chat_reply_envelope(
        parent,
        source=ServiceRef(name="orion-actions"),
        entry=_entry(observer="Juniper"),
    )

    asyncio.run(
        publish_collapse_mirror_chat_reply(
            bus=bus,
            channel=COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
            envelope=env,
        )
    )

    assert bus.rpc_calls == 0
    assert len(bus.published) == 1
    channel, published = bus.published[0]
    assert channel == COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL
    assert published.kind == COLLAPSE_MIRROR_CHAT_REPLY_KIND
    assert published.payload["event_id"] == "evt-1"
    assert published.payload["observer"].lower() == "juniper"
    assert "Collapse Mirror" in published.payload["mirror_text"]
    assert published.payload["entry"]["summary"] == "s"
    assert "verb" not in published.payload


def test_actions_filters_juniper_casefold():
    assert should_trigger(_entry(observer="juniper")) is True
    assert should_trigger(_entry(observer="Juniper")) is True
    assert should_trigger(_entry(observer="JUNIPER")) is True
    assert should_trigger(_entry(observer="orion")) is False


def test_actions_dedupe_prevents_double_dispatch():
    d = ActionDedupe(ttl_seconds=60)
    key = "collapse_123"
    assert d.try_acquire(key) is True
    assert d.try_acquire(key) is False
    d.mark_done(key)
    assert d.try_acquire(key) is False
