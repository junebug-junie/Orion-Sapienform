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
    ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB,
    ActionDedupe,
    build_cortex_orch_envelope,
    dispatch_cortex_request,
    should_trigger,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.published = []
        self.rpc_calls = 0

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))

    async def rpc_request(self, *args, **kwargs):
        self.rpc_calls += 1
        raise AssertionError("rpc_request should not be used for action dispatch")



def _entry(observer: str = "juniper") -> CollapseMirrorEntryV2:
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



def test_actions_emits_cortex_orch_request_not_notify():
    bus = _FakeBus()
    parent = _env()
    env = build_cortex_orch_envelope(
        parent,
        source=ServiceRef(name="orion-actions"),
        entry=_entry(observer="Juniper"),
        session_id="collapse_mirror",
        recipient_group="juniper_primary",
        dedupe_key="evt-1",
        dedupe_window_seconds=86400,
        recall_profile="reflect.v1",
    )

    asyncio.run(dispatch_cortex_request(bus=bus, channel="orion:cortex:request", envelope=env))

    assert bus.rpc_calls == 0
    assert len(bus.published) == 1
    channel, published = bus.published[0]
    assert channel == "orion:cortex:request"
    assert published.kind == "cortex.orch.request"
    assert published.payload["verb"] == ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB
    metadata = published.payload["context"]["metadata"]
    assert metadata["recipient_group"] == "juniper_primary"
    assert metadata["notify_dedupe_key"] == "actions:collapse_reply:evt-1"



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
