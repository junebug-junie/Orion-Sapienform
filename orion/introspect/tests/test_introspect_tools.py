"""IntrospectTools over a fake bus: correct transport, and failures are 'unknown', never empty."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.introspect.tools import RPC_TIMEOUT_SEC, IntrospectTools, IntrospectUnknownError
from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolResultV1
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX

BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="run-1", parent_trace_id="trace-1", memory_allowed=True,
)
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _ok_payload():
    result = IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0)
    return ReadingToolResultV1(ok=True, result=result.model_dump(mode="json")).model_dump(mode="json")


class ReplyBus:
    codec = OrionCodec()

    def __init__(self, payload=None, *, raise_exc=None, wrong_correlation=False):
        self.payload = payload
        self.raise_exc = raise_exc
        self.wrong_correlation = wrong_correlation
        self.sent = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.sent.append((channel, envelope, reply_channel, timeout_sec))
        if self.raise_exc is not None:
            raise self.raise_exc
        reply = BaseEnvelope(
            kind="reading.tool.result.v1",
            correlation_id=uuid4() if self.wrong_correlation else envelope.correlation_id,
            source=ServiceRef(name="orion-hub"),
            payload=self.payload,
        )
        return {"data": self.codec.encode(reply)}


def _invoke(bus, name="reading_results", args=None):
    return asyncio.run(IntrospectTools(bus, BINDING).invoke(name, args or {}))


def test_tool_specs_list_reading_results_then_dreams_then_curiosity():
    assert [s.name for s in IntrospectTools(ReplyBus(), BINDING).tool_specs()] == [
        "reading_results", "dreams", "curiosity",
    ]


def test_reading_results_uses_reading_channel_with_normalized_url():
    bus = ReplyBus(_ok_payload())
    out = _invoke(bus, args={"url": "https://EXAMPLE.org/a#frag", "limit": 3})
    assert out["ok"] is True and out["items"] == [] and out["total_available"] == 0
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == TOOL_CHANNEL
    assert envelope.kind == "reading.tool.request.v1"
    assert envelope.reply_to == reply_channel == f"{TOOL_RESULT_PREFIX}{envelope.correlation_id}"
    assert envelope.source.name == "orion-harness-governor"
    assert timeout == RPC_TIMEOUT_SEC
    assert envelope.payload["operation"] == "reading_result"
    assert envelope.payload["url"] == "https://example.org/a"
    assert envelope.payload["limit"] == 3


@pytest.mark.parametrize("bus", [
    ReplyBus(raise_exc=TimeoutError("no reply")),
    ReplyBus(ReadingToolResultV1(ok=False, error="reading_queue_unavailable").model_dump(mode="json")),
    ReplyBus(_ok_payload(), wrong_correlation=True),
    ReplyBus(ReadingToolResultV1(ok=True, result={"garbage": 1}).model_dump(mode="json")),
    ReplyBus({"not": "a reading tool result"}),
])
def test_every_failure_is_unknown_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        _invoke(bus)


def test_unknown_tool_is_rejected_without_sending():
    bus = ReplyBus(_ok_payload())
    with pytest.raises(ValueError, match="unknown introspect tool"):
        _invoke(bus, name="memories")
    assert bus.sent == []


def test_model_cannot_supply_binding_fields():
    bus = ReplyBus(_ok_payload())
    with pytest.raises(ValidationError):
        _invoke(bus, args={"memory_allowed": True})
    assert bus.sent == []


class CorruptDataBus:
    codec = OrionCodec()

    def __init__(self):
        self.sent = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.sent.append((channel, envelope, reply_channel, timeout_sec))
        return {"data": b"\xff\xfe\x00not-valid-json"}


def test_corrupt_reply_bytes_are_unknown():
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        _invoke(CorruptDataBus())


def test_reading_results_forwards_query_without_url_normalization():
    bus = ReplyBus(_ok_payload())
    out = _invoke(bus, args={"query": "  graphics cards ", "limit": 2})
    assert out["ok"] is True
    [(_, envelope, _, _)] = bus.sent
    assert envelope.payload["query"] == "graphics cards"
    assert envelope.payload["limit"] == 2
    assert "url" not in envelope.payload or envelope.payload["url"] is None


def test_description_leads_with_semantic_query():
    [spec] = [s for s in IntrospectTools(ReplyBus(), BINDING).tool_specs() if s.name == "reading_results"]
    assert spec.description.lower().startswith("search")
    assert "query" in spec.description and "similarity" in spec.description
    assert "query" in spec.arguments.model_json_schema()["properties"]


def test_dreams_description_asks_for_topic_only_query():
    """Dream-worded queries lift every narrative's similarity (task-8 calibration, framed set)."""
    [spec] = [s for s in IntrospectTools(ReplyBus(), BINDING).tool_specs() if s.name == "dreams"]
    assert (
        "Every record here is already a dream, so put only the topic in query -- "
        "'pull requests', not 'a dream about pull requests'."
    ) in spec.description
    assert "similarity" in spec.description and "dream_id=<id>" in spec.description
    assert "items=[] means no dream matched" in spec.description


def test_dreams_description_says_what_each_kind_returns():
    [spec] = [s for s in IntrospectTools(ReplyBus(), BINDING).tool_specs() if s.name == "dreams"]
    assert "kind=narrative returns only the nightly dream narratives" in spec.description
    assert "kind=hypothesis only the sleep-cycle hypotheses already offered to you" in spec.description


class DreamBus(ReplyBus):
    def __init__(self, payload=None, *, kind=RESULT_KIND, **kw):
        super().__init__(payload, **kw)
        self.kind = kind

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.sent.append((channel, envelope, reply_channel, timeout_sec))
        if self.raise_exc is not None:
            raise self.raise_exc
        reply = BaseEnvelope(
            kind=self.kind,
            correlation_id=uuid4() if self.wrong_correlation else envelope.correlation_id,
            source=ServiceRef(name="orion-dream"), payload=self.payload,
        )
        return {"data": self.codec.encode(reply)}


def _dream_ok(**kw):
    return IntrospectResultV1(ok=True, operation="dreams", as_of=NOW, total_available=0, **kw).model_dump(mode="json")


def test_dreams_uses_dream_channel_with_binding_and_clean_args():
    bus = DreamBus(_dream_ok())
    out = _invoke(bus, "dreams", {"query": " vision ", "limit": 2})
    assert out["ok"] is True and out["operation"] == "dreams"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == DREAM_REQUEST_CHANNEL and envelope.kind == REQUEST_KIND
    assert envelope.reply_to == reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    assert timeout == RPC_TIMEOUT_SEC and envelope.source.name == "orion-harness-governor"
    assert envelope.payload["operation"] == "dreams"
    assert envelope.payload["binding"]["parent_run_id"] == "run-1"
    assert envelope.payload["args"] == {"query": "vision", "limit": 2}


def test_dreams_rejects_bad_args_before_transport():
    bus = DreamBus(_dream_ok())
    with pytest.raises(ValidationError):
        _invoke(bus, "dreams", {"arm": "dream"})
    assert bus.sent == []


@pytest.mark.parametrize(
    "bus",
    [
        DreamBus(raise_exc=TimeoutError()),
        DreamBus(_dream_ok(), wrong_correlation=True),
        DreamBus(_dream_ok(), kind="reading.tool.result.v1"),
        DreamBus({"ok": True}),
        DreamBus(IntrospectResultV1(ok=False, operation="dreams", as_of=NOW, error="dreams_unavailable; answer unknown").model_dump(mode="json")),
        DreamBus(IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0).model_dump(mode="json")),
    ],
)
def test_dreams_failures_are_unknown_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="dreams: answer unknown"):
        _invoke(bus, "dreams", {})


def test_dreams_corrupt_reply_bytes_are_unknown():
    bus = CorruptDataBus()
    with pytest.raises(IntrospectUnknownError, match="dreams: answer unknown"):
        _invoke(bus, "dreams", {})
    [(channel, _, _, _)] = bus.sent
    assert channel == DREAM_REQUEST_CHANNEL


from orion.introspect.transport import CURIOSITY_REQUEST_CHANNEL  # noqa: E402


def _curiosity_ok(**kw):
    return IntrospectResultV1(ok=True, operation="curiosity", as_of=NOW, total_available=0, **kw).model_dump(mode="json")


def test_curiosity_uses_its_channel_with_binding_and_clean_args():
    bus = DreamBus(_curiosity_ok())
    out = _invoke(bus, "curiosity", {"query": " stance gate ", "line": "investigate", "limit": 2})
    assert out["ok"] is True and out["operation"] == "curiosity"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == CURIOSITY_REQUEST_CHANNEL and envelope.kind == REQUEST_KIND
    assert envelope.reply_to == reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    assert timeout == RPC_TIMEOUT_SEC
    assert envelope.payload["operation"] == "curiosity"
    assert envelope.payload["args"] == {"query": "stance gate", "line": "investigate", "limit": 2, "kind": "run"}


def test_curiosity_rejects_bad_args_before_transport():
    bus = DreamBus(_curiosity_ok())
    for bad in ({"run_id": "r1", "query": "x"}, {"kind": "self_question", "line": "investigate"}, {"status": "failed"}):
        with pytest.raises(ValidationError):
            _invoke(bus, "curiosity", bad)
    assert bus.sent == []


@pytest.mark.parametrize(
    "bus",
    [
        DreamBus(raise_exc=TimeoutError()),
        DreamBus(_curiosity_ok(), wrong_correlation=True),
        DreamBus({"ok": True}),
        DreamBus(IntrospectResultV1(ok=False, operation="curiosity", as_of=NOW, error="curiosity_unavailable; answer unknown").model_dump(mode="json")),
        DreamBus(_dream_ok()),
    ],
)
def test_curiosity_failures_are_unknown_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="curiosity: answer unknown"):
        _invoke(bus, "curiosity", {})


def test_curiosity_description_names_failures_labels_and_unknown():
    [spec] = [s for s in IntrospectTools(ReplyBus(), BINDING).tool_specs() if s.name == "curiosity"]
    for phrase in ("failures are part of your history", "not established fact", "kind=self_question",
                   "graph_read=false means the hop counts are unknown", "never that no run happened"):
        assert phrase in spec.description
