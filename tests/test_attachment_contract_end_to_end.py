"""An attachment ref must survive every hop of the chat saga unchanged.

Five services touch this value between the browser and the model. Each hop
re-validates against a schema with ``extra="forbid"``, so a field added on one
side and forgotten on another fails here rather than at 2am in a live turn.

Deliberately schema-level, not a live-service test: the point is the contract,
and the contract is the thing that would silently drift.
"""

from __future__ import annotations

import json

import pytest

from orion.core.bus.bus_schemas import AttachmentRefV1, ChatRequestPayload, LLMMessage
from orion.schemas.cortex.contracts import (
    CortexChatRequest,
    CortexClientContext,
)


def make_ref() -> AttachmentRefV1:
    return AttachmentRefV1(
        sha256="c" * 64,
        mime="image/webp",
        bytes=4096,
        width=1024,
        height=768,
        source_url="http://100.92.216.81:8080/api/chat/attachments/" + "c" * 64,
        filename="screenshot.webp",
    )


def test_ref_survives_every_hop_byte_for_byte():
    ref = make_ref()
    original = ref.model_dump(mode="json")

    # Hub -> cortex-gateway
    hub_out = CortexChatRequest(prompt="what is this?", attachments=[ref])

    # cortex-gateway -> cortex-orch
    context = CortexClientContext(
        messages=[LLMMessage(role="user", content="what is this?")],
        attachments=list(hub_out.attachments),
    )

    # cortex-orch -> cortex-exec crosses as plain JSON.
    as_json = json.loads(json.dumps([a.model_dump(mode="json") for a in context.attachments]))

    # cortex-exec re-validates and hands it to the gateway.
    revalidated = [AttachmentRefV1.model_validate(item) for item in as_json]
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content="what is this?")],
        attachments=revalidated,
    )

    assert payload.attachments[0].model_dump(mode="json") == original


def test_every_hop_defaults_to_empty():
    """A turn that never mentions attachments carries none at any layer."""
    assert CortexChatRequest(prompt="hi").attachments == []
    assert CortexClientContext(messages=[]).attachments == []
    assert ChatRequestPayload(messages=[LLMMessage(role="user", content="hi")]).attachments == []


@pytest.mark.parametrize(
    "model",
    [CortexChatRequest, CortexClientContext, ChatRequestPayload],
)
def test_no_hop_accepts_an_unknown_attachment_field(model):
    """extra='forbid' must hold on the ref itself at every hop."""
    bad = make_ref().model_dump(mode="json")
    bad["smuggled"] = "../../etc/passwd"
    kwargs = {"attachments": [bad]}
    if model is CortexChatRequest:
        kwargs["prompt"] = "x"
    if model is CortexClientContext:
        kwargs["messages"] = []
    if model is ChatRequestPayload:
        kwargs["messages"] = [LLMMessage(role="user", content="x")]
    with pytest.raises(Exception):
        model(**kwargs)


def test_message_content_is_still_a_plain_string():
    """The whole design rests on content NOT having been widened.

    If this ever fails, someone has started threading content-part lists through
    the saga and every str-assuming consumer -- Spark ingest, recall,
    chat_history, the memory-graph bridge -- is now at risk.
    """
    with pytest.raises(Exception):
        LLMMessage(role="user", content=[{"type": "text", "text": "hi"}])
