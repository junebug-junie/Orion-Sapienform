from __future__ import annotations

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.collapse_mirror_chat_reply import (
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
from orion.schemas.registry import resolve


def test_collapse_mirror_chat_reply_request_round_trip() -> None:
    entry = CollapseMirrorEntryV2(
        event_id="evt-1",
        observer="juniper",
        trigger="t",
        observer_state=["tired"],
        type="reflect",
        emergent_entity="x",
        summary="felt a shift",
        mantra="stay with it",
    )
    req = CollapseMirrorChatReplyRequestV1(
        event_id="evt-1",
        observer="juniper",
        mirror_text="### Collapse Mirror\n",
        entry=entry,
    )
    dumped = req.model_dump(mode="json")
    again = CollapseMirrorChatReplyRequestV1.model_validate(dumped)
    assert again.event_id == "evt-1"
    assert again.observer == "juniper"
    assert again.entry.summary == "felt a shift"
    assert COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL == "orion:hub:collapse_mirror:chat_reply"
    assert COLLAPSE_MIRROR_CHAT_REPLY_KIND == "collapse.mirror.chat_reply.request.v1"


def test_collapse_mirror_chat_reply_registered() -> None:
    assert resolve("CollapseMirrorChatReplyRequestV1") is CollapseMirrorChatReplyRequestV1
