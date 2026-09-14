"""Actions → Hub: reply to a Juniper Collapse Mirror on the live chat lane.

Design: docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2

COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL = "orion:hub:collapse_mirror:chat_reply"
COLLAPSE_MIRROR_CHAT_REPLY_KIND = "collapse.mirror.chat_reply.request.v1"


class CollapseMirrorChatReplyRequestV1(BaseModel):
    """Thin Hub-bound request: mirror text for the You bubble + full entry for audit."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    observer: str = Field(min_length=1)
    mirror_text: str = Field(min_length=1, description="User-turn text (markdown) for the live session.")
    entry: CollapseMirrorEntryV2
