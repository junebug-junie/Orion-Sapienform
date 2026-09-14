"""Claude room fallback — exactly one RoomClaudeRequest → utterance → PeerBrief.

Live path publishes `RoomClaudeRequestV1` (trigger=auto) and waits for a matching
utterance. Tests inject `publish_request` / `wait_utterance` (or the worker's
`claude=` callable) so nothing hits the bus.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from app.cursor_invoker import build_sealed_prompt, parse_peer_brief_body
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomClaudeUtteranceV1

logger = logging.getLogger("orion-curiosity-peer.claude_fallback")

# Keep in sync with services/orion-room-companion/app/room_prompt.py
CONTRACTOR_PEER_MARKER = "[orion-contractor-peer]"

DEFAULT_ROOM_ID = "curiosity-contractor-peer"
DEFAULT_INVITED_BY = "orion"
DEFAULT_TIMEOUT_SEC = 120.0

PublishRequest = Callable[[RoomClaudeRequestV1], Any]
WaitUtterance = Callable[..., str]


def build_claude_fallback_prompt(
    help_req: HelpRequestV1,
    *,
    context_pack: str = "",
) -> str:
    """Sealed contractor prompt plus the room-companion marker."""
    sealed = build_sealed_prompt(help_req, context_pack=context_pack)
    return f"{CONTRACTOR_PEER_MARKER}\n{sealed}"


def _utterance_text(payload: Any) -> str:
    if isinstance(payload, RoomClaudeUtteranceV1):
        if not payload.ok:
            raise RuntimeError(payload.error or "claude utterance not ok")
        if payload.passed:
            return ""
        return str(payload.text or "")
    if isinstance(payload, dict):
        utt = RoomClaudeUtteranceV1.model_validate(payload)
        return _utterance_text(utt)
    return str(payload or "")


def run_claude_fallback(
    help_req: HelpRequestV1,
    *,
    context_pack: str = "",
    publish_request: Optional[PublishRequest] = None,
    wait_utterance: Optional[WaitUtterance] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    room_id: str = DEFAULT_ROOM_ID,
    invited_by: str = DEFAULT_INVITED_BY,
) -> PeerBriefV1:
    """Publish one auto RoomClaudeRequest and map the utterance to a PeerBrief.

    `publish_request` and `wait_utterance` are required for the live seam;
    omitting either raises so a silent no-op cannot masquerade as success.
    """
    if publish_request is None or wait_utterance is None:
        raise NotImplementedError(
            "run_claude_fallback requires publish_request and wait_utterance "
            "(inject mocks in tests; wire bus seams in the worker)"
        )

    prompt = build_claude_fallback_prompt(help_req, context_pack=context_pack)
    request = RoomClaudeRequestV1(
        trigger="auto",
        invited_by=invited_by,
        room_id=room_id,
        prompt=prompt,
        transcript=[],
        social_memory_summary={},
        correlation_id=help_req.help_id,
    )
    publish_request(request)
    logger.info(
        "curiosity_peer_claude_fallback_request request_id=%s help_id=%s",
        request.request_id,
        help_req.help_id,
    )
    raw = wait_utterance(request.request_id, timeout_sec=timeout_sec)
    body = _utterance_text(raw) if not isinstance(raw, str) else raw
    return parse_peer_brief_body(body, help=help_req, peer="claude_room")
