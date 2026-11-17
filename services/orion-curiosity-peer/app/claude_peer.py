"""Claude helper — exactly one RoomClaudeRequest → utterance → PeerBrief.

Claude (via orion-room-companion) is the only peer this service hires. The
Cursor Agent CLI path was retired 2026-11-17 (#3397): Juniper no longer uses
Cursor, and gating Claude behind a Cursor failure meant no hire ran at all.

Live path publishes `RoomClaudeRequestV1` (trigger=auto) and waits for a matching
utterance. Tests inject `publish_request` / `wait_utterance` (or the worker's
`claude=` callable) so nothing hits the bus.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from app.brief_prompt import build_sealed_prompt, parse_peer_brief_body
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomClaudeUtteranceV1

logger = logging.getLogger("orion-curiosity-peer.claude_peer")

# Keep in sync with services/orion-room-companion/app/room_prompt.py
CONTRACTOR_PEER_MARKER = "[orion-contractor-peer]"

DEFAULT_ROOM_ID = "curiosity-contractor-peer"
DEFAULT_INVITED_BY = "orion"
# Room companion gives a turn with Orion's tools up to
# ROOM_COMPANION_TOOLS_TURN_TIMEOUT_SEC=360 s (consent asks + reads). Waiting
# only 120 s would time out a turn the companion was still allowed to finish.
DEFAULT_TIMEOUT_SEC = 420.0

PublishRequest = Callable[[RoomClaudeRequestV1], Any]

# Plain sentences for `orion.autonomy.ask_claude_trigger.RefusalReason` budget
# values the Claude gate can return. Orion reads these (role-teach progress
# line + soft nudge show the part before the "[codes]" block).
_CLAUDE_REFUSAL_TEXT = {
    "budget_unobserved": (
        "this service cannot see Claude's usage meter, so it refuses rather "
        "than spend blind"
    ),
    "budget_limited": "Claude's usage meter reads limited",
    "budget_unknown": "Claude's usage meter state is unknown",
    "budget_observation_missing": "no Claude usage reading was available",
    "budget_observation_incoherent": "Claude's usage reading contradicted itself",
}


def describe_claude_refusal(refusal: str) -> str:
    text = _CLAUDE_REFUSAL_TEXT.get(refusal, f"budget gate returned {refusal}")
    return f"Claude helper refused: {text}."


def claude_reason(text: str, code: str) -> str:
    """Refusal chain Orion reads: plain sentence first, then the code block."""
    return f"{text.strip()} [{code}]"
WaitUtterance = Callable[..., str]


def build_claude_peer_prompt(
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


def run_claude_peer(
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
            "run_claude_peer requires publish_request and wait_utterance "
            "(inject mocks in tests; wire bus seams in the worker)"
        )

    prompt = build_claude_peer_prompt(help_req, context_pack=context_pack)
    request = RoomClaudeRequestV1(
        trigger="auto",
        invited_by=invited_by,
        room_id=room_id,
        prompt=prompt,
        transcript=[],
        social_memory_summary={},
        correlation_id=help_req.help_id,
    )
    logger.info(
        "curiosity_peer_claude_request request_id=%s help_id=%s",
        request.request_id,
        help_req.help_id,
    )

    # Subscribe-before-publish: Redis pubsub has no backlog. Wait starts the
    # subscription, then calls publish_request once the listener is ready.
    def _publish_after_subscribe() -> None:
        publish_request(request)

    raw = wait_utterance(
        request.request_id,
        timeout_sec=timeout_sec,
        publish_request=_publish_after_subscribe,
    )
    body = _utterance_text(raw) if not isinstance(raw, str) else raw
    return parse_peer_brief_body(body, help=help_req, peer="claude_room")
