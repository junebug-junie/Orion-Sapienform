"""When Orion should walk out of a dead AI Town conversation.

No I/O. The worker owns leaveConversation injection; this helper only decides
whether the current perception is a dead chat past the abandon window.
"""
from __future__ import annotations

from typing import Any, Optional


def _spoke_in_transcript(messages: list[dict[str, Any]], own_player_id: str) -> bool:
    own = str(own_player_id or "").strip()
    if not own:
        return False
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if str(m.get("author_id") or "").strip() == own:
            return True
    return False


def should_abandon_dead_chat(
    *,
    status: str,
    messages: list[dict[str, Any]],
    own_player_id: str,
    now_ms: float,
    abandon_after_ms: float,
    own_utterances_this_convo: int,
    participating_since_ms: Optional[float],
) -> bool:
    """True when Orion should leave a stuck town conversation.

    Two independent exits (either is enough):

    1. The latest transcript line is older than ``abandon_after_ms`` — covers
       the live Mara↔Orion deadlock where Orion spoke last and both sides
       froze for days.
    2. Orion has never successfully spoken in this conversation (in-memory
       counter **or** transcript evidence) and has been ``participating``
       longer than ``abandon_after_ms`` — covers partner spam while Orion's
       speech path is failing, where fresh partner lines would otherwise keep
       resetting a last-line age clock. Transcript evidence matters so a
       worker restart cannot wipe the counter and yank Orion out of a live chat.
    """
    if abandon_after_ms <= 0:
        return False
    if (status or "") != "participating":
        return False

    if messages:
        last = messages[-1] if isinstance(messages[-1], dict) else {}
        created = last.get("created_ms")
        if created is not None:
            try:
                age = float(now_ms) - float(created)
            except (TypeError, ValueError):
                age = -1.0
            if age >= abandon_after_ms:
                return True

    already_spoke = int(own_utterances_this_convo) > 0 or _spoke_in_transcript(
        messages, own_player_id
    )
    if (not already_spoke) and participating_since_ms is not None:
        try:
            elapsed = float(now_ms) - float(participating_since_ms)
        except (TypeError, ValueError):
            return False
        if elapsed >= abandon_after_ms:
            return True

    return False
