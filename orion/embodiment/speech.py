"""Pure helpers for the cortex-generated town speech bridge.

No I/O. The worker owns the cortex RPC + AI Town injection; these helpers only
decide *whether* Orion should speak, *what* to prompt the cortex with, and
*whether* a candidate reply is safe to inject (anti empty-shell guard).
"""
from __future__ import annotations

from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1

_MAX_RECENT_LINES = 4


def _participants(conversation: dict[str, Any]) -> list[str]:
    for key in ("participants", "members", "player_ids"):
        raw = conversation.get(key)
        if isinstance(raw, list) and raw:
            return [str(p) for p in raw]
    return []


def should_speak(perception: WorldPerceptionV1, own_player_id: str) -> bool:
    """True only when ``own_player_id`` is a participant of an active conversation."""
    own = str(own_player_id or "").strip()
    if not own:
        return False
    convo = perception.active_conversation
    if not isinstance(convo, dict) or not convo:
        return False
    return own in _participants(convo)


def _interlocutor_name(perception: WorldPerceptionV1, own_player_id: str) -> str:
    convo = perception.active_conversation or {}
    other = convo.get("other") if isinstance(convo, dict) else None
    if isinstance(other, dict):
        name = other.get("name") or other.get("player_id")
        if name:
            return str(name)
    # Fall back to the nearest known nearby player's name.
    parts = [p for p in _participants(convo) if p != str(own_player_id)]
    by_id = {str(n.get("player_id")): n for n in (perception.nearby_players or [])}
    for pid in parts:
        n = by_id.get(pid)
        if n and n.get("name"):
            return str(n["name"])
        if pid:
            return pid
    return "the other person"


def _recent_lines(conversation: dict[str, Any]) -> list[str]:
    msgs = conversation.get("messages")
    lines: list[str] = []
    if isinstance(msgs, list):
        for m in msgs[-_MAX_RECENT_LINES:]:
            if not isinstance(m, dict):
                continue
            author = m.get("author") or m.get("name") or m.get("player_id") or "?"
            text = str(m.get("text") or m.get("message") or "").strip()
            if text:
                lines.append(f"{author}: {text}")
    return lines


def latest_partner_line(perception: WorldPerceptionV1, own_player_id: str) -> Optional[str]:
    """Return the latest non-Orion utterance in the active town conversation."""
    convo = perception.active_conversation or {}
    if not isinstance(convo, dict):
        return None
    own = str(own_player_id or "").strip()
    msgs = convo.get("messages")
    if not isinstance(msgs, list):
        return None
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        author_id = str(m.get("author_id") or m.get("player_id") or "").strip()
        author = str(m.get("author") or m.get("name") or "").strip().lower()
        if author_id == own or author == "orion":
            continue
        text = str(m.get("text") or m.get("message") or "").strip()
        if text:
            return text
    return None


def build_speech_prompt(perception: WorldPerceptionV1, own_player_id: str) -> str:
    """Prompt for a town utterance anchored on the latest partner line."""
    convo = perception.active_conversation or {}
    interlocutor = _interlocutor_name(perception, own_player_id)
    lines = _recent_lines(convo) if isinstance(convo, dict) else []
    context = "\n".join(lines) if lines else "(no prior lines)"
    latest = latest_partner_line(perception, own_player_id)
    latest_line = latest if latest else "(no partner line yet)"
    return (
        f"You are Orion, embodied in the town, in a conversation with {interlocutor}.\n"
        f"Your task is to answer this latest line from {interlocutor}:\n"
        f"{latest_line}\n\n"
        f"Recent context, reference only:\n{context}\n\n"
        f"Write exactly one short spoken line to {interlocutor} that responds to the latest line above. "
        f"Do not copy, paraphrase, or continue Orion's previous line unless the latest line asks for it. "
        f"If the latest line is a goodbye or departure, acknowledge the departure naturally. "
        f"If there is no partner line worth answering, reply with an empty string."
    )


def is_injectable(reply_text: Optional[str]) -> bool:
    """Anti empty-shell guard: only non-empty, non-whitespace replies are injectable."""
    return bool(reply_text and str(reply_text).strip())
