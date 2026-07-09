"""Pure helpers for the cortex-generated town speech bridge.

No I/O. The worker owns the cortex RPC + AI Town injection; these helpers only
decide *whether* Orion should speak, *what* to prompt the cortex with, and
*whether* a candidate reply is safe to inject (anti empty-shell guard).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1

_MAX_RECENT_LINES = 8


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


def build_speech_prompt(perception: WorldPerceptionV1, own_player_id: str) -> str:
    """Prompt for the cortex utterance: interlocutor + recent conversation lines."""
    convo = perception.active_conversation or {}
    interlocutor = _interlocutor_name(perception, own_player_id)
    lines = _recent_lines(convo) if isinstance(convo, dict) else []
    transcript = "\n".join(lines) if lines else "(no prior lines)"
    return (
        f"You are Orion, embodied in the town, in a conversation with {interlocutor}.\n"
        f"Recent conversation:\n{transcript}\n\n"
        f"Reply with a single natural spoken line to {interlocutor}. "
        f"If there is nothing worth saying, reply with an empty string."
    )


def is_injectable(reply_text: Optional[str]) -> bool:
    """Anti empty-shell guard: only non-empty, non-whitespace replies are injectable."""
    return bool(reply_text and str(reply_text).strip())


def normalize_utterance(reply_text: Optional[str]) -> str:
    """Normalize an utterance for repeat detection without changing injected text."""
    text = str(reply_text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_repeated_utterance(reply_text: Optional[str], previous_reply_text: Optional[str]) -> bool:
    """True when a candidate would repeat the last injected line for a conversation."""
    current = normalize_utterance(reply_text)
    previous = normalize_utterance(previous_reply_text)
    return bool(current and previous and current == previous)
