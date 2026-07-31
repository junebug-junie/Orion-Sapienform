"""Shared ai-town source detection/labeling for every chat_history_log reader.

Root cause (found live 2026-07-31): chat_history_log rows are read by EIGHT
separate call sites across this service (storage/sql_adapter.py,
sql_timeline.py x3, sql_chat.py x2 -- one of which, fetch_chat_turns_by_id,
has three independent downstream consumers: storage/falkor_chat_adapter.py,
storage/falkor_neighborhood_adapter.py, and worker.py), each independently
building its own "User: {prompt}\\nOrion: {response}"-style text with no
awareness of client_meta.external_room.platform -- meaning an ai-town NPC's
line gets labeled "User:" (a false claim that Juniper said it) regardless of
which path served a given recall. A first pass at this fix only touched
sql_adapter.py; this module exists so the other seven don't have to
reimplement (and potentially re-diverge on) the same logic.

See docs/superpowers/specs/2026-07-31-recall-aitown-source-tagging-design.md.
"""
from __future__ import annotations

import json
from typing import Any, Optional

AITOWN_TAG = "aitown"
AITOWN_MARKER = "[ai-town]"


def chat_source_platform(client_meta: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract (platform, participant_name) from a chat_history_log row's
    client_meta. jsonb columns come back as dict via psycopg2's built-in
    adapter; asyncpg does not auto-decode jsonb by default and may hand back
    a JSON-encoded string instead -- handle both. Returns (None, None) for
    rows that predate this field (real hub/direct conversation) or any
    unrecognized shape -- callers treat that as "not ai-town", the correct
    default.
    """
    if isinstance(client_meta, str):
        try:
            client_meta = json.loads(client_meta)
        except Exception:
            return None, None
    if not isinstance(client_meta, dict):
        return None, None
    external_room = client_meta.get("external_room") or {}
    external_participant = client_meta.get("external_participant") or {}
    platform = external_room.get("platform") if isinstance(external_room, dict) else None
    participant_name = (
        external_participant.get("participant_name") if isinstance(external_participant, dict) else None
    )
    return platform, participant_name


def render_labeled_chat_text(prompt: str, response: str, client_meta: Any) -> str:
    """"User: {prompt}\\nOrion: {response}" style, used by sql_adapter.py and
    sql_timeline.py's three chat-branch functions."""
    platform, participant_name = chat_source_platform(client_meta)
    if platform == AITOWN_TAG:
        who = participant_name or "NPC"
        return f"{AITOWN_MARKER} {who}: {prompt}\nOrion (in ai-town): {response}".strip()
    return f"User: {prompt}\nOrion: {response}".strip()


def render_quoted_chat_text(prompt: str, response: str, client_meta: Any) -> str:
    """'ExactUserText: "{prompt}"\\nOrionResponse: "{response}"' style, used by
    sql_chat.py's fetch_chat_history_pairs and storage/falkor_chat_adapter.py
    (via sql_chat.py's fetch_chat_turns_by_id join)."""
    platform, participant_name = chat_source_platform(client_meta)
    if platform == AITOWN_TAG:
        who = participant_name or "NPC"
        return f'{AITOWN_MARKER} {who}: "{prompt}"\nOrion (in ai-town): "{response}"'.strip()
    return f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'.strip()


def chat_source_tags(client_meta: Any, base_tags: list[str]) -> list[str]:
    """base_tags plus "aitown" when the row is ai-town-sourced, else base_tags
    unchanged (same list identity semantics as the call sites' previous
    `tags = [...]` literals -- callers should treat the return value as the
    tags list to use, not mutate base_tags in place)."""
    platform, _ = chat_source_platform(client_meta)
    if platform == AITOWN_TAG:
        return [*base_tags, AITOWN_TAG]
    return list(base_tags)
