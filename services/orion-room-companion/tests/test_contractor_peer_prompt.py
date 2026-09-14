"""Contractor-peer marker selects investigation framing on room prompts."""

from __future__ import annotations

from app.room_prompt import (
    AUTO_INVITE_CLAUSE,
    CONTRACTOR_PEER_CLAUSE,
    CONTRACTOR_PEER_MARKER,
    SYSTEM_PROMPT,
    append_system_prompt_for,
)


def test_append_system_prompt_manual_is_base_only() -> None:
    assert append_system_prompt_for(trigger="manual", prompt="hey") == SYSTEM_PROMPT


def test_append_system_prompt_auto_adds_invite_clause() -> None:
    assert (
        append_system_prompt_for(trigger="auto", prompt="hey")
        == SYSTEM_PROMPT + AUTO_INVITE_CLAUSE
    )


def test_contractor_marker_adds_peer_clause() -> None:
    prompt = f"{CONTRACTOR_PEER_MARKER}\nhelp Orion investigate"
    out = append_system_prompt_for(trigger="auto", prompt=prompt)
    assert out.startswith(SYSTEM_PROMPT)
    assert CONTRACTOR_PEER_CLAUSE in out
    assert AUTO_INVITE_CLAUSE in out
    # Marker alone without trigger=auto still gets contractor clause.
    manual = append_system_prompt_for(trigger="manual", prompt=prompt)
    assert CONTRACTOR_PEER_CLAUSE in manual
    assert AUTO_INVITE_CLAUSE not in manual
