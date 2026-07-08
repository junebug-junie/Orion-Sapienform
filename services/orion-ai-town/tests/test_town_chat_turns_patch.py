"""Gate tests for town NPC chat turn-taking and reply quality patch."""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-town-chat-turns.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_chat_turns_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-town-chat-turns.patch" in text


def test_chat_turns_waits_for_human_before_npc_opener():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "Never talk over them with a synthetic opener" in patch
    assert "otherPlayer.human" in patch


def test_chat_turns_npc_stays_in_human_conversation():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "With a human partner, stay until they leave" in patch
    assert "HUMAN_REPLY_GRACE_MS" in patch


def test_chat_turns_clamps_narration_and_reply_length():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "clampTownReply" in patch
    assert "No scene description, narration" in patch
    assert "max_tokens: 120" in patch
