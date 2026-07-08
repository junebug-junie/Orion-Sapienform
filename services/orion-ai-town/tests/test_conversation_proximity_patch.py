"""Gate tests for conversation proximity patch (NPC walk-up before chat)."""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-conversation-proximity.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_proximity_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-conversation-proximity.patch" in text


def test_proximity_patch_defines_max_invite_distance():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "MAX_INVITE_DISTANCE = 6" in patch
    assert "CONVERSATION_DISTANCE" in patch


def test_proximity_patch_limits_candidate_search_and_start():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "findConversationCandidate" in patch
    assert "d > MAX_INVITE_DISTANCE" in patch
    assert "Players too far apart to start a conversation" in patch
    assert "movePlayer(game, now, player, destination)" in patch


def test_accepted_walking_over_does_not_expire_on_invite_timeout():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "walkingOver, the accepted invite is a contract" in patch
    assert "otherMember.status.kind === 'invited'" in patch
    assert "timing out stale invite" in patch
    assert "walkingOver does not" in patch
