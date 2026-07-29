"""Gate tests for the NPC cooldown tuning patch (load-shedding on orion-llm-gateway)."""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-npc-cooldown-tuning.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_cooldown_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-npc-cooldown-tuning.patch" in text


def test_cooldown_patch_registered_after_patches_that_also_touch_constants():
    text = _APPLY.read_text(encoding="utf-8")
    # This patch's diff hunks are generated against constants.ts *after* the
    # other patches that also touch it -- it must stay last in the array or
    # it will fail to apply / apply against the wrong base.
    order = [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert order[-1] == "orion-npc-cooldown-tuning.patch"
    for other in (
        "orion-town-chat-turns.patch",
        "orion-human-juniper.patch",
        "orion-hub-embed.patch",
    ):
        assert order.index(other) < order.index("orion-npc-cooldown-tuning.patch")


def test_cooldown_patch_raises_conversation_cooldown():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "-export const CONVERSATION_COOLDOWN = 15000;" in patch
    assert "+export const CONVERSATION_COOLDOWN = 45000;" in patch


def test_cooldown_patch_raises_message_cooldown():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "-export const MESSAGE_COOLDOWN = 2000;" in patch
    assert "+export const MESSAGE_COOLDOWN = 8000;" in patch


def test_cooldown_patch_only_touches_constants_file():
    patch = _PATCH.read_text(encoding="utf-8")
    assert patch.count("diff --git") == 1
    assert "convex/constants.ts" in patch
