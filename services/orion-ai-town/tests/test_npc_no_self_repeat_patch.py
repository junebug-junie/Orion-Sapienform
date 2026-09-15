"""Gate: deterministic NPC self-repeat reject (Sofia exact-line dupe, 2026-09-15)."""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-npc-no-self-repeat.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def _apply_order() -> list[str]:
    text = _APPLY.read_text(encoding="utf-8")
    return [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]


def test_no_self_repeat_registered_before_mechanical_leave():
    order = _apply_order()
    assert order.index("orion-resync-agent-descriptions.patch") < order.index(
        "orion-npc-no-self-repeat.patch"
    )
    assert order.index("orion-npc-no-self-repeat.patch") < order.index(
        "orion-mechanical-leave.patch"
    )


def test_no_self_repeat_patch_adds_helpers_and_continue_gate():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "function isSelfRepeat" in patch
    assert "function ownPriorLines" in patch
    assert "out.slice(-4)" in patch
    assert "You already said that" in patch
    assert "SELF_REPEAT_LEAVE_SENTINEL" in patch
    assert "return SELF_REPEAT_LEAVE_SENTINEL" in patch
    assert "convex/agent/conversation.ts" in patch
    assert "convex/aiTown/agentOperations.ts" in patch


def test_no_self_repeat_empty_continue_leaves():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "SELF_REPEAT_LEAVE_SENTINEL" in patch
    assert "text.trim() === SELF_REPEAT_LEAVE_SENTINEL" in patch
    assert "leaveConversationMessage" in patch
    assert "leaveConversation = true" in patch
    assert "Take care." in patch
    # Must not treat every empty continue as leave (transient empty mid-chat).
    assert "args.type === 'continue' && !text.trim()" not in patch


def test_no_self_repeat_patch_hunk_line_counts_match():
    """Catch corrupt @@ headers before deploy apply fails."""
    text = _PATCH.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            parts = line.split()
            old_spec = parts[1]
            new_spec = parts[2]
            old_count = int(old_spec.split(",")[1]) if "," in old_spec else 1
            new_count = int(new_spec.split(",")[1]) if "," in new_spec else 1
            i += 1
            old_seen = new_seen = 0
            while i < len(lines) and not lines[i].startswith("@@"):
                if lines[i].startswith("diff --git"):
                    break
                prefix = lines[i][:1]
                if prefix == " ":
                    old_seen += 1
                    new_seen += 1
                elif prefix == "-":
                    old_seen += 1
                elif prefix == "+":
                    new_seen += 1
                i += 1
            assert old_seen == old_count, f"old count {old_seen} != {old_count}"
            assert new_seen == new_count, f"new count {new_seen} != {new_count}"
            continue
        i += 1
