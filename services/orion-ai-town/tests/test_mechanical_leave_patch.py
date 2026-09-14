"""Gate: NPC duration/count leave must not depend on an LLM goodbye.

Live 2026-09-14 Mara↔Orion deadlock: leave-via-agentGenerateMessage hung on
empty LLM stubs against a bloated transcript, so the chat never ended.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-mechanical-leave.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def _apply_order() -> list[str]:
    text = _APPLY.read_text(encoding="utf-8")
    return [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]


def test_mechanical_leave_patch_registered_last():
    order = _apply_order()
    assert order[-1] == "orion-mechanical-leave.patch"
    assert order.index("orion-town-chat-turns.patch") < order.index(
        "orion-mechanical-leave.patch"
    )


def test_mechanical_leave_calls_conversation_leave_directly():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "conversation.leave(game, now, player)" in patch
    assert "mechanical" in patch.lower()


def test_mechanical_leave_removes_llm_goodbye_path():
    patch = _PATCH.read_text(encoding="utf-8")
    # The old leave path set typing + agentGenerateMessage type leave.
    assert "-              type: 'leave'," in patch
    assert "agentGenerateMessage" in patch  # appears in the removed hunk
    # New path must not start a leave generate operation.
    assert "+            this.startOperation(game, now, 'agentGenerateMessage'" not in patch


def test_mechanical_leave_patch_hunk_line_counts_match():
    """Catch corrupt @@ -a,b +c,d @@ headers before deploy apply fails."""
    text = _PATCH.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            # @@ -old_start,old_count +new_start,new_count @@
            parts = line.split()
            old_spec = parts[1]  # -207,19
            new_spec = parts[2]  # +207,13
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
