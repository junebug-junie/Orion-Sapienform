"""Gate tests for the anti-repetition prompt patch (native NPC dialogue).

See services/orion-llm-gateway's chat_quick.j2 anti-repetition instruction for
the same fix on Orion's own side. Found live 2026-07-30: NPC-to-NPC (and
Orion-to-NPC) conversations degenerating into a repeated metaphor/phrase loop
across many turns (llama.cpp's quick/quick_background workers run with every
anti-repetition sampler at its neutral default -- see
services/orion-llm-gateway/README.md's mesh-LLM-wiring notes -- so nothing in
the sampling config discourages it; this patch compensates via the prompt
instead).
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-anti-repetition-prompt.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-anti-repetition-prompt.patch" in text


def test_patch_adds_anti_repetition_instruction_to_continue_conversation():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "continueConversationMessage" in patch
    assert "+    `Do not repeat a phrase, metaphor, or sentence structure" in patch


def test_patch_only_touches_conversation_file():
    patch = _PATCH.read_text(encoding="utf-8")
    assert patch.count("diff --git") == 1
    assert "convex/agent/conversation.ts" in patch
