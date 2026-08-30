"""Gate tests for the concrete-grounding prompt patch (native NPC dialogue).

Companion to orion-anti-repetition-prompt.patch: found live 2026-07-30/31
that stopping verbatim phrase reuse alone wasn't enough -- NPC-to-NPC
conversations still drifted into vague poetic abstraction (a "light vs
shadows" exchange with no concrete referent) once a character's canonical
catchphrase seeded a metaphor. This patch pushes the model to name an actual
object/task/event instead of extending the metaphor.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-concrete-grounding-prompt.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-concrete-grounding-prompt.patch" in text


def test_patch_registered_after_anti_repetition_patch():
    # This patch's diff hunk is generated against conversation.ts *after*
    # orion-anti-repetition-prompt.patch (its pre-image blob is that patch's
    # post-image blob) -- it must come after it or it will fail to apply /
    # apply against the wrong base.
    text = _APPLY.read_text(encoding="utf-8")
    order = [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert order.index("orion-anti-repetition-prompt.patch") < order.index(
        "orion-concrete-grounding-prompt.patch"
    )
    assert order.index("orion-concrete-grounding-prompt.patch") < order.index(
        "orion-input-counter-contention.patch"
    )


def test_patch_adds_concrete_grounding_instruction_to_continue_conversation():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "continueConversationMessage" in patch
    assert "+    `Answer as your job" in patch


def test_patch_does_not_name_light_bait_words():
    patch = _PATCH.read_text(encoding="utf-8")
    lowered = patch.lower()
    for bait in ("light,", "shadows", "glow", "echoes", "silence"):
        assert bait not in lowered


def test_patch_uses_positive_job_contract():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "Answer as your job" in patch
    assert "Name a person, object, or task" in patch


def test_patch_only_touches_conversation_file():
    patch = _PATCH.read_text(encoding="utf-8")
    assert patch.count("diff --git") == 1
    assert "convex/agent/conversation.ts" in patch
