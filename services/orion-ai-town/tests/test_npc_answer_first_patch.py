"""Gate: NPC continue replies answer the last line instead of hopping props.

Live 2026-08-29 Juniper↔Nico: she asked what was in the pie crumbs, then
said not to speak in riddles. continueConversationMessage told the model
to name a new object and change subject if the topic repeated, so Nico
invented breadbox → oven → elevator. This patch replaces that contract.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-npc-answer-first.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"
_CHARACTER = _SERVICE / "patches" / "orion-character.patch"


def _apply_order() -> list[str]:
    text = _APPLY.read_text(encoding="utf-8")
    return [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]


def test_answer_first_patch_registered_last():
    order = _apply_order()
    assert order[-1] == "orion-npc-answer-first.patch"
    assert order.index("orion-town-continuity-ingest.patch") < order.index(
        "orion-npc-answer-first.patch"
    )


def test_answer_first_replaces_prop_hop_contract():
    patch = _PATCH.read_text(encoding="utf-8")
    # Old contract that caused the hop must leave.
    assert "-    `Answer as your job. Name a person, object, or task" in patch
    assert (
        "-    `Do not repeat a phrase, metaphor, or sentence structure that already appears"
        in patch
    )
    # New contract: answer first, stay on their topic, no quest hook.
    assert "+    `Answer the last thing they said." in patch
    assert "Do not invent a new object, place, or quest" in patch
    assert "isTownFarewell" in patch


def test_answer_first_start_is_not_a_quest_hook():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "Give a short in-character greeting" in patch
    assert "not a quest giver" in patch


def test_nico_identity_is_not_riddle_coded():
    cards = (_SERVICE / "cards" / "town_cards.yaml").read_text(encoding="utf-8")
    patch = _CHARACTER.read_text(encoding="utf-8")
    assert "role: Event promoter\n" in cards
    assert "unreliable narrator" not in cards.split("id: sofia_bell")[0]
    assert "slippery when challenged" not in cards
    assert "unreliable narrator" not in patch
    assert "slippery when challenged" not in patch
    assert "answers with a specific detail first" in patch
