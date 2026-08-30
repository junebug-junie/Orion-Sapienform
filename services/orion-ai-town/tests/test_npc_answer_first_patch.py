"""Gate: the first NPC reply after a human line is continue, not start.

Live 2026-08-29 Juniper↔Nico: after "hi" he opened a crumb-secret hook, then
refused "spill the tea" with a lockbox. Human chats never call
startConversationMessage. The continue contract plus Nico's tease-plan did it
on turn one; the later hop-contract only made the next inventions.
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
    assert "one short goodbye and nothing else" in patch
    # Continue is the human path. Start-only "quest giver" never ran for Juniper.
    assert patch.count("not a quest giver") >= 2
    assert "Do not hide it as a secret, code, key, or later reveal" in patch
    # Empty-shell social-memory topic bags must not be injected as "memory".
    assert "Recent room themes:" in patch
    assert "recent shared topics include" in patch
    assert "usableTownContinuity" in patch


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
    assert "You tell people a specific piece of diner gossip." in cards
    assert "turn it into tonight's event" not in cards
    assert "You tell people a specific piece of diner gossip." in patch
