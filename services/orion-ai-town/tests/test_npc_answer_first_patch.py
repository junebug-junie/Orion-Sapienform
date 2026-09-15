"""Gate: answer-first covers ALL NPCs — plan is background, not a quest.

Aug 30 2026 fixed the human↔NPC path on paper, but after the pair-turn
continuity rewrite the old answer-first hunks targeted a deleted
fetchTownContinuity(otherName) shape and stopped applying. Live 2026-09-15
Mara↔Sofia coffee-loop confirmed Circe was still on plan-as-goals + prop-bait
plans for every NPC↔NPC chat.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-npc-answer-first.patch"
_RESYNC = _SERVICE / "patches" / "orion-resync-agent-descriptions.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"
_CHARACTER = _SERVICE / "patches" / "orion-character.patch"
_CARDS = _SERVICE / "cards" / "town_cards.yaml"


def _apply_order() -> list[str]:
    text = _APPLY.read_text(encoding="utf-8")
    return [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]


def test_answer_first_patch_registered_before_mechanical_leave():
    order = _apply_order()
    assert order.index("orion-npc-answer-first.patch") < order.index(
        "orion-mechanical-leave.patch"
    )
    assert order.index("orion-town-continuity-ingest.patch") < order.index(
        "orion-npc-answer-first.patch"
    )


def test_resync_patch_registered_between_answer_first_and_mechanical_leave():
    order = _apply_order()
    assert order.index("orion-npc-answer-first.patch") < order.index(
        "orion-resync-agent-descriptions.patch"
    )
    assert order.index("orion-resync-agent-descriptions.patch") < order.index(
        "orion-mechanical-leave.patch"
    )


def test_answer_first_replaces_prop_hop_contract():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "-    `Answer as your job. Name a person, object, or task" in patch
    assert (
        "-    `Do not repeat a phrase, metaphor, or sentence structure that already appears"
        in patch
    )
    assert "+    `Answer the last thing they said." in patch
    assert "Do not invent a new quest" in patch
    assert "one short goodbye and nothing else" in patch
    assert patch.count("not a quest giver") >= 2
    assert "Do not hide it as a secret, code, key, or later reveal" in patch


def test_plan_is_background_not_conversation_goal():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "-    prompt.push(`Your goals for the conversation: ${agent.plan}`);" in patch
    assert "background only, not the topic of this chat" in patch


def test_no_forced_previous_conversation_callback():
    patch = _PATCH.read_text(encoding="utf-8")
    assert (
        "-      `Be sure to include some detail or question about a previous conversation"
        in patch
    )
    assert "memoryWithOtherPlayer" in patch  # removed find + use


def test_abstract_circle_can_break_or_leave():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "circling the same abstract idea" in patch
    assert "change the subject to something real you both share, or end the chat" in patch
    assert "bring one real detail from your role background" not in patch
    added = [line for line in patch.splitlines() if line.startswith("+")]
    assert any("Do not repeat a phrase, metaphor, or sentence structure" in line for line in added)
    assert not any("Stay on their topic until they change it" in line for line in added)


def test_resync_sets_descriptions_modified():
    patch = _RESYNC.read_text(encoding="utf-8")
    assert "game.descriptionsModified = true" in patch


def test_all_npc_plans_are_not_prop_quests():
    cards = _CARDS.read_text(encoding="utf-8")
    patch = _CHARACTER.read_text(encoding="utf-8")
    forbidden = [
        "coffee, pie",
        "You run the diner: coffee",
        "You update the town maps:",
        "You tell people a specific piece of diner gossip",
        "You poke a device or a town system",
    ]
    for bait in forbidden:
        assert bait not in cards, bait
        assert bait not in patch, bait
    required = [
        "You notice who depends on whom and say one concrete thing you observed.",
        "You share one concrete thing you heard today, then listen.",
        "You notice who needs something and answer plainly.",
        "You mention one real thing you broke or fixed today.",
    ]
    for line in required:
        assert line in cards, line
        assert line in patch, line


def test_sofia_public_description_not_coffee_menu():
    cards = _CARDS.read_text(encoding="utf-8")
    sofia = cards.split("id: sofia_bell")[1].split("id: ")[0]
    assert "drinks coffee after midnight" not in sofia
    assert "orders pie when they are lying" not in sofia
    assert "notices who" in sofia


def test_resync_patch_adds_engine_input():
    patch = _RESYNC.read_text(encoding="utf-8")
    assert "resyncAgentDescriptions" in patch
    assert "Descriptions.find" in patch
    assert patch.count("diff --git") == 1
    assert "convex/aiTown/agentInputs.ts" in patch


def test_answer_first_start_is_not_a_quest_hook():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "Give a short in-character greeting" in patch
    assert "not a quest giver" in patch


def test_nico_identity_is_not_riddle_coded():
    cards = _CARDS.read_text(encoding="utf-8")
    patch = _CHARACTER.read_text(encoding="utf-8")
    assert "role: Event promoter\n" in cards
    assert "unreliable narrator" not in cards.split("id: sofia_bell")[0]
    assert "slippery when challenged" not in cards
    assert "unreliable narrator" not in patch
    assert "slippery when challenged" not in patch
    assert "answers with a specific detail first" in patch
    assert "You share one concrete thing you heard today, then listen." in cards
    assert "turn it into tonight's event" not in cards
    assert "You share one concrete thing you heard today, then listen." in patch
