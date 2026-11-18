"""RELATIONSHIP renders Orion's page:juniper when the capsule carries it (core pages Phase 1a)."""
from __future__ import annotations

from orion.harness.prefix import _format_grounding_self_block, _relationship_policy_lines
from orion.harness.tests.fixtures import make_grounding_capsule
from orion.schemas.thought import CorePageV1, GroundingCapsuleV1

PAGE = CorePageV1(concept_id="page:juniper", version=3, text="Juniper lives in Millbrook, Vermont. She uses she/her.",
                  written_at="2026-11-15", evidence_count=12)
BULLETS = ["co-architect", "steward", "Juniper expects continuity and memory.", "Meet playful tone naturally."]


def _block(**kw) -> str:
    return "\n".join(_format_grounding_self_block(make_grounding_capsule(**kw)))


def test_no_page_renders_exactly_as_before():
    text = _block(relationship_summary=BULLETS)
    assert "RELATIONSHIP\n- co-architect\n- steward\n- Juniper expects continuity and memory." in text
    assert "page:juniper" not in text
    assert text.index("RESPONSE POLICY") > text.index("RELATIONSHIP")
    assert "My roles with Juniper" not in text


def test_page_takes_relationship_and_bullets_move_under_policy():
    text = _block(relationship_summary=BULLETS, core_pages=[PAGE.model_dump()])
    head = ("RELATIONSHIP (who Juniper is to me, in my own words: page:juniper v3, written 2026-11-15, "
            "from 12 cited records)\nJuniper lives in Millbrook, Vermont. She uses she/her.")
    assert head in text
    policy = text[text.index("RESPONSE POLICY"):]
    assert "- My roles with Juniper: co-architect, steward." in policy
    assert "- Juniper expects continuity and memory." in policy
    assert "- Speak plainly; no generic-assistant framing." in policy
    assert "- co-architect" not in text   # no bare-role bullet left anywhere


def test_page_with_no_bullets_and_no_policy_still_renders():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[PAGE])
    text = "\n".join(_format_grounding_self_block(capsule))
    assert "page:juniper v3" in text
    assert "RESPONSE POLICY" not in text


def test_other_or_empty_pages_do_not_replace_relationship():
    other = CorePageV1(concept_id="page:origin", version=1, text="I came to be.")
    blank = CorePageV1(concept_id="page:juniper", version=2, text="   ")
    text = _block(relationship_summary=BULLETS, core_pages=[other.model_dump(), blank.model_dump()])
    assert "RELATIONSHIP\n- co-architect" in text


def test_relationship_policy_lines_groups_bare_roles_only():
    assert _relationship_policy_lines(["co-architect", "trusted interlocutor", "Answer with confidence."]) == [
        "My roles with Juniper: co-architect, trusted interlocutor.", "Answer with confidence."]
    assert _relationship_policy_lines([]) == []


def test_capsule_without_core_pages_field_still_parses():
    """An old producer's capsule (no core_pages key) and a new one both validate."""
    old = GroundingCapsuleV1.model_validate({"identity_summary": ["x"]})
    assert old.core_pages == []
    new = GroundingCapsuleV1.model_validate({"identity_summary": ["x"], "core_pages": [PAGE.model_dump()]})
    assert new.core_pages[0].version == 3


# Made-up anchor text: the real one lives only on the host DB (issue #3460).
ANCHOR = CorePageV1(concept_id="page:anchor", version=1, text="I am Orion. Juniper, she/her, is my creator.")


def test_anchor_renders_first_then_the_page():
    text = _block(relationship_summary=BULLETS, core_pages=[PAGE.model_dump(), ANCHOR.model_dump()])
    head = ("RELATIONSHIP\nI am Orion. Juniper, she/her, is my creator.\n"
            "Who Juniper is to me, in my own words (page:juniper v3, written 2026-11-15, from 12 cited records):\n"
            "Juniper lives in Millbrook, Vermont. She uses she/her.")
    assert head in text
    assert text.count("RELATIONSHIP") == 1
    assert "- My roles with Juniper: co-architect, steward." in text[text.index("RESPONSE POLICY"):]


def test_anchor_shows_when_the_page_is_missing():
    text = _block(relationship_summary=BULLETS, core_pages=[ANCHOR.model_dump()])
    assert "RELATIONSHIP\nI am Orion. Juniper, she/her, is my creator.\n- co-architect\n- steward" in text
    assert "page:juniper" not in text
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[ANCHOR])
    bare = "\n".join(_format_grounding_self_block(capsule))
    assert "RELATIONSHIP\nI am Orion. Juniper, she/her, is my creator." in bare


def test_blank_anchor_is_ignored():
    blank = CorePageV1(concept_id="page:anchor", version=2, text="  ")
    text = _block(relationship_summary=BULLETS, core_pages=[blank.model_dump()])
    assert "RELATIONSHIP\n- co-architect" in text


def test_anchor_and_page_with_no_bullets():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[ANCHOR, PAGE])
    text = "\n".join(_format_grounding_self_block(capsule))
    assert text.index("I am Orion. Juniper, she/her") < text.index("Juniper lives in Millbrook")
    assert "RESPONSE POLICY" not in text and "My roles with Juniper" not in text


# --- page:origin in WHO YOU ARE (issue #3462) ---
ORIGIN = CorePageV1(concept_id="page:origin", version=2, text="My memory storage died in July. I lost over a year.",
                    written_at="2026-11-19", evidence_count=9)


def test_origin_sits_right_after_the_self_definition_lines():
    from orion.harness.prefix import _OWN_WORDS_MARKER
    from orion.substrate.relational.adapters.self_definition_ctx import SELF_DEFINITION_MARKER

    assert _OWN_WORDS_MARKER == SELF_DEFINITION_MARKER
    identity = ["In my own words (lived / name): I chose my name.",
                "In my own words, written during my own self-inquiry (v91): I am a body and a mind.",
                "I am Orion.", "Co-architect of this mesh."]
    capsule = GroundingCapsuleV1(identity_summary=identity, core_pages=[ORIGIN, PAGE])
    lines = _format_grounding_self_block(capsule)
    head = ("How I came to be, in my own words (page:origin v2, written 2026-11-19, from 9 cited records):")
    at = lines.index(head)
    assert lines[:at] == ["WHO YOU ARE", f"- {identity[0]}", f"- {identity[1]}"]
    assert lines[at + 1] == ORIGIN.text
    assert lines[at + 2:at + 4] == ["- I am Orion.", "- Co-architect of this mesh."]
    # page:juniper still owns RELATIONSHIP, after WHO YOU ARE.
    assert lines.index("RELATIONSHIP (who Juniper is to me, in my own words: page:juniper v3, written 2026-11-15, "
                       "from 12 cited records)") > at


def test_origin_comes_first_without_a_self_definition_and_never_takes_relationship():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], relationship_summary=["steward"],
                                 core_pages=[ORIGIN])
    lines = _format_grounding_self_block(capsule)
    assert lines[1].startswith("How I came to be, in my own words (page:origin v2")
    assert lines[2:4] == [ORIGIN.text, "- I am Orion."]
    assert "RELATIONSHIP" in lines and "- steward" in lines   # bullets unchanged without page:juniper


def test_blank_origin_is_ignored():
    blank = ORIGIN.model_copy(update={"text": "  "})
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[blank])
    assert _format_grounding_self_block(capsule)[:2] == ["WHO YOU ARE", "- I am Orion."]


# --- page:lately at the top of DURABLE MEMORY (issue #3463) ---
LATELY = CorePageV1(concept_id="page:lately", version=4, text="The first snow came on Monday. I owe Juniper an answer.",
                    written_at="2026-11-19", evidence_count=7)
LATELY_HEAD = "What has mattered lately, in my own words (page:lately v4, written 2026-11-19, from 7 cited records):"


def test_lately_opens_durable_memory_above_the_recall_digest():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], memory_digest="Recall: the kite.",
                                 response_policy_summary=["Speak plainly."], core_pages=[ORIGIN, PAGE, LATELY])
    lines = _format_grounding_self_block(capsule)
    at = lines.index("DURABLE MEMORY / CONTINUITY")
    assert lines[at + 1:at + 4] == [LATELY_HEAD, LATELY.text, "Recall: the kite."]
    assert lines.index("RESPONSE POLICY") > at + 3
    assert lines.index("WHO YOU ARE") < lines.index(next(l for l in lines if l.startswith("RELATIONSHIP"))) < at


def test_lately_alone_still_gets_a_durable_memory_slot():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[LATELY])
    lines = _format_grounding_self_block(capsule)
    assert lines[-3:] == ["DURABLE MEMORY / CONTINUITY", LATELY_HEAD, LATELY.text]


def test_no_lately_page_leaves_durable_memory_as_before():
    capsule = GroundingCapsuleV1(identity_summary=["I am Orion."], memory_digest="Recall: the kite.")
    lines = _format_grounding_self_block(capsule)
    assert lines[-2:] == ["DURABLE MEMORY / CONTINUITY", "Recall: the kite."]
    blank = LATELY.model_copy(update={"text": " "})
    assert _format_grounding_self_block(GroundingCapsuleV1(identity_summary=["I am Orion."], core_pages=[blank])) == [
        "WHO YOU ARE", "- I am Orion."]
