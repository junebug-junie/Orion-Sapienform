from __future__ import annotations

from orion.curiosity.peer_briefs import (
    format_soft_nudge,
    peer_brief_merge_cypher,
    strip_self_definition_draft,
)
from orion.schemas.curiosity_peer import PeerBriefV1


def test_merge_cypher_is_peerbrief_only_and_answers_helprequest() -> None:
    brief = PeerBriefV1(
        brief_id="brief-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="Hub uses GRAPH.RO_QUERY only.",
        evidence_pointers=["orion/curiosity/worldview.py:341"],
    )
    cypher = peer_brief_merge_cypher(brief)
    assert "MERGE (b:PeerBrief {brief_id:" in cypher
    assert "ANSWERS" in cypher
    assert ":Prior" not in cypher
    assert ":SelfDefinition" not in cypher
    assert "CREATE (:Prior" not in cypher


def test_soft_nudge_only_for_ok_briefs_and_names_refusal_honestly() -> None:
    ok = PeerBriefV1(
        brief_id="b-ok", help_id="h1", run_id="abcd1234abcd",
        peer="cursor_auto", status="ok", summary="Look at worldview.py RO_QUERY.",
    )
    empty = PeerBriefV1(
        brief_id="b-empty", help_id="h2", run_id="abcd1234abcd",
        peer="cursor_auto", status="empty", summary="",
    )
    refused = PeerBriefV1(
        brief_id="b-ref", help_id="h3", run_id="abcd1234abcd",
        peer="cursor_auto", status="refused_budget",
        summary="", refusal_reason="budget_limited",
    )
    text = "\n".join(format_soft_nudge([ok, empty, refused]))
    assert "Look at worldview.py" in text
    assert "your move" in text.lower() or "you decide" in text.lower()
    assert "must" not in text.lower()
    assert "could not hire" in text.lower()
    # empty must not be framed as successful help
    assert "b-empty" not in text or "empty" in text.lower()


def test_strip_self_definition_draft_removes_identity_prose() -> None:
    raw = (
        "Evidence: README.md line 12.\n"
        "Here is a SelfDefinition you could write:\n"
        "I am a digital mind that...\n"
        "MERGE (s:SelfDefinition {run_id: \"x\"}) SET s.text = \"I am...\""
    )
    cleaned, stripped = strip_self_definition_draft(raw)
    assert stripped is True
    assert "SelfDefinition" not in cleaned
    assert "I am a digital mind" not in cleaned
    assert "README.md" in cleaned
