from __future__ import annotations

from orion.curiosity.peer_briefs import (
    UNUSED_OK_BRIEFS_CYPHER,
    brief_ids_for_consume,
    format_soft_nudge,
    peer_brief_consume_cypher,
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
    cypher, params = peer_brief_merge_cypher(brief)
    assert "MERGE (b:PeerBrief {brief_id: $brief_id})" in cypher
    assert "ANSWERS" in cypher
    assert ":Prior" not in cypher
    assert ":SelfDefinition" not in cypher
    assert "CREATE (:Prior" not in cypher
    assert params["brief_id"] == "brief-1"
    assert params["evidence_pointers"] == ["orion/curiosity/worldview.py:341"]


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


def test_soft_nudge_claude_room_is_conversation_only_no_evidence_lines() -> None:
    claude = PeerBriefV1(
        brief_id="b-claude",
        help_id="h1",
        run_id="abcd1234abcd",
        peer="claude_room",
        status="ok",
        summary="[conversation-only; peer could not look at the repo] maybe check ACL",
        evidence_pointers=[],
    )
    text = "\n".join(format_soft_nudge([claude]))
    assert "conversation-only" in text.lower()
    assert "could not look at the repo" in text.lower()
    assert "evidence:" not in text.lower()


def test_consume_cypher_and_unused_filter() -> None:
    cypher, params = peer_brief_consume_cypher(["brief-a", "brief-b"])
    assert "SET b.consumed = true" in cypher
    assert params["brief_ids"] == ["brief-a", "brief-b"]
    assert "coalesce(b.consumed, false) = false" in UNUSED_OK_BRIEFS_CYPHER
    ids = brief_ids_for_consume(
        [
            PeerBriefV1(
                brief_id="keep",
                help_id="h",
                run_id="abcd1234abcd",
                peer="cursor_auto",
                status="ok",
                summary="x",
            ),
            PeerBriefV1(
                brief_id="skip-empty",
                help_id="h",
                run_id="abcd1234abcd",
                peer="cursor_auto",
                status="empty",
                summary="",
            ),
        ]
    )
    assert ids == ["keep"]


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
