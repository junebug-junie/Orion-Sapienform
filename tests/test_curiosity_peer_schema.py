from __future__ import annotations

from orion.schemas.curiosity_peer import (
    HELP_REQUEST_CHANNEL,
    HELP_REQUEST_KIND,
    PEER_BRIEF_CHANNEL,
    PEER_BRIEF_KIND,
    HelpRequestV1,
    PeerBriefV1,
)


def test_help_request_defaults_and_forbids_extra() -> None:
    row = HelpRequestV1(
        help_id="help-1",
        run_id="abcd1234abcd",
        mode="self_inquiry",
        question="What evidence do I have that I form priors under pressure?",
        tried_summary="Read self_inquiry.py and two hop notes.",
        success_criteria="Pointers to concrete files/tables that support or refute.",
    )
    assert row.schema_version == HELP_REQUEST_KIND
    assert HELP_REQUEST_CHANNEL == "orion:curiosity:help:request"
    dumped = row.model_dump()
    assert dumped["prior_id"] is None
    try:
        HelpRequestV1(
            help_id="h",
            run_id="abcd1234abcd",
            mode="world_curiosity",
            question="q",
            tried_summary="t",
            success_criteria="s",
            unexpected=1,
        )
        assert False, "extra fields must be forbidden"
    except Exception:
        pass


def test_peer_brief_status_literals_and_bounds() -> None:
    brief = PeerBriefV1(
        brief_id="brief-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="x" * 9000,  # must clip
        evidence_pointers=["orion/curiosity/kickoff_prompt.py:528"],
        open_questions=["Does Hub ever write worldview?"],
        suggested_next_looks=["GRAPH.RO_QUERY orion_worldview MATCH (n:Prior) RETURN count(n)"],
    )
    assert brief.schema_version == PEER_BRIEF_KIND
    assert PEER_BRIEF_CHANNEL == "orion:curiosity:peer:brief"
    assert len(brief.summary) <= 4000
    assert brief.peer in ("cursor_auto", "claude_room")
    for status in ("ok", "failed", "refused_budget", "empty"):
        PeerBriefV1(
            brief_id=f"b-{status}",
            help_id="help-1",
            run_id="abcd1234abcd",
            peer="claude_room",
            status=status,
            summary="s",
        )
