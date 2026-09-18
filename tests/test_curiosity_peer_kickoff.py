from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.schemas.curiosity_peer import PeerBriefV1


def _empty_material() -> StudyMaterial:
    return StudyMaterial(generated_at=datetime(2026, 9, 14, tzinfo=timezone.utc))


def test_help_request_cypher_taught_only_when_flag_on_and_writable() -> None:
    material = _empty_material()
    off = build_kickoff_prompt(
        material, run_id="abcd1234abcd", graph_enabled=True, contractor_peer_enabled=False
    )
    assert "HelpRequest" not in off
    on = build_kickoff_prompt(
        material, run_id="abcd1234abcd", graph_enabled=True, contractor_peer_enabled=True
    )
    assert "MERGE (h:HelpRequest" in on
    assert "success_criteria" in on
    assert "do not write :peerbrief" in on.lower()


def test_role_split_taught_when_peer_enabled() -> None:
    on = build_kickoff_prompt(
        _empty_material(),
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    assert "InvestigationRole" in on
    assert "local_crawl" in on
    assert "hire_cursor" in on
    assert "genuinely stuck" not in on.lower()
    assert "not as a default" not in on.lower()
    assert "MERGE (h:HelpRequest" in on
    assert "does not wake" in on.lower() or "not enqueue" in on.lower() or "after" in on.lower()


def test_role_teach_omitted_when_peer_flag_off() -> None:
    off = build_kickoff_prompt(
        _empty_material(),
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    assert "InvestigationRole" not in off
    assert "HelpRequest" not in off


def test_soft_nudge_injected_for_ok_brief() -> None:
    material = _empty_material()
    brief = PeerBriefV1(
        brief_id="brief-9",
        help_id="help-9",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="Check acl.py RO_QUERY.",
    )
    text = build_kickoff_prompt(
        material,
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
        peer_briefs=(brief,),
    )
    assert "Check acl.py RO_QUERY" in text
    assert "must incorporate" not in text.lower()


def test_self_inquiry_forbids_peer_drafting_selfdefinition() -> None:
    text = build_self_inquiry_prompt(
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    assert "HelpRequest" in text
    assert "InvestigationRole" in text
    assert "local_crawl" in text
    assert "hire_cursor" in text
    assert "genuinely stuck" not in text.lower()
    assert "never draft" in text.lower() or "must not draft" in text.lower()
    assert "SelfDefinition" in text  # Orion still writes their own
