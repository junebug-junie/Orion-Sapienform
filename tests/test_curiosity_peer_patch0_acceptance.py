"""Patch 0 acceptance gates for contractor peer (fixture-driven).

Maps spec acceptance checks provable without live hire:
  2 — refused_budget soft-nudge wording
  6 — ok brief nudges; empty brief not framed as success
  7 — self-inquiry forbids / strips identity drafting
  8 — feature flag off omits HelpRequest teach
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.curiosity.peer_briefs import format_soft_nudge, strip_self_definition_draft
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1


def _empty_material() -> StudyMaterial:
    return StudyMaterial(generated_at=datetime(2026, 9, 14, tzinfo=timezone.utc))


def test_flag_off_omits_help_request_teach() -> None:
    """Acceptance 8: flag off → today's curiosity path, no hire teach."""
    material = _empty_material()
    kickoff = build_kickoff_prompt(
        material,
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    self_inquiry = build_self_inquiry_prompt(
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    assert "HelpRequest" not in kickoff
    assert "HelpRequest" not in self_inquiry
    assert "MERGE (h:HelpRequest" not in kickoff
    assert "MERGE (h:HelpRequest" not in self_inquiry


def test_empty_brief_does_not_nudge_as_success() -> None:
    """Acceptance 6: empty brief must not appear as successful peer help."""
    empty = PeerBriefV1(
        brief_id="brief-empty",
        help_id="help-empty",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="empty",
        summary="",
    )
    lines = format_soft_nudge([empty])
    text = "\n".join(lines).lower()
    assert "peer looked" not in text
    assert "brief-empty" not in text
    assert "you decide" not in text

    ok = PeerBriefV1(
        brief_id="brief-ok",
        help_id="help-ok",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="Check worldview RO_QUERY gate.",
    )
    ok_text = "\n".join(format_soft_nudge([ok])).lower()
    assert "peer looked" in ok_text
    assert "check worldview ro_query gate" in ok_text


def test_refused_budget_nudge_says_could_not_hire() -> None:
    """Acceptance 2: budget refusal is honest, not a silent skip."""
    refused = PeerBriefV1(
        brief_id="brief-refused",
        help_id="help-refused",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="refused_budget",
        summary="",
        refusal_reason="budget_limited",
    )
    text = "\n".join(format_soft_nudge([refused]))
    lower = text.lower()
    assert "could not hire" in lower
    assert "refused_budget=1" in lower
    assert "continue alone" in lower
    assert "must incorporate" not in lower


def test_persist_never_emits_prior_merge() -> None:
    """PeerBrief dual-write stays belief-neutral — no :Prior MERGE."""
    brief = PeerBriefV1(
        brief_id="brief-persist-gate",
        help_id="help-persist",
        run_id="abcd1234abcd",
        prior_id="prior-123",
        peer="cursor_auto",
        status="ok",
        summary="Fixture persist path only.",
        evidence_pointers=["orion/curiosity/peer_brief_persist.py"],
    )
    graphs: list[tuple[str, dict | None]] = []
    buses: list[tuple[str, object]] = []

    def graph_execute(cypher: str, params: dict | None = None) -> None:
        graphs.append((cypher, params))

    result = persist_peer_brief(
        brief=brief,
        graph_execute=graph_execute,
        bus_publish=lambda channel, payload: buses.append((channel, payload)),
    )
    assert result == {"graph_ok": True, "bus_ok": True}
    assert len(graphs) == 1
    cypher, params = graphs[0]
    assert "MERGE (b:PeerBrief" in cypher
    assert params is not None and params["brief_id"] == "brief-persist-gate"
    assert ":Prior" not in cypher
    assert "CREATE (:Prior" not in cypher
    assert "MERGE (p:Prior" not in cypher
    assert buses[0][0] == PEER_BRIEF_CHANNEL


def test_self_inquiry_strip_rejects_identity_draft() -> None:
    """Acceptance 7: peer must not draft :SelfDefinition; strip if emitted."""
    prompt = build_self_inquiry_prompt(
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    lower = prompt.lower()
    assert "helprequest" in lower
    assert "never draft" in lower or "must not draft" in lower
    assert "selfdefinition" in lower

    raw = (
        "Evidence: services/orion-hub/app/settings.py.\n"
        "Here is a SelfDefinition you could write:\n"
        "I am a digital mind that traces its own loops.\n"
        'MERGE (s:SelfDefinition {run_id: "x"}) SET s.text = "I am..."'
    )
    cleaned, stripped = strip_self_definition_draft(raw)
    assert stripped is True
    assert "selfdefinition" not in cleaned.lower()
    assert "i am a digital mind" not in cleaned.lower()
    assert "settings.py" in cleaned
