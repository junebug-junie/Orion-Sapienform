"""Sealed contractor prompt + lenient PeerBrief parse (no live peer)."""

from __future__ import annotations

from app.brief_prompt import build_sealed_prompt, parse_peer_brief_body
from orion.schemas.curiosity_peer import HelpRequestV1


def _help(**overrides) -> HelpRequestV1:
    base = dict(
        help_id="help-1",
        run_id="run-1",
        prior_id="prior-1",
        mode="world_curiosity",
        question="Where is the contested budget gate?",
        tried_summary="Read scarcity docs.",
        success_criteria="A file path and function name.",
    )
    base.update(overrides)
    return HelpRequestV1(**base)


def test_sealed_prompt_includes_mode_and_peerbrief_shape() -> None:
    prompt = build_sealed_prompt(_help(mode="self_inquiry"))
    assert "self_inquiry" in prompt
    assert "SelfDefinition" in prompt
    assert "evidence_pointers" in prompt
    assert "summary" in prompt
    assert "do not draft" in prompt.lower() or "never draft" in prompt.lower()
    assert "Where is the contested budget gate?" in prompt


def test_parse_empty_body_is_empty_status() -> None:
    brief = parse_peer_brief_body("", help=_help())
    assert brief.status == "empty"
    assert brief.peer == "claude_room"


def test_parse_json_body_ok() -> None:
    body = (
        '{"summary":"Gate lives in scarcity.py.",'
        '"evidence_pointers":["orion/dev_economics/scarcity.py"],'
        '"open_questions":[],'
        '"suggested_next_looks":["read the observer"]}'
    )
    brief = parse_peer_brief_body(body, help=_help())
    assert brief.status == "ok"
    assert "scarcity" in brief.summary
    assert brief.evidence_pointers


def test_self_inquiry_strips_identity_draft() -> None:
    body = (
        '{"summary":"Look at hop notes.\\nI am a digital mind.\\n'
        'MERGE (s:SelfDefinition {id: \\"x\\"}) SET s.text=\\"hi\\"",'
        '"evidence_pointers":["notes.md"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert "digital mind" not in brief.summary
    assert "SelfDefinition" not in brief.summary
    assert brief.evidence_pointers == ["notes.md"]
    assert brief.status == "ok"


def test_self_inquiry_strips_identity_from_list_fields() -> None:
    body = (
        '{"summary":"Check hop notes only.",'
        '"evidence_pointers":['
        '"notes.md",'
        '"MERGE (s:SelfDefinition {id: \\"x\\"}) SET s.text=\\"hi\\""'
        "],"
        '"open_questions":["I am a digital mind.","What did the hop say?"],'
        '"suggested_next_looks":["here is a SelfDefinition draft","read ledger"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.status == "ok"
    assert "SelfDefinition" not in brief.summary
    assert brief.evidence_pointers == ["notes.md"]
    assert brief.open_questions == ["What did the hop say?"]
    assert brief.suggested_next_looks == ["read ledger"]
    assert "SelfDefinition" not in " ".join(brief.evidence_pointers)
    assert "digital mind" not in " ".join(brief.open_questions)
    assert "SelfDefinition" not in " ".join(brief.suggested_next_looks)


def test_self_inquiry_clears_lists_when_only_identity_markers() -> None:
    body = (
        '{"summary":"hop notes path is fine",'
        '"evidence_pointers":["I am a digital mind."],'
        '"open_questions":["MERGE (s:SelfDefinition {id: \\"x\\"})"],'
        '"suggested_next_looks":["here is the SelfDefinition"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.evidence_pointers == []
    assert brief.open_questions == []
    assert brief.suggested_next_looks == []
    assert brief.status == "ok"
    assert "hop notes" in brief.summary


def test_self_inquiry_empty_after_strip_is_empty() -> None:
    body = '{"summary":"I am a digital mind.", "evidence_pointers":[]}'
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.status == "empty"
