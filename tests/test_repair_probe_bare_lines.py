"""The repair probe reads a bare seven-line YES/NO reply (2026-11-13).

The probe prompt says "Reply with exactly seven lines, one per kind, YES or NO only". On held seats
Bonsai-2-27B (agent-gpu2) and Strata flash-next (agent) sometimes obey that literally and drop the
`kind:` label: `NO\\nNO\\nNO\\nNO\\nNO\\nNO\\nNO` (14 tokens). Four held checks on 11-12 failed as
`probe_unparseable` that way. Line i is kind i of the prompt's list. Anything else still fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import (
    _EVIDENCE_KINDS,
    RepairPressureV2Paradigm,
    RepairProbeUnavailable,
    _evidence_from_probe_response,
    build_repair_probe_prompt,
    parse_probe_answers,
    probe_answer_spans,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "repair_probe_replies"


def _req() -> PreTurnAppraisalRequestV1:
    return PreTurnAppraisalRequestV1(correlation_id="c-bare", session_id="s1",
                                     turn_window=[TurnWindowMessageV1(role="user", content="field note")])


def _token(tok: str, lp: float, tops: list[tuple[str, float]]) -> dict:
    return {"token": tok, "logprob": lp, "top_logprobs": [{"token": t, "logprob": v} for t, v in tops]}


def test_prompt_order_is_the_bare_line_order() -> None:
    prompt = build_repair_probe_prompt([TurnWindowMessageV1(role="user", content="x")])
    listed = [line.split(":")[0] for line in prompt.splitlines() if line.endswith(": YES or NO")]
    assert tuple(listed) == _EVIDENCE_KINDS


def test_bare_seven_lines_map_to_kinds_in_prompt_order() -> None:
    answers = ["NO", "YES", "NO", "NO", "NO", "NO", "YES"]
    assert parse_probe_answers("\n".join(answers)) == dict(zip(_EVIDENCE_KINDS, answers))


def test_bare_lines_allow_bullets_numbers_and_blank_lines() -> None:
    text = "\n- no\n* NO\n\n- YES\n* NO.\nNO\nNO \n  NO\n"
    assert list(parse_probe_answers(text).values()) == ["NO", "NO", "YES", "NO", "NO", "NO", "NO"]
    numbered = "\n".join(f"{i}. {a}" for i, a in enumerate(["NO", "NO", "YES", "NO", "NO", "NO", "NO"], 1))
    assert list(parse_probe_answers(numbered).values()) == ["NO", "NO", "YES", "NO", "NO", "NO", "NO"]


@pytest.mark.parametrize("text", [
    "NO\nNO\nNO\nNO\nNO\nNO",                    # six lines: which kind is missing is unknowable
    "NO\nNO\nNO\nNO\nNO\nNO\nNO\nNO",            # eight lines
    "NO\nNO\nNO\nmaybe\nNO\nNO\nNO",             # a non-answer line
    "No, nothing here needs repair.",            # prose
    "I'd be happy to help!",
    "3. YES\n1. NO\n2. NO\n4. NO\n5. NO\n6. NO\n7. NO",   # reordered numbers: mapping unknowable
    "1. NO\n2. NO\nNO\nNO\nNO\nNO\nNO",                  # some numbered, some not
    "<think>\ntrust_rupture: YES",                            # reasoning cut off, never closed
    "<think>\n" + "\n".join(["NO"] * 7),
])
def test_anything_else_answers_nothing(text) -> None:
    assert parse_probe_answers(text) == {}


def test_labeled_lines_still_win_and_keep_their_names() -> None:
    text = "trust_rupture: YES\nspecificity_demand: NO"
    assert parse_probe_answers(text) == {"trust_rupture": "YES", "specificity_demand": "NO"}


def test_reasoning_before_think_close_is_not_an_answer() -> None:
    text = "<think>\ntrust_rupture: YES\n</think>\n" + "\n".join(["NO"] * 7)
    assert parse_probe_answers(text) == dict.fromkeys(_EVIDENCE_KINDS, "NO")


def test_bare_spans_point_at_the_answer_characters() -> None:
    text = "NO\n\n- YES\nNO\nNO\nNO\nNO\nNO"
    for _kind, answer, at in probe_answer_spans(text):
        assert text[at:at + len(answer)] == answer


def test_bare_reply_with_logprobs_scores_every_kind_from_its_own_token() -> None:
    answers = ["NO", "YES", "NO", "NO", "NO", "NO", "NO"]
    content = []
    for i, a in enumerate(answers):
        other = "YES" if a == "NO" else "NO"
        content.append(_token(a, -0.1, [(a, -0.1), (other, -1.0 - i * 0.5), ("The", -6.0)]))
        content.append(_token("\n", -0.01, [("\n", -0.01), (" ", -5.0)]))
    payload = {"text": "\n".join(answers), "llm_uncertainty": {"available": True},
               "raw": {"choices": [{"logprobs": {"content": content}}]}}
    evidence = _evidence_from_probe_response(payload)
    assert [e.evidence_kind for e in evidence] == list(_EVIDENCE_KINDS)
    assert all(e.detector == "logprob_probe_v2" for e in evidence)
    scores = [round(e.score, 4) for e in evidence]
    assert scores[1] > 0.5 and all(s < 0.5 for i, s in enumerate(scores) if i != 1)
    assert len(set(scores)) > 2   # real per-kind margins, not one fixed fallback value


@pytest.mark.asyncio
async def test_bare_reply_no_longer_fails_the_check() -> None:
    paradigm = RepairPressureV2Paradigm(
        llm_caller=lambda _p: {"text": "\n".join(["NO"] * 7), "llm_uncertainty": {"available": False}})
    slice_ = await paradigm.run(_req())
    assert set(slice_.dimensions) == set(_EVIDENCE_KINDS)


@pytest.mark.asyncio
async def test_six_bare_lines_still_fail_the_check() -> None:
    paradigm = RepairPressureV2Paradigm(
        llm_caller=lambda _p: {"text": "\n".join(["NO"] * 6), "llm_uncertainty": {"available": False}})
    with pytest.raises(RepairProbeUnavailable) as exc:
        await paradigm.run(_req())
    assert exc.value.reason == "probe_unparseable"


def _fixture_payloads() -> list[tuple[str, dict]]:
    return [(p.name, json.loads(p.read_text())) for p in sorted(_FIXTURES.glob("*.json"))]


def test_captured_fixtures_exist() -> None:
    assert _fixture_payloads(), f"no captured replies under {_FIXTURES}"


@pytest.mark.parametrize("name,fixture", _fixture_payloads(), ids=lambda v: v if isinstance(v, str) else "")
def test_captured_replies_score_from_real_logprobs(name, fixture) -> None:
    """Gate: every captured live reply (gateway bus path, probe options) parses and is scored from
    its own answer tokens' logprobs, never the fixed text fallback."""
    evidence = _evidence_from_probe_response(fixture["probe_payload"])
    assert [e.evidence_kind for e in evidence] == list(_EVIDENCE_KINDS), name
    assert all(e.detector == "logprob_probe_v2" for e in evidence), name
    assert all(0.0 < e.score < 1.0 for e in evidence), name


def test_captured_replies_cover_the_bare_shape() -> None:
    texts = [f["probe_payload"]["text"] for _n, f in _fixture_payloads()]
    assert any(":" not in t for t in texts), "no captured bare-line reply"
    assert any(":" in t for t in texts), "no captured labeled reply"
