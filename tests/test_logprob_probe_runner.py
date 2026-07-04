from __future__ import annotations

from orion.substrate.appraisal.probe.logprob_runner import (
    parse_yes_no_lines,
    score_binary_logprob,
    score_kind_from_answer_token,
)


def test_parse_yes_no_lines() -> None:
    text = """
specificity_demand: YES
trust_rupture: NO
coherence_gap: yes
"""
    parsed = parse_yes_no_lines(text)
    assert parsed["specificity_demand"] == "YES"
    assert parsed["trust_rupture"] == "NO"
    assert parsed["coherence_gap"] == "YES"


def test_score_binary_logprob_sigmoid() -> None:
    score = score_binary_logprob(logprob_yes=-0.12, logprob_no=-2.4)
    assert 0.85 < score < 0.99


def test_score_kind_from_answer_token_uses_margin_as_confidence() -> None:
    entry = {
        "token": "YES",
        "logprob": -0.12,
        "top_logprobs": [
            {"token": "YES", "logprob": -0.12},
            {"token": "NO", "logprob": -2.4},
        ],
    }
    scored = score_kind_from_answer_token("specificity_demand", entry)
    assert scored is not None
    assert scored.evidence_kind == "specificity_demand"
    assert scored.score > 0.8
    assert scored.confidence == 2.28
    assert scored.features["logprob_yes"] == -0.12
