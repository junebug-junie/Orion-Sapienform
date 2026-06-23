import math

import pytest

from orion.memory.consolidation_classify import (
    binary_score_from_top_logprobs,
    build_classify_prompt,
    parse_classify_lines,
)


def test_binary_score_from_top_logprobs_yes_wins():
    tops = [{"token": "YES", "logprob": -0.2}, {"token": "NO", "logprob": -2.0}]
    score = binary_score_from_top_logprobs(tops)
    assert score == pytest.approx(math.exp(-0.2) / (math.exp(-0.2) + math.exp(-2.0)), rel=1e-3)


def test_parse_classify_lines():
    text = "MEMORY: YES\nBOUNDARY: NO\n"
    mem, bnd = parse_classify_lines(text)
    assert mem == "YES"
    assert bnd == "NO"


def test_build_classify_prompt_includes_phase():
    p = build_classify_prompt(
        prompt="hi",
        response="hello",
        spark_meta={"conversation_phase": {"phase_change": "same_breath"}},
        baseline_text="",
    )
    assert "same_breath" in p
    assert "NOVEL:" in p


def test_build_classify_prompt_four_lines_no_phi():
    p = build_classify_prompt(
        prompt="hi",
        response="hello",
        spark_meta={"conversation_phase": {"phase_change": "same_breath"}},
        baseline_mode="prior_turn",
        baseline_text="User: earlier\nOrion: reply\n",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "MEMORY:" in p
    assert "BOUNDARY:" in p
    assert "phi_after" not in p
    assert "novelty=" not in p
