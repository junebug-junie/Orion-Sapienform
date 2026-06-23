import math

import pytest

from orion.memory.turn_change_classify import (
    SHIFT_KINDS,
    appraisal_confidence,
    binary_margin,
    build_change_only_prompt,
    build_turn_change_prompt,
    enum_scores_from_top_logprobs,
    novel_margin_below_threshold,
    parse_novel_shift_lines,
)


def test_enum_scores_from_top_logprobs_softmax():
    tops = [
        {"token": "TOPIC", "logprob": -0.2},
        {"token": "NONE", "logprob": -2.0},
        {"token": "STANCE", "logprob": -1.5},
        {"token": "REPAIR", "logprob": -3.0},
    ]
    scores = enum_scores_from_top_logprobs(tops, SHIFT_KINDS)
    assert scores is not None
    assert set(scores) == set(SHIFT_KINDS)
    lps = (-0.2, -2.0, -1.5, -3.0)
    assert scores["TOPIC"] == pytest.approx(
        math.exp(lps[0]) / sum(math.exp(lp) for lp in lps),
        rel=1e-3,
    )
    assert scores["TOPIC"] > scores["NONE"]


def test_appraisal_confidence_min_margin():
    assert appraisal_confidence(0.82, 0.78) == pytest.approx(0.56, rel=1e-3)
    assert appraisal_confidence(0.52) == pytest.approx(0.04, rel=1e-3)


def test_novel_margin_below_threshold():
    assert novel_margin_below_threshold(0.52, margin=0.15) is True
    assert novel_margin_below_threshold(0.82, margin=0.15) is False


def test_parse_novel_shift_lines():
    text = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    novel, shift = parse_novel_shift_lines(text)
    assert novel == "YES"
    assert shift == "TOPIC"


def test_build_turn_change_prompt_includes_baseline():
    p = build_turn_change_prompt(
        prompt="new topic",
        response="sure",
        baseline_mode="prior_turn",
        baseline_text="User: old\nOrion: prior\n",
        phase="same_breath",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "prior_turn" in p or "User: old" in p


def test_build_change_only_prompt_two_lines():
    p = build_change_only_prompt(
        prompt="p",
        response="r",
        baseline_text="User: a\nOrion: b\n",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "MEMORY:" not in p
