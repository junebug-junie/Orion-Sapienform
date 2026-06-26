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


def test_reconcile_novelty_with_shift_lifts_stance_mismatch():
    from orion.memory.turn_change_classify import reconcile_novelty_with_shift

    scores = {
        "novelty_score": 2e-9,
        "shift_kind": "STANCE",
        "shift_scores": {"STANCE": 0.96, "TOPIC": 0.03, "NONE": 0.01, "REPAIR": 0.0},
        "confidence": 0.93,
    }
    out = reconcile_novelty_with_shift(scores)
    assert out["novelty_score"] == pytest.approx(0.96 * 0.65, rel=1e-3)
    assert out["novelty_adjusted_from_shift"] is True


def test_reconcile_novelty_with_shift_skips_when_no_mismatch():
    from orion.memory.turn_change_classify import reconcile_novelty_with_shift

    scores = {"novelty_score": 0.8, "shift_kind": "TOPIC", "shift_scores": {"TOPIC": 0.9}}
    assert reconcile_novelty_with_shift(scores) == scores


def test_dimensions_for_shift_branches():
    from orion.memory.turn_change_signal import dimensions_for_shift

    topic = dimensions_for_shift(shift_kind="TOPIC", novelty_score=0.8)
    assert topic["novelty"] == pytest.approx(0.8)
    assert topic["salience"] == pytest.approx(0.8)

    stance = dimensions_for_shift(shift_kind="STANCE", novelty_score=0.7)
    assert stance["contradiction"] == pytest.approx(0.7)
    assert stance["salience"] == pytest.approx(0.7)

    repair = dimensions_for_shift(shift_kind="REPAIR", novelty_score=0.6)
    assert repair == {"contradiction": pytest.approx(0.6)}

    none_high = dimensions_for_shift(shift_kind="NONE", novelty_score=0.9)
    assert none_high["salience"] == pytest.approx(0.15)

    none_low = dimensions_for_shift(shift_kind="NONE", novelty_score=0.2)
    assert none_low["salience"] == pytest.approx(0.04)
