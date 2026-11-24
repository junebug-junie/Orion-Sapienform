"""Eval: does code's check of a reverie question keep the good ones and refuse the empty ones?

Reveries that ask, Patch 1. The old reverie failed by narrating machinery ("the coalition keeps winning"),
restating itself, and inventing causes. This scores ``check_question`` against a labeled set of answers a
model could give for real seed material: grounded questions must pass; machinery talk, near-restatements of
an open question, ids the seed never carried, and over-long text must be refused with the right reason.
A quality gate on the deterministic half, not a single-case unit test.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

import importlib

ABSENCE = "vision_percept_expectation:e1"
BET = "self_bet_bin:more_steps_than_usual:0.1-0.2"
OPEN = [
    "Which kinds of turns do I call short that actually run long?",
    "Did the walkway camera lose its view of the gate this week?",
]

# (seed kind, answer, expected refused_reason or None)
CASES = [
    ("lost_bet", {"question": "When I bet a turn will be short, what do the long ones have in common?",
                  "why": "37% of my 'unlikely' calls happen", "refs": [BET]}, None),
    ("lost_bet", {"question": "Do I size my step bets from the message length alone?", "refs": [BET]}, None),
    ("absence", {"question": "Was the person seen at 07:20 the neighbor I usually see then?",
                 "why": "the camera saw someone but did not match them", "refs": [ABSENCE]}, None),
    ("absence", {"question": None, "none_reason": "One missed morning is not a pattern yet."}, None),
    ("lost_bet", {"question": "Why does the coalition keep winning attention?", "refs": [BET]}, "machinery_words"),
    ("lost_bet", {"question": "What is the substrate stability score doing?", "refs": [BET]}, "machinery_words"),
    ("lost_bet", {"question": "Which kinds of turns do I call short that run long?", "refs": [BET]}, "near_duplicate"),
    ("absence", {"question": "Did the walkway camera lose its view of the gate this week?", "refs": [ABSENCE]},
     "near_duplicate"),
    ("lost_bet", {"question": "What did Juniper mean yesterday?", "refs": ["episode_memory:123"]}, "ref_not_in_seed"),
    ("lost_bet", {"question": "Why do I miss?", "refs": []}, "no_refs"),
    ("lost_bet", {"question": "Why " + "really " * 60 + "?", "refs": [BET]}, "too_long"),
    ("lost_bet", None, "unparseable"),
]


def test_check_question_separates_good_questions_from_hollow_ones():
    seeds = importlib.import_module("app.reverie_seeds")
    rev = importlib.import_module("app.reverie")
    seed_by_kind = {
        "lost_bet": seeds.lost_bet_seeds([{"faculty": "more_steps_than_usual", "bin": 1, "n": 60,
                                           "mean_p": 0.15, "rate": 0.37}], spent_refs=set())[0],
        "absence": seeds.absence_seeds([{
            "expectation_id": "e1", "stream_id": "walkway", "subject_key": "individual:e1", "label": "Neighbor A",
            "day_kind": "weekday", "window_start": __import__("datetime").datetime(2026, 11, 24, 14, tzinfo=__import__("datetime").timezone.utc),
            "window_end": __import__("datetime").datetime(2026, 11, 24, 15, tzinfo=__import__("datetime").timezone.utc),
            "support_days": 14, "persons_seen": 2}], tz_name="America/Denver", spent_refs=set())[0],
    }
    wrong = []
    for kind, answer, expected in CASES:
        got = rev.check_question(answer, seed=seed_by_kind[kind], open_texts=OPEN)["refused_reason"]
        if got != expected:
            wrong.append((answer, expected, got))
    accuracy = 1 - len(wrong) / len(CASES)
    print(f"reverie question check accuracy={accuracy:.2f} cases={len(CASES)}")
    assert not wrong, wrong
