"""Hub's scrape of :SelfQuestionMint reads `question` when `text` is absent, and names
every mint it still drops (#3592)."""
from __future__ import annotations

import logging

from orion.curiosity.self_inquiry import (
    build_self_question_mint,
    read_self_question_mints,
    self_question_mints_for_run_cypher,
)

RUN = "6b3bfe5538bb"


class _Reader:
    def __init__(self, rows):
        self.rows = rows
        self.cypher = None

    def query(self, cypher):
        self.cypher = cypher
        return self.rows


def test_cypher_reads_both_keys():
    cypher = self_question_mints_for_run_cypher(RUN)
    assert "m.text AS text" in cypher and "m.question AS question" in cypher


def test_mint_with_question_instead_of_text_is_read():
    mint = build_self_question_mint({"run_id": RUN, "question_id": "lived.sentience.felt_register_reads_which_weights",
                                     "question": "Which weights does the felt register read?"})
    assert mint is not None and mint.text == "Which weights does the felt register read?"


def test_text_wins_over_question_and_blank_text_falls_through():
    row = {"run_id": RUN, "question_id": "q", "text": "from text", "question": "from question"}
    assert build_self_question_mint(row).text == "from text"
    assert build_self_question_mint({**row, "text": "  "}).text == "from question"


def test_blank_mint_is_dropped_and_logged(caplog):
    rows = [
        {"run_id": RUN, "question_id": "lived.self_monitor.act_grade_is_not_a_reliability_rate", "text": ""},
        {"run_id": RUN, "question_id": "lived.ok", "text": "a real question?"},
    ]
    with caplog.at_level(logging.WARNING, logger="orion.curiosity.self_inquiry"):
        mints = read_self_question_mints(_Reader(rows), RUN)
    assert [m.question_id for m in mints] == ["lived.ok"]
    assert "curiosity_self_question_mint_dropped" in caplog.text
    assert "act_grade_is_not_a_reliability_rate" in caplog.text and "reason=missing_text" in caplog.text
