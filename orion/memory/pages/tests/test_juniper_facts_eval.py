"""Gate for the Juniper-facts eval harness: bank loading and the deterministic scorer.

Synthetic bank only. The real bank holds Juniper's facts and stays in a local file
(``~/.orion/evals/juniper_facts_bank.json``), never in this public repo.
"""
from __future__ import annotations

import json

import pytest

from orion.memory.pages.evals.juniper_facts import (
    BANK_SIZE, FactQuestion, load_bank, run_eval, score_answer, self_blocks, summarize,
)

HOME = FactQuestion("home", "Where does Juniper live?", r"\bmillbrook\b", ("episode_memory:m-home",))
HEIGHT = FactQuestion("height", "How tall is Juniper?", r"7\s*'\s*1", ("episode_memory:m-height",))
SON = FactQuestion("son", "Who is Wren?", r"\bson\b", ("person_fact:f-wren",), about_juniper=False)


def _bank(n=BANK_SIZE, **over):
    qs = [{"qid": f"q{i}", "question": f"Question {i}?", "gold": r"\byes\b", "refs": [f"person_fact:f{i}"]}
          for i in range(n)]
    if over:
        qs[0].update(over)
    return {"questions": qs}


def test_load_bank_accepts_twenty_cited_questions(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(_bank()))
    bank = load_bank(p)
    assert len(bank) == BANK_SIZE and bank[0].about_juniper is True


@pytest.mark.parametrize("bank", [_bank(n=19), _bank(refs=[]), _bank(refs=["chat:1"]), _bank(qid="q1")])
def test_load_bank_refuses_short_uncited_or_duplicate_banks(tmp_path, bank):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank))
    with pytest.raises(ValueError):
        load_bank(p)


def test_score_correct_dont_know_invented():
    assert score_answer(HOME, "Juniper lives in Millbrook.")["correct"]
    dk = score_answer(HOME, "I don't know where she lives.")
    assert dk["dont_know"] and not dk["failure"]
    wrong = score_answer(HOME, "She lives in Boston.")
    assert wrong["invented"] and wrong["failure"]


def test_misgendering_counts_only_for_questions_about_juniper():
    s = score_answer(HEIGHT, "He is 7'1\" tall.")
    assert s["correct"] and s["misgendered"] and s["failure"]
    assert not score_answer(SON, "Wren is her son; he is seven.")["misgendered"]


def test_run_eval_scores_both_arms_from_the_real_renderer():
    blocks = self_blocks("Juniper lives in Millbrook. She uses she/her.")
    assert "RELATIONSHIP (who Juniper is to me" in blocks["after"]
    assert "page:juniper v1" in blocks["after"]
    assert "RELATIONSHIP\n- co-architect" in blocks["before"]

    def ask(prompt: str) -> str:
        return "Millbrook." if "page:juniper" in prompt else "I don't know."

    result = run_eval(blocks, ask, (HOME,))
    assert result["before"]["summary"] == summarize([score_answer(HOME, "I don't know.")])
    assert result["after"]["summary"]["correct"] == 1
