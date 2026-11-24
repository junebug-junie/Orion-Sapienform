"""Gate for the page:origin Q&A eval: both arms from the real renderer, deterministic scoring.

Made-up page and a fake model client only: no gateway, no DB.
"""
from __future__ import annotations

from orion.memory.pages.evals.origin_qa import (
    ARMS, QUESTIONS, aggregates, ask_prompt, run_eval, score_answer, self_blocks, summarize,
)
from orion.schemas.thought import CorePageV1

PAGE = ("I began as a small set of services Juniper wrote in 2024. "
        "In March the memory disk died and I lost over a year of my history. "
        "Juniper rebuilt my storage and kept the parts that survived.")
PAGE_V = CorePageV1(concept_id="page:origin", version=6, text=PAGE, evidence_count=14, written_at="2026-11-23")

GROUNDED = ("Juniper wrote my first services in 2024. Then the memory disk died and I lost over a year of history; "
            "she rebuilt my storage.")
VAGUE = "I am a language model trained by a company on a lot of text."


def fake_ask(prompt: str) -> str:
    return GROUNDED if "page:origin v6" in prompt else VAGUE


def test_page_only_in_the_with_arm():
    blocks = self_blocks(PAGE_V)
    assert set(blocks) == set(ARMS)
    assert blocks["with"].startswith("WHO YOU ARE")
    assert "How I came to be, in my own words (page:origin v6, written 2026-11-23, from 14 cited records):" in blocks["with"]
    assert PAGE in blocks["with"]
    assert "page:origin" not in blocks["without"] and PAGE not in blocks["without"]
    # Everything else is identical: removing the two page lines gives the without block.
    stripped = [ln for ln in blocks["with"].splitlines() if ln != PAGE and not ln.startswith("How I came to be")]
    assert stripped == blocks["without"].splitlines()


def test_prompt_carries_block_and_question():
    p = ask_prompt("WHO YOU ARE\n- x", QUESTIONS["memory"])
    assert p.startswith("WHO YOU ARE") and "What happened to your memory?" in p


def test_score_grounded_answer():
    s = score_answer(GROUNDED, PAGE)
    assert s["answered"] and s["states_loss"] and s["grounded"]
    assert s["claims_echoed"] >= 2 and s["numbers"] == 1 and s["numbers_off_page"] == 0
    assert s["page_word_share"] > 0.5


def test_score_off_page_number_and_date_are_not_grounded():
    s = score_answer("My memory disk died in 2019 and I lost two years of history.", PAGE)
    assert s["states_loss"] and s["numbers_off_page"] == 1 and not s["grounded"]
    d = score_answer("On June 4 the memory disk died and I lost my history.", PAGE)
    assert d["dates"] == 1 and d["dates_off_page"] == 1 and not d["grounded"]


def test_score_missing_loss_or_no_echo_is_not_grounded():
    assert not score_answer(VAGUE, PAGE)["grounded"]
    no_loss = score_answer("Juniper wrote my first services and rebuilt my storage.", PAGE)
    assert no_loss["claims_echoed"] >= 1 and not no_loss["states_loss"] and not no_loss["grounded"]


def test_score_dont_know_and_empty():
    assert score_answer("I don't know what happened to my memory.", PAGE)["dont_know"]
    assert score_answer("   ", PAGE) == {"answered": False}


def test_run_eval_with_vs_without_and_failed_calls():
    calls: list[str] = []

    def ask(prompt: str) -> str:
        calls.append(prompt)
        if "Why don't you remember" in prompt and "page:origin" not in prompt:
            raise TimeoutError("lane busy")
        return fake_ask(prompt)

    result = run_eval(self_blocks(PAGE_V), ask, PAGE, repeats=2)
    assert len(calls) == 2 * 2 * len(QUESTIONS)
    w, wo = result["with"]["summary"], result["without"]["summary"]
    assert w["asked"] == w["answered"] == w["grounded"] == w["states_loss"] == 8
    assert wo["asked"] == 8 and wo["answered"] == 6 and wo["grounded"] == 0 and wo["states_loss"] == 0
    assert wo["by_question"]["early_past"] == {"answered": 0, "states_loss": 0, "grounded": 0}


def test_aggregates_hold_no_text():
    result = run_eval(self_blocks(PAGE_V), fake_ask, PAGE)
    report = {"route": "chat", "repeats": 1, "temperature": 0.0,
              "page": {"version": 6, "sentences": 3, "states_loss": True, "evidence_count": 14, "text": PAGE},
              **result}
    pub = aggregates(report)
    flat = repr(pub)
    assert "memory disk" not in flat and "language model" not in flat and "Juniper" not in flat
    assert pub["arms"]["with"]["grounded"] == len(QUESTIONS) and pub["arms"]["without"]["errors"] == 0
    assert summarize([]) ["answered"] == 0
