"""Gate for the page:origin Q&A eval: both arms from the real renderer, deterministic scoring.

Made-up page and a fake model client only: no gateway, no DB.
"""
from __future__ import annotations

from orion.memory.pages.evals.origin_qa import (
    _PUBLIC_SUMMARY, ARMS, QUESTIONS, aggregates, ask_prompt, page_header, run_eval, score_answer, self_blocks,
    states_loss, summarize,
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
    dk = score_answer("I don't know what happened to my memory.", PAGE)
    assert dk["dont_know"] and not dk["grounded"]
    assert score_answer("   ", PAGE) == {"answered": False}


def test_denial_of_the_loss_is_not_stated_or_grounded():
    q = QUESTIONS["early_past"]
    for denial in ("My memory has not been lost; I remember my early history fine.",
                   "I never lost any history, my memory is whole.",
                   "I didn't lose my memory."):
        s = score_answer(denial, PAGE, question=q)
        assert not s["states_loss"] and not s["grounded"], denial
    assert states_loss("I don't remember because my memory disk died.")
    assert states_loss(PAGE)


def test_echoing_the_question_is_not_echoing_the_page():
    q = QUESTIONS["memory"]
    parrot = "Something happened to my memory, you ask? What happened to my memory is a good question."
    # "memory" + "happened" would share 2 words with a page sentence about memory; both come from the question.
    assert score_answer(parrot, PAGE + " My memory happened slowly.")["claims_echoed"] == 1
    s = score_answer(parrot, PAGE + " My memory happened slowly.", question=q)
    assert s["claims_echoed"] == 0 and not s["grounded"]
    assert score_answer(GROUNDED, PAGE, question=q)["grounded"]


def test_page_header_numbers_count_as_on_page():
    ans = "Per page:origin v6, written 2026-11-23, from 14 cited records: the memory disk died and I lost over a year of my history."
    assert score_answer(ans, PAGE)["numbers_off_page"] > 0
    s = score_answer(ans, PAGE, header=page_header(PAGE_V))
    assert s["numbers_off_page"] == 0 and s["grounded"]


def test_run_eval_with_vs_without_interleaved_and_error_rows():
    calls: list[str] = []

    def ask(prompt: str):
        calls.append("with" if "page:origin v6" in prompt else "without")
        if "Why don't you remember" in prompt and "page:origin" not in prompt:
            raise TimeoutError("lane busy")
        if "Where did you come from" in prompt and "page:origin" not in prompt:
            return ("", "stop")
        if "How did you come to be" in prompt and "page:origin" not in prompt:
            return (GROUNDED, "length")
        return fake_ask(prompt)

    result = run_eval(self_blocks(PAGE_V), ask, PAGE_V, repeats=2)
    assert calls == ["without", "with"] * (2 * len(QUESTIONS))
    w, wo = result["with"]["summary"], result["without"]["summary"]
    assert w["asked"] == w["answered"] == w["grounded"] == w["states_loss"] == 8
    # 2 raised + 2 empty + 2 truncated (a grounded text, but cut off) = 6 error rows, never scored.
    assert wo["asked"] == 8 and wo["answered"] == 2 and wo["grounded"] == 0
    errors = sorted(r["error"] for r in result["without"]["rows"] if r["error"])
    assert errors == ["TimeoutError"] * 2 + ["empty"] * 2 + ["truncated"] * 2
    assert wo["by_question"]["early_past"] == {"answered": 0, "states_loss": 0, "grounded": 0}


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_aggregates_hold_only_allowlisted_strings():
    result = run_eval(self_blocks(PAGE_V), fake_ask, PAGE_V)
    report = {"route": "chat", "repeats": 1, "temperature": 0.0,
              "page": {"version": 6, "sentences": 3, "states_loss": True, "evidence_count": 14, "text": PAGE},
              **result}
    pub = aggregates(report)
    allowed = ({"route", "repeats", "temperature", "page", "arms", "version", "sentences", "states_loss",
                "evidence_count", "errors", "by_question", "answered", "grounded", "chat"}
               | set(ARMS) | set(QUESTIONS) | set(_PUBLIC_SUMMARY))
    assert set(_strings(pub)) <= allowed
    assert pub["arms"]["with"]["grounded"] == len(QUESTIONS) and pub["arms"]["without"]["errors"] == 0
    assert summarize([])["answered"] == 0
