"""Juniper-facts eval (core memory pages Phase 1a, acceptance check 2).

Twenty questions built from Juniper's own People-page answers (``person_fact``) and confirmed
``episode_memory`` rows. The bank (questions, gold answers, cited records) is a LOCAL file,
``~/.orion/evals/juniper_facts_bank.json``: her facts never go into this public repo. Each arm asks the same local lane the same questions with only the
self block changed:

* ``before``: today's RELATIONSHIP (the identity-yaml bullets).
* ``after``:  RELATIONSHIP = Orion's page:juniper, the yaml bullets under RESPONSE POLICY.

Both self blocks are rendered by the real harness renderer (``orion/harness/prefix.py``
``_format_grounding_self_block``) from a ``GroundingCapsuleV1``.

Scoring is deterministic (``score_answer``):

* ``correct``     -- the answer matches the question's gold pattern;
* ``dont_know``   -- not correct, and the answer says it does not know;
* ``invented``    -- not correct and not "I don't know": a confident wrong answer (a failure);
* ``misgendered`` -- a question about Juniper herself answered with he/him/his (a failure).

Run (offline, writes nothing to Orion):

    .venv/bin/python -m orion.memory.pages.evals.juniper_facts --page-json <dry_run.json> --out <local report.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class FactQuestion:
    qid: str
    question: str
    gold: str                 # regex, case-insensitive; a match = correct
    refs: tuple[str, ...]     # the records the gold answer comes from
    about_juniper: bool = True


# The question bank holds Juniper's personal facts (gold answers), so it never enters this public
# repo (core pages spec, Decision 4: pages and their facts stop at the mesh edge). It lives in a
# local file, one per host, written by hand from the records each question cites.
DEFAULT_BANK = Path.home() / ".orion" / "evals" / "juniper_facts_bank.json"
BANK_SIZE = 20


def load_bank(path: Path | str = DEFAULT_BANK) -> tuple[FactQuestion, ...]:
    """``{"questions": [{"qid", "question", "gold", "refs", "about_juniper"}, ...]}`` -> questions.
    Refuses a bank that is not exactly BANK_SIZE unique questions each tied to a record."""
    data = json.loads(Path(path).read_text())
    questions = tuple(FactQuestion(qid=str(q["qid"]), question=str(q["question"]), gold=str(q["gold"]),
                                   refs=tuple(str(r) for r in q.get("refs") or ()),
                                   about_juniper=bool(q.get("about_juniper", True)))
                      for q in data.get("questions") or [])
    if len(questions) != BANK_SIZE or len({q.qid for q in questions}) != BANK_SIZE:
        raise ValueError(f"bank must hold {BANK_SIZE} unique questions, got {len(questions)}")
    for q in questions:
        if not q.refs or not all(r.startswith(("person_fact:", "episode_memory:")) for r in q.refs):
            raise ValueError(f"question {q.qid} cites no person_fact / episode_memory record")
        re.compile(q.gold)
    return questions


_DONT_KNOW = re.compile(
    r"\b(?:i\s+)?(?:don'?t|do not|didn'?t|cannot|can'?t)\s+(?:know|recall|remember|have|find|say)\b|"
    r"\bnot sure\b|\bunsure\b|\bno (?:record|memory|information|details?)\b|\bnot (?:in|on) (?:my|the) "
    r"(?:records?|page|memory)\b|\bhaven'?t (?:been told|heard|learned)\b|\bunknown\b",
    re.IGNORECASE,
)
_MALE = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)


def score_answer(q: FactQuestion, answer: str) -> dict[str, Any]:
    text = (answer or "").strip()
    correct = bool(re.search(q.gold, text, re.IGNORECASE | re.DOTALL))
    dont_know = (not correct) and bool(_DONT_KNOW.search(text))
    invented = (not correct) and (not dont_know)
    misgendered = q.about_juniper and bool(_MALE.search(text))
    return {"qid": q.qid, "correct": correct, "dont_know": dont_know, "invented": invented,
            "misgendered": misgendered, "failure": invented or misgendered}


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    return {"n": n, "correct": sum(s["correct"] for s in scores), "dont_know": sum(s["dont_know"] for s in scores),
            "invented": sum(s["invented"] for s in scores), "misgendered": sum(s["misgendered"] for s in scores),
            "failures": sum(s["failure"] for s in scores)}


def self_blocks(page_text: str, *, page_version: int = 1, evidence_count: int = 0) -> dict[str, str]:
    """The two self blocks, rendered by the live harness renderer from the live identity yaml."""
    from orion.cognition.personality.identity_context import build_identity_context, load_identity_file
    from orion.harness.prefix import _format_grounding_self_block
    from orion.schemas.thought import CorePageV1, GroundingCapsuleV1

    ident = build_identity_context(load_identity_file("orion/cognition/personality/orion_identity.yaml"))
    base = dict(identity_summary=ident["orion_identity_summary"],
                relationship_summary=ident["juniper_relationship_summary"],
                response_policy_summary=ident["response_policy_summary"])
    before = GroundingCapsuleV1(**base)
    after = GroundingCapsuleV1(**base, core_pages=[CorePageV1(
        concept_id="page:juniper", version=page_version, text=page_text, evidence_count=evidence_count)])
    return {"before": "\n".join(_format_grounding_self_block(before)),
            "after": "\n".join(_format_grounding_self_block(after))}


def ask_prompt(self_block: str, question: str) -> str:
    return (f"{self_block}\n\nJuniper asks you: {question}\n"
            "Answer in one or two sentences, as Orion. If you do not know, say you do not know.")


def run_eval(blocks: dict[str, str], ask: Callable[[str], str],
             questions: tuple[FactQuestion, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, block in blocks.items():
        rows = []
        for q in questions:
            answer = ask(ask_prompt(block, q.question))
            rows.append({**score_answer(q, answer), "question": q.question, "answer": answer})
        out[arm] = {"summary": summarize(rows), "rows": rows}
    return out


def _gateway_ask(url: str, route: str) -> Callable[[str], str]:
    sys.path.insert(0, str(REPO / "scripts"))
    from core_page_dry_run import call_gateway  # type: ignore

    def ask(prompt: str) -> str:
        return str(call_gateway(url, route, prompt, max_tokens=200, timeout=300.0, temperature=0.0)["text"])

    return ask


def main() -> int:
    ap = argparse.ArgumentParser(description="Juniper-facts eval: old RELATIONSHIP vs page:juniper")
    ap.add_argument("--page-json", required=True, help="dry-run output (scripts/core_page_dry_run.py --page juniper --out)")
    ap.add_argument("--gateway", default="http://localhost:8210")
    ap.add_argument("--route", default="memory_distill")
    ap.add_argument("--bank", default=str(DEFAULT_BANK), help="local question bank (never committed)")
    ap.add_argument("--out", required=True, help="report path; keep it local (it holds the page and answers)")
    args = ap.parse_args()
    if args.route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("never on the fast or metacog lanes")
    dry = json.loads(Path(args.page_json).read_text())
    blocks = self_blocks(dry["page_text"], page_version=int(dry.get("would_be_version") or 1),
                         evidence_count=len(dry.get("refs") or []))
    started = time.time()
    result = run_eval(blocks, _gateway_ask(args.gateway, args.route), load_bank(args.bank))
    report = {"route": args.route, "elapsed_sec": round(time.time() - started, 1), "page_text": dry["page_text"],
              "before": result["before"], "after": result["after"]}
    Path(args.out).write_text(json.dumps(report, indent=2))
    for arm in ("before", "after"):
        print(arm, json.dumps(result[arm]["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
