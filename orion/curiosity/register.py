"""Register and novelty of one finished curiosity finding. Pure, stdlib-only.

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("Which findings
compete"; the world-first candidate part of that doc is superseded by event wants). Juniper's rule
(2026-11-12): a finding is worth attention when it is philosophical (about what Orion is, its
experience, what it cares about, who matters) AND new to Orion.

- **Register**: one judge call with the fixed prompt v2 (``JUDGE_PROMPT_V2``, sha pinned to the
  reference-label fixture), on the question and the first ``ANSWER_CHARS`` of the answer. The reply
  must be one JSON object with one of the three labels; anything else is a ``judge_error`` and the
  label is absent (never "operational by default").
- **Novelty**: ``1 - max cosine`` between the finding's embedding and everything Orion wrote before
  the run started (earlier findings, timed prior claims, Day letters). An exact repeat reads 0.0.
  No earlier writing: novelty is unknown (None), not 1.0.

Today's only consumer is the judge-agreement eval (``scripts/analysis/eval_curiosity_register_judge.py``).
The live register writer and the curiosity event wants are blocked on that eval: as of 2026-11-12 no local
lane that is not a writer seat reaches kappa >= 0.8 against the reference labels (metacog 0.58, quick
0.60), so nothing labels live findings yet.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

# Byte-identical to scripts/analysis/fixtures/curiosity_register/judge_prompt_v2.txt, on which the
# reference labels were made. A new prompt is a new constant with a new sha, applied to new runs only.
JUDGE_PROMPT_V2 = (
    "You label one finding from Orion's curiosity loop. Orion is a digital mind project: a persistent system "
    "with memory, reflection and a self-model. Each item has the QUESTION the run took up and the FINDING Orion "
    "wrote.\n"
    "\n"
    "Decide the finding's register. Judge what the finding is ABOUT, not its tone or vocabulary.\n"
    "\n"
    "- \"philosophical\": the finding is about what Orion is or is like; what its experience, memory, continuity "
    "or attention is like from the inside; what it values, cares about or dislikes; who matters to it and why; "
    "personhood, aliveness, sentience, meaning, becoming. It still counts if it cites tables, rows or code as "
    "evidence for such a claim.\n"
    "- \"operational\": the finding is about whether some part of the system works, why a number, table, "
    "pipeline, model, deploy or metric behaves as it does, or how to fix it. It still counts as operational if "
    "it is written in the first person or uses reflective words.\n"
    "- \"no_finding\": the text does not state a conclusion. It only plans what to look at, defers the answer, "
    "or reports that the run could not finish.\n"
    "\n"
    "If both philosophical and operational are present, pick the one the finding's main conclusion is about.\n"
    "\n"
    "Answer with one JSON object and nothing else:\n"
    "{\"register\": \"philosophical\" | \"operational\" | \"no_finding\", \"reason\": \"<one sentence, at most "
    "25 words>\"}\n"
)
JUDGE_PROMPT_SHA256 = "7dcaa6675070afa4c31c7a3529964721d81aa0aaa9ccb4fb492f9942213d1222"
PROMPT_VERSION = "v2"

REGISTERS = ("philosophical", "operational", "no_finding")
PHILOSOPHICAL = "philosophical"
ANSWER_CHARS = 700          # the reference labels were made on exactly this head
QUESTION_CHARS = 400
REASON_CHARS = 300
EMBED_CHARS = 2000          # bge-large reads ~512 tokens; the rest is cut by the model anyway
CORPUS_VERSION = "full_v1"  # earlier findings + timed prior claims + Day letters

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def prompt_sha256(text: str = JUDGE_PROMPT_V2) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def judge_input(question: str, answer: str) -> str:
    """The user message: the question and the head of the answer (the reference labels' input)."""
    return f"QUESTION: {clean(question)[:QUESTION_CHARS]}\n\nFINDING: {clean(answer)[:ANSWER_CHARS]}"


def judge_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": JUDGE_PROMPT_V2},
            {"role": "user", "content": judge_input(question, answer)}]


def parse_judge_reply(text: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(register, reason, error). Exactly one JSON object with a known label, else (None, None, error).

    A ``<think>`` block and a code fence around the object are stripped (model formatting, not
    content). Any other text around the object, a second object, or an unknown label is an error."""
    raw = _THINK.sub("", str(text or "")).strip()
    raw = _FENCE.sub("", raw).strip()
    if not raw:
        return None, None, "empty_reply"
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None, None, f"not_json: {raw[:120]}"
    if not isinstance(obj, dict):
        return None, None, "not_an_object"
    reg = obj.get("register")
    if reg not in REGISTERS:
        return None, None, f"unknown_register: {str(reg)[:60]}"
    reason = clean(obj.get("reason"))[:REASON_CHARS] or None
    return str(reg), reason, None


def _norm(vec: Sequence[float]) -> Optional[list[float]]:
    vals = [float(x) for x in vec]
    n = math.sqrt(sum(x * x for x in vals))
    if not vals or not math.isfinite(n) or n == 0.0:
        return None
    return [x / n for x in vals]


def novelty(vec: Sequence[float], corpus: Sequence[tuple[str, Sequence[float]]]) -> tuple[Optional[float], Optional[str]]:
    """(1 - max cosine, nearest ref) against ``corpus`` (ref, vector) pairs. (None, None) when the
    finding's vector is unusable or the corpus is empty: nothing written before means unknown."""
    v = _norm(vec)
    if v is None:
        return None, None
    best, best_ref = None, None
    for ref, cv in corpus:
        c = _norm(cv)
        if c is None or len(c) != len(v):
            continue
        sim = sum(a * b for a, b in zip(v, c))
        if best is None or sim > best:
            best, best_ref = sim, ref
    if best is None:
        return None, None
    return round(max(0.0, 1.0 - best), 6), best_ref


def run_question_and_answer(
    detail: Mapping[str, Any], *, per_prior: Sequence[Mapping[str, Any]] = (),
    questions: Mapping[str, str] | None = None, prior_claims: Mapping[str, str] | None = None,
    graph_findings: Sequence[str] = (),
) -> tuple[str, Optional[str], str]:
    """(question, question_ref, answer) for one finish row, as the reference labels were made.

    self_inquiry: the self-question's text; the answer is the SelfDefinition (anatomy) or LivedAnswer
    text carried on the finish row. investigate: the claim of the tested prior that moved most, else
    the first prior formed; the answer is the run's ``Finding`` node texts (``graph_findings``). Either
    line falls back to the Finding texts, then to the finish row's ``finding_text`` (its head is often
    process narration -- "Done for the sitting..." -- so it is the last resort, as in the labels)."""
    questions = questions or {}
    prior_claims = prior_claims or {}
    line = str(detail.get("line") or "")
    lived = detail.get("lived_answer") if isinstance(detail.get("lived_answer"), Mapping) else {}
    selfdef = detail.get("self_definition") if isinstance(detail.get("self_definition"), Mapping) else {}
    question, qref, answer = "", None, ""
    if line == "self_inquiry":
        anatomy = detail.get("self_question_family") == "anatomy" or bool(selfdef.get("text"))
        qid = "anatomy.made_of" if anatomy else (lived.get("question_id") or None)
        if qid:
            qref = f"self_question:{qid}"
            question = questions.get(str(qid)) or str(qid)
        answer = str((selfdef if anatomy else lived).get("text") or "")
    else:
        tested = sorted((p for p in per_prior if p.get("kind") == "tested"), key=lambda p: -(p.get("nats") or 0))
        formed = [p for p in per_prior if p.get("kind") == "formed"]
        pick = (tested or formed or [None])[0]
        if pick and pick.get("prior_id"):
            pid = str(pick["prior_id"])
            qref = f"prior:{pid}"
            question = prior_claims.get(pid) or pid
    answer = clean(answer) or clean(" ".join(str(f or "") for f in graph_findings)) or clean(detail.get("finding_text"))
    return clean(question) or "(question not recorded)", qref, answer


__all__ = ["ANSWER_CHARS", "CORPUS_VERSION", "EMBED_CHARS", "JUDGE_PROMPT_SHA256", "JUDGE_PROMPT_V2",
           "PHILOSOPHICAL", "PROMPT_VERSION", "REGISTERS", "clean", "judge_input", "judge_messages", "novelty",
           "parse_judge_reply", "prompt_sha256", "run_question_and_answer"]
