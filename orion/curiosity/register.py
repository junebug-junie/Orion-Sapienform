"""Register and novelty of one finished curiosity finding. Pure, stdlib-only.

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("Which findings
compete"; the world-first candidate part of that doc is superseded by event wants). Juniper's rule
(2026-11-12): a finding is worth attention when it is philosophical (about what Orion is, its
experience, what it cares about, who matters) AND new to Orion.

- **Register**: one judge call with the fixed prompt v2 (``JUDGE_PROMPT_V2``, sha pinned to the
  reference-label fixture), on the question and the first ``ANSWER_CHARS`` of the answer. The reply
  must be one JSON object with one of the three labels; anything else is a ``judge_error`` and the
  label is absent (never "operational by default").
- **Judge model** (Juniper, 2026-11-15): the curiosity lane that wrote the finding judges it. Only a
  model that passed the judge-agreement eval (``PASSED_JUDGE_MODELS``) may label; a reply served by
  any other model is a ``judge_error``. Flash-Next passed on the yes/no question salience uses
  (philosophical or not: kappa 0.856 on 251 reference runs; three-way kappa 0.734, phil-agree 0.87;
  ``scripts/analysis/eval_curiosity_register_judge.py``). Juniper accepted the 0.87.
- **Novelty**: ``1 - max cosine`` between the finding's embedding and everything Orion wrote before
  the run: every answer node in the graph (``Finding``, ``LivedAnswer``, ``SelfDefinition``), every
  prior claim, every Day letter, minus what this run itself wrote (``corpus_items``). An exact repeat
  reads 0.0. No earlier writing: novelty is unknown (None), not 1.0.

Live writer: ``services/orion-hub/scripts/curiosity_finding_register.py`` (one row per finished turn in
``curiosity_finding_register``).
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
CORPUS_VERSION = "graph_v1"  # every graph answer node + prior claims + Day letters, minus this run's own

# The question door (docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md): the same
# judge, prompt v2 unchanged, labels a question Orion wrote for its own self-inquiry BEFORE any sitting
# answers it. The question text goes in as the item (the FINDING slot) and this line goes in as the
# QUESTION, saying what the item is. Fixed: a change here is a change in what the door measures.
# Live check 2026-11-25 (agent seat 8015, all 88 Orion questions): an earlier line ("The item below is not a
# finding ... label what the question is about") drew no_finding on 26 of 88 (a question states no conclusion,
# which is prompt v2's no_finding test), which would have left them unjudged. This line, on those 26 plus the 5
# non-JSON replies: 31 of 31 labelled (26 operational, 5 philosophical); on 19 already-labelled questions, 16
# kept their label and 3 moved philosophical -> operational.
QUESTION_LINE = (
    "(The FINDING below is a question Orion wrote for itself, to sit with in its own self-inquiry. A question "
    "states no conclusion, so never answer no_finding for it: label it philosophical or operational by what "
    "the question asks about.)"
)

# Models that passed the judge-agreement eval. Matched on the served model's file/name, lower case.
PASSED_JUDGE_MODELS = ("qwen3.8-flash-next",)

# Read-only Cypher for the novelty corpus (static, no parameters).
CORPUS_CYPHER = {
    "finding": "MATCH (n:Finding) RETURN n.finding_id AS finding_id, n.run_id AS run_id, n.text AS text",
    "lived_answer": "MATCH (n:LivedAnswer) RETURN n.run_id AS run_id, n.text AS text",
    "self_definition": "MATCH (n:SelfDefinition) RETURN n.run_id AS run_id, n.text AS text",
    "prior": "MATCH (p:Prior) RETURN p.prior_id AS prior_id, p.run_id AS run_id, p.claim AS text",
}

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


def judge_model_passed(model: Optional[str]) -> bool:
    """Whether the served model is one that passed the judge-agreement eval."""
    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return bool(name) and any(m in name for m in PASSED_JUDGE_MODELS)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_items(
    graph_rows: Mapping[str, Sequence[Mapping[str, Any]]], letters: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every piece of Orion's writing the novelty corpus holds: ``{ref, kind, run_id, text, sha}``.

    ``graph_rows`` maps a ``CORPUS_CYPHER`` kind to its rows; ``letters`` are Day letter rows
    ``{run_id, created_at, note_md}``. Text is cleaned and cut to ``EMBED_CHARS`` (what is embedded);
    empty text is dropped. A Finding is keyed by its ``finding_id`` (a few have no run id); the other
    answer nodes by their run (the sha keeps a run's several apart); a prior by its id; a letter by its
    run id."""
    out: list[dict[str, Any]] = []
    for kind, rows in graph_rows.items():
        for r in rows:
            text = clean(r.get("text"))[:EMBED_CHARS]
            run = str(r.get("run_id") or "") or None
            if not text:
                continue
            key = r.get("prior_id") if kind == "prior" else ((r.get("finding_id") or run) if kind == "finding" else run)
            if not key:
                continue
            out.append({"ref": f"{kind}:{key}", "kind": kind, "run_id": run, "text": text,
                        "sha": text_sha256(text), "written_at": None})
    for r in letters:
        text = clean(r.get("note_md"))[:EMBED_CHARS]
        if text and r.get("run_id"):
            out.append({"ref": f"letter:{r['run_id']}", "kind": "letter", "run_id": None, "text": text,
                        "sha": text_sha256(text), "written_at": r.get("created_at")})
    return out


def corpus_before(items: Sequence[Mapping[str, Any]], *, run_id: str, started_at: Any = None) -> list[Mapping[str, Any]]:
    """The items written before this run: drop what the run wrote itself (its answer nodes, the priors
    it formed) and letters written after the run started. A letter with no time is kept."""
    keep = []
    for it in items:
        if it.get("run_id") == run_id:
            continue
        at = it.get("written_at")
        if started_at is not None and at is not None and at >= started_at:
            continue
        keep.append(it)
    return keep


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


__all__ = ["ANSWER_CHARS", "CORPUS_CYPHER", "CORPUS_VERSION", "EMBED_CHARS", "JUDGE_PROMPT_SHA256", "JUDGE_PROMPT_V2",
           "PASSED_JUDGE_MODELS", "PHILOSOPHICAL", "PROMPT_VERSION", "QUESTION_LINE", "REGISTERS", "clean", "corpus_before",
           "corpus_items", "judge_input", "judge_messages", "judge_model_passed", "novelty", "parse_judge_reply",
           "prompt_sha256", "run_question_and_answer", "text_sha256"]
