"""Shared, read-only readers for the reflection-register share eval and the judge anchor set.

Design: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md, sections 6-7. One local
measure for every organ: the finding-register judge (``orion.curiosity.register``: prompt v2, temperature
0, thinking off, a reply counts only when the served model is in ``PASSED_JUDGE_MODELS``), on the first
700 characters of each item, with a question line naming which organ wrote it.

Everything here is SELECT-only (``docker exec orion-athena-sql-db psql``) or ``GRAPH.RO_QUERY``. The judge
is Orion's own local model, called only through an operator lane (``orion/evals/leased_llm.py``): the
caller takes ONE GPU pool hold (``operator_lane_sync``) and ``judge_one`` posts each call to
orion-llm-gateway under it, so the pool sees every call (#3921: this path once sent 286+ unleased calls
straight to the agent seat). There is no direct-URL mode. No outside model is ever called.

Sources (``ORGANS``):

* ``self_inquiry_finding`` / ``investigate_finding``: ``curiosity_finding_register`` rows. The share eval
  reads the label the live register already stored (the same judge); the anchor set rebuilds the judged
  text from the finish row as the register does (``register.run_question_and_answer``).
* ``orion_question``: ``curiosity_self_questions`` rows Orion wrote (``register.QUESTION_LINE``).
* ``outreach``: messages Orion sent (``endogenous_outreach_decisions.outreach``) joined on correlation id to
  the assistant ``chat_message``; ``source`` from the decision's result (none = the outreach tick).
* ``reverie_thought`` (``substrate_reverie_thought.interpretation``), ``reverie_question``
  (``reverie_question.question``), ``day_letter`` (``orion_day_letter.note_md`` paragraphs of 200+ chars).

Sampling is deterministic: items sorted by ``md5(id)``, first N. Not cherry-picked, repeatable.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.curiosity.register import (  # noqa: E402
    ANSWER_CHARS,
    QUESTION_LINE,
    clean,
    judge_messages,
    judge_model_passed,
    parse_judge_reply,
    run_question_and_answer,
)
from orion.evals.leased_llm import LaneCallError  # noqa: E402

# Read-only by the server, not only by convention: every session this opens is a read-only transaction.
PSQL = ["docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "orion-athena-sql-db", "psql",
        "-U", "postgres", "-d", "conjourney", "-Atc"]
GRAPH_HOST, GRAPH_PORT, GRAPH_NAME = "localhost", 6380, "orion_worldview"
JUDGE_MAX_TOKENS = 400  # the live register's value (scripts/curiosity_finding_register.py)
JUDGE_TIMEOUT_SEC = 300
DEFAULT_LANE = "agent"  # the seat these scripts called before they were leased (#3921)
LETTER_PARAGRAPH_MIN = 200

ORGAN_LINES = {
    "orion_question": QUESTION_LINE,
    "outreach": "(A message Orion sent to Juniper on its own. Label what the message is about.)",
    "reverie_thought": "(A thought from Orion's reverie, its idle thinking between tasks. Label what the thought is about.)",
    "reverie_question": "(A question Orion's reverie left for itself. Label what the question is about.)",
    "day_letter": "(A paragraph from Orion's daily letter to Juniper. Label what the paragraph is about.)",
}
ORGANS = ("self_inquiry_finding", "investigate_finding", "orion_question", "outreach", "reverie_thought",
          "reverie_question", "day_letter")


@dataclass(frozen=True)
class Item:
    organ: str
    ref: str
    question: str
    text: str
    sub: Optional[str] = None          # e.g. the outreach source
    stored_label: Optional[str] = None  # the live register's label (curiosity findings only)
    stored_error: Optional[str] = None
    at: Optional[str] = None


def md5_key(ref: str) -> str:
    return hashlib.md5(ref.encode("utf-8")).hexdigest()


def sample(items: Sequence[Item], n: int) -> list[Item]:
    """The first ``n`` items by md5 of their ref: deterministic, independent of time order."""
    return sorted(items, key=lambda it: md5_key(it.ref))[: max(0, int(n))]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def psql_json(sql: str) -> list[dict]:
    """Rows of a ``SELECT json_build_object(...)`` query. Refuses anything that is not a SELECT."""
    flat = " ".join(sql.split())
    if not flat.lower().startswith(("select ", "with ")):
        raise ValueError("read-only: only SELECT / WITH queries")
    out = subprocess.run(PSQL + [flat], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def graph_rows(cypher: str) -> list[list]:
    import redis

    r = redis.Redis(host=GRAPH_HOST, port=GRAPH_PORT, decode_responses=True)
    return r.execute_command("GRAPH.RO_QUERY", GRAPH_NAME, cypher)[1]


def _window(start: str, end: str, col: str) -> str:
    return f"{col} >= '{start}'::timestamptz AND {col} < '{end}'::timestamptz"


def _check_ts(value: str) -> str:
    from datetime import datetime

    datetime.fromisoformat(value.replace("Z", "+00:00"))  # raises on anything that is not a timestamp
    return value


def read_items(organ: str, *, start: str, end: str, with_text: bool = True) -> list[Item]:
    """Every item ``organ`` wrote in [start, end). Read-only."""
    start, end = _check_ts(start), _check_ts(end)
    if organ in ("self_inquiry_finding", "investigate_finding"):
        return _curiosity_items("self_inquiry" if organ == "self_inquiry_finding" else "investigate",
                                organ, start=start, end=end, with_text=with_text)
    if organ == "orion_question":
        rows = psql_json(f"SELECT json_build_object('id', question_id, 'text', text, 'at', created_at) "
                         f"FROM curiosity_self_questions WHERE minted_by = 'orion' AND {_window(start, end, 'created_at')}")
        return [Item(organ, f"self_question:{r['id']}", ORGAN_LINES[organ], clean(r["text"]), at=r["at"]) for r in rows]
    if organ == "outreach":
        rows = psql_json(
            "SELECT json_build_object('id', d.decision_id, 'source', d.result_json->>'source', 'at', d.decided_at, "
            "'text', (SELECT m.content FROM chat_message m WHERE m.correlation_id = d.correlation_id "
            "AND m.role = 'assistant' ORDER BY m.timestamp LIMIT 1)) "
            f"FROM endogenous_outreach_decisions d WHERE d.outreach AND {_window(start, end, 'd.decided_at')}")
        return [Item(organ, f"outreach:{r['id']}", ORGAN_LINES[organ], clean(r["text"]),
                     sub=r.get("source") or "outreach_tick", at=r["at"]) for r in rows if clean(r.get("text"))]
    if organ == "reverie_thought":
        rows = psql_json("SELECT json_build_object('id', thought_id, 'text', interpretation, 'at', created_at) "
                         f"FROM substrate_reverie_thought WHERE interpretation IS NOT NULL AND {_window(start, end, 'created_at')}")
        return [Item(organ, f"reverie_thought:{r['id']}", ORGAN_LINES[organ], clean(r["text"]), at=r["at"])
                for r in rows if clean(r.get("text"))]
    if organ == "reverie_question":
        rows = psql_json("SELECT json_build_object('id', question_id, 'text', question, 'at', created_at) "
                         f"FROM reverie_question WHERE question IS NOT NULL AND {_window(start, end, 'created_at')}")
        return [Item(organ, f"reverie_question:{r['id']}", ORGAN_LINES[organ], clean(r["text"]), at=r["at"])
                for r in rows if clean(r.get("text"))]
    if organ == "day_letter":
        rows = psql_json("SELECT json_build_object('id', run_id, 'text', note_md, 'at', created_at) "
                         f"FROM orion_day_letter WHERE note_md IS NOT NULL AND {_window(start, end, 'created_at')}")
        out = []
        for r in rows:
            paras = [clean(p) for p in str(r["text"] or "").split("\n\n")]
            for k, p in enumerate(x for x in paras if len(x) >= LETTER_PARAGRAPH_MIN):
                out.append(Item(organ, f"letter:{r['id']}#{k}", ORGAN_LINES[organ], p, at=r["at"]))
        return out
    raise ValueError(f"unknown organ {organ!r}")


def _curiosity_items(line: str, organ: str, *, start: str, end: str, with_text: bool) -> list[Item]:
    rows = psql_json(
        "SELECT json_build_object('run_id', run_id, 'register', register, 'judge_error', judge_error, "
        "'question_ref', question_ref, 'at', created_at) FROM curiosity_finding_register "
        f"WHERE line = '{line}' AND {_window(start, end, 'created_at')}")
    if not with_text:
        return [Item(organ, f"curiosity_run:{r['run_id']}", "", "", stored_label=r.get("register"),
                     stored_error=r.get("judge_error"), at=r["at"]) for r in rows]
    ids = ",".join(str(r["run_id"]) for r in rows if str(r["run_id"]).isalnum())
    finish = {r["run_id"]: r for r in psql_json(
        "SELECT json_build_object('run_id', s.run_id, 'detail', s.detail, 'per_prior', o.per_prior) "
        "FROM substrate_durable_run_state s LEFT JOIN curiosity_run_outcomes o USING (run_id) "
        "WHERE s.workflow = 'curiosity.investigate' AND s.node = 'finish' "
        f"AND s.run_id = any(string_to_array('{ids}', ','))")} if ids else {}
    qs = {q["question_id"]: q["text"] for q in psql_json(
        "SELECT json_build_object('question_id', question_id, 'text', text) FROM curiosity_self_questions")}
    claims = {row[0]: row[1] for row in graph_rows("MATCH (p:Prior) RETURN p.prior_id, p.claim") if row[0]}
    findings: dict[str, list[str]] = {}
    for rid, text in graph_rows("MATCH (n:Finding) RETURN n.run_id, n.text"):
        findings.setdefault(str(rid), []).append(text or "")
    out = []
    for r in rows:
        f = finish.get(r["run_id"])
        if f is None:
            continue
        q, _qref, a = run_question_and_answer(f["detail"] or {}, per_prior=f.get("per_prior") or [], questions=qs,
                                              prior_claims=claims, graph_findings=findings.get(r["run_id"], ()))
        if a:
            out.append(Item(organ, f"curiosity_run:{r['run_id']}", q, a, stored_label=r.get("register"),
                            stored_error=r.get("judge_error"), at=r["at"]))
    return out


def served_model(lane: Any, body: dict) -> Optional[str]:
    """The model that served the reply. The lane sends ``model=<operator route>`` (e.g. ``operator_agent``),
    and the gateway rewrites it only for vLLM grants, so a llama.cpp/strata seat may echo the route name
    back. A route name is not a model: then the pool's grant record (``model_file``) names the seat's
    model. Nothing known -> None (``model_not_evaluated: unknown``, never a label)."""
    model = str(body.get("model") or "").strip()
    route = str(getattr(lane, "route", "") or "")
    if model and model != route and not model.startswith("operator"):
        return model
    grant = getattr(getattr(lane, "lease", None), "grant", None)
    return str(getattr(grant, "model_file", "") or "") or None


def judge_one(lane: Any, item: Item) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(label, reason, error, model) from one judge call on ``lane`` (an orion.evals.leased_llm.OperatorLane,
    ``chat(body, timeout_sec)``). A failed call is an error tuple; a model not in PASSED_JUDGE_MODELS is an
    error; a recall (``LaneRecalled``) propagates so the caller stops and writes what it has."""
    try:
        body = lane.chat({
            "messages": judge_messages(item.question, item.text[:ANSWER_CHARS]), "temperature": 0,
            # Pinned: the gateway adds top_p 0.90 / presence_penalty 0.20 on operator classes when a body
            # leaves them unset; the live register's judge sends neither.
            "top_p": 1.0, "presence_penalty": 0,
            "max_tokens": JUDGE_MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}},
            timeout_sec=JUDGE_TIMEOUT_SEC)
        model = served_model(lane, body)
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
    except LaneCallError as exc:
        return None, None, f"call: {exc}"[:200], None
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        return None, None, f"call: bad reply {exc!r}"[:200], None
    if not judge_model_passed(model):
        return None, None, f"model_not_evaluated: {model or 'unknown'}", model
    label, reason, err = parse_judge_reply(content)
    return label, reason, err, model


__all__ = ["DEFAULT_LANE", "JUDGE_TIMEOUT_SEC", "ORGANS", "ORGAN_LINES", "Item", "judge_one", "md5_key", "psql_json", "read_items", "sample", "wilson"]
