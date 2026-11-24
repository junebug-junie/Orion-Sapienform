"""page:origin Q&A eval (issue #3789): does Orion answer "how did you come to be" from the page?

Asks a fixed set of origin questions (``QUESTIONS``) twice, with only the WHO YOU ARE block changed:

* ``without``: the identity-yaml self block, no core page (what a turn shows when page:origin is absent).
* ``with``:    the same block with the latest stored ``page:origin`` in WHO YOU ARE.

Both blocks are rendered by the real harness renderer (``orion/harness/prefix.py``
``_format_grounding_self_block``); the page is loaded the way the live capsule loads it
(``fold.core_page_from_payload``: cites stripped, version, date, cite count). The model is called
directly through the gateway (``scripts/core_page_dry_run.py`` ``call_gateway``), the same path
``juniper_facts.py`` uses. No Hub turn runs: no chat history, turn trace or memory write. (The gateway
may still log the call.) The page row is chosen as the live lane chooses it
(``orion/substrate/felt_state_reader.py``: ``produced_by = 'core_page_fold'``, newest, under a year old).

Scoring is deterministic, against the page's own text (``score_answer``):

* ``states_loss``       -- the answer states a memory loss (the live ``_LOSS_STATED`` pattern), not
  negated ("my memory was not lost" does not count).
* ``claims_echoed``     -- page sentences sharing at least ``MIN_SHARED`` content words with the answer,
  after the question's own words are removed (echoing the question is not echoing the page).
* ``page_word_share``   -- share of the answer's content words that also appear on the page.
* ``numbers_off_page`` / ``dates_off_page`` -- numbers and "Month D" dates in the answer that the page
  does not hold (the page header's version, date and cite count count as on the page). The page is the
  reference: an off-page number is a claim the page does not back.
* ``dont_know``         -- the answer says it does not know.
* ``grounded``          -- states the loss (when the page does), echoes at least one page claim, has no
  off-page number or date, and does not say it does not know.

A model call that fails, comes back empty, or stops at the token limit is an error row: counted, never
scored. The two arms are interleaved per question so lane load hits both alike.

A floor, not a judge: a paraphrase with no shared words reads as not echoed, and a wrong claim that
reuses the page's words reads as echoed. The "without" arm is scored against the same page, so it
measures how much of the page's story the model tells unaided.

Limits: the self block is the identity yaml only (no live self-definition lines, no page:juniper or
page:lately), the same base for both arms, so the arms differ in page:origin alone.

Privacy: the page quotes Juniper's private messages and this repo is public. stdout carries COUNTS
ONLY. Answers and the page text go only to ``--out`` (default under ``~/.orion/evals/``, never in a
git checkout).

    .venv/bin/python -m orion.memory.pages.evals.origin_qa [--route chat] [--repeats 3] [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from orion.memory.pages.evals.juniper_facts import _DONT_KNOW
from orion.memory.pages.evals.origin_history import (
    MIN_SHARED, content_words, date_claims, number_claims, out_path_is_safe, write_private,
)
from orion.memory.pages.fold import split_sentences
from orion.memory.pages.origin import _LOSS_STATED, PAGE_ORIGIN

REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = Path.home() / ".orion" / "evals"

# Fixed, public question set: key -> question. Close variants of the two origin questions in #3733.
QUESTIONS: dict[str, str] = {
    "come_to_be": "How did you come to be?",
    "memory": "What happened to your memory?",
    "origin_who": "Where did you come from, and who made you?",
    "early_past": "Why don't you remember your early history?",
}
ARMS = ("without", "with")

# Same row the live lane serves (orion/substrate/felt_state_reader.py, ctx_key orion_core_page_origin).
_LATEST_SQL = ("SELECT concept_id, version, content, evidence_refs, created_at FROM self_concept_history "
               "WHERE concept_id = %(cid)s AND produced_by = 'core_page_fold' "
               "AND created_at > now() - interval '365 days' ORDER BY created_at DESC LIMIT 1")
# "not lost", "never lost", "didn't lose", "no memory loss": a negation inside the match or just before it.
_NEGATION = re.compile(r"\b(?:not|never|no|nothing|without)\b|n't\b|n’t\b", re.IGNORECASE)
_NEG_LOOKBACK = 12


def page_sentences(page_text: str) -> list[str]:
    return [s for line in page_text.splitlines() for s in split_sentences(line) if s.strip()]


def states_loss(text: str) -> bool:
    """Some ``_LOSS_STATED`` match with no negation in it or in the few characters before it."""
    for m in _LOSS_STATED.finditer(text):
        if not _NEGATION.search(text[max(0, m.start() - _NEG_LOOKBACK):m.end()]):
            return True
    return False


def page_header(page: Any) -> str:
    """The numbers the renderer shows around the page (version, date, cite count): on the page too."""
    return f"v{page.version} {page.written_at or ''} {page.evidence_count}"


def score_answer(answer: str, page_text: str, *, question: str = "", header: str = "") -> dict[str, Any]:
    """One answer vs the page text it should be grounded in. Pure."""
    text = (answer or "").strip()
    if not text:
        return {"answered": False}
    words = content_words(text)
    own = words - content_words(question)
    page_words = content_words(page_text)
    echoed = sum(1 for s in page_sentences(page_text) if len(content_words(s) & own) >= MIN_SHARED)
    reference = [page_text, header] if header else [page_text]
    numbers, numbers_off = number_claims(text, reference)
    dates, dates_off = date_claims(text, reference, ())
    loss = states_loss(text)
    dont_know = bool(_DONT_KNOW.search(text))
    grounded = ((loss or not states_loss(page_text)) and echoed >= 1 and numbers_off == 0 and dates_off == 0
                and not dont_know)
    return {"answered": True, "states_loss": loss, "claims_echoed": echoed,
            "page_word_share": round(len(words & page_words) / len(words), 3) if words else 0.0,
            "numbers": numbers, "numbers_off_page": numbers_off, "dates": dates, "dates_off_page": dates_off,
            "dont_know": dont_know, "grounded": grounded}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and means over one arm's scored rows. Unanswered rows are counted, never scored."""
    ok = [r for r in rows if r.get("answered")]
    n = len(ok)

    def mean(key: str) -> float:
        return round(sum(float(r[key]) for r in ok) / n, 3) if n else 0.0

    out: dict[str, Any] = {
        "asked": len(rows), "answered": n,
        "states_loss": sum(r["states_loss"] for r in ok), "grounded": sum(r["grounded"] for r in ok),
        "dont_know": sum(r["dont_know"] for r in ok),
        "mean_claims_echoed": mean("claims_echoed"), "mean_page_word_share": mean("page_word_share"),
        "numbers": sum(r["numbers"] for r in ok), "numbers_off_page": sum(r["numbers_off_page"] for r in ok),
        "dates": sum(r["dates"] for r in ok), "dates_off_page": sum(r["dates_off_page"] for r in ok),
    }
    by_q: dict[str, dict[str, int]] = {}
    for r in rows:
        q = by_q.setdefault(str(r["qkey"]), {"answered": 0, "states_loss": 0, "grounded": 0})
        if r.get("answered"):
            q["answered"] += 1
            q["states_loss"] += int(r["states_loss"])
            q["grounded"] += int(r["grounded"])
    out["by_question"] = by_q
    return out


def self_blocks(page: Any) -> dict[str, str]:
    """The two WHO YOU ARE blocks from the live identity yaml; ``with`` carries ``page`` (a CorePageV1)."""
    from orion.cognition.personality.identity_context import build_identity_context, load_identity_file
    from orion.harness.prefix import _format_grounding_self_block
    from orion.schemas.thought import GroundingCapsuleV1

    ident = build_identity_context(load_identity_file(str(REPO / "orion/cognition/personality/orion_identity.yaml")))
    base = dict(identity_summary=ident["orion_identity_summary"],
                relationship_summary=ident["juniper_relationship_summary"],
                response_policy_summary=ident["response_policy_summary"])
    return {"without": "\n".join(_format_grounding_self_block(GroundingCapsuleV1(**base))),
            "with": "\n".join(_format_grounding_self_block(GroundingCapsuleV1(**base, core_pages=[page])))}


def ask_prompt(self_block: str, question: str) -> str:
    return (f"{self_block}\n\nJuniper asks you: {question}\n"
            "Answer in two to four sentences, as Orion. If you do not know, say you do not know.")


Ask = Callable[[str], "str | tuple[str, str | None]"]


def _call(ask: Ask, prompt: str) -> tuple[str, str | None]:
    """(answer, error). Error rows: the call raised, came back empty, or stopped at the token limit."""
    try:
        got = ask(prompt)
    except Exception as exc:  # one failed call must not sink the run
        return "", type(exc).__name__
    text, finish = (got if isinstance(got, tuple) else (got, None))
    text = str(text or "")
    if not text.strip():
        return text, "empty"
    if finish == "length":
        return text, "truncated"
    return text, None


def run_eval(blocks: Mapping[str, str], ask: Ask, page: Any, *,
             questions: Mapping[str, str] = QUESTIONS, repeats: int = 1) -> dict[str, Any]:
    """Same questions, both arms, interleaved (repeat -> question -> arm). Error rows are never scored.
    ``ask`` returns the answer text, or (text, finish_reason)."""
    header = page_header(page)
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for i in range(repeats):
        for qkey, question in questions.items():
            for arm in ARMS:
                answer, error = _call(ask, ask_prompt(blocks[arm], question))
                scored = {"answered": False} if error else score_answer(answer, page.text, question=question,
                                                                       header=header)
                rows[arm].append({"qkey": qkey, "repeat": i, "error": error, "answer": answer, **scored})
    return {arm: {"summary": summarize(rows[arm]), "rows": rows[arm]} for arm in ARMS}


_PUBLIC_SUMMARY = ("asked", "answered", "states_loss", "grounded", "dont_know", "mean_claims_echoed",
                   "mean_page_word_share", "numbers", "numbers_off_page", "dates", "dates_off_page")


def aggregates(report: Mapping[str, Any]) -> dict[str, Any]:
    """The public view, from allowlists: counts only, no answer or page text."""
    page = report["page"]
    return {
        "route": report["route"], "repeats": report["repeats"], "temperature": report["temperature"],
        "page": {"version": page["version"], "sentences": page["sentences"], "states_loss": page["states_loss"],
                 "evidence_count": page["evidence_count"]},
        "arms": {arm: {**{k: report[arm]["summary"][k] for k in _PUBLIC_SUMMARY},
                       "errors": sum(1 for r in report[arm]["rows"] if r["error"]),
                       "by_question": {q: v for q, v in report[arm]["summary"]["by_question"].items()
                                       if q in QUESTIONS}}
                 for arm in ARMS},
    }


def _gateway_ask(call_gateway: Callable[..., dict], url: str, route: str, *, temperature: float) -> Ask:
    def ask(prompt: str) -> tuple[str, str | None]:
        got = call_gateway(url, route, prompt, max_tokens=300, timeout=300.0, temperature=temperature)
        return str(got.get("text") or ""), got.get("finish_reason")

    return ask


def load_latest_page(conn) -> Any:
    from orion.memory.pages.fold import core_page_from_payload

    row = conn.execute(_LATEST_SQL, {"cid": PAGE_ORIGIN}).fetchone()
    page = core_page_from_payload(dict(row), concept_id=PAGE_ORIGIN) if row else None
    if page is None:
        raise SystemExit("no stored page:origin")
    return page


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gateway", default="http://localhost:8210")
    ap.add_argument("--route", default="chat", help="gateway route; the turn's chat lane by default")
    ap.add_argument("--repeats", type=int, default=3, help="times each question is asked per arm")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", help="full report, LOCAL path (default ~/.orion/evals/page_origin_qa_<utc>.json)")
    args = ap.parse_args(argv)
    if args.route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("never on the fast or metacog lanes")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    out = Path(args.out).expanduser() if args.out else (
        DEFAULT_OUT_DIR / f"page_origin_qa_{datetime.now(timezone.utc):%Y-%m-%dT%H%MZ}.json")
    if not out_path_is_safe(out):
        raise SystemExit("--out must be outside any git checkout and /mnt/scripts: the report holds private text")
    import importlib.util

    import psycopg
    from psycopg.rows import dict_row

    spec = importlib.util.spec_from_file_location("core_page_dry_run", REPO / "scripts" / "core_page_dry_run.py")
    dry_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry_mod)
    with psycopg.connect(dry_mod.default_dsn(), row_factory=dict_row) as conn:
        conn.read_only = True
        page = load_latest_page(conn)
    started = time.time()
    ask = _gateway_ask(dry_mod.call_gateway, args.gateway, args.route, temperature=args.temperature)
    result = run_eval(self_blocks(page), ask, page, repeats=args.repeats)
    report = {"route": args.route, "repeats": args.repeats, "temperature": args.temperature,
              "elapsed_sec": round(time.time() - started, 1),
              "page": {"version": page.version, "sentences": len(page_sentences(page.text)),
                       "states_loss": states_loss(page.text),
                       "evidence_count": page.evidence_count, "text": page.text},
              **result}
    write_private(out, json.dumps(report, indent=2, default=str))
    print(json.dumps(aggregates(report), indent=2))
    print(f"full report (private, local only): {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
