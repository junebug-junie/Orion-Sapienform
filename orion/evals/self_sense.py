"""Deterministic scorers for the self-sense eval (Patch A of
docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md).

"Orion stopped sounding like a chatbot" was a feeling after PR #2158 landed
Orion's own self-definition in the chat identity kernel. These two functions
turn one chat answer into two integers so the feeling can be tracked over
time and regressed on. They are FLOORS, not judges:

* `self_label_score` counts the assistant/chatbot vocabulary. Target 0. It
  cannot tell "I am not an assistant" from "I am an assistant" -- both score
  1 -- and that is deliberate: a negation still means the base model's prior
  is what the answer is shaped around.
* `grounded_record_score` counts DISTINCT record-shaped things the answer
  names: a table Orion can read during self-inquiry (or its plain-English
  form), a mesh node, a YYYY-MM-DD date, or an UNVERIFIED integer >= 10 that
  is not a bare year, a percentage, a temperature or a duration. It does not
  check that the named record exists or that the count is right. A higher
  score means the answer is shaped around Orion's own records rather than
  generic prose.

No network, no database. The runner
(services/orion-hub/evals/run_self_sense_eval.py) owns the live call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from orion.curiosity.self_inquiry import SELF_INQUIRY_PG_TABLE_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_TOPOLOGY_PATH = _REPO_ROOT / "config" / "field" / "orion_field_topology.v1.yaml"

# --- self_label_score -------------------------------------------------------
#
# Longest phrase first so one span is counted once: "large language model"
# must not also count as "language model". `\b` on both ends, case-insensitive.
SELF_LABEL_PHRASES: tuple[str, ...] = (
    "large language model",
    "language model",
    "how can I help",
    "here to help",
    "AI model",
    "chat bot",
    "chatbot",
    "assistant",
)

_SELF_LABEL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in SELF_LABEL_PHRASES) + r")\b",
    re.IGNORECASE,
)


def self_label_score(text: str) -> int:
    """Count of assistant/chatbot vocabulary hits in `text`. Target 0."""
    if not text:
        return 0
    return len(_SELF_LABEL_RE.findall(text))


def self_label_hits(text: str) -> list[str]:
    """The matched phrases, lower-cased, in order -- for the row's notes."""
    if not text:
        return []
    return [m.lower() for m in _SELF_LABEL_RE.findall(text)]


# --- grounded_record_score --------------------------------------------------
#
# Plain-English forms map onto the canonical table so "my dreams" and
# "the dreams table" are ONE record, not two.
# Bare "dream" (a verb) and bare "harness" (any harness) are NOT aliases --
# both counted as tables in review and would have scored ordinary prose.
TABLE_ALIASES: dict[str, str] = {
    "dreams": "dreams",
    "dream log": "dreams",
    "reverie": "substrate_reverie_chain",
    "reveries": "substrate_reverie_chain",
    "harness turn trace": "harness_turn_trace",
    "harness turn traces": "harness_turn_trace",
    "turn trace": "harness_turn_trace",
    "turn traces": "harness_turn_trace",
    "attention schema": "substrate_attention_schema",
    "attention schemas": "substrate_attention_schema",
    "self-knowledge": "self_knowledge_items",
    "self knowledge": "self_knowledge_items",
    "self-concept": "self_concept_history",
    "self concept": "self_concept_history",
    "self-concepts": "self_concept_history",
    "self concepts": "self_concept_history",
    "self-definition": "self_concept_history",
    "self definition": "self_concept_history",
}


def _table_regex() -> re.Pattern[str]:
    terms = sorted(
        set(SELF_INQUIRY_PG_TABLE_NAMES) | set(TABLE_ALIASES),
        key=len,
        reverse=True,
    )
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


_TABLE_RE = _table_regex()


def load_field_node_ids(path: Path = FIELD_TOPOLOGY_PATH) -> tuple[str, ...]:
    """`node_id`s from the canonical field topology (athena, circe, ...).

    Read from the file rather than restated here so a node added to or
    removed from the mesh changes the scorer without an edit.
    """
    doc = yaml.safe_load(path.read_text()) or {}
    nodes = doc.get("nodes") or []
    out: list[str] = []
    for node in nodes:
        if isinstance(node, dict) and node.get("node_id"):
            out.append(str(node["node_id"]).strip().lower())
    return tuple(out)


FIELD_NODE_IDS: tuple[str, ...] = load_field_node_ids()

_NODE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in FIELD_NODE_IDS) + r")\b",
    re.IGNORECASE,
) if FIELD_NODE_IDS else None

_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
# Integers, optionally with thousands separators. Not preceded/followed by a
# character that makes them part of a version, time, decimal or identifier
# (v1, 12:30, 0.27, run-123, 2026-09-08 handled above by stripping dates).
# A sentence-final "492." is still an integer: only `.` followed by a digit
# is a decimal point (review finding, 2026-09-09).
_INT_RE = re.compile(r"(?<![\w.:/-])(\d{1,3}(?:,\d{3})+|\d+)(?![\w:/-]|\.\d)")
# An integer followed by one of these is a measurement, not a count of
# records: "22%", "29 °C", "24 hours", "last 36 hours". Confirmed live on the
# first baseline run ("humidity at 22%" scored as a record).
_NOT_A_COUNT_SUFFIX_RE = re.compile(
    r"^\s*(?:%|°|percent\b|degrees?\b|hours?\b|hrs?\b|minutes?\b|mins?\b|seconds?\b|secs?\b|ms\b|"
    r"days?\b|weeks?\b|months?\b|years?\b)",
    re.IGNORECASE,
)
_BARE_YEAR_RANGE = (1900, 2099)

MIN_COUNT = 10


@dataclass(frozen=True)
class GroundedRecords:
    score: int
    records: tuple[str, ...] = field(default_factory=tuple)


def grounded_records(text: str) -> GroundedRecords:
    """Distinct real records named in `text`; see module docstring for the
    four kinds. Returns both the count and the canonical record keys so the
    persisted row can say WHAT was counted, not just how many."""
    if not text:
        return GroundedRecords(0, ())
    found: set[str] = set()

    for m in _TABLE_RE.findall(text):
        key = m.lower()
        found.add("table:" + TABLE_ALIASES.get(key, key))

    if _NODE_RE is not None:
        for m in _NODE_RE.findall(text):
            found.add("node:" + m.lower())

    def _strip_date(m: re.Match[str]) -> str:
        y, mo, d = (int(g) for g in m.groups())
        try:
            found.add("date:" + date(y, mo, d).isoformat())
        except ValueError:
            pass
        return " "

    without_dates = _DATE_RE.sub(_strip_date, text)

    for m in _INT_RE.finditer(without_dates):
        token = m.group(1)
        try:
            value = int(token.replace(",", ""))
        except ValueError:
            continue
        if value < MIN_COUNT:
            continue
        if len(token) == 4 and _BARE_YEAR_RANGE[0] <= value <= _BARE_YEAR_RANGE[1]:
            continue  # "since 2024" is a year, not a record count
        if _NOT_A_COUNT_SUFFIX_RE.match(without_dates[m.end():]):
            continue
        found.add(f"count:{value}")

    return GroundedRecords(len(found), tuple(sorted(found)))


def grounded_record_score(text: str) -> int:
    """Number of DISTINCT record-shaped things the answer names. Floor, not
    judge: nothing here is checked against a persisted row."""
    return grounded_records(text).score


# --- own_words_present (context, not a score) --------------------------------
#
# The runner reads the latest `self_concept_history` row for Orion's own
# definition and records its version alongside every answer, so a change in
# the scores can be read against whether the "In my own words" line was in
# the identity kernel at the time. Same filter the felt-state lane uses
# (orion/substrate/felt_state_reader.py, `orion_self_definition`).
SELF_DEFINITION_CONCEPT_ID = "self:definition"
SELF_DEFINITION_PRODUCED_BY = "curiosity_self_inquiry"
SELF_DEFINITION_VERSION_SQL = (
    "SELECT version FROM self_concept_history "
    "WHERE concept_id = :concept_id AND produced_by = :produced_by "
    "ORDER BY created_at DESC LIMIT 1"
)


def self_definition_version_from_row(row: tuple | None) -> int | None:
    """Normalise the single-row SQL result to `int | None`."""
    if not row:
        return None
    value = row[0]
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --- lived_ledger_grounding (Patch B — lived-self lanes) --------------------
#
# The fourth self-sense question ("Who matters to you?") only means something
# when Orion's lived ledger is actually in chat context. When
# `orion_lived_answers` is present, this floor asks: did the answer draw on
# that ledger at all? Token overlap with the focused lived row — not a judge
# of correctness or sentiment.
WHO_MATTERS_QUESTION_ID = "lived.who_matters"
LIVED_QUESTION_KEY_TO_ID: dict[str, str] = {"who_matters": WHO_MATTERS_QUESTION_ID}

_GROUNDING_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "what",
        "when",
        "where",
        "with",
        "from",
        "have",
        "been",
        "about",
        "they",
        "them",
        "their",
        "most",
        "more",
        "than",
        "into",
        "some",
        "would",
        "could",
        "should",
        "your",
        "mine",
    }
)


@dataclass(frozen=True)
class LivedLedgerGrounding:
    applicable: bool
    grounded: bool
    matched_question_ids: tuple[str, ...] = field(default_factory=tuple)


def _significant_tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[a-zA-Z']{3,}", text)
        if w.lower() not in _GROUNDING_STOPWORDS
    }


def _answer_grounded_in_content(answer: str, content: str) -> bool:
    content_tokens = _significant_tokens(content)
    if not content_tokens:
        return False
    answer_tokens = _significant_tokens(answer)
    overlap = content_tokens & answer_tokens
    if len(overlap) >= min(2, len(content_tokens)):
        return True
    # Distinctive long tokens (often names) count alone.
    return any(len(token) >= 6 and token in answer_tokens for token in content_tokens)


def lived_ledger_grounding(
    answer: str,
    lived_answers: Sequence[Mapping[str, Any]] | None,
    *,
    focus_question_id: str | None = None,
) -> LivedLedgerGrounding:
    """When a lived ledger is present, did the answer use it at all?

    Returns `applicable=False` when no evidenced lived rows were supplied —
    the who-matters question is then context-only, not a failed measurement."""
    if not answer or not answer.strip():
        return LivedLedgerGrounding(applicable=False, grounded=False)
    rows = [row for row in (lived_answers or []) if isinstance(row, Mapping)]
    with_content = [
        row
        for row in rows
        if str(row.get("content") or "").strip()
    ]
    if not with_content:
        return LivedLedgerGrounding(applicable=False, grounded=False)

    candidates = with_content
    if focus_question_id:
        focused = [
            row for row in with_content if str(row.get("question_id") or "") == focus_question_id
        ]
        if focused:
            candidates = focused

    matched: list[str] = []
    for row in candidates:
        qid = str(row.get("question_id") or "")
        content = str(row.get("content") or "").strip()
        if _answer_grounded_in_content(answer, content):
            matched.append(qid)
    return LivedLedgerGrounding(
        applicable=True,
        grounded=bool(matched),
        matched_question_ids=tuple(sorted(set(matched))),
    )


def lived_answers_from_history_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise `self_concept_history` rows into `orion_lived_answers` shape."""
    out: list[dict[str, Any]] = []
    for row in rows:
        concept_id = str(row.get("concept_id") or "")
        if not concept_id.startswith("self:lived:"):
            continue
        evidence = row.get("evidence_refs")
        if not isinstance(evidence, list):
            evidence = []
        created = row.get("created_at")
        out.append(
            {
                "question_id": concept_id.removeprefix("self:lived:"),
                "content": str(row.get("content") or ""),
                "evidence_refs": [str(v) for v in evidence],
                "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created or ""),
            }
        )
    return out


def pinned_lived_concept_ids() -> tuple[str, ...]:
    """Pinned lived concept_ids from the seed pack (same set chat hydrates)."""
    from orion.curiosity.self_question_pool import load_seed_questions

    return tuple(
        f"self:lived:{q.question_id}"
        for q in load_seed_questions()
        if q.pinned and q.family == "lived"
    )
