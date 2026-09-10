"""What a human needs to see of Orion's self-definition -- current, history,
and the runs that produced it. Postgres-only counterpart to `atlas.py` (which
answers the same "is the loop working" question for the graph side).

Deliberately does NOT go through the `:TurnOutcome`-keyed run list in
`atlas.py`/`read_atlas()`. A self-inquiry run's `:SelfDefinition` write is
independent of whether it also wrote a `:TurnOutcome` (that node is optional,
"writing nothing is fine and is the normal case" -- worldview.py), and the run
that produced the FIRST self-definition (`282fbb9a08e4`, 2026-09-08) never
wrote one. A panel keyed on TurnOutcome would have shown nothing for it. This
reads `self_concept_history` and `journal_entries` directly, so a definition
is visible the moment it is mirrored, regardless of what else that run wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Keep in sync with orion/curiosity/self_inquiry.py -- not imported from there
# to avoid this Postgres-only module depending on that module's graph/Cypher
# surface for one string constant.
SELF_DEFINITION_CONCEPT_ID = "self:definition"
SELF_INQUIRY_JOURNAL_TITLE = "Self-inquiry"

_JOURNAL_LIMIT = 10
_EVAL_RUN_LIMIT = 20  # generous cap on one run's question rows, not runs
# Self-inquiry caps at 3 runs/day (orion/curiosity/README.md section 13), so
# this comfortably covers several weeks of distinct revisions -- the panel is
# recent-revision history, not a full audit trail. Deeper history is a direct
# query away. Also bounds the page's own poll (POLL_MS in the template) from
# re-fetching and re-rendering an unbounded list every cycle.
_HISTORY_LIMIT = 50


def _coerce_evidence_refs(raw: Any) -> list[str]:
    """`evidence_refs` is a generic JSON column, not JSONB -- asyncpg and
    psycopg2 usually deserialize it to a list already, but tolerate it coming
    back as a JSON string too (same tolerance as
    `self_atlas_cluster_history._coerce_evidence_refs`, kept local here so
    this Postgres-reader module has no import into a Hub script)."""
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    return []


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else None


@dataclass(frozen=True)
class SelfDefinitionVersion:
    version: int
    created_at: Optional[str]
    content: str
    evidence_refs: list[str] = field(default_factory=list)
    produced_by: str = ""


@dataclass(frozen=True)
class SelfInquiryJournalEntry:
    created_at: Optional[str]
    body: str


@dataclass(frozen=True)
class SelfSenseEvalRow:
    question_key: str
    question: str
    self_label_score: Optional[int]
    grounded_record_score: Optional[int]
    answer_source: str


@dataclass(frozen=True)
class SelfPanelView:
    """One read, everything the panel draws -- so the current-definition card
    and the history list cannot disagree about which version is current."""

    history: list[SelfDefinitionVersion] = field(default_factory=list)
    journal_entries: list[SelfInquiryJournalEntry] = field(default_factory=list)
    latest_eval_run_id: Optional[str] = None
    latest_eval: list[SelfSenseEvalRow] = field(default_factory=list)
    unavailable_reason: Optional[str] = None

    @property
    def current(self) -> Optional[SelfDefinitionVersion]:
        return self.history[0] if self.history else None

    @property
    def is_unavailable(self) -> bool:
        return self.unavailable_reason is not None


# Ordered by created_at, NOT version. Every other reader/writer of this table
# treats version as informational rather than authoritative for "what is
# current" -- self_study.py's _next_self_concept_version is a non-transactional
# MAX+1 read (a retried write or a backfill can leave it non-monotonic), the
# table has no unique constraint on (concept_id, version), and the felt-state
# lane that actually feeds every chat turn (orion/substrate/felt_state_reader.py,
# orion_self_definition) orders by created_at. This panel exists to show
# Juniper what the loop is doing, and it must not show a DIFFERENT "current"
# than the one Orion is actually using in chat -- review finding, 2026-09-09.
_HISTORY_SQL = (
    "SELECT version, created_at, content, evidence_refs, produced_by "
    "FROM self_concept_history WHERE concept_id = $1 "
    "ORDER BY created_at DESC LIMIT $2"
)
_JOURNAL_SQL = (
    "SELECT created_at, body FROM journal_entries "
    "WHERE title = $1 ORDER BY created_at DESC LIMIT $2"
)
_LATEST_EVAL_RUN_SQL = (
    "SELECT run_id FROM self_sense_eval_log ORDER BY created_at DESC LIMIT 1"
)
_EVAL_ROWS_SQL = (
    "SELECT question_key, question, self_label_score, grounded_record_score, answer_source "
    "FROM self_sense_eval_log WHERE run_id = $1 ORDER BY created_at ASC LIMIT $2"
)


async def read_self_panel(pool: Any) -> SelfPanelView:
    """Read everything the panel needs from one pool. Never raises -- an
    unreadable store is `unavailable_reason`, the same rule every other
    dashboard reader in this package follows (atlas.py, worldview.py): a
    broken read must not be confused with a mind that has written nothing."""
    if pool is None:
        return SelfPanelView(unavailable_reason="no_pool")
    try:
        async with pool.acquire() as conn:
            history_rows = await conn.fetch(_HISTORY_SQL, SELF_DEFINITION_CONCEPT_ID, _HISTORY_LIMIT)
            journal_rows = await conn.fetch(_JOURNAL_SQL, SELF_INQUIRY_JOURNAL_TITLE, _JOURNAL_LIMIT)
            latest_run_row = await conn.fetchrow(_LATEST_EVAL_RUN_SQL)
            eval_run_id = str(latest_run_row["run_id"]) if latest_run_row else None
            eval_rows = (
                await conn.fetch(_EVAL_ROWS_SQL, eval_run_id, _EVAL_RUN_LIMIT)
                if eval_run_id
                else []
            )
    except Exception as exc:  # noqa: BLE001 -- a dashboard read must never 500
        return SelfPanelView(unavailable_reason=f"{type(exc).__name__}: {str(exc)[:160]}")

    history = [
        SelfDefinitionVersion(
            version=int(r["version"]),
            created_at=_iso(r["created_at"]),
            content=str(r["content"] or ""),
            evidence_refs=_coerce_evidence_refs(r["evidence_refs"]),
            produced_by=str(r["produced_by"] or ""),
        )
        for r in history_rows
    ]
    journal_entries = [
        SelfInquiryJournalEntry(created_at=_iso(r["created_at"]), body=str(r["body"] or ""))
        for r in journal_rows
    ]
    latest_eval = [
        SelfSenseEvalRow(
            question_key=str(r["question_key"] or ""),
            question=str(r["question"] or ""),
            self_label_score=(int(r["self_label_score"]) if r["self_label_score"] is not None else None),
            grounded_record_score=(
                int(r["grounded_record_score"]) if r["grounded_record_score"] is not None else None
            ),
            answer_source=str(r["answer_source"] or ""),
        )
        for r in eval_rows
    ]
    return SelfPanelView(
        history=history,
        journal_entries=journal_entries,
        latest_eval_run_id=eval_run_id,
        latest_eval=latest_eval,
    )


def to_payload(view: SelfPanelView) -> dict[str, Any]:
    """JSON-shaped for the page. `available: False` is a distinct state from
    an empty-but-readable panel (no definition written yet) -- same
    conflation every other reader here refuses (see atlas.py's own docstring)."""
    if view.is_unavailable:
        return {"available": False, "reason": view.unavailable_reason}
    return {
        "available": True,
        "current": (
            {
                "version": view.current.version,
                "created_at": view.current.created_at,
                "content": view.current.content,
                "evidence_refs": view.current.evidence_refs,
                "produced_by": view.current.produced_by,
            }
            if view.current
            else None
        ),
        "history": [
            {
                "version": h.version,
                "created_at": h.created_at,
                "content": h.content,
                "evidence_refs": h.evidence_refs,
                "produced_by": h.produced_by,
            }
            for h in view.history
        ],
        "journal_entries": [
            {"created_at": j.created_at, "body": j.body} for j in view.journal_entries
        ],
        "latest_eval_run_id": view.latest_eval_run_id,
        "latest_eval": [
            {
                "question_key": e.question_key,
                "question": e.question,
                "self_label_score": e.self_label_score,
                "grounded_record_score": e.grounded_record_score,
                "answer_source": e.answer_source,
            }
            for e in view.latest_eval
        ],
    }
