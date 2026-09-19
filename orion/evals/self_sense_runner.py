"""Self-sense eval: the row assembly shared by BOTH runners.

Two things ask Orion the four fixed questions in `orion/schemas/self_sense.py`:

* the host script `services/orion-hub/evals/run_self_sense_eval.py`
  (`make eval-self-sense`), which POSTs `/api/chat` and reads the answer back
  from `harness_turn_trace`; and
* the once-a-day line inside Hub's curiosity scheduler
  (`services/orion-hub/scripts/curiosity_investigation.py`,
  `tick_self_sense_eval`), which runs the turn in-process.

They must produce byte-identical rows for the same answer text, so the answer
source choice, the scoring, the notes and the envelope live here, once. No
network, no database: the two callers own the live call and the reads.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from orion.evals.self_sense import (
    LIVED_QUESTION_KEY_TO_ID,
    SELF_DEFINITION_CONCEPT_ID,
    SELF_DEFINITION_PRODUCED_BY,
    LivedLedgerGrounding,
    grounded_records,
    lived_ledger_grounding,
    self_label_hits,
    self_label_score,
)
from orion.schemas.self_sense import (
    KIND_SELF_SENSE_EVAL_WRITE,
    SelfSenseEvalV1,
    build_entry_id,
)

# The chat session both runners ask under. Fixed so the identity inject and
# any per-session continuity behave the same whether the host script or the
# Hub scheduler asked.
SESSION_ID = "self-sense-eval"
# The catalogue names orion-hub as the producer because Hub owns the chat
# endpoint. The host script executes outside the container but reports the
# same producer name; the Hub scheduler passes its own ServiceRef.
PRODUCER_SERVICE = "orion-hub"
PRODUCER_VERSION = "self-sense-eval/0.1.0"

# asyncpg-style (positional) twins of the SQLAlchemy (named) statements in
# orion/evals/self_sense.py. Hub's memory pool is asyncpg; the host script
# uses SQLAlchemy. Same tables, same filters, same ordering.
SELF_DEFINITION_VERSION_SQL_POSITIONAL = (
    "SELECT version FROM self_concept_history "
    "WHERE concept_id = $1 AND produced_by = $2 "
    "ORDER BY created_at DESC LIMIT 1"
)
SELF_DEFINITION_VERSION_ARGS: tuple[str, str] = (
    SELF_DEFINITION_CONCEPT_ID,
    SELF_DEFINITION_PRODUCED_BY,
)

_LIVED_ANSWERS_SQL_TEMPLATE = (
    "SELECT DISTINCT ON (concept_id) concept_id, content, evidence_refs, created_at "
    "FROM self_concept_history "
    "WHERE concept_id IN ({placeholders}) "
    "AND produced_by = 'curiosity_self_inquiry' "
    "ORDER BY concept_id, created_at DESC"
)


def lived_answers_sql_named(concept_ids: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """SQLAlchemy `text()` form: `:cid0, :cid1, ...` plus its params."""
    placeholders = ", ".join(f":cid{i}" for i in range(len(concept_ids)))
    params = {f"cid{i}": cid for i, cid in enumerate(concept_ids)}
    return _LIVED_ANSWERS_SQL_TEMPLATE.format(placeholders=placeholders), params


def lived_answers_sql_positional(concept_ids: tuple[str, ...]) -> str:
    """asyncpg form: `$1, $2, ...`; pass `*concept_ids` as the args."""
    placeholders = ", ".join(f"${i + 1}" for i in range(len(concept_ids)))
    return _LIVED_ANSWERS_SQL_TEMPLATE.format(placeholders=placeholders)


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _lived_ledger_notes(ledger: LivedLedgerGrounding) -> list[str]:
    if not ledger.applicable:
        return ["lived_ledger=n/a"]
    if ledger.grounded:
        ids = ",".join(ledger.matched_question_ids) or "unknown"
        return [f"lived_ledger=grounded:{ids}"]
    return ["lived_ledger=miss"]


def is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def build_row(
    *,
    run_id: str,
    question_key: str,
    question: str,
    http_text: str | None,
    trace_text: str | None,
    correlation_id: str | None,
    self_definition_version: int | None,
    trace_missing_after_sec: float | None = None,
    lived_answers: list[dict[str, Any]] | None = None,
) -> SelfSenseEvalV1:
    """Pick the answer source (trace beats HTTP; empty is 'none'), score it,
    and build the row. Pure: no network, no database.

    `trace_missing_after_sec`: how long the runner waited for
    harness_turn_trace before giving up, recorded in notes whenever the
    answer had to come from the HTTP body instead."""
    if trace_text and trace_text.strip():
        answer, source = trace_text.strip(), "harness_trace"
    elif http_text and http_text.strip():
        answer, source = http_text.strip(), "http"
    else:
        answer, source = "", "none"

    labels = self_label_hits(answer)
    grounded = grounded_records(answer)
    notes: list[str] = []
    if source == "none":
        notes.append("answer_empty_from_both_sources; scores are not a measurement")
    if source != "harness_trace" and correlation_id and trace_missing_after_sec is not None:
        notes.append(f"trace_missing_after={trace_missing_after_sec:.0f}s")
    if not is_uuid(correlation_id):
        notes.append("envelope_corr=synthetic")
    if labels:
        notes.append("labels=" + ",".join(labels))
    if grounded.records:
        notes.append("records=" + ",".join(grounded.records))

    if question_key in LIVED_QUESTION_KEY_TO_ID:
        focus = LIVED_QUESTION_KEY_TO_ID[question_key]
        ledger = lived_ledger_grounding(answer, lived_answers, focus_question_id=focus)
        notes.extend(_lived_ledger_notes(ledger))

    return SelfSenseEvalV1(
        entry_id=build_entry_id(run_id, question_key),
        run_id=run_id,
        question_key=question_key,  # type: ignore[arg-type]
        question=question,
        answer_text=answer,
        answer_source=source,  # type: ignore[arg-type]
        correlation_id=correlation_id,
        self_label_score=self_label_score(answer),
        grounded_record_score=grounded.score,
        self_definition_version=self_definition_version,
        notes="; ".join(notes) or None,
    )


def envelope_correlation_id(row: SelfSenseEvalV1) -> uuid.UUID:
    """BaseEnvelope requires a UUID. Reuse the chat turn's correlation_id when
    it is one (it is -- Hub mints a uuid4), otherwise derive a deterministic
    uuid5 from the entry_id so a replay carries the same id."""
    if is_uuid(row.correlation_id):
        return uuid.UUID(row.correlation_id)  # type: ignore[arg-type]
    return uuid.uuid5(uuid.NAMESPACE_URL, "orion:self_sense:" + row.entry_id)


def build_envelope(row: SelfSenseEvalV1, *, node: str | None = None, source: Any | None = None):
    """The bus envelope for one row. `source` is a ServiceRef when the caller
    has one (the Hub scheduler); otherwise the nominal host-script producer."""
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(
        kind=KIND_SELF_SENSE_EVAL_WRITE,
        source=source or ServiceRef(name=PRODUCER_SERVICE, version=PRODUCER_VERSION, node=node),
        correlation_id=envelope_correlation_id(row),
        payload=row.model_dump(mode="json"),
    )
