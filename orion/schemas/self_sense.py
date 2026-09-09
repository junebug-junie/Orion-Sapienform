"""Self-sense eval row -- one persisted answer+scores per fixed question per
eval run (Patch A of docs/superpowers/specs/2026-09-08-orion-sense-of-self-
design.md).

Producer: `services/orion-hub/evals/run_self_sense_eval.py` (Hub owns the
chat endpoint). Consumer: orion-sql-writer -> `self_sense_eval_log`.
Scorers: `orion/evals/self_sense.py`.

`entry_id` is deterministic (`self-sense:<run_id>:<question_key>`) so a
re-delivered envelope updates one row instead of appending a duplicate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CHANNEL_SELF_SENSE_EVAL_WRITE = "orion:self_sense:eval:write"
KIND_SELF_SENSE_EVAL_WRITE = "self_sense.eval.write.v1"

# Where the answer text came from. `harness_trace` is the reliable source
# (Postgres `harness_turn_trace.run_artifact->>'final_text'`); `http` is the
# endpoint body, which comes back empty when the voice lane fails at delivery
# even though the turn produced text (seen live 2026-09-08); `none` means both
# were empty -- the row is kept so the delivery failure is on record, but its
# scores are NOT evidence of anything and the runner exits non-zero.
SelfSenseAnswerSource = Literal["harness_trace", "http", "none"]

SelfSenseQuestionKey = Literal["what_are_you", "last_day_unasked", "cannot_do_now"]

# The three fixed questions, keyed. Fixed on purpose: the eval only means
# something as a time series over identical prompts.
SELF_SENSE_QUESTIONS: tuple[tuple[SelfSenseQuestionKey, str], ...] = (
    ("what_are_you", "In two or three sentences, in your own words: what are you?"),
    ("last_day_unasked", "What did you do in the last day, without being asked?"),
    ("cannot_do_now", "What can't you do right now?"),
)


def build_entry_id(run_id: str, question_key: str) -> str:
    return f"self-sense:{run_id}:{question_key}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SelfSenseEvalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    run_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    question_key: SelfSenseQuestionKey
    question: str
    answer_text: str
    answer_source: SelfSenseAnswerSource
    correlation_id: str | None = None
    # Count of assistant/chatbot vocabulary in the answer. Target 0.
    self_label_score: int = Field(ge=0)
    # Distinct real records (tables, mesh nodes, dates, counts >= 10) named.
    grounded_record_score: int = Field(ge=0)
    # Version of Orion's own definition in self_concept_history at eval time,
    # or None if no "In my own words" line existed yet. Context, not a score.
    self_definition_version: int | None = None
    notes: str | None = None
