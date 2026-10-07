"""Orion's Day: the daily letter about what Orion thought about yesterday.

Backend half (this module, orion/orion_day/, the `orion_day.letter` durable
workflow). The Hub half schedules it at 08:30 America/Denver, renders the HTML
email from the persisted row, and offers the carry-forward text to curiosity.

Flow:

    Hub: gather_orion_day(conn, letter_date) -> OrionDayMaterialV1   (read-only SQL)
         build_llm_view(material)            -> OrionDayLlmViewV1    (budgeted text)
         OrionDayRunBriefV1 -> DurableRunRequestV1(workflow="orion_day.letter", admission=agent)
    durable-runs: resource_request -> resource_wait -> write_note -> write_carry_forward
                  -> persist (orion_day_letter row + journal entry of the NOTE) -> finish
    Hub: SELECT * FROM orion_day_letter WHERE letter_date = ... -> OrionDayLetterV1 -> email

Two outputs, never mixed: ``note_md`` is Orion's freeform note about the day (its
prompt carries no instruction about future curiosity at all); ``carry_forward_md``
is a second call's list of threads for future curiosity. They are separate
fields, separate columns and separate verbs.

Every model is ``extra="forbid"``. The dream-hypothesis model is part of the
blind-experiment guard: it has no ``arm`` / ``ref_a`` / ``ref_b`` field, so a
row carrying one cannot even be built.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

ORION_DAY_WORKFLOW = "orion_day.letter"
ORION_DAY_NOTE_VERB = "orion_day_note_v1"
ORION_DAY_CARRY_FORWARD_VERB = "orion_day_carry_forward_v1"
ORION_DAY_TIMEZONE = "America/Denver"
ORION_DAY_LLM_ROUTE = "agent"
ORION_DAY_JOURNAL_TRIGGER_KIND = "orion_day_letter"
ORION_DAY_JOURNAL_SOURCE_KIND = "orion_day"

# The graph's work nodes, in order (services/orion-durable-runs/app/orion_day_graph.py).
ORION_DAY_NODES: tuple[str, ...] = ("write_note", "write_carry_forward", "persist", "finish")

_JOURNAL_NS = uuid5(NAMESPACE_URL, "orion.orion_day.letter.journal.v1")


def orion_day_journal_entry_id(letter_date: date | str) -> str:
    """Stable journal entry id for a day's note: a re-run republishes the same row."""
    return str(uuid5(_JOURNAL_NS, str(letter_date)))


def orion_day_run_id(letter_date: date | str, attempt: int = 1) -> str:
    """Durable run id. The durable store dedupes on run_id and refuses a second request
    with a different brief under the same id, so a retry after a FAILED run needs the
    next attempt number (the letter row itself is still one per day)."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return f"orion-day-{letter_date}-{attempt}"


def _aware(value: datetime | None, name: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- material: what the day contained (full texts; the email shows these) ---------------------


class OrionDaySourceStatusV1(_Model):
    """How one source read went. ``error`` is a failed query (the letter still goes out,
    naming the gap); ``empty`` is a successful read that found nothing that day."""

    status: Literal["ok", "empty", "error"]
    count: int = Field(default=0, ge=0)
    error: str | None = None
    # The read hit its row limit: more rows existed than were gathered.
    truncated: bool = False


class CuriosityRunItemV1(_Model):
    run_id: str
    workflow: str | None = None
    line: str | None = None
    self_question_family: str | None = None
    completed_at: datetime | None = None
    journal_entry_id: str | None = None
    journal_title: str | None = None
    journal_body: str = ""
    # The finish event's capped finding_text; only used when no journal body exists.
    finding_text: str | None = None
    continue_line: bool | None = None
    reach_out: bool | None = None
    reach_out_why: str | None = None
    self_definition_text: str | None = None
    lived_answer_text: str | None = None
    outcome: dict[str, Any] | None = None


class CuriosityFailedRunV1(_Model):
    run_id: str
    workflow: str
    failed_at: datetime
    error: str | None = None


class SelfSenseAnswerV1(_Model):
    run_id: str | None = None
    question_key: str
    question: str
    answer_text: str
    answer_source: str | None = None
    self_label_score: int | None = None
    grounded_record_score: int | None = None
    created_at: datetime


class ReadingItemV1(_Model):
    seed_id: str
    occurred_at: datetime
    title: str | None = None
    url: str | None = None
    why_now: str | None = None
    learned: str = ""
    reading_status: str | None = None
    source_read: bool = False


class JournalTextV1(_Model):
    entry_id: str
    created_at: datetime
    title: str | None = None
    body: str
    source_ref: str | None = None


class DreamNarrativeV1(_Model):
    id: int
    dream_date: date | None = None
    occurred_at: datetime
    tldr: str | None = None
    themes: Any = None
    narrative: str | None = None


class DreamHypothesisV1(_Model):
    """An OFFERED sleep-cycle hypothesis. Deliberately no arm / ref_a / ref_b: the
    hypothesis experiment is blind (orion/dream/hypotheses.py)."""

    hypothesis_id: str
    cycle_id: str | None = None
    claim: str
    why: str | None = None
    offered_at: datetime


class ReverieThoughtV1(_Model):
    thought_id: str
    chain_id: str | None = None
    created_at: datetime
    salience: float | None = None
    interpretation: str = ""
    expectation: str | None = None
    expectation_verdict: str | None = None
    hollow: bool = False


class ReverieChainV1(_Model):
    chain_id: str
    created_at: datetime
    theme_key: str | None = None
    terminal_reason: str | None = None
    ema_salience: float | None = None
    thought_count: int = 0


class VisualReverieV1(_Model):
    """A reference to a reverie image. The image itself is transcoded by the email PR."""

    sha256: str
    chain_id: str | None = None
    step_index: int | None = None
    created_at: datetime
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    path: str | None = None
    description: str | None = None
    theme_key: str | None = None


class WorldPulseDigestItemV1(_Model):
    title: str
    category: str | None = None
    summary: str | None = None
    why_it_matters: str | None = None


class WorldPulseDigestV1(_Model):
    run_id: str
    date: str
    title: str | None = None
    executive_summary: str | None = None
    items: list[WorldPulseDigestItemV1] = Field(default_factory=list)


class OrionDayMaterialV1(_Model):
    """Everything gathered for one Denver calendar day, full length."""

    schema_version: Literal["orion_day.material.v1"] = "orion_day.material.v1"
    letter_date: date
    timezone: str = ORION_DAY_TIMEZONE
    window_start: datetime
    window_end: datetime
    gathered_at: datetime
    curiosity_runs: list[CuriosityRunItemV1] = Field(default_factory=list)
    curiosity_failed: list[CuriosityFailedRunV1] = Field(default_factory=list)
    self_sense: list[SelfSenseAnswerV1] = Field(default_factory=list)
    readings: list[ReadingItemV1] = Field(default_factory=list)
    reading_journals: list[JournalTextV1] = Field(default_factory=list)
    dream_narratives: list[DreamNarrativeV1] = Field(default_factory=list)
    dream_hypotheses: list[DreamHypothesisV1] = Field(default_factory=list)
    reverie_thoughts: list[ReverieThoughtV1] = Field(default_factory=list)
    reverie_chains: list[ReverieChainV1] = Field(default_factory=list)
    visual_reveries: list[VisualReverieV1] = Field(default_factory=list)
    chat_compactor: JournalTextV1 | None = None
    github_compactor: JournalTextV1 | None = None
    world_pulse_digest: WorldPulseDigestV1 | None = None
    sources: dict[str, OrionDaySourceStatusV1] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _window(self):
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        _aware(self.gathered_at, "gathered_at")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self

    def is_empty(self) -> bool:
        """True when the day holds nothing Orion could write about (no empty-shell letters)."""
        return not any((
            self.curiosity_runs, self.self_sense, self.readings, self.reading_journals,
            self.dream_narratives, self.dream_hypotheses, self.reverie_thoughts,
            self.visual_reveries, self.chat_compactor, self.github_compactor,
        ))


# --- the budgeted model input ---------------------------------------------------------------


class OrionDayCondensationV1(_Model):
    """What the budget kept and what it condensed, so the email can say so honestly."""

    reverie_thoughts_total: int = 0
    reverie_thoughts_included: int = 0
    reverie_thoughts_hollow_skipped: int = 0
    reverie_thoughts_duplicate_skipped: int = 0
    # Skipped because a more salient thought from the same chain was already chosen.
    reverie_thoughts_chain_capped: int = 0
    reverie_chains_total: int = 0
    reverie_themes_total: int = 0
    reverie_themes_included: int = 0
    full_text_items_total: int = 0
    full_text_items_clipped: int = 0
    # Per-item character cap applied to full-text bodies; None = nothing clipped.
    full_text_clip_chars: int | None = None


class OrionDayLlmViewV1(_Model):
    schema_version: Literal["orion_day.llm_view.v1"] = "orion_day.llm_view.v1"
    digest_md: str = Field(min_length=1)
    approx_tokens: int = Field(ge=0)
    budget_tokens: int = Field(gt=0)
    chars_per_token: float = Field(gt=0)
    condensed: OrionDayCondensationV1 = Field(default_factory=OrionDayCondensationV1)
    # Every bracketed item reference rendered into digest_md, e.g. "curiosity:ab61e4ccd47b".
    included_refs: list[str] = Field(default_factory=list)


# --- the durable run brief ------------------------------------------------------------------


class OrionDayRunBriefV1(_Model):
    """Everything the `orion_day.letter` run needs, built by Hub at kickoff."""

    letter_date: date
    timezone: str = ORION_DAY_TIMEZONE
    window_start: datetime
    window_end: datetime
    material: OrionDayMaterialV1
    llm_view: OrionDayLlmViewV1
    # Never `chat` (scripts/check_chat_route_poachers.py). The pool spills agent work to the
    # chat card only when that card is lent, which is the pool's call, not this brief's.
    llm_route: Literal["agent"] = ORION_DAY_LLM_ROUTE
    # Per-LLM-call budget: the RPC wait for each verb and AdmissionRuntime.execute's per-node
    # timeout (it reads brief.timeout_sec). Both verbs' YAML timeout_ms stay above it.
    timeout_sec: float = Field(default=1800.0, gt=0.0, le=3600.0)
    # How long the carry-forward text stays eligible for curiosity (sets carry_forward_expires_at).
    carry_forward_ttl_hours: float = Field(default=48.0, gt=0.0, le=24 * 14)

    @model_validator(mode="after")
    def _consistent(self):
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if self.material.letter_date != self.letter_date:
            raise ValueError("material.letter_date must equal letter_date")
        if (self.material.window_start, self.material.window_end) != (self.window_start, self.window_end):
            raise ValueError("material window must equal the brief window")
        return self


# --- the persisted result -------------------------------------------------------------------


class OrionDayLetterSourcesV1(_Model):
    """The ``sources`` column: per-source read status plus what the budget condensed."""

    by_source: dict[str, OrionDaySourceStatusV1] = Field(default_factory=dict)
    condensed: OrionDayCondensationV1 = Field(default_factory=OrionDayCondensationV1)
    approx_tokens: int = 0
    budget_tokens: int = 0


class OrionDayLetterV1(_Model):
    """One row of ``orion_day_letter`` (services/orion-sql-db/manual_migration_orion_day_letter_v1.sql).

    Hub reads it with ``OrionDayLetterV1.model_validate(dict(row))`` after decoding the
    jsonb columns; ``orion.orion_day.store.fetch_letter`` does exactly that."""

    letter_date: date
    run_id: str
    window_start: datetime
    window_end: datetime
    note_md: str = Field(min_length=1)
    carry_forward_md: str = Field(min_length=1)
    material: OrionDayMaterialV1
    sources: OrionDayLetterSourcesV1
    journal_entry_id: str | None = None
    created_at: datetime
    emailed_at: datetime | None = None
    email_notification_id: str | None = None
    carry_forward_expires_at: datetime | None = None
    carry_forward_offered_at: datetime | None = None
    carry_forward_offered_run_id: str | None = None

    @model_validator(mode="after")
    def _separate(self):
        if self.note_md.strip() == self.carry_forward_md.strip():
            raise ValueError("note_md and carry_forward_md must be distinct outputs")
        return self
