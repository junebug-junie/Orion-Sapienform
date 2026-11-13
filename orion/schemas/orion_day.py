"""Orion's Day: the daily letter about what Orion lived through yesterday.

Since 2026-11-12 (living Prime design #3224, Part 3) the letter is Prime's end-of-day turn:

    Prime (orion-mind): the temporal chronicle closed yesterday (GET /temporal-self/day/<date> on
        orion-durable-runs) and Hub says it has no letter (GET /api/orion-day/letters/<date>)
        -> Prime bids a ``make`` act (actor prime) in the body economy; deferred -> no letter
           unless a later bid inside the window is funded (then the next letter says so)
        -> funded, or a capacity deferral worth its price (queues at the GPU pool): Prime opens the act and orders the run (POST Hub /api/orion-day/letter-run,
           OrionDayLetterOrderV1)
    Hub: gather_orion_day(conn, letter_date) -> OrionDayMaterialV1          (read-only SQL)
         render_day_input(material)          -> OrionDayLlmViewV1           (orion/orion_day/day_input.py)
         OrionDayRunBriefV1 -> DurableRunRequestV1(workflow="orion_day.letter", admission=agent)
    durable-runs: resource_request -> resource_wait -> write_letter (ONE unified turn on the
                  durable turn seam, session ``orion_day_letter``; prompt built by
                  orion/orion_day/letter_turn.py) -> persist (orion_day_letter row + journal
                  entry of the NOTE) -> finish
    Hub: emails the persisted row (unchanged)
    Prime: GET /runs/<run_id> -> closes the make act; the letter's threads become Prime concerns

Two outputs, never mixed: ``note_md`` is Orion's letter; ``carry_forward_md`` is the letter's
own closing list of threads (split off the same turn's reply by ``letter_turn.split_letter``).
The threads reach curiosity only through Prime (a concern, then Prime's graded ``ask``).

Every model is ``extra="forbid"``. The dream-hypothesis model is part of the
blind-experiment guard: it has no ``arm`` / ``ref_a`` / ``ref_b`` field, so a
row carrying one cannot even be built.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.schemas.reading_finding import ReadingFindingV1
from orion.schemas.temporal_self import DayCitationCheckV1, TemporalSelfDayBlockV1

ORION_DAY_WORKFLOW = "orion_day.letter"
ORION_DAY_TIMEZONE = "America/Denver"
ORION_DAY_LLM_ROUTE = "agent"
ORION_DAY_JOURNAL_TRIGGER_KIND = "orion_day_letter"
ORION_DAY_JOURNAL_SOURCE_KIND = "orion_day"

# The graph's work nodes, in order (services/orion-durable-runs/app/orion_day_graph.py).
ORION_DAY_NODES: tuple[str, ...] = ("write_letter", "persist", "finish")
# The letter turn's Hub session and turn source. The session id is the one Prime's turn hand-back
# reads as "letter" (orion/prime/handback.py LETTER_SESSION_ID; a test pins the two together).
ORION_DAY_LETTER_SESSION_ID = "orion_day_letter"
ORION_DAY_LETTER_SOURCE_TAG = "orion_day_letter"
# Prime's order to Hub (HTTP, orion-hub scripts/orion_day_order.py).
ORION_DAY_ORDER_PATH = "/api/orion-day/letter-run"
ORION_DAY_STATUS_PATH = "/api/orion-day/letters/{letter_date}"
_RUN_ID_PATTERN = r"^orion-day-\d{4}-\d{2}-\d{2}-\d+$"

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
    # Added 2026-10-13 (reading findings). Default-empty so rows stored before still validate;
    # same rollout rule as ``corrections``: rebuild orion-durable-runs (the forbid reader) with
    # or before orion-hub. ``findings``: what this read changed (orion/world_pulse_read/findings.py).
    request_id: str | None = None
    article_summary: str = ""  # what the piece itself says (Stage 2 ``article_summary``)
    findings: list[ReadingFindingV1] = Field(default_factory=list)


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


class WalkwayDayV1(_Model):
    """The street outside on this day: the deterministic day rollup of the walkway camera
    (``vision_rollup`` grain=day, orion/vision/walkway_rollup.py), as its one template line.
    Counts only -- no LLM wrote it. ``complete`` false = the rollup did not cover the whole day
    (camera down, or the reducer was behind); the line itself says which hours."""

    stream_id: str
    day: date
    line: str
    complete: bool = False


class JuniperChatTurnV1(_Model):
    """One chat turn between Juniper and Orion (``chat_history_log``), full text.

    ``unsolicited`` = Orion wrote to Juniper first (a curiosity outreach: empty prompt,
    ``client_meta.unsolicited``); ``prompt`` is then empty and ``response`` is what Orion sent."""

    turn_id: str
    correlation_id: str | None = None
    at: datetime
    prompt: str = ""
    response: str = ""
    unsolicited: bool = False


class OrionDayCorrectionV1(_Model):
    """A letter claim Juniper and Orion agreed was wrong, as recorded by Orion
    (``orion_day_correction``, orion/orion_day/corrections.py). In a day's material when it was
    recorded that day, so the next letter's writer sees it and does not re-derive the claim.
    The outcome fields are as of the gather: None = not stamped yet, or (moved) unknown."""

    correction_id: str
    created_at: datetime
    letter_date: date
    ref: str
    claim: str
    correction: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    prior_ids: list[str] = Field(default_factory=list)
    offered_run_id: str | None = None
    priors_retested: list[str] | None = None
    priors_moved: list[str] | None = None


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
    # Added 2026-10-11 (letter reread, patch 3). Default-empty so letter rows stored before it
    # still validate. Hub writes it into the durable brief: rebuild orion-durable-runs (the
    # forbid reader) before or with orion-hub.
    corrections: list[OrionDayCorrectionV1] = Field(default_factory=list)
    # Added 2026-10-11 (Temporal Self patch 4): the closed day as Orion's own chronology recorded
    # it (temporal_self_day), rendered by a fixed template with [arc:<id>] refs. None when the day
    # was not closed/chronicled: renders nothing. Same rollout rule as ``corrections``.
    temporal_day: TemporalSelfDayBlockV1 | None = None
    # Added 2026-10-18 (walkway rollups): the walkway day rollup line. None = no row (table not
    # migrated, reducer off, camera not installed). Same rollout rule as ``corrections``.
    walkway_day: WalkwayDayV1 | None = None
    # Added 2026-11-12 (living Prime design #3224, constraint 7): Juniper's chat turns for the day,
    # full text, and Orion's unsolicited messages to her. Default-empty so stored rows still validate.
    # ROLLOUT: unlike ``corrections`` this goes both ways -- Hub writes the brief durable-runs reads,
    # and durable-runs writes the letter row (material + sources.input_*) Hub reads with forbid
    # models. Deploy orion-hub and orion-durable-runs together, outside the letter's 08:30 fire window.
    juniper_chat: list[JuniperChatTurnV1] = Field(default_factory=list)
    # Added 2026-11-12 (constraint 6): the temporal core -- expectations, acts and their graded
    # outcomes, threads with Juniper, the body's day -- rendered from the closed day by a fixed
    # template (orion/orion_day/temporal_core.py), cited by [arc:<id>] / [event:<id>]. Same rule.
    temporal_core: TemporalSelfDayBlockV1 | None = None
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
            any(t.prompt.strip() for t in self.juniper_chat),
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
    # Added 2026-11-12 (honest input, constraint 7). Same two-way rollout as OrionDayMaterialV1.juniper_chat.
    # Self-sense answers gathered vs the one-per-question answers shown.
    self_sense_answers_total: int = 0
    self_sense_answers_shown: int = 0
    # The trimmed tail (readings, journals, dreams, news, code changes, street, visual reveries):
    # items offered, items left out at its cap. Every tail item shown is cut to one short line.
    tail_items_total: int = 0
    tail_items_left_out: int = 0


class OrionDayLlmViewV1(_Model):
    schema_version: Literal["orion_day.llm_view.v1"] = "orion_day.llm_view.v1"
    digest_md: str = Field(min_length=1)
    approx_tokens: int = Field(ge=0)
    budget_tokens: int = Field(gt=0)
    chars_per_token: float = Field(gt=0)
    condensed: OrionDayCondensationV1 = Field(default_factory=OrionDayCondensationV1)
    # Every bracketed item reference rendered into digest_md, e.g. "curiosity:ab61e4ccd47b".
    included_refs: list[str] = Field(default_factory=list)
    # Added 2026-11-12 (constraint 7): characters each source got in digest_md, and that over the
    # digest's length, so the mix can be inspected. Empty on views built before it (build_llm_view).
    source_chars: dict[str, int] = Field(default_factory=dict)
    source_shares: dict[str, float] = Field(default_factory=dict)


# --- the durable run brief ------------------------------------------------------------------


class MissedLetterV1(_Model):
    """A day with no letter, as Prime recorded it (every attempt's run failed, Hub refused the order,
    or the day closed while Prime was down). The next letter's turn is told, once, plainly."""

    letter_date: date
    reason: str = Field(min_length=1, max_length=300)


class OrionDayRunBriefV1(_Model):
    """Everything the `orion_day.letter` run needs. Built by Hub on Prime's order."""

    letter_date: date
    timezone: str = ORION_DAY_TIMEZONE
    window_start: datetime
    window_end: datetime
    material: OrionDayMaterialV1
    llm_view: OrionDayLlmViewV1
    # Never `chat` (scripts/check_chat_route_poachers.py). The pool spills agent work to the
    # chat card only when that card is lent, which is the pool's call, not this brief's.
    llm_route: Literal["agent"] = ORION_DAY_LLM_ROUTE
    # The letter turn's budget per attempt (the attempt timer and Hub's turn limit derive from it).
    timeout_sec: float = Field(default=1800.0, ge=60.0, le=3600.0)
    # Added 2026-11-12 (living Prime Part 3). Prime's make act that ordered this run, and the
    # days before it that got no letter. ROLLOUT: additive on a forbid model. Hub deploys first but
    # sends them only on Prime's order, and orion-mind (Prime) deploys after orion-durable-runs, so
    # the reader always has them before any brief carries them.
    prime_act_id: str | None = Field(default=None, max_length=200)
    missed_letters: list[MissedLetterV1] = Field(default_factory=list, max_length=7)
    # DEPRECATED, IGNORED (living Prime Part 3 retired the curiosity carry-forward offer it timed).
    # Still accepted so a brief built by the retired Hub timer -- an accepted run in flight across
    # the deploy, or an old Hub not yet redeployed -- validates instead of failing load_brief.
    # Nothing reads it; the letter row's carry_forward_expires_at is always written NULL.
    carry_forward_ttl_hours: float | None = Field(default=None, exclude=True)

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
    # Temporal Self patch 4: the citation gate on the note and the carry-forward ("note",
    # "carry_forward"). Unknown [arc:] ids were removed from the stored text and are listed here;
    # label UNVERIFIED = a sentence named the closed day's record without citing an arc.
    arc_citations: dict[str, DayCitationCheckV1] = Field(default_factory=dict)
    # Added 2026-11-12 (constraint 7): per-source characters and shares of the writer's input
    # (OrionDayLlmViewV1.source_chars / source_shares). Empty on rows written before it.
    input_chars: dict[str, int] = Field(default_factory=dict)
    input_shares: dict[str, float] = Field(default_factory=dict)
    # [event:<id>] refs the gate removed because the temporal core did not show them, per text
    # ("note", "carry_forward"). Empty = none removed.
    rejected_event_refs: dict[str, list[str]] = Field(default_factory=dict)
    # Added 2026-11-12 (living Prime Part 3): the unified turn that wrote the letter (find it in
    # cockpit_turn_sighting / harness_turn_trace), Prime's make act, and the missed days the turn
    # was told about. ROLLOUT: Hub (the forbid reader) deploys before orion-durable-runs writes them.
    turn_correlation_id: str | None = None
    prime_act_id: str | None = None
    missed_letters: list[MissedLetterV1] = Field(default_factory=list)


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
    # The retired curiosity carry-forward offer's columns (offer retired 2026-11-12, living Prime
    # Part 3: threads reach curiosity through Prime). Never written any more; kept so old rows load.
    carry_forward_expires_at: datetime | None = None
    carry_forward_offered_at: datetime | None = None
    carry_forward_offered_run_id: str | None = None

    @model_validator(mode="after")
    def _separate(self):
        if self.note_md.strip() == self.carry_forward_md.strip():
            raise ValueError("note_md and carry_forward_md must be distinct outputs")
        return self


# --- Prime's order and Hub's answers (HTTP; living Prime Part 3) ----------------------------


class OrionDayLetterStatusV1(_Model):
    """``GET {HUB}/api/orion-day/letters/{letter_date}``: does the day have a letter yet."""

    schema_version: Literal["orion_day.letter_status.v1"] = "orion_day.letter_status.v1"
    letter_date: date
    letter_exists: bool
    run_id: str | None = None
    emailed: bool = False


class OrionDayLetterOrderV1(_Model):
    """``POST {HUB}/api/orion-day/letter-run``: Prime's make act (tier 0, no bid) orders the day's letter, once per attempt.

    Hub gathers the day, builds the brief and submits the durable run under exactly this ``run_id``
    and ``correlation_id`` (so Prime can follow it with ``GET {DURABLE}/runs/{run_id}``)."""

    schema_version: Literal["orion_day.letter_order.v1"] = "orion_day.letter_order.v1"
    letter_date: date
    run_id: str = Field(pattern=_RUN_ID_PATTERN, max_length=80)
    correlation_id: str = Field(min_length=1, max_length=200)
    prime_act_id: str = Field(min_length=1, max_length=200)
    missed_letters: list[MissedLetterV1] = Field(default_factory=list, max_length=7)
    deadline_at: datetime
    timeout_sec: float = Field(default=1800.0, ge=60.0, le=3600.0)

    @model_validator(mode="after")
    def _checks(self):
        _aware(self.deadline_at, "deadline_at")
        if not self.run_id.startswith(f"orion-day-{self.letter_date.isoformat()}-"):
            raise ValueError("run_id must be orion_day_run_id(letter_date, n)")
        return self


OrionDayOrderStatus = Literal["submitted", "exists", "already_written", "empty_day", "refused", "unavailable"]


class OrionDayLetterOrderResultV1(_Model):
    """Hub's answer to an order. ``submitted`` / ``exists`` (the same run is already in the
    registry): follow ``run_id``. ``already_written``: a letter row exists (``run_id`` = its run).
    ``empty_day``: nothing to write about, no run. ``refused``: durable-runs refused the request
    (``detail``). ``unavailable``: the store or durable-runs could not be reached; try again."""

    schema_version: Literal["orion_day.letter_order_result.v1"] = "orion_day.letter_order_result.v1"
    letter_date: date
    run_id: str | None = None
    status: OrionDayOrderStatus
    detail: str | None = Field(default=None, max_length=500)
