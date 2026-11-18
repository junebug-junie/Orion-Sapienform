"""dream.day: once a night, one durable run in which Orion reads the whole Denver day that just closed and
writes what it wants to remember of it.

Design: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 2 (this module is Patch 2a: the
plan is parsed, ref-checked and stored, but NOT handed to Prime; Patch 2b does that).

    gather -> resource_request -> resource_wait -> dream (one harness turn) -> persist -> finish

* ``gather`` (code, durable-runs): the day's ``dream_cycle`` rows with their hypotheses and pair answers,
  Prime's held concerns (read-only), and ``GET /temporal-self/now?organ=dream`` (the read-back the
  integration gate requires). Stored as ``DreamDayMaterialV1`` in the checkpoint and the ``dream_day`` row.
* ``dream`` (one unified turn on the run's ``memory_distill`` hold): the turn first runs
  ``python -P -m orion.dig dream-day <day>`` inside the governor, the only place the background half of the
  day's digs is readable (orion/dig/digs.py). That command prints the organ roundup and every dig Orion ran
  that day, and writes a read receipt (``DreamDayReadV1``) code checks after the turn.
* ``persist`` (code): every ref must resolve; at most ``DREAM_DAY_MAX_MEMORIES`` memories go through
  orion.memory.episode.validate and store with ``channel='dream'`` in one transaction with the ``dream_day``
  row; the now's cites are recorded with ``record_use``.

ADDITIVE ``DurableWorkflowV1`` value and ``DurableRunRequestV1.brief`` union member on ``extra="forbid"`` /
``Literal`` models: deploy the state consumers (orion-sql-writer, orion-hub, orion-actions) and
orion-cortex-orch before orion-durable-runs submits the first run.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

DREAM_DAY_WORKFLOW = "dream.day"
DREAM_DAY_RUN_PREFIX = "dream-day-"
# One Hub session for every day-dream turn; Hub's resume key (run_id:source_tag) gives each run its own claude
# session, and a retry of the same run resumes it.
DREAM_DAY_SESSION_ID = "dream_day"
DREAM_DAY_SOURCE_TAG = "dream_day"
DREAM_DAY_STEP_KEY = "dream_day"
DREAM_DAY_PROMPT_VERSION = "dream_day.v1"
# The turn's reply is graded against these (Juniper took the bracketed defaults 2026-11-17).
DREAM_DAY_MAX_MEMORIES = 5
DREAM_DAY_MAX_NEW_CONCERNS = 1
DREAM_DAY_TEXT_MAX = 1200
# Ref kinds a memory, a plan item or the day paragraph may cite (spec Patch 2 "Persist"). ``chat`` is the
# kind `dig digs` prints for a chat turn; it resolves exactly like ``chat_turn``.
DREAM_DAY_REF_KINDS: tuple[str, ...] = (
    "experience", "act", "dream_cycle", "dream_hypothesis", "harness_turn_trace", "chat_turn", "chat")
REF_RE = re.compile(r"^\[?\s*(?P<kind>[a-z_]+):(?P<id>[^\s\]]{1,300})\s*\]?$")
# Purposes a dream memory may take. ``follow_up`` is left out on purpose: carrying something into tomorrow is
# the plan's job, and a follow_up memory with no due date would never come due.
DREAM_DAY_PURPOSES: tuple[str, ...] = ("happened", "orion_view", "about_juniper")
# Where the dig writes its read receipt, under the dig's self dir ($ORION_DIG_HOME/self, the same host dir
# orion-durable-runs mounts for the field-note and reminiscence ledgers): <self>/dream_day/<day>.json.
DREAM_DAY_RECEIPT_DIR = "dream_day"


def base_run_id(day: date | str) -> str:
    return f"{DREAM_DAY_RUN_PREFIX}{day if isinstance(day, str) else day.isoformat()}"


def split_ref(raw: Any) -> Optional[tuple[str, str]]:
    """``"act:act_ab12"`` / ``"[experience:dream_cycle:dc-1]"`` -> (kind, id), or None when the kind is not one
    a day dream may cite (or the text is not a ref at all). The id keeps any further colons."""
    m = REF_RE.match(str(raw or "").strip())
    if m is None or m.group("kind") not in DREAM_DAY_REF_KINDS:
        return None
    return m.group("kind"), m.group("id")


class DreamDayBriefV1(BaseModel):
    """The run's whole input at submit. ``dream_day`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    dream_day: Literal["dream.day"]
    # The Denver day the dream reads: the day that closed before the run's night window.
    day: date
    # One turn attempt's hard bound (the attempt timer and Hub's turn limit derive from it).
    timeout_sec: float = Field(ge=60.0, le=3600.0)
    # The pool route the run's hold is asked on (DREAM_DAY_ROUTE). Under a hold every call of the turn runs on
    # the hold's role (Hub names the hold's work-class route).
    llm_route: str = Field(min_length=1, max_length=64)
    # POST /dream-day/run (an operator's hand trigger, outside the night window): shown on the run, nothing else.
    operator_forced: bool = False


class DreamDayHypothesisV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    arm: Optional[str] = None
    claim: str = ""
    why: str = ""


class DreamDayCycleV1(BaseModel):
    """One of the day's sleeps (a ``dream_cycle`` row), as the brief shows it."""

    model_config = ConfigDict(extra="forbid")

    cycle_id: str
    status: str
    started_at: AwareDatetime
    ended_at: Optional[AwareDatetime] = None
    trigger: Optional[str] = None
    hypothesis_count: Optional[int] = None
    no_link_count: Optional[int] = None
    # cycle_json.pair_answers (PR #3636): each pair's refs, outcome and the model's raw answer (cut).
    pair_answers: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[DreamDayHypothesisV1] = Field(default_factory=list)


class DreamDayConcernV1(BaseModel):
    """One concern Prime holds, read-only from ``orion:prime:state``."""

    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str
    age_h: float = Field(ge=0.0)
    # Hours since evidence last touched it (PrimeConcernV1.last_evidence_at): its last action.
    last_evidence_h: float = Field(ge=0.0)
    graded: Optional[str] = None


class DreamDayNowItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    organ: str
    label: str = ""


class DreamDayMaterialV1(BaseModel):
    """What code laid out for the turn before it ran (the half of the brief durable-runs can read; the organ
    roundup and the digs are read inside the turn by ``dig dream-day``, see ``DreamDayReadV1``)."""

    model_config = ConfigDict(extra="forbid")

    day: date
    window_start: AwareDatetime
    window_end: AwareDatetime
    dreams: list[DreamDayCycleV1] = Field(default_factory=list)
    # ok | unreadable (the dream_cycle read failed). Never shown as "no sleeps".
    dreams_status: Literal["ok", "unreadable"] = "ok"
    concerns: list[DreamDayConcernV1] = Field(default_factory=list)
    # ok | absent (no saved state) | unreadable (a read or parse error). Never shown as "no concerns".
    concerns_status: Literal["ok", "absent", "unreadable"] = "ok"
    now_shown_id: Optional[str] = None
    now_items: list[DreamDayNowItemV1] = Field(default_factory=list)
    # ok | unavailable (the now read failed: the run goes on without it, and records no cites).
    now_status: Literal["ok", "unavailable"] = "ok"
    gathered_at: AwareDatetime


class DreamDayReadV1(BaseModel):
    """The read receipt ``dig dream-day <day>`` writes in the governor (``<self>/dream_day/<day>.json``). Code
    checks it after the turn: the turn really read the day, and which parts it could not read."""

    model_config = ConfigDict(extra="forbid")

    day: date
    read_at: AwareDatetime
    roundup_lines: int = Field(ge=0)
    roundup_refs: int = Field(ge=0)
    digs_total: int = Field(ge=0)
    digs_chat: int = Field(ge=0)
    digs_background: int = Field(ge=0)
    # part name -> why it could not be read (the dig's own UNREAD tail). Empty = the whole day was read.
    unread: dict[str, str] = Field(default_factory=dict)


class DreamDayMemoryV1(BaseModel):
    """One memory the turn proposes. Lenient on purpose (model output): code re-checks every field."""

    model_config = ConfigDict(extra="ignore")

    statement: str = Field(min_length=1, max_length=2000)
    purpose: str = "happened"
    refs: list[str] = Field(default_factory=list, max_length=12)


class DreamDayPlanItemV1(BaseModel):
    """keep / drop a held concern (``concern_id``), or one ``new`` concern (``gist``)."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["keep", "drop", "new"]
    concern_id: Optional[str] = None
    gist: Optional[str] = Field(default=None, max_length=500)
    reason: str = Field(default="", max_length=1000)
    refs: list[str] = Field(default_factory=list, max_length=12)


class DreamDayParagraphV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=4000)
    refs: list[str] = Field(default_factory=list, max_length=24)


class DreamDayAnswerV1(BaseModel):
    """The turn's reply, the JSON block it ends with (parsed by orion/memory/dream_day.py ``parse_answer``)."""

    model_config = ConfigDict(extra="ignore")

    memories: list[DreamDayMemoryV1] = Field(default_factory=list)
    plan: list[DreamDayPlanItemV1] = Field(default_factory=list)
    day: Optional[DreamDayParagraphV1] = None


class DreamDayV1(BaseModel):
    """One ``dream_day`` row (services/orion-sql-db/manual_migration_dream_day.sql). One row per Denver day,
    first run to persist wins."""

    model_config = ConfigDict(extra="forbid")

    day: date
    run_id: str
    status: Literal["written"] = "written"
    produced: bool
    # "no lasting trace" when produced is False; else a short count line.
    note: str = ""
    memories_written: int = Field(ge=0, le=DREAM_DAY_MAX_MEMORIES)
    memory_ids: list[str] = Field(default_factory=list, max_length=DREAM_DAY_MAX_MEMORIES)
    # Items (memories, plan items, the day paragraph) dropped for an unresolved or unsupported ref, and how
    # many refs did not resolve. Never written, only counted.
    items_dropped: int = Field(ge=0)
    refs_dropped: int = Field(ge=0)
    # Memories past the cap, refused by the validator, or plan items that name no held concern / a second new.
    memories_rejected: int = Field(ge=0)
    plan_rejected: int = Field(ge=0)
    kept: int = Field(ge=0)
    dropped: int = Field(ge=0)
    new: int = Field(ge=0, le=DREAM_DAY_MAX_NEW_CONCERNS)
    # The parsed, ref-checked plan (Patch 2a stores it; Patch 2b hands it to Prime).
    plan: list[dict[str, Any]] = Field(default_factory=list)
    day_text: Optional[str] = Field(default=None, max_length=DREAM_DAY_TEXT_MAX)
    day_refs: list[str] = Field(default_factory=list)
    material: dict[str, Any] = Field(default_factory=dict)
    read: Optional[dict[str, Any]] = None
    now_shown_id: Optional[str] = None
    turn_correlation_id: Optional[str] = None
    created_at: datetime
