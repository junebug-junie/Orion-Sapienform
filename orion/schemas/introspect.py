"""Introspect tool contracts: Orion reading back their own recorded activity.

Design: docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md.
``reading_result`` travels on the reading contract; every other operation is an
``IntrospectBusOperation`` answered by its owning service over
``orion:introspect:<domain>:request``.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# MAX_ITEMS full-size items stay under ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS
# (12000), the budget the harness proxy enforces on wrapped servers (it cuts
# the JSON mid-item). orion-introspect is not wrapped today; the bound is kept
# so wrapping it later cannot corrupt a result.
MAX_ITEMS = 5
QUERY_CAP = 500
DEFAULT_LIMIT = 5
DEFAULT_TEXT_CAP = 900
SHORT_FIELD_CAP = 200
URL_CAP = 500

FULL_TEXT_CAP = 4000
THEME_CAP = 8
DREAM_ID_PATTERN = r"^(dream:[0-9]{1,12}|dh-[0-9a-f]{6,32})$"

# One curiosity run by id returns its write-up up to this many characters AS
# SERIALIZED JSON (`clip_json_text`), so quote/newline escaping cannot push a
# full item past the 12k MCP budget. Same shape as
# `orion.curiosity.atlas._RUN_ID_RE`; a test pins the two equal.
CURIOSITY_FULL_JSON_BUDGET = 9000
# The run join reads at most this far back (orion-hub curiosity_run_store
# WINDOW_DAYS_MAX, pinned equal by a Hub test); an older `since` is refused
# rather than silently answered from a shorter window.
CURIOSITY_WINDOW_DAYS = 90
CURIOSITY_RUN_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"

IntrospectBusOperation = Literal["dreams", "curiosity", "orion_day"]
IntrospectOperation = Literal["reading_result", "dreams", "curiosity", "orion_day"]

# The day sections of an Orion's Day letter, named as the email headings read
# (services/orion-hub/scripts/orion_day_email.py). Each maps to the material
# refs it is built from in orion.orion_day.letter_parts.SECTION_PREFIXES.
OrionDaySection = Literal[
    "curiosity", "self_sense", "readings", "dreams", "code_changes", "conversations",
    "world_news", "reveries",
]
# A note has ~25 paragraphs and a carry-forward ~10 items; a bound only so a
# typo cannot ask for paragraph 10**9.
ORION_DAY_MAX_INDEX = 500


def clip_text(text: str | None, cap: int = DEFAULT_TEXT_CAP) -> tuple[str, bool]:
    body = (text or "").strip()
    if len(body) <= cap:
        return body, False
    return body[:cap], True


def _json_len(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False))


def clip_json_text(text: str | None, budget: int) -> tuple[str, bool]:
    """Longest prefix of ``text`` whose JSON string encoding fits ``budget``.

    ``clip_text`` counts characters; a write-up full of quotes, backslashes
    and newlines can nearly double when serialized, which is what the MCP
    budget actually measures.
    """
    body = (text or "").strip()
    if _json_len(body) <= budget:
        return body, False
    lo, hi = 0, len(body)
    while lo < hi:  # largest n with _json_len(body[:n]) <= budget
        mid = (lo + hi + 1) // 2
        if _json_len(body[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return body[:lo], True


def _require_tz(value: datetime | None, field: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")


class IntrospectToolBindingV1(BaseModel):
    """Server-authored turn context; never part of model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    invocation_context: Literal["unified_chat", "curiosity"]
    parent_run_id: str
    parent_trace_id: str
    memory_allowed: bool


class IntrospectItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    occurred_at: datetime
    kind: str = Field(min_length=1)
    epistemic_status: Literal["record", "unsettled"]
    text: str
    truncated: bool = False
    sensitivity: Literal["public", "private", "intimate"] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aware(self):
        _require_tz(self.occurred_at, "occurred_at")
        return self


class IntrospectResultV1(BaseModel):
    """``ok=True`` with no items means nothing matched; ``ok=False`` means unknown."""

    model_config = ConfigDict(extra="forbid")
    ok: bool
    operation: IntrospectOperation
    as_of: datetime
    total_available: int | None = Field(default=None, ge=0)
    items: list[IntrospectItemV1] = Field(default_factory=list, max_length=MAX_ITEMS)
    error: str | None = None

    @model_validator(mode="after")
    def _coherent(self):
        _require_tz(self.as_of, "as_of")
        if self.ok:
            if self.error is not None or self.total_available is None:
                raise ValueError("ok result requires total_available and no error")
            if self.total_available < len(self.items):
                raise ValueError("total_available cannot be smaller than items returned")
        elif not self.error or self.items or self.total_available is not None:
            raise ValueError("failed result carries only an error")
        return self


def normalize_query(value: Any) -> Any:
    """Strip before length checks; a blank query is an error, not "recent"."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        raise ValueError("query must not be blank")
    return value


class ReadingResultArguments(BaseModel):
    """Model-supplied arguments for the ``reading_results`` tool."""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    request_id: UUID | None = None
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        return normalize_query(value)

    @model_validator(mode="after")
    def _selectors(self):
        if self.request_id is not None and self.url is not None:
            raise ValueError("reading_results takes at most one of request_id or url")
        if self.query is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("query searches all readings; it cannot be combined with request_id or url")
        _require_tz(self.since, "since")
        if self.since is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("since applies only to query or recent reads (no request_id or url)")
        return self


class IntrospectRequestV1(BaseModel):
    """Harness -> owning service. ``args`` is validated by the owner per operation."""

    model_config = ConfigDict(extra="forbid")
    operation: IntrospectBusOperation
    binding: IntrospectToolBindingV1
    args: dict[str, Any] = Field(default_factory=dict)


class DreamsArguments(BaseModel):
    """Model-supplied arguments for the ``dreams`` tool."""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    dream_id: str | None = Field(default=None, pattern=DREAM_ID_PATTERN)
    kind: Literal["narrative", "hypothesis"] | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        return normalize_query(value)

    @model_validator(mode="after")
    def _selectors(self):
        _require_tz(self.since, "since")
        if self.dream_id is not None and (
            self.query is not None or self.kind is not None or self.since is not None
        ):
            raise ValueError("dream_id fetches one dream; it cannot be combined with query, kind or since")
        return self


class CuriosityArguments(BaseModel):
    """Model-supplied arguments for the ``curiosity`` tool."""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    run_id: str | None = Field(default=None, pattern=CURIOSITY_RUN_ID_PATTERN)
    kind: Literal["run", "self_question"] = "run"
    line: Literal["investigate", "self_inquiry", "self_sense_eval"] | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        return normalize_query(value)

    @model_validator(mode="after")
    def _selectors(self):
        _require_tz(self.since, "since")
        if self.run_id is not None and (
            self.query is not None or self.since is not None or self.line is not None
            or self.kind != "run"
        ):
            raise ValueError("run_id fetches one run; it cannot be combined with query, since, line or kind")
        if (
            self.kind == "run" and self.since is not None
            and self.since < datetime.now(timezone.utc) - timedelta(days=CURIOSITY_WINDOW_DAYS)
        ):
            raise ValueError(
                f"since must be within the last {CURIOSITY_WINDOW_DAYS} days: curiosity runs are "
                f"only readable {CURIOSITY_WINDOW_DAYS} days back"
            )
        if self.kind == "self_question" and (
            self.query is not None or self.run_id is not None or self.line is not None
        ):
            raise ValueError("kind=self_question lists open self-questions; it takes only limit and since")
        return self


class OrionDayArguments(BaseModel):
    """Model-supplied arguments for the ``orion_day`` tool (rereading an Orion's Day letter).

    ``part=list`` (default) is the letter's outline; ``note`` / ``carry_forward`` return
    the exact words, one part by ``index`` or the first ``limit`` parts; ``section`` returns
    the day's records behind one email section. ``query`` searches paragraphs and carry
    items by meaning across letters (``letter_date`` narrows it to one letter, ``part``
    note / carry_forward to one kind).
    """

    model_config = ConfigDict(extra="forbid")
    letter_date: date | None = None
    part: Literal["list", "note", "carry_forward", "section"] = "list"
    index: int | None = Field(default=None, ge=1, le=ORION_DAY_MAX_INDEX)
    section: OrionDaySection | None = None
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        return normalize_query(value)

    @model_validator(mode="after")
    def _selectors(self):
        if self.index is not None and self.part not in ("note", "carry_forward"):
            raise ValueError("index numbers a note paragraph or carry item; it needs part=note or part=carry_forward")
        if self.part == "section" and self.section is None:
            raise ValueError("part=section needs section=<curiosity|self_sense|readings|dreams|code_changes|"
                             "conversations|world_news|reveries>")
        if self.section is not None and self.part != "section":
            raise ValueError("section applies only with part=section")
        if self.query is not None and (self.index is not None or self.part == "section"):
            raise ValueError("query searches note paragraphs and carry items by meaning; it cannot be "
                             "combined with index or part=section")
        return self
