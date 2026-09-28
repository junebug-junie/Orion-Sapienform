"""Introspect tool contracts: Orion reading back their own recorded activity.

Design: docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md.
Slice 1 carries only ``reading_result``; each later slice extends
``IntrospectOperation`` when its owning service gains a responder.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_ITEMS = 10
DEFAULT_LIMIT = 5
# 10 items must fit under ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS (12000) or the
# harness truncates the JSON mid-item.
DEFAULT_TEXT_CAP = 900
SHORT_FIELD_CAP = 200

IntrospectOperation = Literal["reading_result"]


def clip_text(text: str | None, cap: int = DEFAULT_TEXT_CAP) -> tuple[str, bool]:
    body = (text or "").strip()
    if len(body) <= cap:
        return body, False
    return body[:cap], True


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


class ReadingResultArguments(BaseModel):
    """Model-supplied arguments for the ``reading_results`` tool."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID | None = None
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @model_validator(mode="after")
    def _selectors(self):
        if self.request_id is not None and self.url is not None:
            raise ValueError("reading_results takes at most one of request_id or url")
        _require_tz(self.since, "since")
        if self.since is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("since applies only to recent reads (no request_id or url)")
        return self
