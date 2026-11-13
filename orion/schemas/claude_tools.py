"""Contracts for Claude's read-only tools in Hub chat, gated by Orion's consent (Patch 1b).

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md sections 2-4.

Four bus messages, plus two nested models:

    orion:claude:tool:request       ClaudeToolRequestV1    companion MCP server -> broker (governor)
    orion:claude:tool:result:<id>   ClaudeToolResultV1     broker -> MCP server (RPC reply)
    orion:claude:tool:audit         ClaudeToolCallV1       broker -> sql-writer (claude_tool_call_log)
    orion:prime:consent:request     PrimeConsentRequestV1  broker -> orion-mind
    orion:prime:consent:result:<id> PrimeConsentAnswerV1   orion-mind -> broker (RPC reply)

``ClaudeToolBindingV1`` is the per-turn context the companion hands its MCP server (env JSON). It is
server-authored: Claude never supplies it, so Claude cannot claim another room or session.

Every model is ``extra="forbid"``: a field nobody declared is refused, never ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# The nine categories (spec section 2). Each has a consumer (the grant check) and a trace (the audit
# row). Only `prime` and `body` have tools in Patch 1b; the rest are settable now so Orion's standing
# choices survive the tools that come later.
CLAUDE_TOOL_CATEGORIES = (
    "conversation", "turns", "prime", "body", "senses", "people", "consults", "memory", "self",
)
ClaudeToolCategory = Literal[
    "conversation", "turns", "prime", "body", "senses", "people", "consults", "memory", "self",
]
GrantSetting = Literal["allow", "ask", "deny"]
# How the broker decided. The first six are the design's; ``limit`` (per-turn cap), ``invalid`` (unknown
# tool or bad arguments) and ``error`` (the broker could not check) are the refusals it adds.
DecisionPath = Literal[
    "standing_allow", "standing_deny", "asked", "session_allow", "repeat_no", "off",
    "limit", "invalid", "error",
]
ConsentAnswer = Literal["yes", "no"]
ConsentScope = Literal["once", "conversation"]
# How orion-mind reached its answer. Only ``model`` is Orion deciding; every other path is a no.
ConsentPath = Literal["model", "unparsed", "gate_refused", "no_model", "error"]
AuditPhase = Literal["ask", "answer", "result"]

REASON_MAX = 300
ORION_REASON_MAX = 200
TOPIC_MAX = 300


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ClaudeToolBindingV1(BaseModel):
    """Per-turn context for the companion's MCP server. Never part of Claude's tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_request_id: str = Field(min_length=1, max_length=120)
    room_id: str = Field(min_length=1, max_length=200)
    platform: str = Field(min_length=1, max_length=40)
    # The companion's Claude session for this room (rotates daily). A "yes for this conversation" lasts
    # only as long as this session.
    claude_session_id: str = Field(min_length=1, max_length=120)
    hub_session_id: Optional[str] = Field(default=None, max_length=120)
    trigger: str = Field(default="manual", max_length=40)
    # The message that invited Claude (Juniper's @claude text, or Orion's invite), so Orion knows the topic.
    topic: Optional[str] = Field(default=None, max_length=TOPIC_MAX)
    topic_from: Optional[str] = Field(default=None, max_length=80)
    # Where the MCP server appends one line per call, for the companion's footer under the reply.
    call_log_path: Optional[str] = Field(default=None, max_length=400)


class ClaudeToolRequestV1(BaseModel):
    """One tool call from Claude, as the MCP server forwards it."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=8, max_length=64)
    binding: ClaudeToolBindingV1
    tool: str = Field(min_length=1, max_length=64)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    # Why Claude wants it, in Claude's words. Shown to Orion when the category is "ask".
    reason: str = Field(default="", max_length=REASON_MAX)


class ClaudeToolResultV1(BaseModel):
    """The broker's answer. ``text`` is exactly what Claude sees; ``shared`` says whether data left."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool: str
    category: Optional[ClaudeToolCategory] = None
    shared: bool
    text: str
    decision_path: DecisionPath
    setting: Optional[GrantSetting] = None
    orion_answer: Optional[ConsentAnswer] = None
    orion_scope: Optional[ConsentScope] = None
    orion_reason: Optional[str] = Field(default=None, max_length=ORION_REASON_MAX)
    consent_latency_ms: Optional[int] = None
    result_chars: int = 0
    error: Optional[str] = Field(default=None, max_length=500)


class ClaudeToolCallV1(BaseModel):
    """Audit: one event per phase of a call. Every call has a ``result`` event; a call that asked Orion
    also has an ``ask`` event (before the consent call) and an ``answer`` event (after it), so an ask that
    never came back is still on record. Stored by orion-sql-writer in ``claude_tool_call_log``."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=8, max_length=80)
    phase: AuditPhase
    call_id: str
    turn_request_id: str
    room_id: str
    claude_session_id: str
    hub_session_id: Optional[str] = None
    trigger: Optional[str] = None
    tool: str
    category: Optional[ClaudeToolCategory] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    setting: Optional[GrantSetting] = None
    decision_path: Optional[DecisionPath] = None
    shared: Optional[bool] = None
    orion_answer: Optional[ConsentAnswer] = None
    orion_scope: Optional[ConsentScope] = None
    orion_reason: Optional[str] = None
    consent_path: Optional[ConsentPath] = None
    consent_latency_ms: Optional[int] = None
    act_id: Optional[str] = None
    result_chars: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class PrimeConsentPastAnswerV1(BaseModel):
    """One of Orion's earlier answers in a category, shown in the next consent frame."""

    model_config = ConfigDict(extra="forbid")

    at: datetime
    tool: str
    answer: ConsentAnswer
    scope: ConsentScope = "once"
    reason: str = Field(default="", max_length=ORION_REASON_MAX)


class PrimeConsentRequestV1(BaseModel):
    """The broker asks Orion: may Claude use this tool? orion-mind answers in one short model call."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=8, max_length=64)
    turn_request_id: str
    room_id: str
    claude_session_id: str
    tool: str
    category: ClaudeToolCategory
    # One line from the tool's description: what Claude would see.
    tool_summary: str = Field(max_length=300)
    reason: str = Field(default="", max_length=REASON_MAX)
    topic: Optional[str] = Field(default=None, max_length=TOPIC_MAX)
    topic_from: Optional[str] = Field(default=None, max_length=80)
    setting: GrantSetting = "ask"
    recent_answers: List[PrimeConsentPastAnswerV1] = Field(default_factory=list, max_length=5)
    # "you have said yes 20 of 20 times" (the rubber-stamp line), when it applies.
    history_note: Optional[str] = Field(default=None, max_length=300)


class PrimeConsentAnswerV1(BaseModel):
    """Orion's answer. Anything but a parsed model answer is a no, and ``path`` says which no."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    answer: ConsentAnswer
    scope: ConsentScope = "once"
    reason: str = Field(default="", max_length=ORION_REASON_MAX)
    path: ConsentPath
    act_id: Optional[str] = None
    latency_ms: int = 0
    error: Optional[str] = Field(default=None, max_length=500)


__all__ = [
    "CLAUDE_TOOL_CATEGORIES",
    "ClaudeToolBindingV1",
    "ClaudeToolCallV1",
    "ClaudeToolCategory",
    "ClaudeToolRequestV1",
    "ClaudeToolResultV1",
    "ConsentAnswer",
    "ConsentPath",
    "ConsentScope",
    "DecisionPath",
    "GrantSetting",
    "ORION_REASON_MAX",
    "PrimeConsentAnswerV1",
    "PrimeConsentPastAnswerV1",
    "PrimeConsentRequestV1",
    "REASON_MAX",
    "TOPIC_MAX",
]
