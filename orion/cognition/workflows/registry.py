from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.discussion_window.timeframe import parse_journal_discussion_lookback_seconds


ExecutionMode = Literal["sync", "async_capable"]


class WorkflowStepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    description: str
    adapter: str


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    display_name: str
    description: str
    aliases: List[str] = Field(default_factory=list)
    user_invocable: bool = True
    autonomous_invocable: bool = False
    execution_mode: ExecutionMode = "sync"
    may_call_actions: bool = False
    persistence_policy: str
    result_surface: str
    steps: List[WorkflowStepDefinition] = Field(default_factory=list)
    feature_flags: List[str] = Field(default_factory=list)
    settings_guards: List[str] = Field(default_factory=list)
    planner_hints: List[str] = Field(default_factory=list)


class WorkflowInvocationMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    matched_alias: str
    normalized_prompt: str
    confidence: float = 1.0
    resolver: Literal["alias_registry"] = "alias_registry"


def _normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"^[\s,;:.-]*(hey|hi|hello)\s+orion[\s,;:.-]*", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


_WORKFLOWS: tuple[WorkflowDefinition, ...] = (
    WorkflowDefinition(
        workflow_id="dream_cycle",
        display_name="Dream Cycle",
        description="Bounded cognition workflow that gathers dream context, runs the dream routine, interprets the result, and may surface a concise completion summary.",
        aliases=[
            "run your dream cycle",
            "dream now",
            "do a dream pass",
            "run a dream pass",
            "dream cycle",
        ],
        user_invocable=True,
        autonomous_invocable=True,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="dream artifact persistence follows the existing dream_cycle verb and downstream dream.result.v1 contracts.",
        result_surface="Return a concise user-visible summary with dream outcome and any persisted artifact notes.",
        steps=[
            WorkflowStepDefinition(step_id="gather_context", description="Gather bounded dream context via existing recall profile.", adapter="verb:dream_cycle"),
            WorkflowStepDefinition(step_id="dream_routine", description="Run the existing dream routine.", adapter="verb:dream_cycle"),
            WorkflowStepDefinition(step_id="interpret", description="Interpret and reflect on the resulting dream output.", adapter="verb:dream_cycle"),
        ],
        planner_hints=["Use when the user explicitly asks Orion to run its dream cycle or dream pass."],
    ),
    WorkflowDefinition(
        workflow_id="journal_pass",
        display_name="Journal Pass",
        description="Bounded journaling workflow that reuses the existing structured journal compose path and append-only journal persistence boundary.",
        aliases=[
            "do a journal pass",
            "journal now",
            "write a journal entry",
            "run a journal reflection",
            "journal pass",
        ],
        user_invocable=True,
        autonomous_invocable=False,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="Compose through journal.compose and persist only through the existing append-only journal.entry.write.v1 boundary.",
        result_surface="Return the drafted journal title/body summary and whether an entry was persisted.",
        steps=[
            WorkflowStepDefinition(step_id="normalize_trigger", description="Build a manual journal trigger from the chat request.", adapter="journaler:manual_trigger"),
            WorkflowStepDefinition(step_id="compose", description="Invoke the existing journal.compose path.", adapter="journaler:compose_request"),
            WorkflowStepDefinition(step_id="persist", description="Persist through journal.entry.write.v1.", adapter="journaler:append_only_write"),
        ],
        planner_hints=["Use when the user explicitly asks for a journal pass or journal entry."],
    ),
    WorkflowDefinition(
        workflow_id="journal_discussion_window_pass",
        display_name="Journal discussion window",
        description=(
            "Bounded journaling workflow that reads a time window from persisted chat_history_log "
            "(skills.chat.discussion_window.v1), then composes and persists via journal.compose and journal.entry.write.v1."
        ),
        aliases=[
            "journal the last 34 minutes",
            "journal the last hour",
            "journal our chat discussion for the last day",
            "journal our chat discussion",
            "do a journal discussion pass",
        ],
        user_invocable=True,
        autonomous_invocable=False,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="Compose through journal.compose and persist only through journal.entry.write.v1; chat turns are read-only from SQL.",
        result_surface="Return window bounds, turn count, persisted journal entry id, and a short summary.",
        steps=[
            WorkflowStepDefinition(
                step_id="discussion_window",
                description="Fetch bounded discussion turns from chat_history_log.",
                adapter="verb:skills.chat.discussion_window.v1",
            ),
            WorkflowStepDefinition(step_id="compose", description="Invoke journal.compose with prompt_seed from transcript.", adapter="journaler:compose_request"),
            WorkflowStepDefinition(step_id="persist", description="Persist through journal.entry.write.v1.", adapter="journaler:append_only_write"),
        ],
        planner_hints=[
            "Use when the user asks to journal recent chat or discussion for an explicit minutes/hours/day window.",
        ],
    ),
    WorkflowDefinition(
        workflow_id="self_review",
        display_name="Self Review",
        description="Bounded metacognitive workflow that reuses self-study reflection pathways to produce a structured self-review.",
        aliases=[
            "run a self review",
            "do a self review",
            "reflect on yourself",
            "run self study",
            "do a metacognitive review",
            "self review",
        ],
        user_invocable=True,
        autonomous_invocable=True,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="Reuse existing self-study reflective graph/journal writebacks when the underlying adapter performs them.",
        result_surface="Return a structured self-review summary with findings counts and persistence notes.",
        steps=[
            WorkflowStepDefinition(step_id="gather_self_context", description="Gather authoritative self-study context.", adapter="verb:self_concept_reflect"),
            WorkflowStepDefinition(step_id="reflect", description="Run bounded reflective self-study synthesis.", adapter="verb:self_concept_reflect"),
        ],
        planner_hints=["Use when the user explicitly asks Orion for a self review or metacognitive review."],
    ),
    WorkflowDefinition(
        workflow_id="concept_induction_pass",
        display_name="Concept Induction Pass",
        description="Bounded concept review workflow that inspects existing concept induction profiles and synthesizes a compact review without uncontrolled mutation.",
        aliases=[
            "run concept induction",
            "do a concept induction pass",
            "run through your concept induction graphs",
            "review your concept induction graph",
            "inspect concept induction profiles",
            "concept induction pass",
        ],
        user_invocable=True,
        autonomous_invocable=True,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="Read existing concept profile state; do not mutate graph/profile state unless an existing safe bounded path is explicitly invoked elsewhere.",
        result_surface="Return a bounded profile review covering concepts, clusters, tensions, and profile freshness when available.",
        steps=[
            WorkflowStepDefinition(step_id="load_profiles", description="Load current concept induction profiles from the existing profile store.", adapter="concept_store:load_profiles"),
            WorkflowStepDefinition(step_id="review", description="Synthesize a compact concept review from loaded profiles.", adapter="concept_store:review_profiles"),
        ],
        planner_hints=["Use when the user explicitly asks to run or inspect concept induction graphs/profiles."],
    ),
)

_WORKFLOW_INDEX = {workflow.workflow_id: workflow for workflow in _WORKFLOWS}
_ALIAS_INDEX: dict[str, WorkflowDefinition] = {}
for workflow in _WORKFLOWS:
    for alias in workflow.aliases:
        _ALIAS_INDEX[_normalize_text(alias)] = workflow


def list_workflows(*, user_invocable_only: bool = False) -> List[WorkflowDefinition]:
    workflows = list(_WORKFLOWS)
    if user_invocable_only:
        workflows = [workflow for workflow in workflows if workflow.user_invocable]
    return workflows


def get_workflow_definition(workflow_id: str) -> Optional[WorkflowDefinition]:
    return _WORKFLOW_INDEX.get(workflow_id)


def workflow_registry_payload(*, user_invocable_only: bool = False) -> List[Dict[str, Any]]:
    return [workflow.model_dump(mode="json") for workflow in list_workflows(user_invocable_only=user_invocable_only)]


def _journal_discussion_window_command_intent(lowered: str) -> bool:
    """True when the prompt is clearly a user-directed journal/log command, not a passing mention of journaling."""
    s = (lowered or "").strip()
    if re.search(r"\bjournal\s+discussion\s+pass\b", s) or re.search(r"\bdo\s+a\s+journal\s+discussion\s+pass\b", s):
        return True
    return bool(re.match(r"^(please\s+)?(journal|log(\s+our\s+chat)?)\b", s))


def resolve_user_workflow_invocation(prompt: str, *, allow_workflow_ids: Iterable[str] | None = None) -> Optional[WorkflowInvocationMatch]:
    normalized_prompt = _normalize_text(prompt)
    if not normalized_prompt:
        return None

    allowed = set(allow_workflow_ids or _WORKFLOW_INDEX.keys())
    best: Optional[WorkflowInvocationMatch] = None
    for alias, workflow in _ALIAS_INDEX.items():
        if workflow.workflow_id not in allowed:
            continue
        if normalized_prompt == alias or f" {alias} " in f" {normalized_prompt} ":
            candidate = WorkflowInvocationMatch(
                workflow_id=workflow.workflow_id,
                matched_alias=alias,
                normalized_prompt=normalized_prompt,
            )
            if best is None or len(alias) > len(best.matched_alias):
                best = candidate

    # Time-bounded chat discussion journaling: explicit durations (any minute count) and related phrases.
    if "journal_discussion_window_pass" in allowed:
        lowered = (prompt or "").strip().lower()
        if parse_journal_discussion_lookback_seconds(lowered) is not None and _journal_discussion_window_command_intent(lowered):
            synthetic = "journal_discussion_window_time_bounded_sql_v1"
            candidate = WorkflowInvocationMatch(
                workflow_id="journal_discussion_window_pass",
                matched_alias=synthetic,
                normalized_prompt=normalized_prompt,
            )
            if best is None or len(candidate.matched_alias) > len(best.matched_alias):
                best = candidate
    return best
