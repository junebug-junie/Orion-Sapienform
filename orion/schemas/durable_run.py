"""Durable cognition runs -- the contract between cortex, the runner, and Hub.

Design: docs/superpowers/specs/2026-09-06-durable-cognition-runs-from-cortex-design.md
(step 3 of the attention-schema-surface ordering, option A).

A curiosity investigation is Orion's longest single act of cognition (10-40
minutes) and, before this, lived entirely in one Hub process's memory: every
Hub redeploy killed it, burned a daily-cap slot, and left nothing to resume.
`orion-durable-runs` owns the run's state machine under a Postgres
checkpointer; Hub keeps owning what only Hub can do (scheduling, the material
and worldview reads, the prompt, executing the harness turn, outreach), and
cortex is where every run is kicked off, so cortex knows cognition is
happening.

Flow:

    Hub tick -> orion:cortex:request (context.metadata.durable_run = DurableRunRequestV1)
      -> orion-cortex-orch dispatches -> orion:durable:run:request (single consumer: the runner)
      -> runner graph: harness_turn (RPC back to Hub: orion:curiosity:turn:request)
                       -> read_turn_result -> publish_attention_row -> journal -> finish
      -> every transition: orion:durable:run:state -> orion-sql-writer -> substrate_durable_run_state
                                                   -> Hub (completed + reach_out -> outreach)

Every message here is additive-safe by construction: the runner is the only
consumer of the request, the state is consumed by extra="ignore"-tolerant
readers via `model_validate` on this exact model, and the turn RPC is Hub's
alone. No field on any existing schema changed (see the 2026-09-06 incident
on FieldGoalProvenanceV1 for why that matters).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.resource_admission import ResourceLeaseV1, ResourceRequirementV1

DURABLE_RUN_REQUEST_CHANNEL = "orion:durable:run:request"
DURABLE_RUN_STATE_CHANNEL = "orion:durable:run:state"
CURIOSITY_TURN_REQUEST_CHANNEL = "orion:curiosity:turn:request"
CURIOSITY_TURN_REPLY_PREFIX = "orion:curiosity:turn:reply"

DURABLE_RUN_REQUEST_KIND = "durable.run.request.v1"
DURABLE_RUN_STATE_KIND = "durable.run.state.v1"
DURABLE_RUN_RECEIPT_KIND = "durable.run.receipt.v1"
DURABLE_RUN_REPLY_PREFIX = "orion:durable:run:reply"
CURIOSITY_TURN_REQUEST_KIND = "curiosity.turn.request.v1"
CURIOSITY_TURN_RESULT_KIND = "curiosity.turn.result.v1"

DurableWorkflowV1 = Literal["curiosity.investigate", "self_sense_eval", "self_study.reflect"]

# The runner's node names, in order. `attention_reason` on the surface lane
# walks this list for a run; `DurableRunStateV1.node` is always one of them.
CURIOSITY_NODES: tuple[str, ...] = (
    "harness_turn",
    "read_turn_result",
    "publish_attention_row",
    "journal",
    "finish",
)

# self_sense_eval's own graph (services/orion-durable-runs/app/self_sense_graph.py):
# no material/worldview read, no journal, no outreach -- ask the four fixed
# questions, score and publish self_sense_eval_log rows, done.
SELF_SENSE_EVAL_NODES: tuple[str, ...] = ("ask_questions", "publish", "finish")

# self_study.reflect's own graph (services/orion-durable-runs/app/reflect_graph.py):
# ONE LLM call, no publish -- finding validation and journal/self_concept_history
# writes stay in cortex-exec (they need the full snapshot/concepts for
# evidence-chain construction, not just the small input this graph carries),
# unlike self_sense_eval's publish, which is simple enough to run here.
# cortex-exec dispatches, awaits this run's completion synchronously, and
# does the rest itself -- same external contract `_call_self_study_reflect_llm`
# always had.
SELF_STUDY_REFLECT_NODES: tuple[str, ...] = ("llm_call", "finish")

DurableRunStatusV1 = Literal[
    "accepted", "queued", "waiting_resource", "admitted", "running", "paused",
    "resumed", "retrying", "completed", "failed", "cancelled", "abandoned",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CuriosityMaterialBriefV1(BaseModel):
    """The five facts the journal entry reports about what Orion was shown.
    Counts only -- the cards themselves stay in Hub; the prompt already
    carries what Orion saw."""

    model_config = ConfigDict(extra="forbid")

    approved_total: int = 0
    approved_by_kind: dict[str, int] = Field(default_factory=dict)
    crystallization_count: int = 0
    relation_total: int = 0
    relation_count: int = 0


class CuriosityRunBriefV1(BaseModel):
    """Everything the runner needs to carry one curiosity run to completion
    without Hub's in-memory objects. Built by Hub at kickoff, checkpointed by
    the runner as the run's initial state."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    session_id: str
    fcc_model_label: str | None = None
    timeout_sec: float = Field(gt=0.0)
    graph_configured: bool = False
    material: CuriosityMaterialBriefV1 = Field(default_factory=CuriosityMaterialBriefV1)
    source_tag: str = "curiosity_investigation"
    # Which curiosity line this run belongs to. `self_inquiry` runs are the
    # same graph with one extra read (`:SelfDefinition` or `:LivedAnswer`)
    # and a different journal title; Hub mirrors the write on the `finish`
    # event. `self_sense_eval` runs a DIFFERENT graph entirely (see
    # `DurableRunRequestV1.workflow`) -- this field still carries it for
    # the same reason `CuriosityTurnRequestV1.source_tag` does: a uniform
    # place a listener checks "which line is this" without branching on
    # workflow name. ADDITIVE FIELD ON A `forbid` MODEL: deploy
    # orion-durable-runs before orion-hub, or an old runner rejects the
    # brief and Hub falls back to running the turn in-process (logged as
    # curiosity_durable_dispatch_fell_back).
    line: Literal["investigate", "self_inquiry", "self_sense_eval", "reflect"] = "investigate"
    # self_sense_eval only, additive: the fixed (question_key, question_text)
    # pairs to ask, in order -- Hub owns `orion.schemas.self_sense.SELF_SENSE_QUESTIONS`
    # as the source of truth and just carries a copy here so the runner
    # doesn't need its own import of a Hub-side schema module.
    questions: list[tuple[str, str]] | None = None
    # self_sense_eval only, additive: read by Hub BEFORE dispatch (same
    # "Hub does the DB reads, runner just executes" split as `material`
    # above) so `build_row` can score each answer's self-report grounding.
    self_definition_version: int | None = None
    lived_answers: list[dict[str, Any]] | None = None
    # reflect only, additive: cortex-exec's own
    # `_self_study_reflect_input(snapshot, concepts)` dict (snapshot_id,
    # counts_by_kind, concepts) -- the ENTIRE input the LLM call needs.
    # Unlike self_sense_eval/investigate, reflect's downstream finding
    # validation and publish (evidence-chain construction against the real
    # snapshot/concepts) stay in cortex-exec, unmoved -- this brief only
    # carries what the runner's `llm_call` node needs to make the actual
    # LLM request; cortex-exec awaits the run's completion synchronously
    # (see self_study.py's `_call_self_study_reflect_llm`) and does the
    # rest itself, same as it always has.
    self_study_reflect_input: dict[str, Any] | None = None
    llm_route: str | None = None


class DurableRunRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["durable.run.request.v1"] = "durable.run.request.v1"
    run_id: str = Field(min_length=6)
    workflow: DurableWorkflowV1
    correlation_id: str
    requested_at: datetime = Field(default_factory=_utc_now)
    brief: CuriosityRunBriefV1
    admission: ResourceRequirementV1 | None = None


class DurableRunReceiptV1(BaseModel):
    """Runner acknowledgment emitted only after durable registration commits."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["durable.run.receipt.v1"] = DURABLE_RUN_RECEIPT_KIND
    run_id: str
    status: DurableRunStatusV1
    workflow_kind: DurableWorkflowV1
    requested_resource: str


class DurableRunStateV1(BaseModel):
    """One row per node transition. `thread_id` is the checkpointer thread
    (== run_id for this workflow). `resumed_from_node` is set on the first
    transition after a restart picked the run back up."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["durable.run.state.v1"] = "durable.run.state.v1"
    # Row identity for the writer (one row per transition; a republish of the
    # same event is an ON CONFLICT no-op, never a duplicate row).
    entry_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    workflow: DurableWorkflowV1
    thread_id: str
    node: str
    next_node: str | None = None
    status: DurableRunStatusV1
    resumed_from_node: str | None = None
    correlation_id: str
    generated_at: datetime = Field(default_factory=_utc_now)
    # Small, bounded facts a listener needs without re-reading the checkpoint:
    # on `finish`: reach_out (bool), reach_out_why, finding_text (capped),
    # continue_line; on `failed`: error. Never the prompt, never the material.
    detail: dict[str, Any] = Field(default_factory=dict)


class CuriosityTurnRequestV1(BaseModel):
    """Runner -> Hub: execute the harness turn for this run. Hub is the only
    process that can (the unified-turn saga needs Hub's own clients), so the
    one node that cannot leave Hub is invoked over RPC and retried by the
    runner if Hub goes away mid-turn."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.turn.request.v1"] = "curiosity.turn.request.v1"
    run_id: str
    correlation_id: str
    prompt: str = Field(min_length=1)
    fcc_model_label: str | None = None
    timeout_sec: float = Field(gt=0.0)
    source_tag: str = "curiosity_investigation"
    attempt: int = Field(default=1, ge=1)
    lease: ResourceLeaseV1 | None = None
    assigned_lane: str | None = None
    # Stage 4: the run's GPU pool hold. Hub validates it with the pool's ``status`` verb and runs
    # the turn under it (every LLM call attaches to the hold). Consumer first: Hub must accept this
    # before durable-runs 4.5 sends it (extra="forbid").
    gpu_lease: GpuLeaseRefV1 | None = None
    # Additive: an explicit session to run this turn under, distinct from
    # curiosity's own shared investigation session. self_sense_eval needs
    # this -- its answers must land in the SAME clean session
    # `make eval-self-sense` and the in-process scheduler line both use
    # (`orion.evals.self_sense_runner.SESSION_ID`), or scheduled and ad hoc
    # runs stop being comparable rows in the same table (the exact review
    # finding PR #2247 fixed for the in-process path; this closes the same
    # gap for the durable path). `None` keeps every existing caller's
    # behavior unchanged -- Hub's `_turn_result_for` falls back to its own
    # shared session when this is absent.
    session_id: str | None = None


class CuriosityTurnResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.turn.result.v1"] = "curiosity.turn.result.v1"
    run_id: str
    correlation_id: str
    text: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
