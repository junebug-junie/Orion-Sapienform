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

DURABLE_RUN_REQUEST_CHANNEL = "orion:durable:run:request"
DURABLE_RUN_STATE_CHANNEL = "orion:durable:run:state"
CURIOSITY_TURN_REQUEST_CHANNEL = "orion:curiosity:turn:request"
CURIOSITY_TURN_REPLY_PREFIX = "orion:curiosity:turn:reply"

DURABLE_RUN_REQUEST_KIND = "durable.run.request.v1"
DURABLE_RUN_STATE_KIND = "durable.run.state.v1"
CURIOSITY_TURN_REQUEST_KIND = "curiosity.turn.request.v1"
CURIOSITY_TURN_RESULT_KIND = "curiosity.turn.result.v1"

DurableWorkflowV1 = Literal["curiosity.investigate"]

# The runner's node names, in order. `attention_reason` on the surface lane
# walks this list for a run; `DurableRunStateV1.node` is always one of them.
CURIOSITY_NODES: tuple[str, ...] = (
    "harness_turn",
    "read_turn_result",
    "publish_attention_row",
    "journal",
    "finish",
)

DurableRunStatusV1 = Literal["running", "resumed", "completed", "failed", "abandoned"]


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


class DurableRunRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["durable.run.request.v1"] = "durable.run.request.v1"
    run_id: str = Field(min_length=6)
    workflow: DurableWorkflowV1
    correlation_id: str
    requested_at: datetime = Field(default_factory=_utc_now)
    brief: CuriosityRunBriefV1


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


class CuriosityTurnResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.turn.result.v1"] = "curiosity.turn.result.v1"
    run_id: str
    correlation_id: str
    text: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
