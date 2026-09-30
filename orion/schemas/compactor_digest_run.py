"""compactor.digest: the LLM half of the daily compactors as an admitted durable run.

Why: ``github_compactor_pass`` and ``chat_history_compactor_pass`` (cortex-orch) digest a full
Denver calendar day with several LLM calls (map-reduce, PR #2422). They used to make those calls
in-process from inside one synchronous workflow RPC, each through a one-inference gateway lease;
at 06:00 the GPU pool is busy and those leases failed with ``gpu_pool_unavailable:deadline``, and
the only answer was an in-process retry and a 3600 s scheduler wait. As a durable run every call
holds the run's GPU pool hold: a busy pool is a checkpointed wait (never an attempt), a restart
resumes at the next unfinished call, and a failed call is one of
``DURABLE_RUNS_RETRY_MAX_ATTEMPTS`` bounded attempts.

Flow (who owns what):

    orion-actions scheduler -> cortex-orch workflow (fetch PRs / chat turns, build chunk inputs;
        quiet day -> finalize inline, no LLM) -> DurableRunRequestV1(workflow="compactor.digest",
        brief=CompactorDigestRunBriefV1, admission=agent) -> reply "accepted" at once
    orion-durable-runs graph (compactor_digest_graph.py):
        resource_request -> resource_wait -> digest (one chunk digest or the merge per node run,
        looping) -> finalize -> finish
    finalize: hold released, then CompactorDigestResultV1 sent back to cortex-orch as
        ``workflow_request.durable_digest`` -> memory card + journal write + workflow notify
    orion-actions: the terminal ``orion:durable:run:state`` row settles the schedule run.

``run_id`` is deterministic per (workflow, window, repo, input hash): a re-dispatch of the same
window finds the existing run instead of starting a second one. ADDITIVE on ``extra="forbid"``
/ ``Literal`` contracts: deploy the ``DurableRunStateV1`` consumers (orion-sql-writer, orion-hub)
and orion-durable-runs before cortex-orch (orch submits these), and cortex-orch before
orion-actions (its 600 s dispatch wait is too short for an old in-process orch).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

COMPACTOR_DIGEST_WORKFLOW = "compactor.digest"
# The graph's node order, admitted shell included (for state events / Hub views).
COMPACTOR_DIGEST_NODES: tuple[str, ...] = ("resource_request", "resource_wait", "digest", "finalize", "finish")
# The workflow_request key cortex-orch reads a finished digest from (finalize path).
DURABLE_DIGEST_KEY = "durable_digest"

CompactorWorkflowIdV1 = Literal["github_compactor_pass", "chat_history_compactor_pass"]
_KIND_FOR_WORKFLOW = {"github_compactor_pass": "github", "chat_history_compactor_pass": "chat"}
# Juniper's reserved interactive lanes: background digest work never names them
# (scripts/check_chat_route_poachers.py).
_RESERVED_ROUTES = frozenset({"chat", "harness"})


class CompactorDigestRunBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["github", "chat"]
    workflow_id: CompactorWorkflowIdV1
    window_label: str = Field(min_length=1)
    # The chunk inputs cortex-orch built (orion.cognition.*_compactor.digest), in order. Every
    # item of the window is in exactly one; nothing is dropped to make a chunk fit.
    inputs: list[dict[str, Any]] = Field(min_length=1)
    llm_route: str = "agent"
    # Per digest call (verb budget + bus slack): the RPC wait and the admitted node's timeout.
    timeout_sec: float = Field(gt=0)
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    # Opaque to durable-runs: what cortex-orch needs to finish the day without re-fetching
    # (repo, window bounds, coverage, fetch facts, notify policy). Echoed back verbatim.
    finalize: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self):
        if _KIND_FOR_WORKFLOW[self.workflow_id] != self.kind:
            raise ValueError("compactor kind and workflow_id must agree")
        if self.llm_route.strip().lower() in _RESERVED_ROUTES:
            raise ValueError(f"llm_route {self.llm_route!r} is a reserved interactive lane")
        return self


class CompactorDigestResultV1(BaseModel):
    """durable-runs -> cortex-orch (``workflow_request.durable_digest``): the finished digest."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    kind: Literal["github", "chat"]
    workflow_id: CompactorWorkflowIdV1
    window_label: str
    digest: dict[str, Any]
    chunk_count: int = Field(ge=1)
    merge_mode: Literal["single", "llm_merge", "concatenated"]
    merge_skipped_reason: str | None = None
    trimmed_fields: list[str] = Field(default_factory=list)
    # One row per LLM call attempt: {"step", "ok", "error"?, "role"?}. Pool waits are not rows.
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    llm_route: str
    # The GPU pool roles the calls actually ran on (the hold's role per call, de-duplicated).
    gpu_roles: list[str] = Field(default_factory=list)
    finalize: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self):
        if _KIND_FOR_WORKFLOW[self.workflow_id] != self.kind:
            raise ValueError("compactor kind and workflow_id must agree")
        return self
