"""The `reverie.visual` admitted durable workflow: one image, checkpointed per stage.

Design: docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md.

durable-runs drives the graph; orion-thought executes each stage and owns every
visual-chain side effect (prompt plan, image file, chain row, receipts). The
durable run only carries ids and the stage results it needs to route:

    prepare (no hold) -> resource_request -> resource_wait -> generate (diffusion hold)
      -> caption (no hold) -> finish

A step result's ``status`` tells the graph what to do next:

* ``done``     -- stage complete; advance.
* ``retry``    -- a deferral or transient failure (thermal, capacity busy, resource
                  deferred, generation/transport error). Release any hold, back off,
                  resume at the same stage. Never spends the run's attempt budget.
* ``terminal`` -- the request can never produce an image in this run (baseline already
                  satisfied elsewhere, dispatch replay mismatch, ineligible). Finish now;
                  ``outcome`` says why.

``reason == "needs_generate"`` on a caption retry means the recorded image is gone from
disk: the graph goes back through the hold to generate instead of retrying caption.

``abandon`` is sent whenever a run ends without completing (run deadline, operator cancel)
and retried until thought acknowledges it, so thought closes the attempt; an attempt left
``active`` blocks every later claim. It may omit ``attempt_id`` (resolved by dispatch_id).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reverie_visual import VisualRunOutcome, VisualRunRequestV1

REVERIE_VISUAL_WORKFLOW = "reverie.visual"
REVERIE_VISUAL_HOLD_LANE = "diffusion"
REVERIE_VISUAL_STEP_CHANNEL = "orion:reverie:visual:step:request"
REVERIE_VISUAL_STEP_REPLY_PREFIX = "orion:reverie:visual:step:reply"
REVERIE_VISUAL_STEP_REQUEST_KIND = "reverie.visual.step.request.v1"
REVERIE_VISUAL_STEP_RESULT_KIND = "reverie.visual.step.result.v1"

# Longest retry window a run may be given. orion-thought releases attempts older than its
# ORION_VISUAL_CHAIN_ATTEMPT_MAX_AGE_SEC (default 7200, asserted larger than this), so a live
# run must never outlast it: its attempt would be released under it.
REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC = 6600.0

REVERIE_VISUAL_NODES: tuple[str, ...] = ("prepare", "generate", "caption", "finish")

ReverieVisualStep = Literal["prepare", "generate", "caption", "abandon"]
NEEDS_GENERATE = "needs_generate"
ReverieVisualStepStatus = Literal["done", "retry", "terminal"]


def reverie_visual_run_id(dispatch_id: str) -> str:
    """Deterministic run id: a resubmitted dispatch dedupes at the durable store."""
    from uuid import NAMESPACE_URL, uuid5

    return "reverie-visual-" + uuid5(NAMESPACE_URL, f"reverie.visual:{dispatch_id}").hex


class ReverieVisualRunBriefV1(BaseModel):
    """What the durable run needs to drive thought's stages. No context text, ever:
    context selection (including memory crystallizations) stays inside thought."""

    model_config = ConfigDict(extra="forbid")

    visual_request: VisualRunRequestV1
    # Per-step RPC budget (AdmissionRuntime.execute reads brief.timeout_sec for the
    # held generate step). Covers diffusion under the run's hold (no separate GPU wait since stage 5.4).
    timeout_sec: float = Field(default=360.0, gt=0)
    session_id: str | None = None

    @model_validator(mode="after")
    def dispatch_identity(self):
        if not self.visual_request.dispatch_id:
            raise ValueError("reverie.visual runs require visual_request.dispatch_id")
        return self


class ReverieVisualStepRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reverie.visual.step.request.v1"] = REVERIE_VISUAL_STEP_REQUEST_KIND
    run_id: str
    correlation_id: str
    step: ReverieVisualStep
    visual_request: VisualRunRequestV1
    # Returned by prepare; every later step names it (it is also the chain_id).
    attempt_id: str | None = None
    # Required for generate only. Never forwarded to interpret/caption calls: the pool
    # attaches a child call to the hold's own role.
    gpu_lease: GpuLeaseRefV1 | None = None

    @model_validator(mode="after")
    def step_shape(self):
        # abandon may omit attempt_id: a lost prepare reply must not strand a claimed attempt,
        # so thought resolves it from visual_request.dispatch_id.
        if self.step not in ("prepare", "abandon") and not self.attempt_id:
            raise ValueError(f"{self.step} requires attempt_id from prepare")
        if self.step == "generate" and self.gpu_lease is None:
            raise ValueError("generate requires the run's diffusion hold")
        if self.step != "generate" and self.gpu_lease is not None:
            raise ValueError("only generate may carry the diffusion hold")
        return self


class ReverieVisualStepResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reverie.visual.step.result.v1"] = REVERIE_VISUAL_STEP_RESULT_KIND
    run_id: str
    correlation_id: str
    step: ReverieVisualStep
    status: ReverieVisualStepStatus
    attempt_id: str | None = None
    # Machine-readable why for retry/terminal (thermal_refused, resource_deferred:..., hold_invalid:..., ...).
    reason: str | None = None
    # Set on terminal and on caption done. None while the run is still in flight.
    outcome: VisualRunOutcome | None = None
    chain_id: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Wall seconds thought spent doing this step's real work (generate: permit wait +
    # diffusion + disk write). Summed into the run's visual_elapsed_sec.
    elapsed_sec: float | None = Field(default=None, ge=0)
    # Seconds to wait before retrying, when thought knows better than the backoff.
    retry_after_sec: float | None = Field(default=None, ge=0)
    execution_receipt: dict[str, Any] | None = None

    @model_validator(mode="after")
    def status_shape(self):
        if self.status == "terminal" and self.outcome is None:
            raise ValueError("terminal step results must name an outcome")
        if self.status == "retry" and not self.reason:
            raise ValueError("retry step results must name a reason")
        if self.status == "done" and self.step in ("prepare", "generate") and not self.attempt_id:
            raise ValueError("done prepare/generate results must carry attempt_id")
        return self
