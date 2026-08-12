from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionDispatchCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str

    source_decision_id: str
    source_proposal_id: str

    dispatch_status: Literal[
        "prepared",
        "dry_run",
        "blocked",
        "prepared_for_dispatch",
        "dispatched",
        "skipped",
    ]

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ]

    dispatch_kind: Literal[
        "inspect",
        "summarize",
        "observe",
        "noop",
        # 2026-08-12: the first MUTATING dispatch kind. Every member above is
        # read-only by construction. Kept a closed Literal so adding another
        # mutating kind stays a deliberate schema change, not a config typo.
        "maintain",
    ]

    target_id: str
    target_kind: str

    cortex_verb: str | None = None
    cortex_mode: str | None = None

    request_envelope: dict[str, object] = Field(default_factory=dict)

    constraints: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)

    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    result_ref: str | None = None
    dispatch_error: str | None = None
    dispatched_at: datetime | None = None

    @model_validator(mode="after")
    def _dispatched_requires_evidence(self) -> "ExecutionDispatchCandidateV1":
        if self.dispatch_status == "dispatched":
            if self.dispatched_at is None or (self.result_ref is None and self.dispatch_error is None):
                raise ValueError(
                    "dispatch_status='dispatched' requires dispatched_at and one of "
                    "result_ref/dispatch_error as evidence a send was actually attempted; "
                    "use 'prepared_for_dispatch' for a candidate that has not been sent yet"
                )
        return self


class ExecutionDispatchFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution.dispatch.frame.v1"] = "execution.dispatch.frame.v1"

    frame_id: str
    generated_at: datetime

    source_policy_frame_id: str
    source_proposal_frame_id: str
    # source_self_state_id -> source_field_tick_id 2026-07-22, SelfStateV1
    # burn -- field was always the real upstream tick.
    source_field_tick_id: str | None = None

    execution_dispatch_policy_id: str = "execution_dispatch_policy.v1"

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ] = "dry_run"

    candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    blocked_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    dispatched_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)

    dispatch_attempted: bool = False
    dispatch_count: int = 0
    blocked_count: int = 0

    warnings: list[str] = Field(default_factory=list)

    # 2026-07-29: self-calibrating daily risk ceiling (replaces the old fixed
    # ORION_DISPATCH_MAX_RISK_PER_DAY constant). EWMA baseline over *uncapped*
    # daily demand -- see services/orion-execution-dispatch-runtime/app/
    # store.py::sum_uncapped_risk_for_day for why this must not be fed from
    # sum_risk_dispatched_today() (right-censored at whatever cap was in
    # force that day). Same naming/shape convention as
    # ExecutionTrajectoryProjectionV1.prediction_error_baseline_ewma/_var/_n
    # (orion/schemas/execution_projection.py), new domain. Every saved frame
    # carries the current baseline state forward (not just the ones that
    # update it) so the next tick's store.load_latest_daily_risk_baseline()
    # always finds the latest state regardless of which tick it reads.
    # Defaults are the correct cold-start value for both a fresh frame and an
    # older persisted row that predates these fields.
    daily_risk_baseline_ewma: float = 0.0
    daily_risk_baseline_ewma_var: float = 0.0
    daily_risk_baseline_ewma_n: int = 0
    # Plain ISO date string ("2026-07-28"), not a bare `date` object -- this
    # store's json_serializer=json.dumps (store.py) can't serialize a bare
    # `date`.
    daily_risk_baseline_last_day: str | None = None

    # 2026-07-30 (docs/superpowers/specs/2026-07-30-execution-dispatch-
    # staleness-discard-design.md): backlog-pressure signal. EWMA over "how
    # many consecutive stale (past their staleness window) policy frames did
    # this tick's FIFO drain discard before finding one fresh enough to
    # process, or running out." A real, direct measure of execution-dispatch
    # backlog depth -- 0 in steady state (healthy: consumption keeps pace
    # with production), rising when production of new policy_decision_frames
    # outpaces this service's real per-tick dispatch throughput (bounded by
    # a synchronous cortex-exec RPC per real send, ~7-11s measured live
    # 2026-07-30). Independent of daily_risk_baseline_* above (that measures
    # risk-budget demand; this measures queue-depth/consumption-lag) -- same
    # carried-forward-on-every-frame convention, same EWMA mechanism
    # (orion.bus.ewma.compute_ewma_update), new domain. NOT persisted on a
    # tick where the policy-decision queue is fully empty from the start (no
    # frame at all to carry a value=0.0 sample on) -- see
    # STALENESS_DISCARD_EWMA_ALPHA's own comment in app/worker.py for why
    # this is a disclosed, deliberate under-sample of the true-idle case
    # rather than a synthetic no-op frame invented to close the gap.
    staleness_discard_count_ewma: float = 0.0
    staleness_discard_count_ewma_var: float = 0.0
    staleness_discard_count_ewma_n: int = 0
