"""GPU pool contracts: one lease queue for every GPU on circe.

Design: ``docs/superpowers/specs/2026-09-24-gpu-pool-design.md``. Callers speak in work
classes, the pool speaks in roles and cards; model identity is only ever the *discovered*
``llm_profiles.yaml`` profile, never a name baked into config.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

GPU_POOL_LEASE_REQUEST_CHANNEL = "orion:gpu_pool:lease:request"
GPU_POOL_EVENT_CHANNEL = "orion:gpu_pool:event"
GPU_POOL_STATE_CHANNEL = "orion:gpu_pool:state"
GPU_POOL_STATE_REQUEST_CHANNEL = "orion:gpu_pool:state:request"
GPU_POOL_CONTROL_REQUEST_CHANNEL = "orion:gpu_pool:control:request"
GPU_POOL_ACTUATE_REQUEST_CHANNEL = "orion:gpu_pool:actuate:request"
GPU_POOL_ACTUATE_RESULT_CHANNEL = "orion:gpu_pool:actuate:result"
LLM_WORKER_ANNOUNCE_CHANNEL = "orion:llm:worker:announce"
GPU_POOL_LEASE_REPLY_PREFIX = "orion:gpu_pool:reply:"
GPU_POOL_STATE_REPLY_PREFIX = "orion:gpu_pool:state:reply:"
GPU_POOL_CONTROL_REPLY_PREFIX = "orion:gpu_pool:control:reply:"

GPU_LEASE_REQUEST_KIND = "gpu_pool.lease.request.v1"
GPU_LEASE_REPLY_KIND = "gpu_pool.lease.reply.v1"
GPU_POOL_EVENT_KIND = "gpu_pool.event.v1"
GPU_POOL_STATE_KIND = "gpu_pool.state.v1"
GPU_POOL_STATE_REQUEST_KIND = "gpu_pool.state.request.v1"
GPU_POOL_CONTROL_KIND = "gpu_pool.control.v1"
GPU_POOL_CONTROL_REPLY_KIND = "gpu_pool.control.reply.v1"
GPU_ACTUATE_KIND = "gpu_pool.actuate.v1"
GPU_ACTUATE_RESULT_KIND = "gpu_pool.actuate.result.v1"
GPU_LEASE_REF_KIND = "gpu_pool.lease.ref.v1"
LLM_WORKER_ANNOUNCE_KIND = "llm.worker.announce.v1"

Priority = Literal["interactive", "system", "background"]
LeaseKind = Literal["request", "hold"]
LeaseStatus = Literal[
    "queued", "backlogged", "granted", "recalling", "retry_wait",
    "released", "unavailable", "aborted", "expired", "dead_letter",
]
TERMINAL_STATUSES = frozenset({"released", "unavailable", "dead_letter"})
ReleaseOutcome = Literal["ok", "upstream_error", "timeout", "cancelled"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GpuLeaseRequestV1(BaseModel):
    """One verb against the pool. ``acquire`` is idempotent on ``request_id``.

    Stage 4 (docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md):
    ``attach`` asks for a child request lease under a hold (``hold_lease_id`` + ``hold_generation``;
    the child's own ``lease_id`` does not exist yet, so it is not sent). Named ``hold_*``, not
    ``parent_*``: the pool's request dict already uses ``parent_lease_id`` for dead-letter replay
    lineage (services/orion-gpu-pool/app/runtime.py), and the two must not collide;
    ``status`` is a read of ``lease_id`` with no side effect (resume after restart, Door-A).
    A pool that predates the engine for them answers ``unavailable reason=verb_not_supported:<verb>``.
    """

    model_config = ConfigDict(extra="forbid")

    verb: Literal["acquire", "heartbeat", "release", "cancel", "attach", "status"]
    request_id: str | None = Field(None, min_length=1, max_length=128)
    lease_id: str | None = Field(None, min_length=1, max_length=128)
    holder: str | None = Field(None, min_length=1, max_length=256)
    work_class: str | None = Field(None, min_length=1, max_length=64)
    priority: Priority = "system"
    kind: LeaseKind = "request"
    min_ctx_tokens: int = Field(0, ge=0)
    deadline_at: datetime | None = None
    # The turn this lease serves. The lease RPC itself travels on a fresh correlation id so
    # queue wait never enters the turn's bus-synaptic chain (spec: "waiting in line is not
    # transport"); this field is how the panel and history join back to the turn.
    turn_correlation_id: str | None = Field(None, max_length=128)
    outcome: ReleaseOutcome | None = None
    detail: str | None = Field(None, max_length=2000)
    # True only when someone will actually use a re-grant: a durable run that resumes by
    # lease_id, or (stage 3) the gateway re-dispatching ``replay_payload`` itself. Without it a
    # failed/expired/aborted lease ends instead of retrying, and "backlog" behaves like "wait" --
    # otherwise the pool would grant GPU slots to callers that have already gone away.
    retryable: bool = False
    # Size-capped caller payload the pool may re-dispatch on backlog replay.
    replay_payload: dict[str, Any] | None = None
    # verb=attach only: the hold this call runs under, and the hold generation the caller was
    # granted (a stale generation means the hold was re-granted and the caller must not attach).
    hold_lease_id: str | None = Field(None, min_length=1, max_length=128)
    hold_generation: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _stage4_verb_shapes(self):
        has_hold = self.hold_lease_id is not None or self.hold_generation is not None
        if self.verb == "attach":
            if self.hold_lease_id is None or self.hold_generation is None:
                raise ValueError("attach needs hold_lease_id and hold_generation")
            if self.lease_id is not None:
                raise ValueError("attach names the hold in hold_lease_id; lease_id is the child's, not sent")
            if self.request_id is None or self.work_class is None:
                raise ValueError("attach needs request_id (idempotency) and work_class")
        elif has_hold:
            raise ValueError(f"hold_lease_id/hold_generation are only valid on attach, not {self.verb}")
        if self.verb == "status" and self.lease_id is None:
            raise ValueError("status needs lease_id")
        return self


class GpuLeaseGrantV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str
    generation: int = Field(ge=1)
    role: str
    cards: list[str]
    url: str
    profile_name: str | None = None
    model_file: str | None = None
    ctx_per_slot: int | None = None
    served_by: str


class GpuLeaseReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["granted", "queued", "backlogged", "unavailable", "ok", "unknown_lease", "recall"]
    lease_id: str | None = None
    grant: GpuLeaseGrantV1 | None = None
    position: int | None = None
    reason: str | None = None
    recall_by: datetime | None = None


class GpuPoolEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["gpu_pool.event.v1"] = GPU_POOL_EVENT_KIND
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event: Literal[
        "admitted", "queued", "granted", "backlogged", "recalled", "aborted", "expired",
        "retried", "dead_lettered", "replayed", "released", "unavailable", "cancelled",
        "swap_requested", "swap_started", "swapped", "swap_failed", "actuate_refused", "lent", "unlent", "discovery_mismatch", "discovery_confirmed",
    ]
    lease_id: str | None = None
    holder: str | None = None
    work_class: str | None = None
    priority: Priority | None = None
    role: str | None = None
    cards: list[str] = Field(default_factory=list)
    turn_correlation_id: str | None = None
    attempt: int | None = None
    waited_ms: float | None = None
    held_ms: float | None = None
    reason: str | None = None
    generated_at: datetime = Field(default_factory=_now)
    detail: dict[str, Any] = Field(default_factory=dict)


class DiscoveredRoleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    kind: Literal["llm", "service"]
    cards: list[str]
    url: str
    status: Literal["confirmed", "mismatch", "silent", "down", "unloaded", "evicted", "static"]
    profile_name: str | None = None
    model_file: str | None = None
    model_path: str | None = None   # as llama.cpp /props reports it (durable-runs compares full paths)
    slots: int = 0
    ctx_per_slot: int | None = None
    vision: bool | None = None
    vram_gb: float | None = None
    detail: str | None = None
    checked_at: datetime | None = None


class GpuCardStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card: str
    vram_gb: float
    lendable: bool = False
    lent: bool = False
    swapped_in: list[str] = Field(default_factory=list)
    # fault: a load failed and the actuator could not restore the evicted residents; no grants on
    # any role of the card until an operator clears it or discovery sees the residents healthy.
    swap_state: Literal["idle", "loading", "unloading", "fault"] = "idle"
    cooldown_until: datetime | None = None
    # Stage 4.3 actuation engine (all optional; a pre-4.3 pool sends none of them):
    swap_role: str | None = None              # the seat a loading/unloading/fault state is about
    residency_until: datetime | None = None   # evicted residents stay until this after an unload
    loaded_at: datetime | None = None         # when the pool loaded (or adopted) the seat here
    actuated_roles: list[str] = Field(default_factory=list)   # seats on this card the pool may actuate
    # The current or last GpuActuateV1 for this card set: action_id, role, action, generation,
    # sent_at, acked_at, deadline_at, phase, outcome, reason.
    actuation: dict[str, Any] | None = None


class GpuLeaseRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str
    request_id: str
    holder: str
    work_class: str
    priority: Priority
    kind: LeaseKind
    status: LeaseStatus
    role: str | None = None
    attempt: int = 1
    created_at: datetime
    granted_at: datetime | None = None
    recall_by: datetime | None = None
    turn_correlation_id: str | None = None
    generation: int = 0
    hold_lease_id: str | None = None   # a child call: the durable-run hold whose slot it runs in


class GpuPoolStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["gpu_pool.state.v1"] = GPU_POOL_STATE_KIND
    generated_at: datetime = Field(default_factory=_now)
    mode: Literal["observe", "enforce"] = "observe"
    config_digest: str
    cards: list[GpuCardStateV1]
    roles: list[DiscoveredRoleV1]
    unclaimed_servers: list[str] = Field(default_factory=list)
    leases: list[GpuLeaseRowV1] = Field(default_factory=list)
    queue_depth: dict[str, int] = Field(default_factory=dict)
    backlog_depth: dict[str, int] = Field(default_factory=dict)
    # Swap-load guards as the pool last read them: name -> None when clear, else why it blocks.
    swap_guards: dict[str, str | None] = Field(default_factory=dict)
    # Filled only on request (GpuPoolStateRequestV1), never on the periodic broadcast:
    config: dict[str, Any] | None = None          # parsed config/gpu_pool.yaml (the Hub picture)
    config_yaml: str | None = None                # the file as written (the Hub "raw YAML" view)
    history_lease_id: str | None = None
    history: list[dict[str, Any]] | None = None   # that lease's path through the lease graph


class GpuPoolStateRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_leases: bool = True
    include_config: bool = False
    history_for: str | None = Field(None, max_length=128)


class GpuPoolControlV1(BaseModel):
    """Operator verbs. No token (Juniper, 2026-09-24): the pool trusts the bus like every other Orion
    service does. Every verb is logged with its ``actor`` and published as a pool event."""

    model_config = ConfigDict(extra="forbid")

    # clear_fault (stage 4.3): take `card` out of swap_state=fault. The pool reconciles with the
    # actuator (`status`) and adopts what it reports; with no answer it settles from discovery.
    verb: Literal["lend", "unlend", "replay", "cancel", "backfill", "hold", "release", "clear_fault"]
    card: str | None = None
    lease_id: str | None = None
    backfill: dict[str, Any] | None = None
    actor: str = "operator"
    work_class: str | None = None   # verb=hold: an operator hold (e.g. the multi-card experiment seat)


class GpuPoolControlReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class GpuLeaseRefV1(BaseModel):
    """A pool lease a caller carries on each LLM call made under it (stage 4: a durable run's
    hold). The gateway turns it into ``attach`` instead of taking a lease of its own -- a call
    whose run already holds the role's only slot would otherwise queue behind itself forever.
    Wire: HTTP header ``X-Orion-Gpu-Lease`` (orion.llm.resource_lease) or bus ``options.gpu_lease``."""

    model_config = ConfigDict(extra="forbid")

    lease_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=1)
    role: str = Field(min_length=1, max_length=64)
    holder: str = Field(min_length=1, max_length=256)


ActuateAction = Literal["load", "unload", "status"]


class GpuActuateV1(BaseModel):
    """Pool -> host actuator: put ``role`` on (or off) ``cards``. The message names a role, never a
    container: the actuator only touches compose services its OWN copy of config/gpu_pool.yaml lists
    under its own actuator name, and refuses when ``launch_digest`` differs from that copy's
    (orion.gpu_pool.config.launch_digest)."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=128)   # idempotency key: a replay returns the recorded result
    generation: int = Field(ge=1)       # pool-issued, increasing per card set; actuator rejects <= last seen
    actuator: str = Field(min_length=1, max_length=64)      # the host actuator that must act; others ignore it
    role: str = Field(min_length=1, max_length=64)
    action: ActuateAction
    cards: list[str] = Field(min_length=1)
    profile: str | None = None          # llm_profiles.yaml profile; stage 4 always None (compose default)
    launch_digest: str = Field(min_length=1, max_length=128)
    deadline_at: datetime
    reason: str = Field(min_length=1, max_length=256)    # demand | idle | max_hold | operator | reconcile ...


class GpuActuateResultV1(BaseModel):
    """Host actuator -> pool. ``accepted`` within ``defaults.actuate_ack_sec``, then ``progress``,
    then exactly one terminal ``succeeded`` | ``failed`` | ``refused`` per ``action_id``."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=1)
    role: str = Field(min_length=1, max_length=64)
    action: ActuateAction
    status: Literal["accepted", "progress", "succeeded", "failed", "refused"]
    phase: Literal["draining", "stopping", "starting", "ready_wait", "rolling_back"] | None = None
    restored: bool | None = None        # failed load only: were the evicted residents put back?
    elapsed_ms: float | None = Field(None, ge=0)
    reason: str | None = Field(None, max_length=2000)
    # role -> container state after the action, so a restarted pool can reconcile from it.
    observed: dict[str, Literal["running", "exited", "absent", "unknown"]] = Field(default_factory=dict)
    # action=status only (stage 4.3; consumer-first: the pool reads them, the actuator may start
    # sending them after that pool is deployed). in_flight: True while an action for this card set
    # is still running (the pool keeps polling and never faults the card for a missed deadline),
    # False when none is, None when the actuator did not say. last_action_id: the last action it
    # finished for this card set. Structured so the pool never parses `reason` for state.
    in_flight: bool | None = None
    last_action_id: str | None = Field(None, max_length=128)

    @model_validator(mode="after")
    def _restored_only_on_failed_load(self):
        if self.restored is not None and (self.status != "failed" or self.action != "load"):
            raise ValueError("restored is only meaningful on a failed load")
        if (self.in_flight is not None or self.last_action_id is not None) and self.action != "status":
            raise ValueError("in_flight/last_action_id are only meaningful on a status reply")
        return self


class LlmWorkerAnnounceV1(BaseModel):
    """A llama.cpp worker saying which role and ``llm_profiles.yaml`` profile it serves."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["llm.worker.announce.v1"] = LLM_WORKER_ANNOUNCE_KIND
    host: str
    role: str
    profile_name: str
    port: int = Field(ge=1, le=65535)
    cuda_visible_devices: str | None = None
    service_name: str | None = None
    started_at: datetime | None = None
    announced_at: datetime = Field(default_factory=_now)
