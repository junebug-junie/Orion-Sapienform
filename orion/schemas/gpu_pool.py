"""GPU pool contracts: one lease queue for every GPU on circe.

Design: ``docs/superpowers/specs/2026-09-24-gpu-pool-design.md``. Callers speak in work
classes, the pool speaks in roles and cards; model identity is only ever the *discovered*
``llm_profiles.yaml`` profile, never a name baked into config.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

GPU_POOL_LEASE_REQUEST_CHANNEL = "orion:gpu_pool:lease:request"
GPU_POOL_EVENT_CHANNEL = "orion:gpu_pool:event"
GPU_POOL_STATE_CHANNEL = "orion:gpu_pool:state"
GPU_POOL_STATE_REQUEST_CHANNEL = "orion:gpu_pool:state:request"
GPU_POOL_CONTROL_REQUEST_CHANNEL = "orion:gpu_pool:control:request"
GPU_POOL_ACTUATE_REQUEST_CHANNEL = "orion:gpu_pool:actuate:request"
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
    """One verb against the pool. ``acquire`` is idempotent on ``request_id``."""

    model_config = ConfigDict(extra="forbid")

    verb: Literal["acquire", "heartbeat", "release", "cancel"]
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
        "swap_requested", "swapped", "lent", "unlent", "discovery_mismatch", "discovery_confirmed",
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
    swap_state: Literal["idle", "loading", "unloading"] = "idle"
    cooldown_until: datetime | None = None


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
    """Operator verbs. ``operator_token`` is checked by the pool; never logged."""

    model_config = ConfigDict(extra="forbid")

    verb: Literal["lend", "unlend", "replay", "cancel", "backfill", "hold", "release"]
    operator_token: str = Field(min_length=1, repr=False)
    card: str | None = None
    lease_id: str | None = None
    work_class: str | None = None   # verb=hold: an operator hold (e.g. the multi-card experiment seat)
    backfill: dict[str, Any] | None = None
    actor: str = "operator"


class GpuPoolControlReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class GpuActuateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(default_factory=lambda: uuid4().hex)
    target: str  # the YAML swap verb, e.g. "gpu2/agent"
    role: str
    cards: list[str]


class GpuActuateResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    ok: bool
    elapsed_ms: float | None = None
    reason: str | None = None


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
