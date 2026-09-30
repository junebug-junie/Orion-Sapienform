"""LLM-inference substrate lane projection (orion-llm-gateway reporting on itself).

The gateway aggregates its own chat calls (bus RPC and the HTTP passthroughs) into fixed windows and publishes
one grammar trace per window (``llm_gateway.inference:<gateway>:<window_id>``,
services/orion-llm-gateway/app/grammar_emit.py). The llm_inference reducer
(orion/substrate/llm_inference_loop/) folds each window into one state per
serving field node. Counts only -- no prompt or response text ever reaches here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---- Wire contract shared by the producer (orion-llm-gateway) and the reducer.
# Lives here, not in orion/substrate/, so the gateway can import it without
# executing orion/substrate/__init__.py (graph store, materializer -- the
# import that crash-looped two thin services on 2026-08-19).
LLM_INFERENCE_SOURCE_SERVICE = "orion-llm-gateway"
LLM_INFERENCE_TRACE_PREFIX = "llm_gateway.inference:"
# One atom per serving node per window (self-contained, so a reducer batch
# boundary can split a window between nodes but never inside one node's reading).
ROLE_NODE_WINDOW = "llm_inference_window_observed"
ROLE_WINDOW_COMPLETED = "llm_gateway_window_completed"

OUTCOME_SERVED = "served"
# The backend was asked and did not produce an answer. Only these count
# toward inference_failure_pressure.
UPSTREAM_FAILURE_CLASSES = frozenset(
    {
        "upstream_timeout",
        "upstream_connect",
        "upstream_http_5xx",
        "upstream_not_found",
        "upstream_error",
    }
)
# The gateway itself declined before any backend work. Admission and lease
# refusals belong to the GPU pool's telemetry (gpu-pool spec stage 3 moves
# them); counted for inspection only, never wired to the field here.
REFUSAL_CLASSES = frozenset(
    {
        "gateway_overloaded",
        "gateway_capacity_rejected",
        "resource_lease_rejected",
        "route_operator_closed",
        # lane resolver declined (probe status or background-lane policy);
        # carries no route target, so it is never node-attributed anyway
        "llm_route_unavailable",
        # GPU pool placement (gpu-pool stage 3): the pool gave no card in time, the route has
        # no pool class, no grant was held, or the pool took the card back mid-call (recall /
        # lost lease) -- none of these is the serving node failing.
        "gpu_pool_unavailable",
        "route_not_in_gpu_pool",
        "no_pool_grant",
        "gpu_pool_recalled",
    }
)
# The caller's request was unusable (bad attachment, unknown route). Not the
# backend's health either way.
# upstream_http_4xx lives here, not in UPSTREAM_FAILURE_CLASSES: llama.cpp answers
# 400 for an oversized or malformed request from one caller while the node is fine
# (review finding, 2026-09-25).
# context_overflow: the prompt did not fit any slot the route's class has (llama.cpp's own
# "exceeds the available context size"); one caller's request, not a node fault.
REQUEST_INVALID_CLASSES = frozenset(
    {"request_invalid", "route_not_configured", "upstream_http_4xx", "context_overflow"})


# Retired 2026-09-30 (gpu-pool stage 6.2): one clock started before the pool lease, so it
# measured queue wait + model time as one number, per machine. Replaced by the per-role
# ``by_role`` clocks below. Dropped on load (never re-emitted) so the persisted projection row
# written by the previous reducer still validates under extra="forbid" -- the 2026-07-24
# crash-loop class (scripts/check_substrate_projection_schema_drift.py).
RETIRED_NODE_STATE_FIELDS = frozenset({"latency_p50_ms", "latency_p95_ms"})


class LlmInferenceRoleStateV1(BaseModel):
    """One granted GPU-pool role's window on one serving node (gpu-pool stage 6.2).

    Two disjoint clocks: ``wait_*`` is lease request -> grant (the line; every call that
    waited, served or not), ``model_*`` is grant -> reply (the worker; served calls only).
    ``decode_tps_p50`` is llama.cpp's own ``timings.predicted_per_second`` median over served
    calls that reported it -- a per-token speed, independent of answer length. Every clock is
    ``None`` when nothing measured it this window: "not measured", never a fake 0.
    Projection/debug only: not wired to any field channel (spec G2, pending the 48 h check).
    """

    # protected_namespaces=(): "model_p50_ms" is the model's clock, not a pydantic attribute.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    calls: int = 0
    # of ``calls``, how many came through the HTTP passthroughs (/v1/messages,
    # /v1/chat/completions). Those reach by_role only: the node-level counts above feed
    # inference_failure_pressure and stay bus-RPC calls.
    http_calls: int = 0
    served: int = 0
    upstream_failed: int = 0
    refused: int = 0
    request_invalid: int = 0
    wait_p50_ms: int | None = None
    wait_p95_ms: int | None = None
    model_p50_ms: int | None = None
    model_p95_ms: int | None = None
    decode_tps_p50: float | None = Field(default=None, gt=0.0)
    # served calls whose reply carried timings.predicted_per_second (the p50's sample size)
    decode_tps_samples: int = 0
    # Covariates so a per-role baseline does not read normal slot sharing as a degraded worker
    # (stage 7 input): llama.cpp decodes every busy slot in one batch, so decode speed is banded
    # by occupancy at grant -- solo (the only call in flight on the role) vs shared.
    decode_tps_solo_p50: float | None = Field(default=None, gt=0.0)
    decode_tps_solo_samples: int = 0
    decode_tps_shared_p50: float | None = Field(default=None, gt=0.0)
    decode_tps_shared_samples: int = 0
    # this gateway's calls in flight on the role at each grant (this one included). Counts
    # calls, not pool leases: an idle durable-run hold keeps a slot but decodes nothing.
    busy_p50: int | None = None
    busy_max: int | None = None
    # the pool's discovered slot count for the role at window end (None: pool unreachable)
    slots: int | None = None
    # llama.cpp timings.prompt_n / cache_n summed over served calls that reported both:
    # prompt tokens processed vs reused from the slot's KV cache. None when none reported.
    prompt_n: int | None = None
    cache_n: int | None = None
    cache_reports: int = 0


class LlmInferenceNodeStateV1(BaseModel):
    """Everything one gateway window saw about one serving field node."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["llm_inference.node_state.v1"] = "llm_inference.node_state.v1"

    target_id: str  # "llm_node:<node>"
    node_id: str  # field node key without the "node:" prefix, e.g. "circe"
    gateway_node: str
    sample_window_id: str
    source_trace_id: str
    window_sec: float = 0.0

    calls: int = 0
    served: int = 0
    upstream_failed: int = 0
    refused: int = 0
    request_invalid: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # granted pool role -> that role's wait/model clocks and decode speed (stage 6.2).
    # "ungranted" collects calls that never held a lease (pool refusal, plan error).
    by_role: dict[str, LlmInferenceRoleStateV1] = Field(default_factory=dict)
    # gateway worker labels that served this node this window (bounded list)
    served_by_labels: list[str] = Field(default_factory=list)
    # outcome class -> count, e.g. {"served": 9, "upstream_timeout": 1}
    outcome_classes: dict[str, int] = Field(default_factory=dict)

    # Rolling failure share, NOT this window alone (2026-09-29): over the last
    # FAILURE_WINDOW_SEC of this node's windows, upstream failures / max(attempts,
    # FAILURE_MIN_DENOMINATOR), 0.0 until FAILURE_MIN_COUNT failures -- the RPC
    # delivery bridge's rule (orion/substrate/rpc_delivery.py::hop_pressure). The
    # worse of the node-pooled reading and the worst single worker's. None when
    # nothing was sent upstream in the rolling window: "not measured", never a
    # fake calm 0.0. See orion/substrate/llm_inference_loop/failure_window.py.
    inference_failure_pressure: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_event_ids: list[str] = Field(default_factory=list)
    observed_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and RETIRED_NODE_STATE_FIELDS.intersection(data):
            return {k: v for k, v in data.items() if k not in RETIRED_NODE_STATE_FIELDS}
        return data


class LlmInferenceWindowCountV1(BaseModel):
    """One gateway window's upstream counts for one serving node, kept on the
    projection so the failure reading can span several windows (2026-09-29)."""

    model_config = ConfigDict(extra="forbid")

    window_id: str
    window_end: datetime
    served: int = 0
    upstream_failed: int = 0
    # gateway worker label -> calls sent upstream (served + upstream failures) /
    # upstream failures. Empty when the gateway predates per-worker counts.
    worker_attempted: dict[str, int] = Field(default_factory=dict)
    worker_failed: dict[str, int] = Field(default_factory=dict)


class LlmInferenceProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["llm_inference.projection.v1"] = "llm_inference.projection.v1"
    projection_id: str
    generated_at: datetime
    nodes: dict[str, LlmInferenceNodeStateV1] = Field(default_factory=dict)
    # Calls the gateway could not attribute to a known field node (no route
    # target, or a served_by label outside the known-node convention), last window.
    last_unattributed_calls: int = 0
    last_window_id: str | None = None
    # target_id -> this node's recent windows, oldest first, pruned to the rolling
    # failure window (2026-09-29). Only the substrate runtime reads this row.
    recent_windows: dict[str, list[LlmInferenceWindowCountV1]] = Field(default_factory=dict)
