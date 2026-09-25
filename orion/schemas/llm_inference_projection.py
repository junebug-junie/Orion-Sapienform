"""LLM-inference substrate lane projection (orion-llm-gateway reporting on itself).

The gateway aggregates its own bus-RPC chat calls into fixed windows and publishes
one grammar trace per window (``llm_gateway.inference:<gateway>:<window_id>``,
services/orion-llm-gateway/app/grammar_emit.py). The llm_inference reducer
(orion/substrate/llm_inference_loop/) folds each window into one state per
serving field node. Counts only -- no prompt or response text ever reaches here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    latency_p50_ms: int | None = None
    latency_p95_ms: int | None = None
    # gateway worker labels that served this node this window (bounded list)
    served_by_labels: list[str] = Field(default_factory=list)
    # outcome class -> count, e.g. {"served": 9, "upstream_timeout": 1}
    outcome_classes: dict[str, int] = Field(default_factory=dict)

    # upstream_failed / (served + upstream_failed). None when nothing was sent
    # upstream this window: "not measured", never a fake calm 0.0.
    inference_failure_pressure: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_event_ids: list[str] = Field(default_factory=list)
    observed_at: datetime


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
