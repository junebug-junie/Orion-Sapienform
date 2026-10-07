"""stance_context_prepare: build the stance context while orion-mind runs.

Unified-turn latency L4 (docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md).

orion-thought used to await orion-mind (~11 s) and only then send stance_react to
cortex-exec, which then spent ~9 s building chat_stance_inputs. The build never
reads mind's output, so orion-thought now fires this request at the same time as
the mind call. cortex-exec builds the context, holds it in an in-process cache
keyed by correlation id, and the later stance_react consumes it instead of
building again.

Both requests must reach the same cortex-exec container (the cache is
in-process). orion-thought therefore derives this channel from the exact exec
request channel it will send stance_react on: each cortex-exec lane container
subscribes to the prepare channel matching its own request channel.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.cortex.schemas import PlanExecutionRequest

STANCE_CONTEXT_PREPARE_REQUEST_KIND = "cortex.stance_context_prepare.request.v1"
STANCE_CONTEXT_PREPARE_RESULT_KIND = "cortex.stance_context_prepare.result.v1"

EXEC_REQUEST_CHANNEL_BASE = "orion:cortex:exec:request"
STANCE_CONTEXT_PREPARE_CHANNEL_BASE = "orion:cortex:exec:stance_prepare"
STANCE_CONTEXT_PREPARE_RESULT_PREFIX = "orion:cortex:exec:stance_prepare_result"

# ctx key orion-thought sets on the stance_react request when it fired a
# prepare for the same turn. cortex-exec only waits for a prepared context when
# this is present, so stance_react callers that never prepare pay nothing.
STANCE_PREPARE_REQUESTED_CTX_KEY = "stance_prepare_requested"


def stance_context_prepare_channel(exec_request_channel: str) -> Optional[str]:
    """Prepare channel served by the container that listens on ``exec_request_channel``.

    ``orion:cortex:exec:request`` -> ``orion:cortex:exec:stance_prepare``;
    ``orion:cortex:exec:request:chat`` -> ``orion:cortex:exec:stance_prepare:chat``.
    Returns None for a channel outside that family: there is no matching prepare
    listener, so the caller must not prepare.
    """
    channel = str(exec_request_channel or "").strip()
    if channel == EXEC_REQUEST_CHANNEL_BASE:
        return STANCE_CONTEXT_PREPARE_CHANNEL_BASE
    prefix = EXEC_REQUEST_CHANNEL_BASE + ":"
    if channel.startswith(prefix) and channel[len(prefix):] and ":" not in channel[len(prefix):]:
        return f"{STANCE_CONTEXT_PREPARE_CHANNEL_BASE}:{channel[len(prefix):]}"
    return None


class StanceContextPrepareRequestV1(BaseModel):
    """Build stance_react's brain reply context ahead of the stance_react request.

    ``plan_request`` is the same PlanExecutionRequest orion-thought will send as
    stance_react (without mind coloring, which the build does not read), so
    cortex-exec builds from the same ctx the stance request would produce.
    """

    model_config = ConfigDict(extra="forbid")

    correlation_id: str = Field(..., min_length=1)
    plan_request: PlanExecutionRequest


class StanceContextPrepareResultV1(BaseModel):
    """Outcome of one prepare. ``ready``: cached for stance_react.
    ``duplicate``: a prepare for this correlation id already exists on this
    container. ``failed``: the build raised; stance_react builds inline."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    status: Literal["ready", "duplicate", "failed"]
    build_ms: Optional[float] = None
    lane: Optional[str] = None
    error: Optional[str] = None
