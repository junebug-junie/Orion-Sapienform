"""The gateway marks a reply that is not a served answer as degraded (2026-10-13).

A refusal (gpu_pool_unavailable, ...) comes back as a normal llm.chat.result with empty
content and a typed raw.error. Live 2026-10-08 every such reply on the GPU shed was timed
by the caller's rpc_request() as a fast success; the marker makes the caller count it
degraded instead (orion/core/bus/rpc_degraded.py)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef  # noqa: E402
from orion.core.bus.rpc_degraded import begin_collecting, current_degraded_reason, end_collecting  # noqa: E402
from orion.schemas.llm_inference_projection import (  # noqa: E402
    REFUSAL_CLASSES,
    REQUEST_INVALID_CLASSES,
    UPSTREAM_FAILURE_CLASSES,
)

from app import grammar_emit  # noqa: E402


@pytest.mark.parametrize("err", sorted(REFUSAL_CLASSES | REQUEST_INVALID_CLASSES))
def test_every_declined_work_class_is_degraded(err: str) -> None:
    assert grammar_emit.degraded_outcome({"text": "", "raw": {"error": err}}) == err


@pytest.mark.parametrize("err", sorted(UPSTREAM_FAILURE_CLASSES | {"timeout", "gateway_exception"}))
def test_a_backend_failure_stays_timed(err: str) -> None:
    """Review finding: the backend was asked and the wait was real -- an upstream
    timeout is the slowness the transport baseline exists to see."""
    assert grammar_emit.degraded_outcome({"text": "", "raw": {"error": err}}) is None


def test_a_served_or_empty_answer_is_not_degraded() -> None:
    assert grammar_emit.degraded_outcome({"text": "hello", "raw": {}}) is None
    # upstream_empty is unmeasured at rest: deliberately not called degraded.
    assert grammar_emit.degraded_outcome({"text": "", "raw": {}}) is None
    assert grammar_emit.degraded_outcome({"text": "", "reasoning_content": "thought", "raw": {}}) is None
    # a framed backend failure is a failure, but not declined work: timed
    assert grammar_emit.degraded_outcome({"text": "[Error: llamacpp timed out]", "raw": {}}) is None
    assert grammar_emit.degraded_outcome({"text": "[Error: route x not configured]", "raw": {}}) == "route_not_configured"


def _req() -> BaseEnvelope:
    return BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="quick").model_dump(
            mode="json"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,expected",
    [
        ({"text": "", "raw": {"error": "gpu_pool_unavailable", "details": {"reason": "deadline"}}}, "gpu_pool_unavailable"),
        ({"text": "a real answer", "raw": {}, "served_by": "circe-worker-1"}, None),
    ],
)
async def test_handle_chat_notes_degradation_for_the_chassis_to_stamp(result: dict, expected) -> None:
    from app import main

    async def _dispatch(body, *, correlation_id, **_kw):
        return dict(result)

    token = begin_collecting()
    try:
        with patch.object(main, "_dispatch_chat", _dispatch), patch.object(
            main.settings, "llm_gateway_grammar_enabled", False
        ):
            out = await main.handle_chat(_req())
        assert current_degraded_reason() == expected
    finally:
        end_collecting(token)
    assert out.kind == "llm.chat.result"  # the reply itself is unchanged
