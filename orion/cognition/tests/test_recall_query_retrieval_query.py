"""cortex-orch's recall builder (Mind prefetch, autonomy) sends the caller's search text.

Phase 3 of docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from orion.cognition.recall_prefetch import prefetch_recall_bundle_for_projection
from orion.cognition.recall_query import (
    build_recall_query_v1,
    cap_retrieval_query,
    recall_query_mode_from_cfg,
    retrieval_query_from_ctx,
)
from orion.core.bus.bus_schemas import ServiceRef


def test_builder_prefers_ctx_retrieval_query_and_keeps_fragment() -> None:
    req = build_recall_query_v1(
        {"user_message": "long instruction prompt " * 50, "retrieval_query": " the standing question "},
        correlation_id="c",
        deadline_ms=12_000,
    )
    assert req is not None
    assert req.retrieval_query == "the standing question"
    assert req.fragment.startswith("long instruction prompt")
    assert req.deadline_ms == 12_000
    assert req.mode == "retrieve"


def test_builder_without_retrieval_query_is_unchanged() -> None:
    req = build_recall_query_v1({"user_message": "hello"}, correlation_id="c")
    assert req is not None
    assert req.retrieval_query is None
    assert req.deadline_ms is None
    assert req.fragment == "hello"


def test_builder_still_refuses_an_empty_retrieve_query() -> None:
    assert build_recall_query_v1({"user_message": ""}, correlation_id="c") is None


def test_builder_allows_context_only_without_text() -> None:
    req = build_recall_query_v1({}, correlation_id="c", recall_cfg={"query_mode": "context_only"})
    assert req is not None
    assert req.mode == "context_only"
    assert req.fragment == ""


def test_helpers_cap_and_normalize() -> None:
    assert cap_retrieval_query("x" * 2000) == "x" * 1000
    assert cap_retrieval_query("   ") is None
    assert cap_retrieval_query(None) is None
    assert retrieval_query_from_ctx({"retrieval_query": 5}) is None
    assert recall_query_mode_from_cfg({"query_mode": "CONTEXT_ONLY"}) == "context_only"
    assert recall_query_mode_from_cfg({"query_mode": "nope"}) == "retrieve"
    assert recall_query_mode_from_cfg(None) == "retrieve"
    # recall_cfg["mode"] is RecallDirective.mode, a different thing: never read here.
    assert recall_query_mode_from_cfg({"mode": "context_only"}) == "retrieve"


def test_prefetch_sends_retrieval_query_and_deadline() -> None:
    bus = MagicMock()
    bus.rpc_request = AsyncMock(side_effect=TimeoutError("stop after capture"))
    ctx: dict[str, Any] = {"verb": "chat_general", "user_message": "a turn", "retrieval_query": "what to search"}

    asyncio.run(
        prefetch_recall_bundle_for_projection(
            bus,
            source=ServiceRef(name="cortex-orch", version="0", node="t"),
            ctx=ctx,
            correlation_id=str(uuid4()),
            recall_enabled=True,
            recall_profile="reflect.v1",
            recall_channel="orion:exec:request:RecallService",
            timeout_sec=12.5,
        )
    )

    env = bus.rpc_request.call_args.args[1]
    assert env.payload["retrieval_query"] == "what to search"
    assert env.payload["fragment"] == "a turn"
    assert env.payload["deadline_ms"] == 12_500


def _prefetch(ctx: dict[str, Any], recall_cfg: dict[str, Any]) -> tuple[MagicMock, dict]:
    bus = MagicMock()
    bus.rpc_request = AsyncMock(side_effect=TimeoutError("stop after capture"))
    _merge, diag = asyncio.run(
        prefetch_recall_bundle_for_projection(
            bus,
            source=ServiceRef(name="cortex-orch", version="0", node="t"),
            ctx=ctx,
            correlation_id=str(uuid4()),
            recall_enabled=True,
            recall_profile="reflect.v1",
            recall_cfg=recall_cfg,
            recall_channel="orion:exec:request:RecallService",
            timeout_sec=5.0,
        )
    )
    return bus, diag


def test_prefetch_and_builder_agree_context_only_needs_no_text() -> None:
    """PR #2423 review: prefetch refused context_only with no text while the builder
    allowed it. Both now allow it."""
    ctx: dict[str, Any] = {"verb": "reverie_narrate"}
    assert build_recall_query_v1(ctx, correlation_id="c", recall_cfg={"query_mode": "context_only"}) is not None
    bus, diag = _prefetch(ctx, {"query_mode": "context_only"})
    assert diag["reason"] != "empty_query_text"
    env = bus.rpc_request.call_args.args[1]
    assert env.payload["mode"] == "context_only"


def test_prefetch_still_refuses_an_empty_retrieve_query() -> None:
    bus, diag = _prefetch({"verb": "chat_general"}, {})
    assert diag["reason"] == "empty_query_text"
    bus.rpc_request.assert_not_called()
