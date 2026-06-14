from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from orion.cognition.projection_builder import build_cognitive_projection_for_mind_with_diagnostics
from orion.cognition.projection_context import enrich_projection_context, inject_identity_context_for_projection
from orion.cognition.recall_prefetch import prefetch_recall_bundle_for_projection
from orion.cognition.recall_query import recall_ctx_merge_from_reply
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.contracts.recall import MemoryBundleV1, MemoryItemV1, RecallReplyV1


def _fake_reply(*, n: int = 2) -> RecallReplyV1:
    items = [
        MemoryItemV1(
            id=f"mem-{i}",
            source="journal",
            snippet=f"memory snippet {i} about autonomy",
            score=0.8,
        )
        for i in range(n)
    ]
    return RecallReplyV1(bundle=MemoryBundleV1(items=items, rendered="digest text"))


def _fake_bus(reply: RecallReplyV1 | None = None, *, exc: Exception | None = None) -> MagicMock:
    bus = MagicMock()
    if exc is not None:

        async def _rpc(*_a: Any, **_k: Any) -> Any:
            raise exc

    else:
        reply_payload = reply.model_dump(mode="json") if reply is not None else {"error": "fail"}

        class _Decoded:
            ok = True
            error = None

            class _Env:
                payload = reply_payload

            envelope = _Env()

        async def _rpc(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"data": b"x"}

        bus.codec.decode.return_value = _Decoded()
    bus.rpc_request = AsyncMock(side_effect=_rpc)
    return bus


def test_successful_recall_prefetch_populates_ctx() -> None:
    ctx: dict[str, Any] = {
        "verb": "chat_general",
        "session_id": "sess-a",
        "user_message": "What did we decide?",
        "messages": [{"role": "user", "content": "What did we decide?"}],
    }
    inject_identity_context_for_projection(
        ctx,
        plan_metadata={"personality_file": "orion/cognition/personality/orion_identity.yaml"},
    )
    merge, diag = asyncio.run(
        prefetch_recall_bundle_for_projection(
            _fake_bus(_fake_reply(n=2)),
            source=ServiceRef(name="cortex-orch", version="0.2.0", node="test"),
            ctx=ctx,
            correlation_id=str(uuid4()),
            recall_enabled=True,
            recall_profile="reflect.v1",
            recall_channel="orion:exec:request:RecallService",
            timeout_sec=12.0,
        )
    )
    assert merge is not None
    ctx.update(merge)
    assert ctx["recall_bundle"]["fragments"]
    assert diag["ok"] is True
    assert diag["result_count"] == 2
    assert diag["recall_bundle_present_after_write"] is True
    from orion.cognition.projection_context import summarize_projection_inputs

    summary = summarize_projection_inputs(ctx, phase="orch_mind_preflight")
    assert summary["recall_bundle_present"] is True
    assert summary["recall_fragment_count"] == 2


def test_recall_timeout_degrades_cleanly() -> None:
    ctx: dict[str, Any] = {
        "verb": "chat_general",
        "user_message": "hello",
        "orion_identity_summary": ["Oríon is present."],
        "juniper_relationship_summary": ["Juniper co-architect."],
        "response_policy_summary": ["Answer first."],
    }
    merge, diag = asyncio.run(
        prefetch_recall_bundle_for_projection(
            _fake_bus(exc=TimeoutError("RPC timeout waiting on reply")),
            source=ServiceRef(name="cortex-orch", version="0.2.0", node="test"),
            ctx=ctx,
            correlation_id=str(uuid4()),
            recall_enabled=True,
            recall_profile="reflect.v1",
            recall_channel="orion:exec:request:RecallService",
            timeout_sec=6.0,
        )
    )
    assert merge is None
    assert "recall_bundle" not in ctx or not (ctx.get("recall_bundle") or {}).get("fragments")
    assert diag["timed_out"] is True
    assert diag["ok"] is False
    assert ctx["orion_identity_summary"]
    assert not (ctx.get("recall_bundle") or {}).get("fragments")
    projection, build_diag = build_cognitive_projection_for_mind_with_diagnostics(
        ctx,
        publish_tier_outcomes=False,
        build_path="test.recall_timeout",
    )
    assert build_diag["input_summary"]["recall_bundle_present"] is False
    assert build_diag["input_summary"]["identity_yaml_inputs"]["orion_identity_summary"] >= 1


def test_recall_normalization_matches_exec_shape() -> None:
    reply = _fake_reply(n=1)
    orch_merge = recall_ctx_merge_from_reply(reply)
    assert set(orch_merge.keys()) >= {"recall_bundle", "memory_digest", "recall_fragments", "memory_used"}
    bundle = orch_merge["recall_bundle"]
    assert "fragments" in bundle and "citations" in bundle and "rendered" in bundle
    frag = bundle["fragments"][0]
    assert frag["source"] == "journal"
    assert frag["snippet"]
    assert "items" in orch_merge["memory_bundle"]

    safe_merge = recall_ctx_merge_from_reply(reply, prompt_safe_ctx=True)
    assert safe_merge["memory_bundle"] == {"rendered": "digest text"}
    assert safe_merge["recall_memory_bundle_debug"]["items"]


def test_projection_producer_emits_recall_derived_items() -> None:
    ctx: dict[str, Any] = {
        "verb": "chat_general",
        "orion_identity_summary": ["Oríon."],
        "juniper_relationship_summary": ["Juniper."],
        "response_policy_summary": ["Policy."],
        "recall_bundle": {
            "fragments": [
                {
                    "source": "journal",
                    "snippet": "Prior reflection on shared architecture decisions.",
                },
                {
                    "source": "tension:growth",
                    "snippet": "Tension between speed and care.",
                    "subject": "relationship",
                },
            ],
            "citations": [],
        },
    }
    enrich_projection_context(ctx)
    projection, diag = build_cognitive_projection_for_mind_with_diagnostics(
        ctx,
        publish_tier_outcomes=False,
        build_path="test.recall_projection",
    )
    assert projection is not None
    assert int(projection.item_count or 0) >= 2
    source_counts = diag.get("source_counts") or {}
    assert source_counts.get("orion", 0) >= 1
    assert "recall" in diag.get("projection_sources_returned", [])
    assert "recall:snapshot_ephemeral" in (diag.get("lineage") or [])
