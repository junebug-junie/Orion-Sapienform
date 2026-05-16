"""Late Mind shadow refresh for chat stance turns.

Orch invokes Mind before Exec has prepared the full chat stance substrate. That can
leave the persisted Mind run with an empty CognitiveProjectionV1 while Exec later
builds a populated projection for ChatStanceDebug.

This module performs a bounded, non-authoritative refresh from Exec after the
real chat stance projection exists. It does not change routing, replace the
legacy ChatStanceBrief, or relax stance skip gates. Its output is merged only as
Mind metadata for Hub comparison/inspection surfaces and is also published as a
Mind artifact so the Mind runs modal can show the refreshed run.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1, MindRunResultV1
from orion.schemas.mind.artifact import MindRunArtifactV1

logger = logging.getLogger("orion.cortex.exec.late_mind_shadow")

MIND_ARTIFACT_CHANNEL = "orion:mind:artifact"
MIND_ARTIFACT_KIND = "mind.run.artifact.v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _projection_item_count(projection: dict[str, Any] | None) -> int:
    if not isinstance(projection, dict):
        return 0
    try:
        return int(projection.get("item_count") or 0)
    except Exception:
        return 0


def _mind_enabled(ctx: dict[str, Any]) -> bool:
    metadata = _as_dict(ctx.get("metadata"))
    return metadata.get("mind_enabled") is True


def _should_refresh_late_mind(ctx: dict[str, Any]) -> bool:
    if ctx.get("exec_late_mind_shadow_refreshed") is True:
        return False
    if not _mind_enabled(ctx):
        return False
    projection = _as_dict(ctx.get("chat_cognitive_projection"))
    projection_count = _projection_item_count(projection)
    if projection_count <= 0:
        return False

    metadata = _as_dict(ctx.get("metadata"))
    existing_count = 0
    try:
        existing_count = int(metadata.get("mind.cognitive_projection_item_count") or 0)
    except Exception:
        existing_count = 0
    existing_shadow = bool(metadata.get("mind_shadow_synthesis_present"))
    existing_projection_id = str(metadata.get("mind.cognitive_projection_id") or "")
    projection_id = str(projection.get("projection_id") or "")

    return (
        not existing_shadow
        or existing_count < projection_count
        or (projection_id and existing_projection_id and existing_projection_id != projection_id)
    )


def _mind_base_url() -> str:
    return (os.getenv("ORION_MIND_BASE_URL") or "http://orion-mind:6611").rstrip("/")


def _mind_timeout_sec() -> float:
    raw = os.getenv("ORION_MIND_TIMEOUT_SEC") or os.getenv("EXEC_LATE_MIND_TIMEOUT_SEC") or "45"
    try:
        return max(0.5, float(raw))
    except Exception:
        return 45.0


def _ctx_user_text(ctx: dict[str, Any]) -> str:
    for key in ("user_message", "raw_user_text", "message"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = ctx.get("messages") if isinstance(ctx.get("messages"), list) else []
    for msg in reversed(messages):
        if isinstance(msg, dict) and str(msg.get("role") or "").lower() == "user":
            value = msg.get("content") or msg.get("text") or ""
            if str(value).strip():
                return str(value).strip()
    return ""


def _messages_tail(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    messages = ctx.get("messages") if isinstance(ctx.get("messages"), list) else []
    tail: list[dict[str, Any]] = []
    for msg in messages[-8:]:
        if isinstance(msg, dict):
            tail.append(dict(msg))
        elif hasattr(msg, "model_dump"):
            tail.append(msg.model_dump(mode="json"))
    return tail


def _build_request(ctx: dict[str, Any], correlation_id: str) -> MindRunRequestV1:
    metadata = _as_dict(ctx.get("metadata"))
    policy_raw = _as_dict(metadata.get("mind_policy"))
    projection = _as_dict(ctx.get("chat_cognitive_projection"))
    policy = MindRunPolicyV1(
        n_loops_max=int(policy_raw.get("n_loops_max") or os.getenv("MIND_N_LOOPS_DEFAULT") or 1),
        wall_time_ms_max=int(policy_raw.get("wall_time_ms_max") or os.getenv("MIND_WALL_MS_DEFAULT") or 120000),
        llm_enabled_per_loop=list(policy_raw.get("llm_enabled_per_loop") or []),
        router_profile_id=str(policy_raw.get("router_profile_id") or "default"),
    )
    snapshot = {
        "user_text": _ctx_user_text(ctx)[:20_000],
        "messages_tail": _messages_tail(ctx),
        "facets": {
            "cognitive_projection": projection,
        },
    }
    orion_state = metadata.get("orion_state")
    if isinstance(orion_state, dict):
        snapshot["orion_state"] = orion_state
    return MindRunRequestV1(
        correlation_id=correlation_id,
        session_id=ctx.get("session_id"),
        trace_id=ctx.get("trace_id"),
        trigger="user_turn",
        snapshot_inputs=snapshot,
        policy=policy,
        upstream_artifacts={
            "source": "cortex_exec_late_mind_shadow",
            "previous_mind_projection_id": metadata.get("mind.cognitive_projection_id"),
            "chat_stance_projection_id": projection.get("projection_id"),
        },
    )


def _mind_quality(result: MindRunResultV1) -> str:
    if isinstance(getattr(result.brief, "mind_quality", None), str):
        return result.brief.mind_quality
    if isinstance(getattr(result, "mind_quality", None), str):
        return result.mind_quality
    return "empty"


def _merge_result_into_ctx(ctx: dict[str, Any], result: MindRunResultV1) -> None:
    metadata = ctx.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    for key, value in (result.brief.machine_contract or {}).items():
        metadata[str(key)] = value
    metadata["mind_handoff"] = result.brief.model_dump(mode="json")
    metadata["mind_quality"] = _mind_quality(result)
    metadata["mind_run_ok"] = bool(result.ok)
    metadata["mind_contract_only"] = _mind_quality(result) in {"fallback_contract_only", "shadow_synthesis"}
    metadata["mind_skip_stance_synthesis"] = False
    metadata["mind_authorized_for_stance_skip"] = False
    metadata["exec_late_mind_shadow_refreshed"] = True
    metadata["exec_late_mind_run_id"] = str(result.mind_run_id)
    shadow = result.brief.shadow_synthesis
    if shadow is not None:
        metadata["mind_shadow_synthesis"] = shadow.model_dump(mode="json")
        metadata["mind_shadow_synthesis_present"] = bool(shadow.present)
        metadata["mind_authorized_for_stance_skip"] = bool(shadow.authorized_for_stance_skip)
    else:
        metadata["mind_shadow_synthesis_present"] = False
    ctx["exec_late_mind_shadow_refreshed"] = True
    ctx["exec_late_mind_result"] = result.model_dump(mode="json")


async def _publish_artifact(
    bus: Any,
    *,
    source: ServiceRef,
    correlation_id: str,
    ctx: dict[str, Any],
    req: MindRunRequestV1,
    result: MindRunResultV1,
) -> None:
    artifact = MindRunArtifactV1(
        mind_run_id=result.mind_run_id,
        correlation_id=correlation_id,
        session_id=ctx.get("session_id"),
        trigger=req.trigger,
        ok=result.ok,
        error_code=result.error_code,
        snapshot_hash=result.snapshot_hash,
        router_profile_id=req.policy.router_profile_id,
        result_jsonb=result.model_dump(mode="json"),
        request_summary_jsonb={
            "correlation_id": correlation_id,
            "verb": ctx.get("verb"),
            "mode": ctx.get("mode"),
            "session_id": ctx.get("session_id"),
            "source": "cortex_exec_late_mind_shadow",
            "projection_id": (_as_dict(ctx.get("chat_cognitive_projection"))).get("projection_id"),
            "projection_item_count": (_as_dict(ctx.get("chat_cognitive_projection"))).get("item_count"),
        },
        created_at_utc=datetime.now(timezone.utc),
    )
    env = BaseEnvelope(
        kind=MIND_ARTIFACT_KIND,
        source=source,
        correlation_id=correlation_id,
        payload=artifact.model_dump(mode="json"),
    )
    await bus.publish(MIND_ARTIFACT_CHANNEL, env)


async def maybe_refresh_mind_shadow_after_projection(
    *,
    bus: Any,
    source: ServiceRef,
    ctx: dict[str, Any],
    correlation_id: str,
) -> bool:
    """Run Mind after Exec has a populated chat stance projection.

    Returns True when a late run succeeded and metadata was refreshed. Failures are
    logged and stored in metadata but never fail the chat stance step.
    """
    if not _should_refresh_late_mind(ctx):
        return False
    metadata = ctx.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return False
    req = _build_request(ctx, correlation_id)
    url = f"{_mind_base_url()}/v1/mind/run"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_mind_timeout_sec())) as client:
            resp = await client.post(url, json=req.model_dump(mode="json"))
            resp.raise_for_status()
            result = MindRunResultV1.model_validate(resp.json())
    except Exception as exc:
        logger.warning("exec_late_mind_shadow_failed corr=%s err=%s", correlation_id, exc)
        metadata["exec_late_mind_shadow_failed"] = str(exc)
        return False

    _merge_result_into_ctx(ctx, result)
    try:
        await _publish_artifact(bus, source=source, correlation_id=correlation_id, ctx=ctx, req=req, result=result)
    except Exception as exc:
        logger.warning("exec_late_mind_artifact_publish_failed corr=%s err=%s", correlation_id, exc)
        metadata["exec_late_mind_artifact_publish_failed"] = str(exc)
    logger.info(
        "exec_late_mind_shadow_refreshed corr=%s mind_run_id=%s quality=%s projection_id=%s item_count=%s shadow=%s",
        correlation_id,
        result.mind_run_id,
        _mind_quality(result),
        (_as_dict(ctx.get("chat_cognitive_projection"))).get("projection_id"),
        (_as_dict(ctx.get("chat_cognitive_projection"))).get("item_count"),
        bool(result.brief.shadow_synthesis),
    )
    return True
