"""Orion → orion-mind HTTP integration (Orch is the only canonical caller for binding runs)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.mind.v1 import MindRunRequestV1, MindRunResultV1, MindRunPolicyV1
from orion.schemas.chat_stance import ChatStanceBrief
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.mind.artifact import MindRunArtifactV1

from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch.mind")

MIND_ARTIFACT_CHANNEL = "orion:mind:artifact"
MIND_ARTIFACT_KIND = "mind.run.artifact.v1"


def _mind_enabled_exact(metadata: dict[str, Any] | None) -> bool:
    return metadata is not None and metadata.get("mind_enabled") is True


def build_mind_run_request(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
) -> MindRunRequestV1:
    """Construct a bounded v1 request; snapshot is inline JSON only (v1)."""
    meta = client_request.context.metadata if isinstance(client_request.context.metadata, dict) else {}
    trigger = meta.get("mind_trigger")
    if trigger not in ("user_turn", "scheduled", "operator", "replay"):
        trigger = "user_turn"
    user_text = (
        (client_request.context.user_message or client_request.context.raw_user_text or "") or ""
    ).strip()
    messages = [m for m in (client_request.context.messages or [])]
    messages_tail = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages[-8:]]
    orion_state = (plan_request.context or {}).get("metadata", {}) if isinstance(plan_request.context, dict) else {}
    orion_state = orion_state.get("orion_state") if isinstance(orion_state, dict) else None
    snapshot: dict[str, Any] = {
        "user_text": user_text[:20_000],
        "messages_tail": messages_tail,
    }
    if orion_state is not None:
        snapshot["orion_state"] = orion_state
    extra_policy = meta.get("mind_policy") if isinstance(meta.get("mind_policy"), dict) else {}
    n_loops = int(extra_policy.get("n_loops_max", get_settings().mind_n_loops_default))
    wall_ms = int(extra_policy.get("wall_time_ms_max", get_settings().mind_wall_ms_default))
    router_profile = str(extra_policy.get("router_profile_id") or "default")
    policy = MindRunPolicyV1(
        n_loops_max=n_loops,
        wall_time_ms_max=wall_ms,
        llm_enabled_per_loop=list(extra_policy.get("llm_enabled_per_loop") or []),
        router_profile_id=router_profile,
    )
    return MindRunRequestV1(
        correlation_id=correlation_id,
        session_id=client_request.context.session_id,
        trace_id=client_request.context.trace_id,
        trigger=trigger,  # type: ignore[arg-type]
        snapshot_inputs=snapshot,
        policy=policy,
    )


def merge_mind_brief_into_plan_metadata(plan_request: PlanExecutionRequest, result: MindRunResultV1) -> None:
    meta = plan_request.context.setdefault("metadata", {})
    if not isinstance(meta, dict):
        return
    for k, v in (result.brief.machine_contract or {}).items():
        meta[str(k)] = v
    meta["mind_handoff"] = result.brief.model_dump(mode="json")
    skip_llm_stance = False
    if result.ok:
        meta["mind_run_ok"] = True
        sp = result.brief.stance_payload if isinstance(result.brief.stance_payload, dict) else {}
        try:
            ChatStanceBrief.model_validate(sp)
            skip_llm_stance = True
        except Exception:
            meta["mind_stance_payload_invalid"] = True
            skip_llm_stance = False
        meta["mind_skip_stance_synthesis"] = skip_llm_stance
    else:
        meta["mind_skip_stance_synthesis"] = False
        meta["mind_run_ok"] = False
        meta["mind_error_code"] = result.error_code


async def call_orion_mind_http(req: MindRunRequestV1) -> MindRunResultV1:
    s = get_settings()
    base = (s.orion_mind_base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("orion_mind_unconfigured")
    url = f"{base}/v1/mind/run"
    timeout_sec = float(s.orion_mind_timeout_sec)
    timeout = httpx.Timeout(
        connect=min(10.0, timeout_sec),
        read=timeout_sec,
        write=min(30.0, timeout_sec),
        pool=5.0,
    )
    max_body = int(s.orion_mind_max_response_bytes)
    limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        resp = await client.post(url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        raw = resp.content
        if len(raw) > max_body:
            raise RuntimeError(f"mind_response_too_large:{len(raw)}")
        return MindRunResultV1.model_validate(resp.json())


async def publish_mind_run_artifact(
    bus: Any,
    *,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    client_request: CortexClientRequest,
    mind_req: MindRunRequestV1,
    mind_res: MindRunResultV1,
) -> None:
    summary = {
        "correlation_id": correlation_id,
        "verb": client_request.verb,
        "mode": client_request.mode,
        "session_id": client_request.context.session_id,
    }
    artifact = MindRunArtifactV1(
        mind_run_id=mind_res.mind_run_id,
        correlation_id=correlation_id,
        session_id=client_request.context.session_id,
        trigger=mind_req.trigger,
        ok=mind_res.ok,
        error_code=mind_res.error_code,
        snapshot_hash=mind_res.snapshot_hash,
        router_profile_id=mind_req.policy.router_profile_id,
        result_jsonb=mind_res.model_dump(mode="json"),
        request_summary_jsonb=summary,
        created_at_utc=datetime.now(timezone.utc),
    )
    env = BaseEnvelope(
        kind=MIND_ARTIFACT_KIND,
        source=source,
        correlation_id=correlation_id,
        causality_chain=list(causality_chain or []),
        trace=dict(trace or {}),
        payload=artifact.model_dump(mode="json"),
    )
    await bus.publish(MIND_ARTIFACT_CHANNEL, env)
