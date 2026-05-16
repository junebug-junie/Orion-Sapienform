"""Orion → orion-mind HTTP integration (Orch is the only canonical caller for binding runs)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.cognition.projection_builder import build_cognitive_projection_for_context
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


def _mind_result_quality(result: MindRunResultV1) -> str:
    brief_quality = getattr(result.brief, "mind_quality", None)
    if isinstance(brief_quality, str) and brief_quality:
        return brief_quality
    result_quality = getattr(result, "mind_quality", None)
    if isinstance(result_quality, str) and result_quality:
        return result_quality
    summary = (result.brief.summary_one_paragraph or "").strip().lower()
    if summary in {
        "deterministic mind run (v1).",
        "fallback contract only — no meaningful mind synthesis produced.",
    }:
        return "fallback_contract_only"
    if not result.ok:
        return "error"
    return "empty"


def _mind_result_is_deterministic_contract_only(result: MindRunResultV1) -> bool:
    quality = _mind_result_quality(result)
    if quality == "fallback_contract_only":
        return True
    patches = list(result.trajectory.patches or [])
    if patches and all(getattr(p.provenance, "model_id", "") == "deterministic" for p in patches):
        return True
    summary = (result.brief.summary_one_paragraph or "").strip()
    return summary == "Deterministic mind run (v1)."


async def fetch_substrate_telemetry_facet_for_mind(correlation_id: str) -> dict[str, Any] | None:
    """GET latest row from telemetry service.

    Returns None when the row is absent (404) or when the fetch is misconfigured, times out,
    or otherwise fails — telemetry is optional and must not break the verb path.
    """
    s = get_settings()
    base = (s.orion_substrate_telemetry_base_url or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/v1/substrate/tier-outcomes/latest"
    params = {"correlation_id": correlation_id}
    headers: dict[str, str] = {}
    tok = (s.orion_substrate_telemetry_read_token or "").strip()
    if tok:
        headers["X-Telemetry-Token"] = tok
    timeout_sec = max(0.1, min(10.0, float(s.orion_substrate_telemetry_timeout_sec)))
    timeout = httpx.Timeout(timeout_sec)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 400:
                logger.warning(
                    "substrate_telemetry_fetch_http_status corr=%s status=%s",
                    correlation_id,
                    resp.status_code,
                )
                return None
            row = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("substrate_telemetry_fetch_http_err corr=%s err=%s", correlation_id, exc)
        return None
    except ValueError as exc:
        logger.warning("substrate_telemetry_fetch_bad_json corr=%s err=%s", correlation_id, exc)
        return None

    if not isinstance(row, dict):
        logger.warning(
            "substrate_telemetry_fetch_non_object corr=%s type=%s",
            correlation_id,
            type(row).__name__,
        )
        return None

    return {
        "status": "present",
        "generated_at": row.get("generated_at"),
        "cold_anchors": row.get("cold_anchors"),
        "tier_outcomes": row.get("tier_outcomes"),
        "degraded_producers": row.get("degraded_producers"),
        "source_service": row.get("source_service"),
        "source_node": row.get("source_node"),
        "received_at_utc": row.get("received_at_utc"),
        "row_id": row.get("id"),
    }


def _inline_cognitive_projection_facet(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return caller-supplied cognitive projection for Mind shadow mode."""
    for key in ("cognitive_projection_facet", "cognitive_projection"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return None


def _plan_projection_context(client_request: CortexClientRequest, plan_request: PlanExecutionRequest, correlation_id: str) -> dict[str, Any]:
    ctx = dict(plan_request.context or {}) if isinstance(plan_request.context, dict) else {}
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    ctx["metadata"] = dict(metadata)
    ctx["correlation_id"] = correlation_id
    ctx["requested_verb"] = client_request.verb or plan_request.plan.verb_name
    ctx["verb"] = client_request.verb or plan_request.plan.verb_name
    ctx["mode"] = client_request.mode
    ctx["messages"] = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in (client_request.context.messages or [])]
    ctx["raw_user_text"] = client_request.context.raw_user_text or None
    ctx["user_message"] = client_request.context.user_message or None
    ctx["session_id"] = client_request.context.session_id
    ctx["trace_id"] = client_request.context.trace_id
    ctx["user_id"] = client_request.context.user_id
    return ctx


def _build_cognitive_projection_facet(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
) -> dict[str, Any] | None:
    meta = client_request.context.metadata if isinstance(client_request.context.metadata, dict) else {}
    inline = _inline_cognitive_projection_facet(meta)
    if inline is not None:
        return inline
    try:
        projection = build_cognitive_projection_for_context(
            _plan_projection_context(client_request, plan_request, correlation_id),
            publish_tier_outcomes=False,
        )
    except Exception as exc:
        logger.warning("mind_cognitive_projection_build_failed corr=%s err=%s", correlation_id, exc)
        return None
    if projection is None:
        return None
    return projection.model_dump(mode="json")


def build_mind_run_request(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
    *,
    substrate_telemetry_facet: dict[str, Any] | None = None,
    cognitive_projection_facet: dict[str, Any] | None = None,
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
    facets: dict[str, Any] = {}
    if substrate_telemetry_facet is not None:
        facets["substrate_telemetry"] = substrate_telemetry_facet
    cognitive_projection = cognitive_projection_facet or _build_cognitive_projection_facet(
        client_request,
        plan_request,
        correlation_id,
    )
    if cognitive_projection is not None:
        facets["cognitive_projection"] = cognitive_projection
    if facets:
        snapshot["facets"] = facets
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
    meta["mind_quality"] = _mind_result_quality(result)
    skip_llm_stance = False
    if result.ok:
        meta["mind_run_ok"] = True
        if _mind_result_quality(result) == "meaningful_synthesis" and not _mind_result_is_deterministic_contract_only(result):
            sp = result.brief.stance_payload if isinstance(result.brief.stance_payload, dict) else {}
            try:
                ChatStanceBrief.model_validate(sp)
                skip_llm_stance = True
            except Exception:
                meta["mind_stance_payload_invalid"] = True
                skip_llm_stance = False
        else:
            meta["mind_contract_only"] = True
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
