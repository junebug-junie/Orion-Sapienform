"""Orion → orion-mind HTTP integration (Orch is the only canonical caller for binding runs)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.cognition.projection import project_unified_beliefs_for_mind
from orion.cognition.projection_builder import (
    build_cognitive_projection_for_mind_with_diagnostics,
    build_projection_unification_registry,
    summarize_projection_build,
)
from orion.cognition.projection_context import enrich_projection_context, summarize_projection_inputs
from orion.cognition.recall_prefetch import (
    log_mind_projection_prebuild_ctx_summary,
    prefetch_recall_bundle_for_projection,
)
from orion.cognition.recall_query import recall_cfg_from_recall_directive
from orion.substrate.relational import CognitiveUnificationLayer
from orion.substrate.store import InMemorySubstrateGraphStore
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

MIND_PROJECTION_RESOLUTION_KEYS: tuple[str, ...] = (
    "inline_projection_missing",
    "inline_projection_empty",
    "inline_projection_nonempty",
    "warm_projection_missing",
    "warm_projection_empty",
    "cold_rebuild_attempted",
    "cold_rebuild_succeeded",
    "cold_rebuild_failed",
)


def _mind_enabled_exact(metadata: dict[str, Any] | None) -> bool:
    return metadata is not None and metadata.get("mind_enabled") is True


def _mind_result_quality(result: MindRunResultV1) -> str:
    brief_quality = getattr(result.brief, "mind_quality", None)
    if isinstance(brief_quality, str) and brief_quality and brief_quality != "empty":
        return brief_quality
    result_quality = getattr(result, "mind_quality", None)
    if isinstance(result_quality, str) and result_quality and result_quality != "empty":
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


def _projection_item_count(projection: dict[str, Any] | None) -> int:
    if not isinstance(projection, dict):
        return 0
    try:
        count = int(projection.get("item_count") or 0)
    except Exception:
        count = 0
    if count > 0:
        return count
    anchors = projection.get("anchors") if isinstance(projection.get("anchors"), dict) else {}
    derived = 0
    for anchor_payload in anchors.values():
        if isinstance(anchor_payload, dict):
            derived += len(anchor_payload.get("items") or [])
    return derived


def _raw_inline_projection(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("cognitive_projection_facet", "cognitive_projection"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return None


def _fresh_projection_resolution() -> dict[str, Any]:
    resolution: dict[str, Any] = {key: False for key in MIND_PROJECTION_RESOLUTION_KEYS}
    resolution["resolved_item_count"] = 0
    resolution["resolution_path"] = None
    resolution["orch_invoked_before_exec"] = True
    return resolution


def _attach_resolution_diagnostics(
    projection_payload: dict[str, Any],
    *,
    resolution: dict[str, Any],
    build_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_build = dict(build_diagnostics or {})
    merged_build.update({key: resolution.get(key) for key in MIND_PROJECTION_RESOLUTION_KEYS})
    merged_build["resolution_path"] = resolution.get("resolution_path")
    merged_build["resolved_item_count"] = resolution.get("resolved_item_count")
    merged_build["orch_invoked_before_exec"] = resolution.get("orch_invoked_before_exec")
    projection_payload["projection_build_diagnostics"] = merged_build
    projection_payload["mind_projection_resolution"] = dict(resolution)
    return projection_payload


def _plan_projection_context(client_request: CortexClientRequest, plan_request: PlanExecutionRequest, correlation_id: str) -> dict[str, Any]:
    ctx = dict(plan_request.context or {}) if isinstance(plan_request.context, dict) else {}
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    ctx["metadata"] = dict(metadata)
    lane = metadata.get("execution_lane")
    if isinstance(lane, str) and lane.strip():
        ctx["execution_lane"] = lane.strip()
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
    ctx["orch_invoked_before_exec"] = True
    ctx["recall_enabled"] = bool(client_request.recall.enabled)
    plan_metadata = plan_request.plan.metadata if isinstance(plan_request.plan.metadata, dict) else {}
    if plan_metadata.get("personality_file") and not ctx.get("personality_file"):
        ctx["personality_file"] = plan_metadata.get("personality_file")
    enrich_projection_context(ctx, plan_metadata=plan_metadata)
    return ctx


async def prepare_plan_context_for_mind_projection(
    bus: Any,
    *,
    source: ServiceRef,
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
) -> None:
    """Enrich plan ctx with Exec-parity producer inputs before Mind preflight projection."""
    ctx = _plan_projection_context(client_request, plan_request, correlation_id)
    settings = get_settings()
    recall_prefetch_diag: dict[str, Any] | None = None
    if settings.mind_recall_prefetch_enabled and client_request.recall.enabled:
        recall_merge, recall_prefetch_diag = await prefetch_recall_bundle_for_projection(
            bus,
            source=source,
            ctx=ctx,
            correlation_id=correlation_id,
            recall_enabled=True,
            recall_profile=client_request.recall.profile,
            recall_channel=settings.channel_recall_intake,
            timeout_sec=float(settings.mind_recall_prefetch_timeout_sec),
            recall_cfg=recall_cfg_from_recall_directive(client_request.recall),
            recall_reply_prefix=settings.channel_recall_reply_prefix,
        )
        if isinstance(recall_merge, dict):
            ctx.update(recall_merge)
    elif not settings.mind_recall_prefetch_enabled:
        recall_prefetch_diag = {
            "correlation_id": correlation_id,
            "enabled": False,
            "reason": "MIND_RECALL_PREFETCH_ENABLED=false",
            "ok": False,
        }
    elif not client_request.recall.enabled:
        recall_prefetch_diag = {
            "correlation_id": correlation_id,
            "enabled": False,
            "reason": "client_recall_disabled",
            "ok": False,
        }
    prebuild_summary = log_mind_projection_prebuild_ctx_summary(
        correlation_id=correlation_id,
        ctx=ctx,
        recall_prefetch=recall_prefetch_diag,
    )
    plan_ctx = plan_request.context if isinstance(plan_request.context, dict) else {}
    plan_ctx.update(ctx)
    plan_request.context = plan_ctx
    meta = plan_ctx.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["orch_preflight_input_summary"] = summarize_projection_inputs(ctx, phase="orch_mind_preflight")
        if recall_prefetch_diag is not None:
            meta["recall_prefetch"] = recall_prefetch_diag
        meta["mind_projection_prebuild_ctx_summary"] = prebuild_summary


def _build_cold_cognitive_projection_facet(ctx: dict[str, Any], correlation_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Force cold producer fan-out on a fresh in-memory durable store for this turn."""
    build_path = "orion.cortex.orch.mind_runtime._build_cold_cognitive_projection_facet"
    try:
        registry = build_projection_unification_registry()
        layer = CognitiveUnificationLayer(registry=registry, store=InMemorySubstrateGraphStore())
        beliefs = layer.beliefs_for_stance(ctx=ctx, timeout_sec=5.0)
        projection = project_unified_beliefs_for_mind(beliefs)
        diagnostics = summarize_projection_build(ctx, beliefs=beliefs, projection=projection, build_path=build_path)
    except Exception as exc:
        logger.warning("mind_cognitive_projection_cold_build_failed corr=%s err=%s", correlation_id, exc)
        diagnostics = summarize_projection_build(ctx, beliefs=None, projection=None, build_path=build_path)
        diagnostics["producer_errors"] = list(diagnostics.get("producer_errors") or []) + [f"cold_build_failed:{exc}"]
        return None, diagnostics
    if projection is None:
        return None, diagnostics
    payload = projection.model_dump(mode="json")
    logger.info(
        "mind_cognitive_projection_cold_build corr=%s projection_id=%s item_count=%s cold_anchors=%s degraded=%s",
        correlation_id,
        payload.get("projection_id"),
        payload.get("item_count"),
        payload.get("cold_anchors"),
        payload.get("degraded_producers"),
    )
    return payload, diagnostics


def resolve_cognitive_projection_for_mind(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve a non-empty cognitive projection for Mind preflight using the shared spine.

    Order:
    1. Non-empty inline metadata projection
    2. Warm/durable projection via ``build_cognitive_projection_for_mind_with_diagnostics``
    3. Isolated cold rebuild before Mind is invoked

    Returns ``(projection_facet_or_none, resolution_diagnostics)``. When starved, the
    projection facet is ``None`` and resolution diagnostics carry the empty shell.
    """
    ctx = _plan_projection_context(client_request, plan_request, correlation_id)
    input_summary = summarize_projection_inputs(ctx, phase="orch_mind_preflight")
    resolution = _fresh_projection_resolution()
    resolution["orch_preflight_input_summary"] = input_summary
    meta = client_request.context.metadata if isinstance(client_request.context.metadata, dict) else {}
    build_diagnostics: dict[str, Any] = {}

    raw_inline = _raw_inline_projection(meta)
    if raw_inline is None:
        resolution["inline_projection_missing"] = True
    elif _projection_item_count(raw_inline) <= 0:
        resolution["inline_projection_empty"] = True
    else:
        resolution["inline_projection_nonempty"] = True
        resolution["resolution_path"] = "inline_metadata"
        resolution["resolved_item_count"] = _projection_item_count(raw_inline)
        inline_payload = dict(raw_inline)
        inline_build = (
            inline_payload.get("projection_build_diagnostics")
            if isinstance(inline_payload.get("projection_build_diagnostics"), dict)
            else summarize_projection_build(
                ctx,
                beliefs=None,
                projection=None,
                build_path="orion.cortex.orch.mind_runtime.resolve.inline_metadata",
            )
        )
        inline_build["projection_sources_returned"] = ["inline_metadata"]
        inline_build["item_count"] = _projection_item_count(inline_payload)
        inline_payload = _attach_resolution_diagnostics(
            inline_payload,
            resolution=resolution,
            build_diagnostics=inline_build,
        )
        return inline_payload, resolution

    warm_payload: dict[str, Any] | None = None
    try:
        projection, build_diagnostics = build_cognitive_projection_for_mind_with_diagnostics(
            ctx,
            publish_tier_outcomes=True,
            build_path="orion.cortex.orch.mind_runtime.resolve.warm_shared_spine",
        )
        if projection is None:
            resolution["warm_projection_missing"] = True
        else:
            warm_payload = projection.model_dump(mode="json")
            if _projection_item_count(warm_payload) <= 0:
                resolution["warm_projection_empty"] = True
            else:
                resolution["resolution_path"] = "warm_shared_spine"
                resolution["resolved_item_count"] = _projection_item_count(warm_payload)
                return _attach_resolution_diagnostics(warm_payload, resolution=resolution, build_diagnostics=build_diagnostics), resolution
    except Exception as exc:
        logger.warning("mind_cognitive_projection_warm_build_failed corr=%s err=%s", correlation_id, exc)
        resolution["warm_projection_missing"] = True
        build_diagnostics = summarize_projection_build(
            ctx,
            beliefs=None,
            projection=None,
            build_path="orion.cortex.orch.mind_runtime.resolve.warm_shared_spine",
        )
        build_diagnostics["producer_errors"] = list(build_diagnostics.get("producer_errors") or []) + [f"warm_build_failed:{exc}"]

    resolution["cold_rebuild_attempted"] = True
    cold_payload, cold_diag = _build_cold_cognitive_projection_facet(ctx, correlation_id)
    build_diagnostics = cold_diag
    if _projection_item_count(cold_payload) > 0:
        resolution["cold_rebuild_succeeded"] = True
        resolution["resolution_path"] = "cold_isolated_store"
        resolution["resolved_item_count"] = _projection_item_count(cold_payload)
        assert cold_payload is not None
        return _attach_resolution_diagnostics(cold_payload, resolution=resolution, build_diagnostics=build_diagnostics), resolution

    resolution["cold_rebuild_failed"] = True
    resolution["resolution_path"] = "starved_before_exec"
    resolution["orch_preflight_producer_outcomes"] = build_diagnostics
    shell_source = cold_payload if isinstance(cold_payload, dict) else warm_payload
    empty_shell: dict[str, Any] = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": (shell_source or {}).get("projection_id") or "cog-proj-mind-starved",
        "generated_at": (shell_source or {}).get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "source": "cognitive_unification_layer",
        "anchors": (shell_source or {}).get("anchors") if isinstance((shell_source or {}).get("anchors"), dict) else {},
        "item_count": 0,
        "notes": list((shell_source or {}).get("notes") or []) + ["mind_projection_starved_at_orch_preflight"],
    }
    resolution["empty_projection_shell"] = _attach_resolution_diagnostics(
        empty_shell,
        resolution=resolution,
        build_diagnostics=build_diagnostics,
    )
    resolution["projection_build_diagnostics"] = empty_shell["projection_build_diagnostics"]
    logger.warning(
        "mind_cognitive_projection_starved corr=%s resolution=%s",
        correlation_id,
        {key: resolution.get(key) for key in MIND_PROJECTION_RESOLUTION_KEYS},
    )
    return None, resolution


def _build_cognitive_projection_facet(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
) -> dict[str, Any] | None:
    projection, _resolution = resolve_cognitive_projection_for_mind(client_request, plan_request, correlation_id)
    return projection


def _share_cognitive_projection_with_plan(
    plan_request: PlanExecutionRequest,
    projection: dict[str, Any],
    *,
    resolution: dict[str, Any] | None = None,
) -> None:
    ctx = plan_request.context if isinstance(plan_request.context, dict) else {}
    metadata = ctx.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    metadata["cognitive_projection_facet"] = projection
    metadata["cognitive_projection"] = projection
    metadata["cognitive_projection_source"] = "orion_cortex_orch_mind_runtime"
    if isinstance(resolution, dict):
        metadata["mind_projection_resolution"] = dict(resolution)


def _record_mind_projection_resolution(plan_request: PlanExecutionRequest, resolution: dict[str, Any]) -> None:
    ctx = plan_request.context if isinstance(plan_request.context, dict) else {}
    metadata = ctx.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    metadata["mind_projection_resolution"] = dict(resolution)
    metadata["mind_projection_starved"] = _projection_item_count(
        resolution.get("empty_projection_shell") if isinstance(resolution.get("empty_projection_shell"), dict) else None
    ) <= 0 and bool(
        resolution.get("cold_rebuild_failed")
        or resolution.get("warm_projection_empty")
        or resolution.get("inline_projection_empty")
    )
    metadata["mind_orch_before_exec"] = True


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

    cognitive_projection: dict[str, Any] | None = None
    resolution = _fresh_projection_resolution()

    if isinstance(cognitive_projection_facet, dict) and _projection_item_count(cognitive_projection_facet) > 0:
        resolution["inline_projection_nonempty"] = True
        resolution["resolution_path"] = "explicit_facet_argument"
        resolution["resolved_item_count"] = _projection_item_count(cognitive_projection_facet)
        cognitive_projection = _attach_resolution_diagnostics(dict(cognitive_projection_facet), resolution=resolution)
    else:
        cognitive_projection, resolution = resolve_cognitive_projection_for_mind(
            client_request,
            plan_request,
            correlation_id,
        )

    _record_mind_projection_resolution(plan_request, resolution)

    parity = {
        "orch_preflight_input_summary": resolution.get("orch_preflight_input_summary")
        or summarize_projection_inputs(
            _plan_projection_context(client_request, plan_request, correlation_id),
            phase="orch_mind_preflight",
        ),
        "orch_preflight_producer_outcomes": resolution.get("orch_preflight_producer_outcomes")
        or (cognitive_projection or {}).get("projection_build_diagnostics")
        or resolution.get("projection_build_diagnostics"),
    }
    plan_meta = plan_request.context.setdefault("metadata", {}) if isinstance(plan_request.context, dict) else {}
    if isinstance(plan_meta, dict):
        plan_meta["projection_parity_diagnostics"] = parity

    if cognitive_projection is not None and _projection_item_count(cognitive_projection) > 0:
        facets["cognitive_projection"] = cognitive_projection
        facets["projection_parity_diagnostics"] = parity
        _share_cognitive_projection_with_plan(plan_request, cognitive_projection, resolution=resolution)
    else:
        facets["mind_projection_resolution"] = dict(resolution)
        facets["projection_parity_diagnostics"] = parity
        empty_shell = resolution.get("empty_projection_shell")
        if isinstance(empty_shell, dict):
            facets["cognitive_projection_degraded"] = empty_shell

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
    shadow = result.brief.shadow_synthesis
    if shadow is not None:
        meta["mind_shadow_synthesis"] = shadow.model_dump(mode="json")
        meta["mind_shadow_synthesis_present"] = bool(shadow.present)
        meta["mind_authorized_for_stance_skip"] = bool(shadow.authorized_for_stance_skip)
    else:
        meta["mind_shadow_synthesis_present"] = False
        meta["mind_authorized_for_stance_skip"] = False
    skip_llm_stance = False
    if result.ok:
        meta["mind_run_ok"] = True
        if (
            _mind_result_quality(result) == "meaningful_synthesis"
            and bool(getattr(result.brief, "mind_authorized_for_stance_skip", False))
            and not _mind_result_is_deterministic_contract_only(result)
        ):
            sp = result.brief.stance_payload if isinstance(result.brief.stance_payload, dict) else {}
            try:
                ChatStanceBrief.model_validate(sp)
                skip_llm_stance = True
            except Exception:
                meta["mind_stance_payload_invalid"] = True
                skip_llm_stance = False
        else:
            meta["mind_contract_only"] = _mind_result_quality(result) in {"fallback_contract_only", "shadow_synthesis"}
        meta["mind_skip_stance_synthesis"] = skip_llm_stance
    else:
        meta["mind_skip_stance_synthesis"] = False
        meta["mind_run_ok"] = False
        meta["mind_error_code"] = result.error_code


def mind_http_base_url() -> str:
    return (get_settings().orion_mind_base_url or "").rstrip("/")


def log_mind_http_failure(
    *,
    correlation_id: str,
    session_id: str | None,
    trigger: str | None,
    exc: BaseException,
    base_url: str | None = None,
) -> None:
    configured = (base_url if base_url is not None else mind_http_base_url()) or "(unset)"
    status_code: int | None = None
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
    logger.warning(
        "mind_http_failed correlation_id=%s session_id=%s trigger=%s mind_base_url=%s exc_type=%s status_code=%s err=%s",
        correlation_id,
        session_id,
        trigger,
        configured,
        type(exc).__name__,
        status_code,
        exc,
    )


async def call_orion_mind_http(req: MindRunRequestV1) -> MindRunResultV1:
    base = mind_http_base_url()
    if not base:
        raise RuntimeError("orion_mind_unconfigured")
    url = f"{base}/v1/mind/run"
    s = get_settings()
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
    logger.info(
        "mind_run_artifact_publish mind_run_id=%s correlation_id=%s session_id=%s trigger=%s ok=%s error_code=%s router_profile_id=%s",
        artifact.mind_run_id,
        correlation_id,
        artifact.session_id,
        artifact.trigger,
        artifact.ok,
        artifact.error_code,
        artifact.router_profile_id,
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
