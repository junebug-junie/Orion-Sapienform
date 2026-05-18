"""Deterministic Mind cognition engine (service-local only)."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

import yaml

from orion.mind.validation import hash_snapshot_inputs, validate_merged_stance_brief_optional
from orion.mind.v1 import (
    MindControlDecisionV1,
    MindHandoffBriefV1,
    MindRunRequestV1,
    MindRunResultV1,
    MindShadowSynthesisV1,
    MindStancePatchV1,
    MindStanceTrajectoryV1,
    MindProvenanceV1,
)

logger = logging.getLogger("orion-mind.engine")

_FACET_ORDER = (
    "cognitive_projection",
    "autonomy",
    "substrate",
    "substrate_telemetry",
    "collapse",
    "concept_induction",
    "recall_digest",
    "social",
    "metacog",
    "tool_outcomes",
    "misc",
)

_TECHNICAL_PATTERNS = (
    r"\bapi\b",
    r"\bbug\b",
    r"\bcode\b",
    r"\bcursor\b",
    r"\bdebug\b",
    r"\bdeploy\b",
    r"\bdocker\b",
    r"\berror\b",
    r"\bgraphdb\b",
    r"\blog(s)?\b",
    r"\bpython\b",
    r"\bschema\b",
    r"\bservice\b",
    r"\bsql\b",
    r"\bstack\b",
    r"\btimeout\b",
)
_PLANNING_PATTERNS = (
    r"\bbuild\b",
    r"\bdesign\b",
    r"\bimplement\b",
    r"\bnext step\b",
    r"\bplan\b",
    r"\bprompt\b",
    r"\broadmap\b",
)
_REFLECTIVE_PATTERNS = (
    r"\bautonomy\b",
    r"\bcognition\b",
    r"\bdream\b",
    r"\bfeel\b",
    r"\bidentity\b",
    r"\bmeaning\b",
    r"\bmind\b",
    r"\borion\b",
    r"\brelationship\b",
)
_RELATIONAL_PATTERNS = (
    r"\bamanda\b",
    r"\bbike(s)?\b",
    r"\bfamily\b",
    r"\bfriend\b",
    r"\bgoing to\b",
    r"\bhusband\b",
    r"\bkid(s)?\b",
    r"\blol\b",
    r"\bshow\b",
    r"\bthanks\b",
    r"\bwatch\b",
    r"\bwife\b",
    r"[:;]-?\)",
)


def _json_blob_size(obj: Any) -> int:
    return len(json.dumps(obj, default=str).encode("utf-8"))


def build_bounded_snapshot_inputs(raw: dict[str, Any], max_bytes: int) -> tuple[dict[str, Any], bool]:
    if _json_blob_size(raw) <= max_bytes:
        return raw, False
    facets: dict[str, Any] = {}
    if isinstance(raw.get("facets"), dict):
        facets = dict(raw["facets"])
    else:
        facets = {"misc": raw}
    trimmed: dict[str, Any] = {"facets": {}}
    order = list(_FACET_ORDER) + sorted(k for k in facets if k not in _FACET_ORDER)
    for key in order:
        if key not in facets:
            continue
        blob = facets[key]
        if isinstance(blob, dict):
            stub = {"truncated": True, "preview": str(blob)[:512]}
        else:
            stub = {"truncated": True, "preview": str(blob)[:512]}
        candidate = dict(trimmed)
        candidate["facets"][key] = stub
        if _json_blob_size(candidate) <= max_bytes:
            trimmed = candidate
    if _json_blob_size(trimmed) > max_bytes:
        return {}, True
    return trimmed, True


def _router_profile(router_profiles_dir: Path, profile_id: str) -> dict[str, Any]:
    path = router_profiles_dir / "router_profiles.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles") or {}
    return dict(profiles.get(profile_id) or profiles.get("default") or {})


def deterministic_merge_stance(patch_structured_layers: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in patch_structured_layers:
        if isinstance(layer, dict):
            merged.update(layer)
    return merged


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _deterministic_frame_from_user_text(user_text: str) -> tuple[str, str]:
    """Tiny safe fallback classifier for contract-only Mind runs.

    This is not meaningful synthesis. It exists only to avoid actively wrong
    boilerplate such as marking every casual turn as technical.
    """
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return "mixed", "direct_response"
    if _matches_any(lowered, _TECHNICAL_PATTERNS):
        return "technical", "technical_collaboration"
    if _matches_any(lowered, _PLANNING_PATTERNS):
        return "planning", "direct_response"
    if _matches_any(lowered, _REFLECTIVE_PATTERNS):
        return "reflective", "reflective_dialogue"
    if _matches_any(lowered, _RELATIONAL_PATTERNS):
        return "playful_relational", "playful_exchange"
    return "mixed", "direct_response"


def _default_stance_from_user_text(user_text: str) -> dict[str, Any]:
    ut = (user_text or "").strip()[:2000]
    conversation_frame, task_mode = _deterministic_frame_from_user_text(ut)
    return {
        "conversation_frame": conversation_frame,
        "task_mode": task_mode,
        "identity_salience": "medium",
        "user_intent": ut or "(empty turn)",
        "self_relevance": "contract_only: no meaningful Mind synthesis produced.",
        "juniper_relevance": "contract_only: preserve latest user turn without adding invented context.",
        "answer_strategy": "DirectAnswer",
        "stance_summary": f"Deterministic contract-only stance seed ({len(ut)} chars).",
    }


def _snapshot_facets(snapshot: dict[str, Any]) -> dict[str, Any]:
    facets = snapshot.get("facets") if isinstance(snapshot.get("facets"), dict) else {}
    return dict(facets)


def _mind_projection_resolution(snapshot: dict[str, Any]) -> dict[str, Any]:
    facets = _snapshot_facets(snapshot)
    for key in ("mind_projection_resolution",):
        value = facets.get(key)
        if isinstance(value, dict):
            return dict(value)
    projection = facets.get("cognitive_projection")
    if isinstance(projection, dict):
        nested = projection.get("mind_projection_resolution")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def _cognitive_projection(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    facets = _snapshot_facets(snapshot)
    projection = facets.get("cognitive_projection")
    if isinstance(projection, dict) and _projection_items(projection):
        return projection
    degraded = facets.get("cognitive_projection_degraded")
    if isinstance(degraded, dict):
        return degraded
    if isinstance(projection, dict):
        return projection
    return None


def _cognitive_projection_debug(snapshot: dict[str, Any]) -> dict[str, Any]:
    facets = _snapshot_facets(snapshot)
    projection = facets.get("cognitive_projection")
    if not isinstance(projection, dict):
        projection = facets.get("cognitive_projection_degraded")
    resolution = _mind_projection_resolution(snapshot)
    if not isinstance(projection, dict):
        if resolution:
            empty_shell = resolution.get("empty_projection_shell")
            if isinstance(empty_shell, dict):
                projection = empty_shell
        if not isinstance(projection, dict):
            return {"present": False, "resolution": resolution}
    anchors = projection.get("anchors") if isinstance(projection.get("anchors"), dict) else {}
    build_diag = projection.get("projection_build_diagnostics")
    if not isinstance(build_diag, dict):
        build_diag = resolution.get("projection_build_diagnostics") if isinstance(resolution.get("projection_build_diagnostics"), dict) else {}
    item_count = projection.get("item_count")
    try:
        counted = int(item_count or 0)
    except Exception:
        counted = 0
    if counted <= 0:
        counted = len(_projection_items(projection))
    return {
        "present": True,
        "schema_version": projection.get("schema_version"),
        "projection_id": projection.get("projection_id"),
        "generated_at": projection.get("generated_at"),
        "item_count": counted,
        "anchor_count": len(anchors),
        "cold_anchors": projection.get("cold_anchors") if isinstance(projection.get("cold_anchors"), list) else [],
        "degraded_producers": projection.get("degraded_producers") if isinstance(projection.get("degraded_producers"), list) else [],
        "notes": projection.get("notes") if isinstance(projection.get("notes"), list) else [],
        "build_diagnostics": build_diag if isinstance(build_diag, dict) else {},
        "resolution": resolution,
    }


def _projection_starvation_machine_keys(build_diag: dict[str, Any], resolution: dict[str, Any] | None = None) -> dict[str, Any]:
    """Surface Orch/build-time starvation diagnostics on the Mind machine contract."""
    if not build_diag and not resolution:
        return {}
    keys: dict[str, Any] = {
        "mind.projection_sources_requested": list(build_diag.get("projection_sources_requested") or []),
        "mind.projection_sources_returned": list(build_diag.get("projection_sources_returned") or []),
        "mind.projection_source_counts": dict(build_diag.get("source_counts") or {}),
        "mind.projection_dropped_counts_by_reason": dict(build_diag.get("dropped_counts_by_reason") or {}),
        "mind.projection_producer_errors": list(build_diag.get("producer_errors") or []),
        "mind.projection_short_circuit_policy_active": bool(build_diag.get("short_circuit_policy_active")),
        "mind.projection_build_path": build_diag.get("build_path"),
    }
    resolution = resolution if isinstance(resolution, dict) else {}
    for flag in (
        "inline_projection_missing",
        "inline_projection_empty",
        "inline_projection_nonempty",
        "warm_projection_missing",
        "warm_projection_empty",
        "cold_rebuild_attempted",
        "cold_rebuild_succeeded",
        "cold_rebuild_failed",
    ):
        if flag in resolution:
            keys[f"mind.projection_resolution.{flag}"] = bool(resolution.get(flag))
    if resolution.get("resolution_path"):
        keys["mind.projection_resolution.path"] = resolution.get("resolution_path")
    if resolution.get("orch_invoked_before_exec"):
        keys["mind.projection_resolution.orch_before_exec"] = True
    return keys


def _projection_starvation_summary(build_diag: dict[str, Any], item_count: int, resolution: dict[str, Any] | None = None) -> str:
    if item_count > 0:
        return ""
    resolution = resolution if isinstance(resolution, dict) else {}
    path = resolution.get("resolution_path") or build_diag.get("resolution_path")
    short_circuit = bool(build_diag.get("short_circuit_policy_active"))
    dropped = build_diag.get("dropped_counts_by_reason") if isinstance(build_diag.get("dropped_counts_by_reason"), dict) else {}
    producer_errors = list(build_diag.get("producer_errors") or [])
    reasons = ", ".join(f"{key}={value}" for key, value in sorted(dropped.items())[:4]) or "no_active_projection_items"
    resolution_bits = [
        bit
        for bit, active in (
            ("inline_empty", resolution.get("inline_projection_empty")),
            ("warm_empty", resolution.get("warm_projection_empty")),
            ("cold_failed", resolution.get("cold_rebuild_failed")),
        )
        if active
    ]
    resolution_note = f"; resolution={','.join(resolution_bits)}" if resolution_bits else ""
    if short_circuit:
        return f"Degraded Mind: projection short-circuited ({reasons}){resolution_note}."
    if producer_errors:
        return f"Degraded Mind: projection starved at Orch preflight ({reasons}); path={path}; producer_errors={producer_errors[:6]}{resolution_note}."
    return f"Degraded Mind: projection starved at Orch preflight ({reasons}); path={path}{resolution_note}."


def _projection_items(projection: dict[str, Any] | None, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(projection, dict):
        return []
    anchors = projection.get("anchors") if isinstance(projection.get("anchors"), dict) else {}
    items: list[dict[str, Any]] = []
    for anchor, anchor_payload in anchors.items():
        if not isinstance(anchor_payload, dict):
            continue
        for item in anchor_payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            enriched.setdefault("anchor", anchor)
            items.append(enriched)
    return sorted(
        items,
        key=lambda item: (float(item.get("salience") or 0.0), float(item.get("confidence") or 0.0), str(item.get("label") or "")),
        reverse=True,
    )[:limit]


def _label_for_projection_item(item: dict[str, Any]) -> str:
    for key in ("label", "summary", "node_id"):
        value = item.get(key)
        if value:
            return str(value).strip()[:160]
    return "projection item"


def _build_shadow_synthesis(snapshot: dict[str, Any], base_stance: dict[str, Any]) -> MindShadowSynthesisV1 | None:
    projection = _cognitive_projection(snapshot)
    items = _projection_items(projection)
    if not projection or not items:
        return None

    focus_items = items[:4]
    labels = [_label_for_projection_item(item) for item in focus_items]
    refs = [str(item.get("node_id") or item.get("item_id") or label) for item, label in zip(focus_items, labels)]
    relationship_labels = [
        _label_for_projection_item(item)
        for item in focus_items
        if str(item.get("anchor") or "").lower() == "relationship" or "relationship" in str(item.get("bucket") or "").lower()
    ]
    curiosity = []
    if labels:
        curiosity.append(f"Notice whether the turn connects to: {labels[0]}")
    if len(labels) > 1:
        curiosity.append(f"Consider a light follow-up around: {labels[1]}")

    stance_candidate = dict(base_stance)
    stance_candidate.update(
        {
            "stance_summary": "Shadow synthesis candidate from CognitiveProjectionV1; non-authoritative.",
            "response_priorities": [
                "use cognitive projection as context, not as truth",
                "preserve current user turn",
                "prefer concise, situated response",
            ],
            "active_relationship_facets": relationship_labels[:3],
            "reflective_themes": labels[:4],
        }
    )
    return MindShadowSynthesisV1(
        present=True,
        authorized_for_stance_skip=False,
        stance_candidate=stance_candidate,
        attention_focus=labels,
        curiosity_candidate=curiosity,
        relationship_frame=relationship_labels[0] if relationship_labels else None,
        projection_refs_used=refs,
        hazards=[
            "shadow synthesis is non-authoritative",
            "do not skip legacy chat stance from this candidate",
            "do not invent facts beyond projection/source text",
        ],
        confidence=min(0.75, 0.35 + 0.05 * len(items)),
        rationale="Deterministic shadow candidate extracted from top CognitiveProjectionV1 items for operator comparison.",
    )


def _elapsed_ms_wall_clock(t_run_start: float) -> float:
    return (time.perf_counter() - t_run_start) * 1000


def _error_result(
    *,
    mind_run_id,
    error_code: str,
    diagnostics: list[str],
    snapshot_hash: str,
    trajectory: MindStanceTrajectoryV1 | None = None,
    brief: MindHandoffBriefV1 | None = None,
    timing_ms_by_phase: dict[str, float] | None = None,
) -> MindRunResultV1:
    return MindRunResultV1(
        mind_run_id=mind_run_id,
        ok=False,
        error_code=error_code,
        diagnostics=diagnostics,
        snapshot_hash=snapshot_hash,
        trajectory=trajectory or MindStanceTrajectoryV1(patches=[], merged_stance_brief={}, merge_policy="deterministic_merge"),
        decision=MindControlDecisionV1(route_kind="no_chat", refusals=[{"code": error_code}]),
        brief=brief or MindHandoffBriefV1(mind_quality="error"),
        mind_quality="error",
        timing_ms_by_phase=timing_ms_by_phase or {},
    )


def run_mind_deterministic(
    req: MindRunRequestV1,
    *,
    router_profiles_dir: Path,
    snapshot_max_bytes: int,
) -> MindRunResultV1:
    t_run_start = time.perf_counter()
    mind_run_id = uuid4()
    phases: dict[str, float] = {}
    wall_budget_ms = float(req.policy.wall_time_ms_max)

    logger.info(
        "mind_run_start mind_run_id=%s correlation_id=%s trigger=%s",
        mind_run_id,
        req.correlation_id,
        req.trigger,
    )

    bounded, _truncated = build_bounded_snapshot_inputs(dict(req.snapshot_inputs or {}), snapshot_max_bytes)
    cognitive_projection_debug = _cognitive_projection_debug(bounded)
    snap_hash = hash_snapshot_inputs(bounded)
    phases["snapshot_ms"] = _elapsed_ms_wall_clock(t_run_start)

    if _elapsed_ms_wall_clock(t_run_start) > wall_budget_ms:
        logger.info("mind_run_end mind_run_id=%s ok=False error=loop_budget_exceeded phase=snapshot", mind_run_id)
        return _error_result(
            mind_run_id=mind_run_id,
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_after_snapshot"],
            snapshot_hash=snap_hash,
            timing_ms_by_phase=phases,
        )

    user_text = ""
    if isinstance(bounded.get("user_text"), str):
        user_text = bounded["user_text"]
    elif isinstance(bounded.get("messages_tail"), list) and bounded["messages_tail"]:
        last = bounded["messages_tail"][-1]
        if isinstance(last, dict):
            user_text = str(last.get("content") or last.get("text") or "")

    n_loops = min(max(req.policy.n_loops_max, 1), 32)

    patches: list[MindStancePatchV1] = []
    layers: list[dict[str, Any]] = []
    t_loop = time.perf_counter()
    base = _default_stance_from_user_text(user_text)
    shadow_synthesis = _build_shadow_synthesis(bounded, base)
    for i in range(n_loops):
        if _elapsed_ms_wall_clock(t_run_start) > wall_budget_ms:
            logger.info(
                "mind_run_end mind_run_id=%s ok=False error=loop_budget_exceeded phase=loop loop_index=%s",
                mind_run_id,
                i,
            )
            return _error_result(
                mind_run_id=mind_run_id,
                error_code="loop_budget_exceeded",
                diagnostics=[f"wall_time_exceeded_during_loop_at_index_{i}"],
                snapshot_hash=snap_hash,
                trajectory=MindStanceTrajectoryV1(patches=patches, merged_stance_brief={}, merge_policy="deterministic_merge"),
                timing_ms_by_phase={**phases, "loops_partial_ms": (time.perf_counter() - t_loop) * 1000},
            )
        # llm_flags[i] True means LLM would run — v1 is deterministic only; no extra merge keys (avoid polluting ChatStanceBrief).
        delta: dict[str, Any] = {
            "stance_summary": f"{base['stance_summary']} [loop {i + 1}/{n_loops}]",
        }
        if i == 0:
            delta.update(base)
            if cognitive_projection_debug.get("present"):
                delta["cognitive_projection_seen"] = True
                delta["cognitive_projection_id"] = cognitive_projection_debug.get("projection_id")
                delta["cognitive_projection_item_count"] = cognitive_projection_debug.get("item_count")
            if shadow_synthesis is not None:
                delta["mind_shadow_synthesis_present"] = True
                delta["mind_shadow_projection_refs_used"] = list(shadow_synthesis.projection_refs_used)
        else:
            delta["user_intent"] = base["user_intent"]
        layers.append(delta)
        patches.append(
            MindStancePatchV1(
                loop_index=i,
                structured=delta,
                provenance=MindProvenanceV1(
                    model_id="deterministic",
                    input_hash=hash_snapshot_inputs({"loop": i, "corr": req.correlation_id}),
                ),
            )
        )
    phases["loops_ms"] = (time.perf_counter() - t_loop) * 1000

    if _elapsed_ms_wall_clock(t_run_start) > wall_budget_ms:
        logger.info("mind_run_end mind_run_id=%s ok=False error=loop_budget_exceeded phase=post_loops", mind_run_id)
        merged_partial = deterministic_merge_stance(layers)
        return _error_result(
            mind_run_id=mind_run_id,
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_after_loops_before_merge"],
            snapshot_hash=snap_hash,
            trajectory=MindStanceTrajectoryV1(
                patches=patches,
                merged_stance_brief=merged_partial,
                merge_policy="deterministic_merge",
            ),
            timing_ms_by_phase=phases,
        )

    merged = deterministic_merge_stance(layers)
    t_merge = time.perf_counter()
    valid, err = validate_merged_stance_brief_optional(merged)
    phases["merge_ms"] = (time.perf_counter() - t_merge) * 1000

    if valid is None:
        logger.info("mind_run_end mind_run_id=%s ok=False error=stance_merge_invalid", mind_run_id)
        return _error_result(
            mind_run_id=mind_run_id,
            error_code="stance_merge_invalid",
            diagnostics=[err or "stance_merge_invalid", f"merged_keys={list(merged.keys())}"],
            snapshot_hash=snap_hash,
            trajectory=MindStanceTrajectoryV1(patches=patches, merged_stance_brief=merged, merge_policy="deterministic_merge"),
            brief=MindHandoffBriefV1(mind_quality="error"),
            timing_ms_by_phase=phases,
        )

    prof = _router_profile(router_profiles_dir, req.policy.router_profile_id)
    mode_suggestion = str(prof.get("mode_suggestion") or "brain")
    if mode_suggestion not in ("brain", "agent", "workflow_only", "no_chat"):
        mode_suggestion = "brain"
    mode_binding = str(prof.get("mode_binding") or "advisory")
    if mode_binding not in ("advisory", "mandatory"):
        mode_binding = "advisory"
    decision = MindControlDecisionV1(
        route_kind=str(prof.get("route_kind") or "brain"),
        allowed_verbs=list(prof.get("allowed_verbs") or ["chat_general"]),
        recall_profile_override=prof.get("recall_profile_override"),
        mode_suggestion=mode_suggestion,  # type: ignore[arg-type]
        mode_binding=mode_binding,  # type: ignore[arg-type]
        budgets={"wall_ms_remaining": float(req.policy.wall_time_ms_max), "truncated": _truncated},
    )
    quality = "shadow_synthesis" if shadow_synthesis is not None else "fallback_contract_only"
    build_diag = cognitive_projection_debug.get("build_diagnostics") if isinstance(cognitive_projection_debug.get("build_diagnostics"), dict) else {}
    resolution = cognitive_projection_debug.get("resolution") if isinstance(cognitive_projection_debug.get("resolution"), dict) else {}
    item_count = int(cognitive_projection_debug.get("item_count") or 0)
    projection_starved = bool(cognitive_projection_debug.get("present")) and item_count <= 0
    machine = {
        "mind.route_kind": decision.route_kind,
        "mind.allowed_verbs": decision.allowed_verbs,
        "mind.mode_suggestion": decision.mode_suggestion,
        "mind.mode_binding": decision.mode_binding,
        "mind.quality": quality,
        "mind.cognitive_projection_seen": bool(cognitive_projection_debug.get("present")),
        "mind.shadow_synthesis_present": shadow_synthesis is not None,
        "mind.authorized_for_stance_skip": False,
        "mind.projection_starved": projection_starved,
        "mind.contract_only_degraded": quality == "fallback_contract_only",
    }
    if cognitive_projection_debug.get("present"):
        machine["mind.cognitive_projection_id"] = cognitive_projection_debug.get("projection_id")
        machine["mind.cognitive_projection_item_count"] = item_count
    if projection_starved:
        machine.update(_projection_starvation_machine_keys(build_diag, resolution))
    if shadow_synthesis is not None:
        machine["mind.shadow_projection_refs_used"] = list(shadow_synthesis.projection_refs_used)
        machine["mind.shadow_confidence"] = shadow_synthesis.confidence
    starvation_summary = _projection_starvation_summary(build_diag, item_count, resolution)
    brief = MindHandoffBriefV1(
        summary_one_paragraph=(
            "Shadow synthesis candidate produced from CognitiveProjectionV1; not authorized for stance skip."
            if shadow_synthesis is not None
            else (
                starvation_summary
                if starvation_summary
                else "Fallback contract only — no meaningful Mind synthesis produced."
            )
        ),
        machine_contract=machine,
        mandatory_keys=["mind.route_kind", "mind.allowed_verbs"],
        advisory_keys=["mind.mode_suggestion", "mind.quality", "mind.cognitive_projection_seen", "mind.shadow_synthesis_present"],
        stance_payload=valid.model_dump(mode="json"),
        mind_quality=quality,  # type: ignore[arg-type]
        shadow_synthesis=shadow_synthesis,
        mind_authorized_for_stance_skip=False,
    )

    phases["total_ms"] = _elapsed_ms_wall_clock(t_run_start)
    logger.info(
        "mind_run_end mind_run_id=%s ok=True snapshot_hash=%s quality=%s cognitive_projection_seen=%s shadow_synthesis_present=%s",
        mind_run_id,
        snap_hash,
        quality,
        bool(cognitive_projection_debug.get("present")),
        shadow_synthesis is not None,
    )

    return MindRunResultV1(
        mind_run_id=mind_run_id,
        ok=True,
        snapshot_hash=snap_hash,
        trajectory=MindStanceTrajectoryV1(
            patches=patches,
            merged_stance_brief=valid.model_dump(mode="json"),
            merge_policy="deterministic_merge",
        ),
        decision=decision,
        brief=brief,
        mind_quality=quality,  # type: ignore[arg-type]
        timing_ms_by_phase=phases,
    )


def _validate_phase_route(route: str, *, phase: str) -> str | None:
    cleaned = (route or "").strip()
    if not cleaned:
        return f"invalid_route:{phase}:empty"
    return None


def _llm_machine_contract(
    *,
    decision: MindControlDecisionV1,
    synthesis,
    frontier,
    handoff,
    cognitive_projection_debug: dict[str, Any],
    shadow_synthesis: MindShadowSynthesisV1 | None,
    llm_enabled: bool,
    llm_error: str | None,
    phase_telemetry: list | None = None,
    fallback_reason: list[str] | None = None,
    semantic_route: str | None = None,
    appraisal_route: str | None = None,
    stance_route: str | None = None,
) -> dict[str, Any]:
    claim_labels = [c.label for c in (synthesis.claims if synthesis else [])[:8]]
    top_labels = [m.label for m in (frontier.selected if frontier else [])[:6]]
    machine: dict[str, Any] = {
        "mind.route_kind": decision.route_kind,
        "mind.allowed_verbs": decision.allowed_verbs,
        "mind.mode_suggestion": decision.mode_suggestion,
        "mind.mode_binding": decision.mode_binding,
        "mind.quality": handoff.mind_quality if handoff else "empty",
        "mind.cognitive_projection_seen": bool(cognitive_projection_debug.get("present")),
        "mind.shadow_synthesis_present": shadow_synthesis is not None,
        "mind.authorized_for_stance_skip": bool(handoff.authorized_for_stance_use) if handoff else False,
        "mind.llm_synthesis_enabled": llm_enabled,
        "mind.semantic_synthesis_seen": synthesis is not None,
        "mind.semantic_claim_count": len(synthesis.claims) if synthesis else 0,
        "mind.semantic_claim_labels": claim_labels,
        "mind.active_frontier_seen": frontier is not None,
        "mind.active_frontier_selected_count": len(frontier.selected) if frontier else 0,
        "mind.active_frontier_top_labels": top_labels,
        "mind.stance_handoff_seen": handoff is not None,
        "mind.authorized_for_stance_use": bool(handoff.authorized_for_stance_use) if handoff else False,
        "mind.authorization_reasons": list(handoff.authorization_reasons) if handoff else [],
    }
    if llm_error:
        machine["mind.llm_synthesis_error"] = llm_error
    if semantic_route:
        machine["mind.semantic_route"] = semantic_route
    if appraisal_route:
        machine["mind.appraisal_route"] = appraisal_route
    if stance_route:
        machine["mind.stance_route"] = stance_route
    if phase_telemetry:
        from .phase_telemetry import phase_telemetry_machine_keys

        machine.update(phase_telemetry_machine_keys(phase_telemetry))
    if fallback_reason:
        machine["mind.fallback_reason"] = list(fallback_reason)[:8]
    return machine


MindLLMSynthesisOutcome = Union[MindRunResultV1, "MindLLMFailOpenRecord", None]


def run_mind_llm_synthesis(
    req: MindRunRequestV1,
    *,
    router_profiles_dir: Path,
    snapshot_max_bytes: int,
    mind_settings: Any,
) -> MindLLMSynthesisOutcome:
    from .llm_fail_open import MindLLMFailOpenRecord
    from .appraisal import run_active_frontier_judge
    from .budget import MindRunBudget
    from .evidence import build_evidence_pack
    from .guardrails import build_handoff
    from .llm_client import get_llm_client
    from .llm_context import MindLLMRequestContext
    from .phase_telemetry import MindPhaseTelemetry
    from .settings import settings as mind_settings_module
    from .stance_handoff import run_stance_handoff
    from .synthesis import run_semantic_synthesis

    s = mind_settings if mind_settings is not None else mind_settings_module
    if not bool(getattr(s, "MIND_LLM_SYNTHESIS_ENABLED", False)):
        return None

    t_run_start = time.perf_counter()
    mind_run_id = uuid4()
    phases: dict[str, float] = {}
    budget = MindRunBudget(
        float(req.policy.wall_time_ms_max),
        safety_ms=float(getattr(s, "MIND_LLM_PHASE_SAFETY_MS", 50.0)),
    )
    bounded, _truncated = build_bounded_snapshot_inputs(dict(req.snapshot_inputs or {}), snapshot_max_bytes)
    cognitive_projection_debug = _cognitive_projection_debug(bounded)
    snap_hash = hash_snapshot_inputs(bounded)

    logger.info(
        "mind_llm_run_start mind_run_id=%s correlation_id=%s session_id=%s trace_id=%s trigger=%s",
        mind_run_id,
        req.correlation_id,
        req.session_id,
        req.trace_id,
        req.trigger,
    )

    pack = build_evidence_pack(
        bounded,
        max_messages=int(getattr(s, "MIND_EVIDENCE_MAX_MESSAGES", 8)),
        max_recall_fragments=int(getattr(s, "MIND_EVIDENCE_MAX_RECALL_FRAGMENTS", 8)),
        max_projection_items=int(getattr(s, "MIND_EVIDENCE_MAX_PROJECTION_ITEMS", 16)),
        max_total_chars=int(getattr(s, "MIND_EVIDENCE_MAX_CHARS", 12_000)),
    )
    phases["evidence_pack_ms"] = _elapsed_ms_wall_clock(t_run_start)

    fail_open = bool(getattr(s, "MIND_LLM_FAIL_OPEN_LEGACY", True))
    configured_timeout = float(getattr(s, "MIND_LLM_TIMEOUT_SEC", 90.0))
    semantic_route = str(getattr(s, "MIND_SEMANTIC_MODEL_ROUTE", "quick"))
    appraisal_route = str(getattr(s, "MIND_APPRAISAL_MODEL_ROUTE", "metacog"))
    stance_route = str(getattr(s, "MIND_STANCE_MODEL_ROUTE", "chat"))
    phase_records: list[MindPhaseTelemetry] = []

    def _fail_open_or_error(
        *,
        error_code: str,
        diagnostics: list[str],
        failed_phase: str | None = None,
    ) -> MindRunResultV1 | MindLLMFailOpenRecord:
        if fail_open:
            return MindLLMFailOpenRecord(
                mind_run_id=mind_run_id,
                snapshot_hash=snap_hash,
                error_code=error_code,
                diagnostics=list(diagnostics),
                failed_phase=failed_phase,
                semantic_route=semantic_route,
                appraisal_route=appraisal_route,
                stance_route=stance_route,
                phase_telemetry=list(phase_records),
                timing_ms_by_phase=dict(phases),
                fallback_reason=[error_code, *diagnostics[:4]],
            )
        return _error_result(
            mind_run_id=mind_run_id,
            error_code=error_code,
            diagnostics=diagnostics,
            snapshot_hash=snap_hash,
            timing_ms_by_phase=phases,
        )

    if budget.over_budget():
        return _fail_open_or_error(
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_before_llm"],
            failed_phase="pre_llm",
        )

    for phase_name, route in (
        ("semantic_synthesis", semantic_route),
        ("active_frontier_judge", appraisal_route),
        ("stance_handoff", stance_route),
    ):
        route_err = _validate_phase_route(route, phase=phase_name)
        if route_err:
            return _fail_open_or_error(
                error_code="invalid_llm_route",
                diagnostics=[route_err],
                failed_phase=phase_name,
            )

    client = get_llm_client()
    llm_errors: list[str] = []

    def _phase_context(phase_name: str) -> MindLLMRequestContext:
        return MindLLMRequestContext(
            correlation_id=req.correlation_id,
            mind_run_id=str(mind_run_id),
            phase_name=phase_name,
            session_id=req.session_id,
            trace_id=req.trace_id,
            router_profile_id=req.policy.router_profile_id,
            trigger=str(req.trigger),
        )

    if not budget.can_run_phase():
        rec = MindPhaseTelemetry(
            phase_name="semantic_synthesis",
            route=semantic_route,
            skipped=True,
            skip_reason="wall_budget_insufficient",
        )
        phase_records.append(rec)
        return _fail_open_or_error(
            error_code="loop_budget_exceeded",
            diagnostics=["wall_budget_insufficient_before_semantic"],
            failed_phase="semantic_synthesis",
        )

    t_sem = time.perf_counter()
    synthesis, sem_err, sem_telemetry = run_semantic_synthesis(
        pack,
        client=client,
        route=semantic_route,
        model_id=semantic_route,
        max_tokens=int(getattr(s, "MIND_LLM_MAX_TOKENS_SEMANTIC", 2048)),
        context=_phase_context("semantic_synthesis"),
        timeout_sec=budget.phase_timeout_sec(configured_timeout),
    )
    phase_records.append(sem_telemetry)
    phases["semantic_synthesis_ms"] = (time.perf_counter() - t_sem) * 1000
    if sem_err:
        llm_errors.append(sem_err)
    if budget.over_budget():
        return _fail_open_or_error(
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_after_semantic"],
            failed_phase="semantic_synthesis",
        )
    if synthesis is None or not synthesis.claims:
        return _fail_open_or_error(
            error_code="semantic_synthesis_failed",
            diagnostics=llm_errors or ["semantic_synthesis_empty"],
            failed_phase="semantic_synthesis",
        )

    frontier = None
    if synthesis.claims and budget.can_run_phase():
        t_ap = time.perf_counter()
        frontier, ap_err, ap_telemetry = run_active_frontier_judge(
            synthesis,
            pack,
            client=client,
            route=appraisal_route,
            model_id=appraisal_route,
            max_tokens=int(getattr(s, "MIND_LLM_MAX_TOKENS_APPRAISAL", 3072)),
            thinking=bool(getattr(s, "MIND_LLM_THINKING_APPRAISAL", True)),
            context=_phase_context("active_frontier_judge"),
            timeout_sec=budget.phase_timeout_sec(configured_timeout),
        )
        phase_records.append(ap_telemetry)
        phases["appraisal_ms"] = (time.perf_counter() - t_ap) * 1000
        if ap_err:
            llm_errors.append(ap_err)
        if budget.over_budget():
            return _fail_open_or_error(
                error_code="loop_budget_exceeded",
                diagnostics=["wall_time_exceeded_after_appraisal"],
                failed_phase="active_frontier_judge",
            )
    elif synthesis.claims:
        phase_records.append(
            MindPhaseTelemetry(
                phase_name="active_frontier_judge",
                route=appraisal_route,
                skipped=True,
                skip_reason="wall_budget_insufficient",
            )
        )
        llm_errors.append("appraisal_skipped_wall_budget")

    user_text = pack.current_user_text
    base = _default_stance_from_user_text(user_text)
    stance_payload = dict(base)
    if frontier and frontier.selected and synthesis and budget.can_run_phase():
        t_st = time.perf_counter()
        stance_payload, st_err, st_telemetry = run_stance_handoff(
            frontier,
            synthesis,
            pack,
            client=client,
            route=stance_route,
            model_id=stance_route,
            max_tokens=int(getattr(s, "MIND_LLM_MAX_TOKENS_STANCE", 1536)),
            context=_phase_context("stance_handoff"),
            timeout_sec=budget.phase_timeout_sec(configured_timeout),
        )
        phase_records.append(st_telemetry)
        phases["stance_handoff_ms"] = (time.perf_counter() - t_st) * 1000
        if st_err:
            llm_errors.append(st_err)
    elif frontier and frontier.selected:
        phase_records.append(
            MindPhaseTelemetry(
                phase_name="stance_handoff",
                route=stance_route,
                skipped=True,
                skip_reason="wall_budget_insufficient",
            )
        )
        llm_errors.append("stance_handoff_skipped_wall_budget")

    handoff = build_handoff(
        synthesis=synthesis,
        frontier=frontier,
        stance_payload=stance_payload,
        model_id=stance_route,
        llm_errors=llm_errors,
    )
    valid, err = validate_merged_stance_brief_optional(stance_payload)
    if valid is None:
        if bool(getattr(s, "MIND_LLM_FAIL_OPEN_LEGACY", True)):
            return _fail_open_or_error(
                error_code="stance_handoff_invalid",
                diagnostics=[err or "invalid_stance_payload", *llm_errors],
                failed_phase="stance_handoff",
            )
        return _error_result(
            mind_run_id=mind_run_id,
            error_code="stance_handoff_invalid",
            diagnostics=[err or "invalid_stance_payload", *llm_errors],
            snapshot_hash=snap_hash,
            timing_ms_by_phase=phases,
        )

    shadow_synthesis = _build_shadow_synthesis(bounded, stance_payload)
    prof = _router_profile(router_profiles_dir, req.policy.router_profile_id)
    decision = MindControlDecisionV1(
        route_kind=str(prof.get("route_kind") or "brain"),
        allowed_verbs=list(prof.get("allowed_verbs") or ["chat_general"]),
        recall_profile_override=prof.get("recall_profile_override"),
        mode_suggestion=str(prof.get("mode_suggestion") or "brain"),  # type: ignore[arg-type]
        mode_binding=str(prof.get("mode_binding") or "advisory"),  # type: ignore[arg-type]
        budgets={"wall_ms_remaining": float(req.policy.wall_time_ms_max), "truncated": _truncated},
    )
    quality = handoff.mind_quality
    authorized = bool(handoff.authorized_for_stance_use)
    fallback_reason = list(handoff.authorization_reasons) if not authorized else []
    if llm_errors and not authorized:
        fallback_reason = [*(fallback_reason or []), *llm_errors[:4]]
    machine = _llm_machine_contract(
        decision=decision,
        synthesis=synthesis,
        frontier=frontier,
        handoff=handoff,
        cognitive_projection_debug=cognitive_projection_debug,
        shadow_synthesis=shadow_synthesis,
        llm_enabled=True,
        llm_error=llm_errors[0] if llm_errors else None,
        phase_telemetry=phase_records,
        fallback_reason=fallback_reason or None,
        semantic_route=semantic_route,
        appraisal_route=appraisal_route,
        stance_route=stance_route,
    )
    machine["mind.wall_time_ms_max"] = float(req.policy.wall_time_ms_max)
    machine["mind.wall_time_elapsed_ms"] = budget.elapsed_ms()
    summary = (
        "Meaningful Mind synthesis authorized for stance handoff."
        if authorized
        else f"Mind LLM synthesis degraded ({', '.join(handoff.authorization_reasons[:4])})."
    )
    brief = MindHandoffBriefV1(
        summary_one_paragraph=summary,
        machine_contract=machine,
        mandatory_keys=["mind.route_kind", "mind.allowed_verbs"],
        advisory_keys=["mind.quality", "mind.authorized_for_stance_use", "mind.semantic_synthesis_seen"],
        stance_payload=valid.model_dump(mode="json"),
        mind_quality=quality,  # type: ignore[arg-type]
        shadow_synthesis=shadow_synthesis,
        mind_authorized_for_stance_skip=authorized,
        semantic_synthesis=synthesis,
        active_frontier=frontier,
        stance_handoff=handoff,
    )
    phases["total_ms"] = _elapsed_ms_wall_clock(t_run_start)
    patch = MindStancePatchV1(
        loop_index=0,
        structured=valid.model_dump(mode="json"),
        provenance=MindProvenanceV1(model_id=stance_route, input_hash=snap_hash),
        narrative_notes="llm_synthesis_pipeline",
    )
    return MindRunResultV1(
        mind_run_id=mind_run_id,
        ok=True,
        snapshot_hash=snap_hash,
        trajectory=MindStanceTrajectoryV1(
            patches=[patch],
            merged_stance_brief=valid.model_dump(mode="json"),
            merge_policy="deterministic_merge",
        ),
        decision=decision,
        brief=brief,
        mind_quality=quality,  # type: ignore[arg-type]
        timing_ms_by_phase=phases,
    )


def _llm_fail_open_from_exception(
    exc: BaseException,
    *,
    req: MindRunRequestV1,
    mind_settings: Any,
    snapshot_max_bytes: int,
) -> "MindLLMFailOpenRecord":
    from .llm_fail_open import MindLLMFailOpenRecord

    bounded, _ = build_bounded_snapshot_inputs(dict(req.snapshot_inputs or {}), snapshot_max_bytes)
    return MindLLMFailOpenRecord(
        mind_run_id=uuid4(),
        snapshot_hash=hash_snapshot_inputs(bounded),
        error_code="llm_synthesis_exception",
        diagnostics=[str(exc)],
        failed_phase="unknown",
        semantic_route=str(getattr(mind_settings, "MIND_SEMANTIC_MODEL_ROUTE", "quick")),
        appraisal_route=str(getattr(mind_settings, "MIND_APPRAISAL_MODEL_ROUTE", "metacog")),
        stance_route=str(getattr(mind_settings, "MIND_STANCE_MODEL_ROUTE", "chat")),
        fallback_reason=["llm_synthesis_exception", str(exc)],
    )


def run_mind(
    req: MindRunRequestV1,
    *,
    router_profiles_dir: Path,
    snapshot_max_bytes: int,
    mind_settings: Any | None = None,
) -> MindRunResultV1:
    from .llm_fail_open import MindLLMFailOpenRecord
    from .settings import settings as default_settings

    s = mind_settings if mind_settings is not None else default_settings
    fail_open_record: MindLLMFailOpenRecord | None = None
    if bool(getattr(s, "MIND_LLM_SYNTHESIS_ENABLED", False)):
        try:
            llm_result = run_mind_llm_synthesis(
                req,
                router_profiles_dir=router_profiles_dir,
                snapshot_max_bytes=snapshot_max_bytes,
                mind_settings=s,
            )
            if isinstance(llm_result, MindRunResultV1):
                return llm_result
            if isinstance(llm_result, MindLLMFailOpenRecord):
                fail_open_record = llm_result
        except Exception as exc:  # noqa: BLE001
            logger.warning("mind_llm_synthesis_failed correlation_id=%s err=%s", req.correlation_id, exc)
            if not bool(getattr(s, "MIND_LLM_FAIL_OPEN_LEGACY", True)):
                mind_run_id = uuid4()
                return _error_result(
                    mind_run_id=mind_run_id,
                    error_code="llm_synthesis_failed",
                    diagnostics=[str(exc)],
                    snapshot_hash=hash_snapshot_inputs(dict(req.snapshot_inputs or {})),
                )
            fail_open_record = _llm_fail_open_from_exception(
                exc,
                req=req,
                mind_settings=s,
                snapshot_max_bytes=snapshot_max_bytes,
            )
    det = run_mind_deterministic(
        req,
        router_profiles_dir=router_profiles_dir,
        snapshot_max_bytes=snapshot_max_bytes,
    )
    if fail_open_record is not None:
        return fail_open_record.merge_into_deterministic(det)
    return det
