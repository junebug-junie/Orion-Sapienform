"""Deterministic Mind cognition engine (service-local only)."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any
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
