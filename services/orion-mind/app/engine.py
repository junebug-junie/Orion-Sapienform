"""Deterministic Mind cognition engine (service-local only)."""

from __future__ import annotations

import json
import logging
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
    MindStancePatchV1,
    MindStanceTrajectoryV1,
    MindProvenanceV1,
)

logger = logging.getLogger("orion-mind.engine")

_FACET_ORDER = (
    "autonomy",
    "substrate",
    "collapse",
    "concept_induction",
    "recall_digest",
    "social",
    "metacog",
    "tool_outcomes",
    "misc",
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


def _default_stance_from_user_text(user_text: str) -> dict[str, Any]:
    ut = (user_text or "").strip()[:2000]
    return {
        "conversation_frame": "technical",
        "task_mode": "direct_response",
        "identity_salience": "medium",
        "user_intent": ut or "(empty turn)",
        "self_relevance": "Maintain coherence with Orion continuity.",
        "juniper_relevance": "Collaborate with Juniper on the active thread.",
        "answer_strategy": "DirectAnswer",
        "stance_summary": f"Deterministic stance seed ({len(ut)} chars).",
    }


def _elapsed_ms_wall_clock(t_run_start: float) -> float:
    return (time.perf_counter() - t_run_start) * 1000


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
    snap_hash = hash_snapshot_inputs(bounded)
    phases["snapshot_ms"] = _elapsed_ms_wall_clock(t_run_start)

    if _elapsed_ms_wall_clock(t_run_start) > wall_budget_ms:
        logger.info("mind_run_end mind_run_id=%s ok=False error=loop_budget_exceeded phase=snapshot", mind_run_id)
        return MindRunResultV1(
            mind_run_id=mind_run_id,
            ok=False,
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_after_snapshot"],
            snapshot_hash=snap_hash,
            trajectory=MindStanceTrajectoryV1(patches=[], merged_stance_brief={}, merge_policy="deterministic_merge"),
            decision=MindControlDecisionV1(route_kind="no_chat", refusals=[{"code": "loop_budget_exceeded"}]),
            brief=MindHandoffBriefV1(),
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
    for i in range(n_loops):
        if _elapsed_ms_wall_clock(t_run_start) > wall_budget_ms:
            logger.info(
                "mind_run_end mind_run_id=%s ok=False error=loop_budget_exceeded phase=loop loop_index=%s",
                mind_run_id,
                i,
            )
            return MindRunResultV1(
                mind_run_id=mind_run_id,
                ok=False,
                error_code="loop_budget_exceeded",
                diagnostics=[f"wall_time_exceeded_during_loop_at_index_{i}"],
                snapshot_hash=snap_hash,
                trajectory=MindStanceTrajectoryV1(patches=patches, merged_stance_brief={}, merge_policy="deterministic_merge"),
                decision=MindControlDecisionV1(route_kind="no_chat", refusals=[{"code": "loop_budget_exceeded"}]),
                brief=MindHandoffBriefV1(),
                timing_ms_by_phase={**phases, "loops_partial_ms": (time.perf_counter() - t_loop) * 1000},
            )
        # llm_flags[i] True means LLM would run — v1 is deterministic only; no extra merge keys (avoid polluting ChatStanceBrief).
        delta: dict[str, Any] = {
            "stance_summary": f"{base['stance_summary']} [loop {i + 1}/{n_loops}]",
        }
        if i == 0:
            delta.update(base)
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
        return MindRunResultV1(
            mind_run_id=mind_run_id,
            ok=False,
            error_code="loop_budget_exceeded",
            diagnostics=["wall_time_exceeded_after_loops_before_merge"],
            snapshot_hash=snap_hash,
            trajectory=MindStanceTrajectoryV1(
                patches=patches,
                merged_stance_brief=merged_partial,
                merge_policy="deterministic_merge",
            ),
            decision=MindControlDecisionV1(route_kind="no_chat", refusals=[{"code": "loop_budget_exceeded"}]),
            brief=MindHandoffBriefV1(),
            timing_ms_by_phase=phases,
        )

    merged = deterministic_merge_stance(layers)
    t_merge = time.perf_counter()
    valid, err = validate_merged_stance_brief_optional(merged)
    phases["merge_ms"] = (time.perf_counter() - t_merge) * 1000

    if valid is None:
        logger.info("mind_run_end mind_run_id=%s ok=False error=stance_merge_invalid", mind_run_id)
        return MindRunResultV1(
            mind_run_id=mind_run_id,
            ok=False,
            error_code="stance_merge_invalid",
            diagnostics=[err or "stance_merge_invalid", f"merged_keys={list(merged.keys())}"],
            snapshot_hash=snap_hash,
            trajectory=MindStanceTrajectoryV1(patches=patches, merged_stance_brief=merged, merge_policy="deterministic_merge"),
            decision=MindControlDecisionV1(route_kind="no_chat", refusals=[{"code": "stance_merge_invalid", "detail": str(err)}]),
            brief=MindHandoffBriefV1(),
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
    machine = {
        "mind.route_kind": decision.route_kind,
        "mind.allowed_verbs": decision.allowed_verbs,
        "mind.mode_suggestion": decision.mode_suggestion,
        "mind.mode_binding": decision.mode_binding,
    }
    brief = MindHandoffBriefV1(
        summary_one_paragraph="Deterministic mind run (v1).",
        machine_contract=machine,
        mandatory_keys=["mind.route_kind", "mind.allowed_verbs"],
        advisory_keys=["mind.mode_suggestion"],
        stance_payload=valid.model_dump(mode="json"),
    )

    phases["total_ms"] = _elapsed_ms_wall_clock(t_run_start)
    logger.info("mind_run_end mind_run_id=%s ok=True snapshot_hash=%s", mind_run_id, snap_hash)

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
        timing_ms_by_phase=phases,
    )
