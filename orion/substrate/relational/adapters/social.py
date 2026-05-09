"""Social memory adapter — snapshot_ephemeral tier.

Maps social bridge ctx keys into StateSnapshotNodeV1 nodes anchored to
'relationship'.  Replaces ``_social_summary`` and ``_social_bridge_summary``
as substrate-backed ephemeral contributions.  pull_on_cold=False (ctx-based).

Reads:
  ctx["social_inspection_snapshot"]
  ctx["social_stance_snapshot"]
  ctx["social_turn_policy"]
  ctx["social_peer_style_hint"]
  ctx["social_context_window"]
  ctx["social_thread_routing"]
  ctx["social_repair_decision"]
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)

from orion.substrate.adapters._common import make_temporal

_TIER_RANK = 4  # snapshot_ephemeral

_TECHNICAL_TURN_PATTERNS = (
    r"\bgpu\b",
    r"\bvram\b",
    r"\bllamacpp\b",
    r"\bqwen\b",
    r"\boffline\b",
    r"\bworkflow(s)?\b",
    r"\bcarrier board\b",
    r"\bupgrade\b",
    r"\bdebug\b",
    r"\btriage\b",
)


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="social_bridge",
        source_channel="ctx.social_bridge",
        producer="social_adapter",
        tier_rank=_TIER_RANK,
    )


def _compact(value: Any, *, limit: int = 220) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def map_social_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map social bridge ctx keys → StateSnapshotNodeV1 (anchor=relationship)."""
    ctx = ctx if isinstance(ctx, dict) else {}

    turn_policy = ctx.get("social_turn_policy") if isinstance(ctx.get("social_turn_policy"), dict) else {}
    stance_snapshot = ctx.get("social_stance_snapshot") if isinstance(ctx.get("social_stance_snapshot"), dict) else {}
    peer_style = ctx.get("social_peer_style_hint") if isinstance(ctx.get("social_peer_style_hint"), dict) else {}
    inspection = ctx.get("social_inspection_snapshot") if isinstance(ctx.get("social_inspection_snapshot"), dict) else {}
    context_window = ctx.get("social_context_window") if isinstance(ctx.get("social_context_window"), dict) else {}
    routing = ctx.get("social_thread_routing") if isinstance(ctx.get("social_thread_routing"), dict) else {}
    repair = ctx.get("social_repair_decision") if isinstance(ctx.get("social_repair_decision"), dict) else {}

    if not any([turn_policy, stance_snapshot, peer_style, inspection, context_window, routing, repair]):
        return None

    posture: list[str] = []
    hazards: list[str] = []
    framing: list[str] = []
    relationship_facets: list[str] = []

    reasons = [str(r).strip() for r in (turn_policy.get("reasons") or []) if str(r).strip()]
    reasons_blob = " ".join(reasons).lower()
    user_message = _compact(ctx.get("user_message") or "", limit=400).lower()
    orientation = _compact(
        stance_snapshot.get("recent_social_orientation_summary") or stance_snapshot.get("summary") or "", limit=180
    ).lower()
    style_hint = _compact(peer_style.get("style_hints_summary") or "", limit=180).lower()
    routing_decision = str(routing.get("routing_decision") or "").strip().lower()

    def _add_posture(cond: bool, tag: str) -> None:
        if cond:
            posture.append(tag)

    _add_posture("direct" in orientation or "direct" in style_hint or "direct" in reasons_blob, "direct")
    _add_posture("warm" in orientation or "warm" in style_hint, "warm")
    _add_posture("playful" in orientation or "playful" in style_hint, "playful")
    _add_posture("reflect" in orientation or "reflect" in reasons_blob, "reflective")
    _add_posture("strain" in orientation or "repair" in reasons_blob or "friction" in reasons_blob, "strained")
    _add_posture(
        "technical" in orientation
        or "technical" in style_hint
        or any(re.search(p, user_message) for p in _TECHNICAL_TURN_PATTERNS),
        "technical",
    )
    _add_posture(bool(turn_policy.get("addressed")), "addressed")
    _add_posture(bool(turn_policy.get("should_speak")), "engaged_turn")
    _add_posture(routing_decision == "reply_to_peer", "peer_reply_mode")
    _add_posture(routing_decision == "reply_to_room", "room_reply_mode")
    _add_posture(str(repair.get("decision") or "").strip().lower() in {"yield", "reset_thread"}, "deescalate")

    # Inspection snapshot contributes to relationship facets
    for key in ("summary", "stance", "relationship_posture"):
        val = inspection.get(key)
        if isinstance(val, str) and val.strip():
            relationship_facets.append(val.strip()[:140])

    for candidate in (context_window.get("selected_candidates") or [])[:6]:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("candidate_kind") or "").strip().lower()
        decision = str(candidate.get("inclusion_decision") or "include").strip().lower()
        summary = _compact(candidate.get("summary"), limit=120)
        if kind:
            framing.append(f"{kind}:{decision}")
        if summary:
            framing.append(summary)
        if decision in {"exclude", "soften"}:
            hazards.append(f"context_{decision}:{kind or 'unknown'}")

    for reason in reasons[:8]:
        lo = reason.lower()
        if "cooldown" in lo:
            hazards.append("cooldown_active")
        if "duplicate" in lo:
            hazards.append("duplicate_message")
        if "self-loop" in lo:
            hazards.append("self_message_loop")

    turn_decision = _compact(turn_policy.get("decision"), limit=40)
    orientation_summary = _compact(stance_snapshot.get("recent_social_orientation_summary"), limit=140)
    style_summary = _compact(peer_style.get("style_hints_summary"), limit=140)

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()

    snapshot = StateSnapshotNodeV1(
        anchor_scope="relationship",
        temporal=temporal,
        provenance=prov,
        signals=SubstrateSignalBundleV1(confidence=0.75, salience=0.5),
        snapshot_source="social_bridge",
        dimensions={
            "has_turn_policy": 1.0 if turn_policy else 0.0,
            "addressed": 1.0 if turn_policy.get("addressed") else 0.0,
            "should_speak": 1.0 if turn_policy.get("should_speak") else 0.0,
        },
        metadata={
            "posture": posture,
            "hazards": hazards,
            "framing": framing,
            "relationship_facets": relationship_facets,
            "turn_decision": turn_decision,
            "orientation_summary": orientation_summary,
            "style_summary": style_summary,
        },
    )

    return SubstrateGraphRecordV1(anchor_scope="relationship", nodes=[snapshot])
