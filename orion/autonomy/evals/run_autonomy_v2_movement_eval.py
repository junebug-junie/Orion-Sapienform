#!/usr/bin/env python3
"""Trace-proven AutonomyStateV2 pressure movement from typed chat evidence.

Enable bar for AUTONOMY_STATE_V2_REDUCER_ENABLED: do not flip production until
this eval exits 0.
"""
from __future__ import annotations

import sys
from datetime import datetime

from orion.autonomy.evidence_compiler import compile_autonomy_evidence
from orion.autonomy.models import AutonomyStateV2
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state


def _cold(**overrides: object) -> AutonomyStateV2:
    base = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="eval",
        generated_at=datetime(2026, 7, 10, 12, 0, 0),
        schema_version="autonomy.state.v2",
        confidence=0.5,
        unknowns=[],
        evidence_refs=[],
        freshness={},
        attention_items=[],
        candidate_impulses=[],
        inhibited_impulses=[],
        last_action_outcomes=[],
        drive_pressures={
            "coherence": 0.05,
            "continuity": 0.05,
            "relational": 0.05,
            "autonomy": 0.05,
            "capability": 0.05,
            "predictive": 0.05,
        },
        dominant_drive=None,
        active_drives=[],
        tension_kinds=[],
    )
    return base.model_copy(update=overrides)


def main() -> int:
    fixed = datetime(2026, 7, 10, 16, 0, 0)
    compiled = compile_autonomy_evidence(
        user_message="are you looping?",
        social={"hazards": ["self_message_loop", "context_excluded:x"]},
        social_bridge={"hazards": ["cooldown_active"]},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=True,
        autonomy_debug={"orion": {"availability": "available"}},
        now=fixed,
    )
    assert any(e.kind == "reasoning_quality" for e in compiled.evidence)
    assert any(e.dimension == "self_message_loop" for e in compiled.evidence)
    assert any(
        e.summary == "context_excluded:x"
        and e.signal_kind == "chat_social_hazard"
        and e.dimension == "context_excluded:x"
        for e in compiled.evidence
    )

    baseline = _cold()
    before = dict(baseline.drive_pressures)
    before_dom = baseline.dominant_drive

    result = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=baseline,
            evidence=compiled.evidence,
            action_outcomes=[],
            now=fixed,
        )
    )
    after = result.state.drive_pressures
    moved = {
        k: round(after[k] - before[k], 6)
        for k in before
        if abs(after[k] - before[k]) > 1e-9
    }
    print("omitted=", compiled.omitted)
    print("moved=", moved)
    print("dominant_before=", before_dom, "dominant_after=", result.state.dominant_drive)
    print("new_tensions=", result.delta.new_tensions)

    if not moved:
        print("FAIL: no drive_pressures movement")
        return 1
    if after.get("relational", 0.0) <= before.get("relational", 0.0):
        print("FAIL: relational did not increase from mapped hazards")
        return 1
    if after.get("coherence", 0.0) <= before.get("coherence", 0.0):
        print("FAIL: coherence did not increase from reasoning fallback")
        return 1
    # Dominant may change once pressures clear the 0.15 threshold.
    if result.state.dominant_drive is None and max(after.values()) >= 0.15:
        print("FAIL: expected dominant_drive once pressure >= 0.15")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
