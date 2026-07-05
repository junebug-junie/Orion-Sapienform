from __future__ import annotations

from orion.harness.finalize import maybe_quick_lane_verdict, quick_lane_block_reason
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought
from orion.thought.policy_refusal import TRUST_RUPTURE_DEFER_THRESHOLD


def _eligible_inputs():
    return (
        make_thought(),
        make_appraisal(),
        make_repair_overlay(),
    )


def test_quick_lane_blocked_on_high_surprise() -> None:
    thought, appraisal, overlay = _eligible_inputs()
    appraisal = make_appraisal(surprise_level=0.5)
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
        epsilon=0.08,
    ) == "surprise_level"
    assert maybe_quick_lane_verdict(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=overlay,
        epsilon=0.08,
    ) is None


def test_quick_lane_blocked_on_alignment_hints() -> None:
    thought, _, overlay = _eligible_inputs()
    appraisal = make_appraisal(alignment_hints=["tone mismatch"])
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "alignment_hints"


def test_quick_lane_blocked_on_strain_shift_refs() -> None:
    thought, _, overlay = _eligible_inputs()
    appraisal = make_appraisal(strain_shift_refs=["strain-9"])
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "strain_shift_refs"


def test_quick_lane_blocked_on_open_loop_pressure() -> None:
    thought, _, overlay = _eligible_inputs()
    appraisal = make_appraisal(open_loop_pressure=0.25)
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "open_loop_pressure"


def test_quick_lane_blocked_on_repair_pressure() -> None:
    thought, appraisal, overlay = _eligible_inputs()
    thought = make_thought(repair_pressure_level=0.35)
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "repair_pressure_level"


def test_quick_lane_blocked_on_trust_rupture() -> None:
    thought, appraisal, overlay = _eligible_inputs()
    thought = make_thought(trust_rupture_score=TRUST_RUPTURE_DEFER_THRESHOLD)
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "trust_rupture_score"


def test_quick_lane_blocked_on_boundary_register() -> None:
    thought, appraisal, overlay = _eligible_inputs()
    thought = make_thought(boundary_register=True)
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "boundary_register"


def test_quick_lane_blocked_on_non_default_repair_overlay() -> None:
    thought, appraisal, _ = _eligible_inputs()
    overlay = make_repair_overlay(mode="repair_concrete", rule_lines=["be concrete"])
    assert quick_lane_block_reason(
        substrate_appraisal=appraisal,
        thought=thought,
        repair_overlay=overlay,
    ) == "repair_overlay_mode"


def test_quick_lane_allowed_when_all_criteria_pass() -> None:
    thought, appraisal, overlay = _eligible_inputs()
    reflection = maybe_quick_lane_verdict(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=overlay,
        epsilon=0.08,
    )
    assert reflection is not None
    assert reflection.reflection_source == "deterministic_quick_gate"
    assert reflection.quick_lane_skipped_llm is True
