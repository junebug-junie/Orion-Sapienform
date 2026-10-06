"""draft_preview_hold_reason / draft_preview_display_text (spec L8)."""
from __future__ import annotations

from orion.harness.finalize import (
    draft_preview_display_text,
    draft_preview_hold_reason,
    quick_lane_block_reason,
    sensitive_turn_reason,
)
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought


def test_calm_turn_shows_the_draft() -> None:
    assert draft_preview_hold_reason(thought=make_thought(), repair_overlay=make_repair_overlay()) is None


def test_boundary_and_trust_turns_are_held() -> None:
    overlay = make_repair_overlay()
    assert (
        draft_preview_hold_reason(thought=make_thought(boundary_register=True), repair_overlay=overlay)
        == "sensitive:boundary_register"
    )
    assert (
        draft_preview_hold_reason(thought=make_thought(trust_rupture_score=0.99), repair_overlay=overlay)
        == "sensitive:trust_rupture_score"
    )


def test_repair_overlay_mode_is_held() -> None:
    overlay = make_repair_overlay(mode="concrete_bias")
    assert sensitive_turn_reason(thought=make_thought(), repair_overlay=overlay) == "repair_overlay_mode"
    assert draft_preview_hold_reason(thought=make_thought(), repair_overlay=overlay) == "sensitive:repair_overlay_mode"


def test_structured_and_cut_short_drafts_are_held() -> None:
    kw = {"thought": make_thought(), "repair_overlay": make_repair_overlay()}
    assert draft_preview_hold_reason(**kw, preserve_structured_output=True) == "structured_output"
    assert draft_preview_hold_reason(**kw, cut_short=True) == "cut_short"


def test_quick_lane_keeps_its_reasons_after_the_shared_extraction() -> None:
    appraisal = make_appraisal()
    overlay = make_repair_overlay()
    assert quick_lane_block_reason(substrate_appraisal=appraisal, thought=make_thought(), repair_overlay=overlay) is None
    assert (
        quick_lane_block_reason(
            substrate_appraisal=appraisal, thought=make_thought(boundary_register=True), repair_overlay=overlay
        )
        == "boundary_register"
    )
    # Substrate reasons still win first, as before.
    assert (
        quick_lane_block_reason(
            substrate_appraisal=make_appraisal(surprise_level=0.99),
            thought=make_thought(boundary_register=True),
            repair_overlay=overlay,
        )
        == "surprise_level"
    )


def test_display_text_without_receipts_is_the_draft() -> None:
    assert draft_preview_display_text("hello", None) == "hello"
    assert draft_preview_display_text("hello", []) == "hello"
