"""Tests for quality evaluator (answer depth overhaul)."""

from __future__ import annotations

from orion.cognition.quality_evaluator import looks_like_meta_plan, should_rewrite_for_instructional


def test_meta_plan_flagged():
    """Meta-plan output is flagged."""
    assert looks_like_meta_plan("First gather requirements, then create a guide, and finally review and refine.")
    assert looks_like_meta_plan("You should develop a plan and implement the solution.")
    assert not looks_like_meta_plan("Step 1: Run `docker compose up`. Step 2: Configure env vars.")


def test_should_rewrite_for_instructional():
    """Instructional mode + meta-plan -> should rewrite."""
    should, reason = should_rewrite_for_instructional(
        "Gather requirements, create a guide, test deployment.",
        "implementation_guide",
    )
    assert should is True
    assert reason == "meta_plan_detected"
