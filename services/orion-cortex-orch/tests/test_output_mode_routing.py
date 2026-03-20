"""Tests for output-mode routing (answer depth overhaul)."""

from __future__ import annotations

from app.output_mode_classifier import classify_output_mode


def test_discord_deploy_routes_to_implementation_guide():
    """Please provide instructions on how to deploy you onto Discord -> implementation_guide."""
    result = classify_output_mode("Please provide instructions on how to deploy you onto Discord.")
    assert result.output_mode == "implementation_guide"
    assert result.response_profile == "technical_delivery"


def test_compare_routes_to_comparative():
    """Compare Discord vs Slack deployment for Orion -> comparative_analysis."""
    result = classify_output_mode("Compare Discord vs Slack deployment for Orion.")
    assert result.output_mode == "comparative_analysis"


def test_decide_routes_to_decision_support():
    """Help me decide whether to build the Discord bridge -> decision_support."""
    result = classify_output_mode("Help me decide whether to build the Discord bridge now or later.")
    assert result.output_mode == "decision_support"
