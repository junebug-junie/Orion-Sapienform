from __future__ import annotations

from orion.cognition.delivery_grounding import build_delivery_grounding_context
from orion.cognition.quality_evaluator import detect_generic_delivery_drift, should_rewrite_for_instructional


def test_orion_discord_request_uses_repo_architecture_grounding() -> None:
    grounding = build_delivery_grounding_context(
        user_text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        output_mode="implementation_guide",
    )

    assert grounding["delivery_grounding_mode"] == "orion_repo_architecture"
    assert "PlannerReact/AgentChain" in grounding["grounding_context"]
    assert "Discord" in grounding["grounding_context"]


def test_generic_flask_answer_detected_as_drift_for_orion_discord_request() -> None:
    drifted, reason = detect_generic_delivery_drift(
        "Deploy a Flask app on Ubuntu with Gunicorn and Nginx.",
        request_text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        grounding_mode="orion_repo_architecture",
    )

    assert drifted is True
    assert reason == "generic_architecture_drift"


def test_instructional_rewrite_triggers_on_generic_drift() -> None:
    should_rewrite, reason = should_rewrite_for_instructional(
        "Deploy a Flask app on Ubuntu with Gunicorn and Nginx.",
        "implementation_guide",
        request_text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        grounding_mode="orion_repo_architecture",
    )

    assert should_rewrite is True
    assert reason == "generic_architecture_drift"
