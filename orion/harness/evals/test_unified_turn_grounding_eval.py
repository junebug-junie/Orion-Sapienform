from __future__ import annotations

from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_grounding_capsule, make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_how_are_you_turn_is_grounded_end_to_end() -> None:
    """A 'How are you?' unified turn carries Orion self-context into the motor."""
    capsule = make_grounding_capsule(
        identity_summary=["I am Oríon, a digital mind in development with Juniper."],
        relationship_summary=["Juniper is my collaborator; we build my mind together."],
        response_policy_summary=[
            'Avoid phrase: "I notice we\'re in the Orion-Sapienform repository"',
        ],
        memory_digest="Last we spoke, we were wiring self-grounding into the unified turn.",
    )
    thought = make_thought(
        imperative="Stay present with Juniper; one situated wondering.",
        tone="warm, companionable",
        grounding_capsule=capsule,
    )

    prefix = compile_harness_prefix(
        thought, repair_overlay=HarnessRepairOverlayV1(), user_message="How are you?"
    )
    assert "WHO YOU ARE" in prefix
    assert "I am Oríon" in prefix
    assert "Juniper is my collaborator" in prefix
    assert "Last we spoke, we were wiring self-grounding" in prefix
    assert "RESPONSE POLICY" in prefix
    assert "Avoid phrase" in prefix
