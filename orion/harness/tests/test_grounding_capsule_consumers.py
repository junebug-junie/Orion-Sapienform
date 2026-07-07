from __future__ import annotations

from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_grounding_capsule, make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_prefix_renders_compact_self_block_when_capsule_present() -> None:
    capsule = make_grounding_capsule()
    thought = make_thought(imperative="Stay present.", grounding_capsule=capsule)
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )
    assert "WHO YOU ARE" in prompt
    assert "I am Oríon" in prompt
    assert "Juniper is my collaborator" in prompt
    assert "We were mid-way through the grounding refactor." in prompt
    # Response policy is reserved for the voice pass (motor budget discipline).
    assert "no generic-assistant framing" not in prompt
    # Self block precedes the imperative.
    assert prompt.index("WHO YOU ARE") < prompt.index("Imperative:")


def test_prefix_no_ops_when_capsule_absent() -> None:
    thought = make_thought(imperative="Stay present.")
    assert thought.grounding_capsule is None
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )
    assert "WHO YOU ARE" not in prompt
