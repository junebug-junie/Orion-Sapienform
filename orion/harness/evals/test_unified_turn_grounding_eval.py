from __future__ import annotations

from pathlib import Path

import jinja2

from orion.harness.finalize import build_voice_finalize_context
from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_grounding_capsule,
    make_reflection,
    make_thought,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_how_are_you_turn_is_grounded_end_to_end() -> None:
    """A 'How are you?' unified turn carries Orion self-context into both passes."""
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

    # Motor prefix: compact self is present, no repository framing leaks in.
    prefix = compile_harness_prefix(
        thought, repair_overlay=HarnessRepairOverlayV1(), user_message="How are you?"
    )
    assert "WHO YOU ARE" in prefix
    assert "I am Oríon" in prefix

    # Voice finalize: full self (incl. response policy banning generic framing).
    ctx = build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="I notice we're in the Orion-Sapienform repository — a fascinating project.",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="How are you?",
    )
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    rendered = env.from_string(template_path.read_text(encoding="utf-8")).render(**ctx)

    assert "You are Oríon" in rendered
    assert "I am Oríon, a digital mind in development with Juniper." in rendered
    assert "Juniper is my collaborator" in rendered
    assert "Last we spoke, we were wiring self-grounding" in rendered
    # The banned generic-assistant phrase is present as a policy instruction to avoid,
    # proving the voice pass is told not to reproduce it.
    assert "Avoid phrase" in rendered
