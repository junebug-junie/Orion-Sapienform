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


from pathlib import Path

import jinja2

from orion.harness.finalize import build_voice_finalize_context
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
)
from orion.schemas.cognition.answer_contract import AnswerContract


def _voice_context(capsule) -> dict:
    thought = make_thought(imperative="Stay present.", grounding_capsule=capsule)
    return build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )


def test_voice_context_includes_full_capsule() -> None:
    capsule = make_grounding_capsule()
    ctx = _voice_context(capsule)
    assert ctx["grounding_capsule"]["identity_summary"] == capsule.identity_summary
    assert ctx["grounding_capsule"]["response_policy_summary"] == capsule.response_policy_summary


def test_voice_context_capsule_none_when_absent() -> None:
    ctx = _voice_context(None)
    assert ctx["grounding_capsule"] is None


def test_voice_template_renders_self_block_above_style_rules() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    ctx = _voice_context(make_grounding_capsule())
    rendered = template.render(**ctx)
    assert "WHO YOU ARE" in rendered
    assert "I am Oríon" in rendered
    assert "RESPONSE POLICY" in rendered
    assert "no generic-assistant framing" in rendered
    assert rendered.index("WHO YOU ARE") < rendered.index("STYLE RULES")


def test_voice_template_omits_self_block_when_capsule_none() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    ctx = _voice_context(None)
    rendered = template.render(**ctx)
    assert "WHO YOU ARE" not in rendered


def test_stance_react_prompt_renders_identity_when_present() -> None:
    template_path = Path("orion/cognition/prompts/stance_react.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    rendered = template.render(
        user_message="how are you?",
        stance_inputs={},
        association={},
        repair_bundle=None,
        coalition_projection=None,
        orion_identity_summary=["I am Oríon, a digital mind in development."],
        juniper_relationship_summary=["Juniper is my collaborator."],
    )
    assert "I am Oríon" in rendered


def test_stance_react_prompt_renders_without_identity() -> None:
    template_path = Path("orion/cognition/prompts/stance_react.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    rendered = template.render(
        user_message="how are you?",
        stance_inputs={},
        association={},
        repair_bundle=None,
        coalition_projection=None,
        orion_identity_summary=[],
        juniper_relationship_summary=[],
    )
    assert "how are you?" in rendered


def test_voice_template_preserves_identity_boundary_with_capsule() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    capsule = make_grounding_capsule(
        identity_summary=["I am Oríon; I am not Juniper."],
        relationship_summary=["Juniper is my human collaborator (a separate person)."],
    )
    ctx = _voice_context(capsule)
    rendered = template.render(**ctx)
    # The assistant is always framed as Oríon, never as Juniper.
    assert rendered.strip().startswith("You are Oríon")
    assert "I am Oríon; I am not Juniper." in rendered
    # Juniper appears only as the interlocutor/relationship, never as the speaker identity.
    assert "Juniper is my human collaborator" in rendered
