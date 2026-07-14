from __future__ import annotations

import pytest

from app.grounding_capsule import (
    assemble_stance_grounding,
    build_grounding_capsule,
    context_provenance_for_ctx,
    stance_slice_brief_from_step_text,
)
from app.settings import Settings


def test_build_grounding_capsule_from_ctx() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": ["Juniper is my collaborator."],
        "response_policy_summary": ["Speak plainly."],
        "continuity_digest": "We were mid-refactor.",
        "belief_digest": "Orion values continuity.",
        "memory_digest": "We were mid-refactor.\n\nOrion values continuity.",
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=True)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.relationship_summary == ["Juniper is my collaborator."]
    assert capsule.response_policy_summary == ["Speak plainly."]
    assert capsule.memory_digest == "We were mid-refactor.\n\nOrion values continuity."
    assert capsule.provenance["identity_source"] == "configured_yaml"
    assert capsule.provenance["pcr_ran"] is True
    assert capsule.context_provenance["orion_identity_summary"] == "static_identity_config"
    assert capsule.context_provenance["belief_digest"] == "derived_summary"
    assert capsule.context_provenance["memory_digest"] == "memory_recall"


def test_context_provenance_for_ctx_only_includes_present_keys() -> None:
    ctx = {"self_state": {"overall_condition": "steady"}, "user_message": "hi"}
    provenance = context_provenance_for_ctx(ctx)
    assert provenance == {
        "self_state": "live_runtime_projection",
        "user_message": "user_input",
    }


def test_context_provenance_for_ctx_omits_unregistered_keys() -> None:
    ctx = {"some_new_key_nobody_classified_yet": "value"}
    assert context_provenance_for_ctx(ctx) == {}


def test_build_grounding_capsule_identity_only_when_pcr_missing() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": [],
        "response_policy_summary": [],
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=False)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.continuity_digest is None
    assert capsule.belief_digest is None
    assert capsule.memory_digest is None
    assert capsule.provenance["pcr_ran"] is False


def test_stance_slice_brief_from_step_text_extracts_mode_and_frame() -> None:
    text = (
        '{"imperative":"Stay present.","tone":"warm",'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )
    brief = stance_slice_brief_from_step_text(text)
    assert brief["task_mode"] == "reflective_dialogue"
    assert brief["conversation_frame"] == "reflective"


def test_stance_slice_brief_from_step_text_tolerates_garbage() -> None:
    assert stance_slice_brief_from_step_text("not json") == {}
    assert stance_slice_brief_from_step_text("") == {}


@pytest.mark.asyncio
async def test_assemble_ships_identity_only_capsule_when_pcr_raises(monkeypatch) -> None:
    from app import grounding_capsule as gc
    from app.settings import Settings

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("pcr down")

    monkeypatch.setattr(gc, "run_pcr_phase3", _boom)
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=True, CHAT_PCR_ENABLED=True)
    ctx: dict = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": ["Juniper is my collaborator."],
        "response_policy_summary": ["Speak plainly."],
        "identity_kernel_source": "configured_yaml",
    }
    capsule = await assemble_stance_grounding(
        bus=None,
        source=None,
        ctx=ctx,
        correlation_id="c-1",
        recall_cfg={},
        stance_step_text='{"stance_harness_slice":{"task_mode":"reflective_dialogue","conversation_frame":"reflective"}}',
        exec_settings=cfg,
    )
    assert capsule is not None
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.provenance["pcr_ran"] is False
    assert capsule.memory_digest is None


@pytest.mark.asyncio
async def test_assemble_returns_none_when_flag_off() -> None:
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=False)
    ctx: dict = {"orion_identity_summary": ["I am Oríon."]}
    result = await assemble_stance_grounding(
        bus=None,
        source=None,
        ctx=ctx,
        correlation_id="c-1",
        recall_cfg={},
        stance_step_text="{}",
        exec_settings=cfg,
    )
    assert result is None
    # Flag off ⇒ short-circuit before any stance-brief / PCR side effect on ctx.
    assert "grounding_capsule" not in ctx
    assert "chat_stance_brief" not in ctx
