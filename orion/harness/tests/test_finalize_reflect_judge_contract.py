"""Finalize 5b judge: the call asks for a grade, under a schema, with the draft quoted as data.

Live 2026-11-03..11 (harness_turn_trace + cognition_traces), 74 of 1,689 judged turns fell
back to "reflect_llm_failed". 46 of them were the model not writing a FinalizeReflectionV1:
Bonsai thinking to the 8000-token cap, Bonsai misspelling keys ("imperial"), and
Qwen3.8-Flash-Next continuing a reading turn's JSON draft ("what_i_learned"). These tests pin
the request shape that removes those, and keep the failure path honest.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from jinja2 import Environment

from orion.harness.finalize import (
    FINALIZE_REFLECT_MODEL_FIELDS,
    FINALIZE_REFLECT_USER_TURN,
    build_finalize_reflect_context,
    finalize_reflect_json_schema,
    run_finalize_reflection,
)
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought
from orion.schemas.harness_finalize import REFLECTION_SOURCE_JUDGE_FAILED, FinalizeReflectionV1

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "orion/cognition/prompts/harness_finalize_reflect.j2"

# A reading turn's draft: the motor's own output contract, JSON with what_i_learned.
READING_DRAFT = json.dumps({
    "what_i_learned": "Fetched the page once. The paper says latent actions can be read out of a "
                      "pretrained video diffusion model.",
    "sources": ["https://example.org/paper"],
    "open_questions": ["Does it transfer across embodiments?"],
})
READING_REQUEST = ("Fetch the url below with WebFetch before answering. "
                   "A turn with no successful fetch of this url is discarded.\nurl=https://example.org/paper")

# What Flash-Next returned on 2026-11-05 (cognition_traces, verb harness_finalize_reflect):
# the draft's shape, continued, instead of a grade.
MOTOR_SHAPE_COPY = json.dumps({
    "what_i_learned": "Fetch log first: the exact seed URL failed twice ...",
    "sources": ["https://example.org/paper"],
})

GOOD_JUDGEMENT = {
    "imperative": "Report what the fetched paper says.",
    "tone": "steady",
    "strain_refs": [],
    "alignment_notes": ["draft reports the fetched page; receipts show one WebFetch"],
    "alignment_verdict": "aligned",
    "strain_unresolved": False,
    "world_contact_opportunity": False,
    "recommended_tool": None,
    "recommended_tool_reason": None,
    "promises_made": [],
    "offers_accepted": [],
    "promises_kept": [],
}


def _ctx(draft: str = READING_DRAFT, user: str = READING_REQUEST) -> dict:
    return build_finalize_reflect_context(
        correlation_id="c-judge", draft_text=draft, thought=make_thought(),
        substrate_appraisal=make_appraisal(), repair_overlay=make_repair_overlay(), user_message=user)


def _render(ctx: dict) -> str:
    return Environment(autoescape=False).from_string(TEMPLATE.read_text(encoding="utf-8")).render(**ctx)


def test_schema_is_the_judgement_fields_of_finalize_reflection_v1() -> None:
    schema = finalize_reflect_json_schema()
    model_fields = set(FinalizeReflectionV1.model_fields)
    assert set(schema["properties"]) == set(FINALIZE_REFLECT_MODEL_FIELDS) <= model_fields
    assert schema["required"] == list(FINALIZE_REFLECT_MODEL_FIELDS)
    assert schema["additionalProperties"] is False
    # Identity and provenance stay with the harness.
    for owned in ("correlation_id", "thought_event_id", "substrate_appraisal_id", "draft_hash",
                  "reflection_source", "quick_lane_skipped_llm", "finalize_changed", "schema_version"):
        assert owned not in schema["properties"]
    # Notes before verdict: with thinking off, the reasons are written first.
    order = list(schema["properties"])
    assert order.index("alignment_notes") < order.index("alignment_verdict")


def test_schema_refuses_the_live_failure_shapes_and_accepts_a_judgement() -> None:
    schema = finalize_reflect_json_schema()
    jsonschema.validate(GOOD_JUDGEMENT, schema)
    misspelled = {("imperial" if k == "imperative" else k): v for k, v in GOOD_JUDGEMENT.items()}
    partial = {k: v for k, v in GOOD_JUDGEMENT.items() if k != "strain_unresolved"}
    for bad in (json.loads(MOTOR_SHAPE_COPY), misspelled, partial):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)


def test_context_requests_schema_thinking_off_and_a_grade_turn() -> None:
    ctx = _ctx()
    assert ctx["structured_output_schema"] == finalize_reflect_json_schema()
    assert ctx["structured_output_method"] == "json_object_schema"
    assert ctx["structured_output_thinking_policy"] == "disabled_for_artifact"
    assert ctx["chat_template_kwargs"] == {"enable_thinking": False}
    assert ctx["messages"] == [{"role": "user", "content": FINALIZE_REFLECT_USER_TURN}]
    # Spark and gateway logs still see the turn's real request.
    assert ctx["raw_user_text"] == READING_REQUEST


def test_rendered_prompt_quotes_the_draft_and_request_as_data() -> None:
    rendered = _render(_ctx())
    draft_block = rendered.split("<draft_text>\n", 1)[1].split("\n</draft_text>", 1)[0]
    assert draft_block == READING_DRAFT
    request_block = rendered.split("<user_message>\n", 1)[1].split("\n</user_message>", 1)[0]
    assert request_block == READING_REQUEST
    assert "what_i_learned or summary" in rendered  # names the draft-shape trap
    # The required-fields list matches the schema the gateway enforces.
    required = rendered.split("\nREQUIRED JSON FIELDS (", 1)[1].split("\nFAIL-CLOSED", 1)[0]
    listed = [line[2:].split(":", 1)[0] for line in required.splitlines() if line.startswith("- ")]
    assert listed == list(FINALIZE_REFLECT_MODEL_FIELDS)


@pytest.mark.asyncio
async def test_motor_shape_copy_is_a_labelled_failure_never_a_verdict() -> None:
    """The 2026-11-05 shape: a reading draft in, the draft's own keys back out.

    Regression guard for the failure path (it passes on main too): whatever the request
    shape, a judge that writes the draft's keys must be a labelled failure, never a verdict."""
    async def _copies_the_draft(_req: object) -> dict[str, object]:
        return {"final_text": MOTOR_SHAPE_COPY}

    reflection, quick, _ = await run_finalize_reflection(
        correlation_id="c-copy", draft_text=READING_DRAFT, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=READING_REQUEST, cortex_client=_copies_the_draft)

    assert quick is False
    assert reflection.reflection_source == REFLECTION_SOURCE_JUDGE_FAILED
    assert reflection.alignment_verdict == "misaligned"
    assert reflection.alignment_notes[0].startswith("reflect_llm_failed:")


@pytest.mark.asyncio
async def test_schema_shaped_judgement_validates_and_the_harness_owns_provenance() -> None:
    sent: list = []

    async def _judges(req: object) -> dict[str, object]:
        sent.append(req)
        # A model that also tries to label itself; the harness overwrites provenance.
        return {"final_text": json.dumps({**GOOD_JUDGEMENT, "reflection_source": REFLECTION_SOURCE_JUDGE_FAILED,
                                          "quick_lane_skipped_llm": True})}

    reflection, quick, _ = await run_finalize_reflection(
        correlation_id="c-ok", draft_text=READING_DRAFT, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=READING_REQUEST, cortex_client=_judges)

    assert quick is False
    assert reflection.reflection_source == "substrate_informed_pass"
    assert reflection.quick_lane_skipped_llm is False
    assert reflection.alignment_verdict == "aligned"
    assert reflection.correlation_id == "c-ok"
    assert sent[0].context["structured_output_schema"]["required"] == list(FINALIZE_REFLECT_MODEL_FIELDS)
