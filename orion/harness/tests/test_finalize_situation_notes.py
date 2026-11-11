"""5b sees the Situation block the draft was written from (2026-11-03).

Live case: the notes said "Rowan usually comes around 11:39"; the draft said
"Rowan right on her usual 11:39"; asked where, the next draft invented a
vehicle "slot" it had "started calling Rowan". 5b called that draft
accurate: it never saw the notes, so it had nothing to check it against.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from orion.harness import finalize
from orion.harness.finalize import build_finalize_reflect_context, build_finalize_reflect_plan_request
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought

TEMPLATE = Path(finalize.__file__).resolve().parents[1] / "cognition" / "prompts" / "harness_finalize_reflect.j2"
NOTES = ("Situation:\n- Walkway rhythm: no match for Rowan (a person, name from Juniper, 10-25) yet today "
         "(usually around 11:39). A person match is the camera's guess, not a confirmed sighting.")


def _ctx(notes):
    return build_finalize_reflect_context(
        correlation_id="c-notes", draft_text="Rowan right on her usual 11:39.", thought=make_thought(),
        substrate_appraisal=make_appraisal(), repair_overlay=make_repair_overlay(),
        user_message="where did you spot Rowan?", situation_notes=notes,
    )


def _render(ctx) -> str:
    return Environment(autoescape=False).from_string(TEMPLATE.read_text()).render(**ctx)


def test_reflect_context_carries_the_situation_notes() -> None:
    assert _ctx(NOTES)["situation_notes"] == NOTES
    assert _ctx(None)["situation_notes"] == ""


def test_plan_request_threads_the_notes_through() -> None:
    req = build_finalize_reflect_plan_request(
        correlation_id="c-notes", draft_text="d", thought=make_thought(), substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(), user_message="u", situation_notes=NOTES,
    )
    assert req.context["situation_notes"] == NOTES


def test_a_runaway_fragment_is_capped() -> None:
    long = "x" * (finalize.FINALIZE_SITUATION_NOTES_MAX_CHARS + 50)
    got = _ctx(long)["situation_notes"]
    assert got.endswith("[notes cut]")
    assert len(got) <= finalize.FINALIZE_SITUATION_NOTES_MAX_CHARS + len(" [notes cut]")


def test_prompt_shows_the_notes_and_the_claims_check_only_when_there_are_notes() -> None:
    with_notes = _render(_ctx(NOTES))
    assert "no match for Rowan (a person, name from Juniper, 10-25)" in with_notes
    assert "PERCEPTION CLAIMS CHECK" in with_notes
    assert "unsupported claim:" in with_notes
    without = _render(_ctx(None))
    assert "PERCEPTION CLAIMS CHECK" not in without
    assert "situation_notes" not in without


REPAIR_TEMPLATE = TEMPLATE.parent / "orion_response_repair.j2"


def test_repair_gets_the_same_notes_and_the_unsupported_claims_rule() -> None:
    from orion.harness.finalize import build_response_repair_context, build_response_repair_plan_request
    from orion.harness.tests.fixtures import make_reflection

    ctx = build_response_repair_context(
        correlation_id="c-notes", draft_text="d", reflection=make_reflection(), user_message="u",
        situation_notes=NOTES,
    )
    assert ctx["situation_notes"] == NOTES
    req = build_response_repair_plan_request(
        correlation_id="c-notes", draft_text="d", reflection=make_reflection(), user_message="u",
        situation_notes=NOTES,
    )
    assert req.context["situation_notes"] == NOTES
    rendered = Environment(autoescape=False).from_string(REPAIR_TEMPLATE.read_text()).render(**ctx)
    assert "SITUATION NOTES" in rendered and "UNSUPPORTED CLAIMS" in rendered
    bare = Environment(autoescape=False).from_string(REPAIR_TEMPLATE.read_text()).render(
        **build_response_repair_context(correlation_id="c", draft_text="d", reflection=make_reflection(), user_message="u"))
    assert "UNSUPPORTED CLAIMS" not in bare
