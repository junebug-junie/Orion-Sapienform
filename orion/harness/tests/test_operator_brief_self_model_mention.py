"""Patch A of docs/superpowers/specs/2026-09-10-self-report-tool-discipline-design.md:
the harness motor's unified operator brief now names the self-model Postgres
tables and the credential that already reaches every turn. See that design
doc's Arsonist summary for the live failure this closes (asked "what can't
you do right now," Orion answered with generic-assistant boilerplate because
nothing told it these records existed).

This mention is prompt-only -- it does not touch stance or the relational/
instrumental gate (`is_relational_motor_stance`). A relational-bucket turn
still gets told not to reach for tools at all; these tests confirm that split
is untouched.
"""

from __future__ import annotations

import pytest

from orion.curiosity.self_inquiry import SELF_INQUIRY_PG_TABLES
from orion.harness.operator_brief import (
    HARNESS_RELATIONAL_TOOL_DISCIPLINE,
    HARNESS_SELF_MODEL_ACCESS_BRIEF,
    HARNESS_UNIFIED_OPERATOR_BRIEF,
    harness_motor_instruction,
)
from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1


def test_unified_brief_names_the_credential_and_both_tables() -> None:
    assert "$ORION_CURIOSITY_PG_DSN" in HARNESS_UNIFIED_OPERATOR_BRIEF
    assert "self_concept_history" in HARNESS_UNIFIED_OPERATOR_BRIEF
    assert "self_sense_eval_log" in HARNESS_UNIFIED_OPERATOR_BRIEF


def test_mention_reuses_self_inquiry_descriptions_not_a_hardcoded_copy() -> None:
    """Ties the brief to self_inquiry.py's own account of these tables so the
    two cannot silently drift -- if a description there changes, this mention
    changes with it instead of going stale."""
    descriptions = dict(SELF_INQUIRY_PG_TABLES)
    assert descriptions["self_concept_history"] in HARNESS_SELF_MODEL_ACCESS_BRIEF
    assert descriptions["self_sense_eval_log"] in HARNESS_SELF_MODEL_ACCESS_BRIEF


def test_a_renamed_source_table_fails_loud_not_silent() -> None:
    """If SELF_INQUIRY_PG_TABLES ever drops one of the two named tables, the
    mention must break at build time, not quietly render without it."""
    import orion.harness.operator_brief as operator_brief

    original = dict(operator_brief._SELF_INQUIRY_TABLE_DESCRIPTIONS)
    try:
        operator_brief._SELF_INQUIRY_TABLE_DESCRIPTIONS.pop("self_sense_eval_log", None)
        with pytest.raises(KeyError):
            operator_brief._self_model_access_line()
    finally:
        operator_brief._SELF_INQUIRY_TABLE_DESCRIPTIONS.clear()
        operator_brief._SELF_INQUIRY_TABLE_DESCRIPTIONS.update(original)


def test_repo_and_runtime_briefs_are_unaffected() -> None:
    """Patch A only touches the unified brief -- the repo/technical and
    runtime/debug briefs (used by other harness entry points) are untouched."""
    from orion.harness.operator_brief import (
        HARNESS_REPO_OPERATOR_BRIEF,
        HARNESS_RUNTIME_OPERATOR_BRIEF,
    )

    assert "ORION_CURIOSITY_PG_DSN" not in HARNESS_REPO_OPERATOR_BRIEF
    assert "ORION_CURIOSITY_PG_DSN" not in HARNESS_RUNTIME_OPERATOR_BRIEF


def test_relational_bucket_still_forbids_tools_regardless_of_the_mention() -> None:
    """The mention reaches every unified turn via compile_harness_prefix, but
    a relational-bucket turn's motor *instruction* is untouched -- it still
    says don't reach for tools. Patch A does not decide that; Patch B would."""
    thought = make_thought(
        imperative="Just be present.",
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="reflective",
            interaction_regime="relational",
            answer_strategy="companion_presence",
        ),
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you feeling today?",
    )
    instruction = harness_motor_instruction(thought=thought)

    assert "$ORION_CURIOSITY_PG_DSN" in prompt  # the door is mentioned
    assert HARNESS_RELATIONAL_TOOL_DISCIPLINE.strip() in instruction  # but still closed here
