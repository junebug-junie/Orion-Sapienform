from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.thought.stance_quality import enforce_thought_stance_quality


def _relational_thought() -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-rel",
        correlation_id="c-rel",
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="Stay present; one situated question.",
        tone="warm",
        strain_refs=["loop-1"],
        evidence_refs=["loop-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="reflective",
            response_priorities=["companion_presence", "situated_curiosity", "hold_space"],
            response_hazards=["avoid_task_tracking", "avoid_next_steps"],
            answer_strategy="RelationalHoldSpace",
        ),
    )


def test_stance_react_relational_survives_compressor() -> None:
    thought = _relational_thought()
    stance_inputs = {"user_message": "just talk to me, im lonely"}
    enriched, changed = enforce_thought_stance_quality(thought, stance_inputs)
    assert "companion_presence" in enriched.stance_harness_slice.response_priorities
    assert "avoid_task_tracking" in enriched.stance_harness_slice.response_hazards
    assert enriched.stance_harness_slice.answer_strategy == "RelationalHoldSpace"
    assert changed is False


def test_stance_react_instrumental_compression_preserved() -> None:
    thought = ThoughtEventV1(
        event_id="t-tri",
        correlation_id="c-tri",
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="List concrete file paths and line numbers.",
        tone="direct",
        strain_refs=["node-x"],
        evidence_refs=["node-x"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="triage",
            conversation_frame="mixed",
            response_priorities=["triage_operational_blockers_first"],
            answer_strategy="direct",
        ),
    )
    enriched, _ = enforce_thought_stance_quality(thought, {"user_message": "deploy is broken again"})
    assert enriched.stance_harness_slice.task_mode == "triage"
    assert "triage_operational_blockers_first" in enriched.stance_harness_slice.response_priorities
    assert "self_intro_on_operational_turn" in enriched.stance_harness_slice.response_hazards
    assert "generic_sympathy_script" in enriched.stance_harness_slice.response_hazards


def test_stance_react_playful_exchange_survives_compressor() -> None:
    thought = _relational_thought()
    thought = thought.model_copy(
        update={
            "stance_harness_slice": thought.stance_harness_slice.model_copy(
                update={
                    "task_mode": "playful_exchange",
                    "conversation_frame": "playful_relational",
                }
            )
        }
    )
    enriched, changed = enforce_thought_stance_quality(thought, {"user_message": "someone to talk. im lonely"})
    assert enriched.stance_harness_slice.task_mode == "playful_exchange"
    assert "companion_presence" in enriched.stance_harness_slice.response_priorities
    assert changed is False


def test_stance_react_empty_imperative_defers() -> None:
    thought = _relational_thought()
    thought = thought.model_copy(update={"imperative": "   "})
    enriched, changed = enforce_thought_stance_quality(thought, {"user_message": "hey"})
    assert enriched.disposition == "defer"
    assert "empty_imperative" in enriched.disposition_reasons
    assert changed is True


def test_stance_react_missing_evidence_refs_defers() -> None:
    thought = _relational_thought()
    enriched, changed = enforce_thought_stance_quality(
        thought.model_copy(update={"evidence_refs": []}),
        {"user_message": "hey"},
    )
    assert enriched.disposition == "defer"
    assert "missing_evidence_refs" in enriched.disposition_reasons
    assert changed is True
