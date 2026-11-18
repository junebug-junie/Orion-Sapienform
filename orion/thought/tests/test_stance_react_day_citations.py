"""Temporal Self patch 4: the stance's references to Orion's own day go through the citation gate."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1
from orion.schemas.temporal_self import TemporalSelfStanceCueV1
from orion.schemas.thought import HubAssociationBundleV1, StanceHarnessSliceV1, StanceReactRequestV1, ThoughtEventV1
from orion.thought.stance_react import apply_stance_react_pipeline, parse_stance_react_payload

NOW = datetime(2026, 10, 11, 17, 20, tzinfo=timezone.utc)
SHOWN = "ae0a3cb57ae73ee4"
CUE = TemporalSelfStanceCueV1(
    text=f"Morning, Sun 11:20 local (day record as of 11:16). On now: attention on Bus synaptic prediction "
         f"error for 28 min [arc:{SHOWN}].",
    day_id="2026-10-11", rendered_at=NOW, arc_ids=[SHOWN],
    labels={"Bus synaptic prediction error": [SHOWN]},
)


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how is your day?",
        association=HubAssociationBundleV1(
            correlation_id="c-1", read_source="felt_state_reader", broadcast_stale=False,
            broadcast=AttentionBroadcastProjectionV1(frame=AttentionFrameV1(), attended_node_ids=["node-a"]),
        ),
        repair_bundle=None, stance_inputs={"user_message": "how is your day?"},
    )


def _thought(**overrides) -> ThoughtEventV1:
    base = dict(
        event_id="t-1", correlation_id="c-1", session_id="s-1", created_at=NOW,
        imperative="Answer warmly.", tone="present", strain_refs=[], evidence_refs=["node-a"],
        stance_harness_slice=StanceHarnessSliceV1(task_mode="reflective_dialogue", conversation_frame="reflective",
                                                  answer_strategy="share"),
    )
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def test_shown_arc_in_evidence_is_kept_and_proceeds():
    out = apply_stance_react_pipeline(_thought(evidence_refs=["node-a", f"arc:{SHOWN}"]), _request(), day_cue=CUE)
    assert out.evidence_refs == ["node-a", f"arc:{SHOWN}"]
    assert out.disposition == "proceed"
    assert out.day_citations.cited_valid == [f"arc:{SHOWN}"] and out.day_citations.label == "verified"


def test_unknown_arc_is_removed_and_counted_everywhere():
    out = apply_stance_react_pipeline(_thought(
        evidence_refs=["node-a", "arc:deadbeefdeadbeef"], strain_refs=["arc:0123456789abcdef"],
        imperative="Mention the morning [arc:feedfacefeedface] briefly.",
    ), _request(), day_cue=CUE)
    assert out.evidence_refs == ["node-a"] and out.strain_refs == []
    assert out.imperative == "Mention the morning briefly."
    assert out.day_citations.rejected_unknown == [
        "arc:deadbeefdeadbeef", "arc:0123456789abcdef", "arc:feedfacefeedface"]
    assert out.day_citations.action == "stripped"
    assert out.disposition == "proceed"  # never blocked


def test_unknown_arc_in_strain_refs_cannot_launder_itself():
    out = apply_stance_react_pipeline(_thought(evidence_refs=["arc:deadbeefdeadbeef"],
                                               strain_refs=["arc:deadbeefdeadbeef"]), _request(), day_cue=CUE)
    assert "arc:deadbeefdeadbeef" not in out.evidence_refs + out.strain_refs


def test_arc_refs_without_a_cue_are_all_unknown():
    out = apply_stance_react_pipeline(_thought(evidence_refs=["node-a", f"arc:{SHOWN}"]), _request(), day_cue=None)
    assert out.evidence_refs == ["node-a"]
    assert out.day_citations.rejected_unknown == [f"arc:{SHOWN}"] and out.day_citations.shown == 0


def test_uncited_day_sentence_is_labelled_unverified_not_blocked():
    out = apply_stance_react_pipeline(
        _thought(imperative="Say the Bus synaptic prediction error held me all morning."), _request(), day_cue=CUE)
    assert out.disposition == "proceed"
    assert out.day_citations.label == "UNVERIFIED" and out.day_citations.uncited_day_sentences == 1
    assert out.imperative == "Say the Bus synaptic prediction error held me all morning."


def test_label_cited_through_evidence_is_verified():
    out = apply_stance_react_pipeline(_thought(
        imperative="Say the Bus synaptic prediction error held me all morning.",
        evidence_refs=["node-a", f"arc:{SHOWN}"]), _request(), day_cue=CUE)
    assert out.day_citations.label == "verified"


def test_cue_shown_but_unused_records_no_day_claims():
    out = apply_stance_react_pipeline(_thought(), _request(), day_cue=CUE)
    assert out.day_citations.label == "no_day_claims" and out.day_citations.shown == 1


def test_no_cue_no_arc_leaves_day_citations_none():
    assert apply_stance_react_pipeline(_thought(), _request()).day_citations is None
    skipped = CUE.model_copy(update={"text": "", "skip_reason": "stale"})
    assert apply_stance_react_pipeline(_thought(), _request(), day_cue=skipped).day_citations is None


def test_model_supplied_day_citations_is_dropped():
    raw = {"imperative": "x", "tone": "y", "strain_refs": [], "evidence_refs": ["hub:turn"],
           "stance_harness_slice": {"task_mode": "direct_response", "conversation_frame": "mixed",
                                    "answer_strategy": "d"},
           "day_citations": {"label": "verified", "cited_valid": ["arc:deadbeefdeadbeef"]}}
    assert parse_stance_react_payload(raw, correlation_id="c-1").day_citations is None


def test_bracketed_evidence_ref_from_the_live_model_is_cited_not_dropped():
    """Live 2026-10-12 (qwen 35B, relational turn): evidence_refs came back as '[arc:<id>]'."""
    out = apply_stance_react_pipeline(_thought(evidence_refs=["node-a", f"[arc:{SHOWN}]"]), _request(), day_cue=CUE)
    assert out.evidence_refs == ["node-a", f"arc:{SHOWN}"]
    assert out.day_citations.cited_valid == [f"arc:{SHOWN}"] and out.day_citations.label == "verified"


def test_stored_check_describes_the_text_that_ships():
    """Enforcement runs after the gate; it must not rewrite imperative/tone, or the stored
    uncited examples would describe text that never shipped (review finding)."""
    imperative = "Say the Bus synaptic prediction error held me all morning."
    out = apply_stance_react_pipeline(_thought(imperative=imperative), _request(), day_cue=CUE)
    assert out.imperative == imperative
    assert out.day_citations.uncited_examples == [imperative]
