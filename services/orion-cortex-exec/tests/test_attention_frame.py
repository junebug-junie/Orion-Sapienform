from __future__ import annotations

from app.attention_frame import build_attention_frame
from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.detectors.autonomy import AutonomySignalDetector


class _FakeDetector:
    detector_id = "fake_detector"

    def detect(self, ctx, inputs, belief_lineage):
        return [
            AttentionSignalV1(
                signal_id="fake-signal-1",
                source=self.detector_id,
                target_text="NonRegex Observatory",
                target_type_hint="concept",
                signal_kind="fake_open_loop",
                salience=0.9,
                confidence=0.9,
                evidence_refs=["fake:evidence"],
                provenance={"belief_lineage": belief_lineage},
            )
        ]


def _inputs(**overrides):
    base = {
        "identity": {"orion": [], "juniper": [], "response_policy": []},
        "concept_induction": {"self": [], "relationship": [], "growth": [], "tension": []},
        "social": {"social_posture": [], "relationship_facets": [], "hazards": []},
        "reflective": {"themes": [], "tensions": [], "dream_motifs": []},
        "autonomy": {"summary": {"top_drives": [], "active_tensions": []}, "debug": {}},
        "reasoning_summary": {"hazards": [], "tensions": [], "fallback_recommended": False},
        "situation": {},
    }
    base.update(overrides)
    return base


def test_novel_unresolved_activity_creates_open_loop() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "I am debugging the carrier board bringup around the LVDS rail."},
        inputs=_inputs(),
        belief_lineage=["recall:snapshot_ephemeral"],
    )
    assert frame.open_loops
    assert any("carrier board" in loop.description.lower() or "lvds" in loop.description.lower() for loop in frame.open_loops)
    assert frame.debug["belief_lineage"] == ["recall:snapshot_ephemeral"]


def test_generic_reciprocity_is_suppressed() -> None:
    frame = build_attention_frame(ctx={"user_message": "What about you?"}, inputs=_inputs())
    assert any(s.reason == "generic_reciprocity" for s in frame.suppressions)
    assert frame.selected_action is not None
    assert frame.selected_action.action_type != "ask"


def test_already_known_fact_suppresses_redundant_question() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "Tell me about Project Silver Loom", "memory_digest": "Project Silver Loom is already known."},
        inputs=_inputs(),
    )
    assert any(loop.already_known for loop in frame.open_loops)
    assert any(s.reason == "already_known" for s in frame.suppressions)


def test_high_value_open_loop_selects_single_ask() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "I am planning next week's migration around Zephyr Bridge."},
        inputs=_inputs(
            autonomy={
                "summary": {"top_drives": ["predictive", "continuity"], "active_tensions": ["tension.continuity_gap.v1"]},
                "debug": {},
            },
        ),
    )
    asks = [a for a in frame.candidate_actions if a.action_type == "ask"]
    assert frame.selected_action is not None
    assert frame.selected_action.action_type == "ask"
    assert len([a for a in asks if a.question_text]) <= 1


def test_low_value_open_loop_selects_non_ask() -> None:
    frame = build_attention_frame(ctx={"user_message": "Please implement this plan for Orion."}, inputs=_inputs())
    assert frame.selected_action is not None
    assert frame.selected_action.action_type in {"watch", "defer", "suppress", "none"}
    assert any(s.reason == "user_needs_direct_answer" for s in frame.suppressions)


def test_concept_and_autonomy_pressure_influence_ranking() -> None:
    low = build_attention_frame(ctx={"user_message": "I am exploring Blue Lattice."}, inputs=_inputs())
    high = build_attention_frame(
        ctx={"user_message": "I am exploring Blue Lattice."},
        inputs=_inputs(
            concept_induction={"self": ["lattice coherence"], "relationship": [], "growth": [], "tension": ["fragmentation"]},
            autonomy={
                "summary": {"top_drives": ["coherence", "predictive"], "active_tensions": ["tension.coherence_break.v1"]},
                "state_v2": {"attention_items": [{"summary": "dominant_drive=coherence"}]},
                "debug": {},
            },
        ),
    )
    assert high.candidate_actions[0].score >= low.candidate_actions[0].score


def test_absent_upstream_signals_fail_open_without_noisy_question() -> None:
    frame = build_attention_frame(ctx={"user_message": ""}, inputs={})
    assert frame.open_loops == []
    assert frame.selected_action is not None
    assert frame.selected_action.action_type == "none"

def test_detector_registry_accepts_fake_detector_without_regex() -> None:
    frame = build_attention_frame(ctx={"user_message": ""}, inputs=_inputs(), detectors=[_FakeDetector()])
    assert frame.open_loops
    assert frame.open_loops[0].description == "NonRegex Observatory"
    assert frame.open_loops[0].provenance["signal_source"] == "fake_detector"


def test_autonomy_detector_emits_attention_item_signal() -> None:
    signals = AutonomySignalDetector().detect(
        {},
        _inputs(
            autonomy={
                "summary": {"top_drives": [], "active_tensions": []},
                "state_v2": {"attention_items": [{"summary": "dominant_drive=coherence", "salience": 0.81}]},
                "debug": {},
            }
        ),
        ["autonomy:graphdb_durable"],
    )
    assert signals
    assert signals[0].source == "autonomy_attention_v1"
    assert signals[0].target_text == "dominant_drive=coherence"
