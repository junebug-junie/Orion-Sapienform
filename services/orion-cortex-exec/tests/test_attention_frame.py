from __future__ import annotations

from app.attention_frame import build_attention_frame
from orion.schemas.attention_frame import AttentionSignalV1


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
    # CurrentTurnSignalDetector no longer extracts phrases from user_text
    # itself (that was the deleted LegacyRegexSignalDetector's job) -- it
    # reads pre-populated LLM-judged candidates from
    # ctx["current_turn_llm_signals"] (populated by
    # services/orion-cortex-exec/app/current_turn_llm_signals.py, upstream
    # of build_attention_frame() in the real chat_stance.py flow).
    frame = build_attention_frame(
        ctx={
            "user_message": "I am debugging the carrier board bringup around the LVDS rail.",
            "current_turn_llm_signals": [{"phrase": "carrier board bringup", "type": "activity"}],
        },
        inputs=_inputs(),
        belief_lineage=["recall:snapshot_ephemeral"],
    )
    assert frame.open_loops
    assert any("carrier board" in loop.description.lower() for loop in frame.open_loops)
    assert frame.debug["belief_lineage"] == ["recall:snapshot_ephemeral"]


_READ_SHARING = {"ok": True, "wants_direct_answer": False}
_READ_DIRECT = {"ok": True, "wants_direct_answer": True}


class _BackgroundDetector:
    """Stands in for Orion's own background threads (concept/situation detectors),
    e.g. the "Biometrics prediction error" loop that was the only candidate on
    corr beab81a3."""

    detector_id = "concept_induction_v1"

    def __init__(self, salience: float = 0.9, confidence: float = 0.9) -> None:
        self._salience = salience
        self._confidence = confidence

    def detect(self, ctx, inputs, belief_lineage):
        return [
            AttentionSignalV1(
                signal_id="background-signal-1",
                source=self.detector_id,
                target_text="Biometrics prediction error",
                target_type_hint="anomaly",
                signal_kind="concept_pressure",
                salience=self._salience,
                confidence=self._confidence,
                evidence_refs=["concept:biometrics", "substrate:biometrics"],
            )
        ]


def _detectors_with_background(**kw):
    from orion.substrate.attention.detectors import CurrentTurnSignalDetector

    return [CurrentTurnSignalDetector(), _BackgroundDetector(**kw)]


def test_question_to_orion_is_answered_not_deflected_into_curiosity() -> None:
    """Replaces the generic-reciprocity regex: the turn read judges that the user
    asked Orion something, so Orion's own background threads are not askable."""
    frame = build_attention_frame(
        ctx={"user_message": "What about you?", "current_turn_llm_read": _READ_DIRECT},
        inputs=_inputs(),
        detectors=_detectors_with_background(),
    )
    assert any(s.reason == "user_needs_direct_answer" for s in frame.suppressions)
    assert frame.selected_action is not None
    assert frame.selected_action.action_type != "ask"


def test_shared_life_news_selects_the_natural_follow_up_as_the_ask() -> None:
    frame = build_attention_frame(
        ctx={
            "user_message": "I'll be pretty busy the next few days with work travel.",
            "current_turn_llm_read": _READ_SHARING,
            "current_turn_llm_signals": [
                {"phrase": "work travel", "type": "plan", "natural_question": "Where are you headed?"}
            ],
        },
        inputs=_inputs(),
        detectors=_detectors_with_background(),
    )
    assert frame.selected_action is not None
    assert frame.selected_action.action_type == "ask"
    assert frame.selected_action.question_text == "Where are you headed?"


def test_shared_life_news_is_askable_even_with_nothing_competing() -> None:
    """A lone current-turn loop's Borda fallback score tops out near 0.5 -- below
    min_ask 0.65 -- so on a quiet turn a disclosure could never become a question."""
    frame = build_attention_frame(
        ctx={
            "user_message": "finally signed up for that pottery class",
            "current_turn_llm_read": _READ_SHARING,
            "current_turn_llm_signals": [
                {"phrase": "pottery class", "type": "activity", "natural_question": "What made you finally do it?"}
            ],
        },
        inputs=_inputs(),
    )
    assert frame.selected_action.action_type == "ask"
    assert frame.selected_action.question_text == "What made you finally do it?"


def test_follow_up_on_what_juniper_shared_outranks_orions_own_background_thread() -> None:
    frame = build_attention_frame(
        ctx={
            "user_message": "my sister's coming to stay this weekend",
            "current_turn_llm_read": _READ_SHARING,
            "current_turn_llm_signals": [
                {"phrase": "sister's visit", "type": "person", "natural_question": "How long is she staying?"}
            ],
        },
        inputs=_inputs(),
        detectors=_detectors_with_background(salience=1.0, confidence=1.0),
    )
    assert frame.selected_action.question_text == "How long is she staying?"


def test_direct_request_still_allows_follow_up_on_what_they_shared() -> None:
    """'traveling this week, any tips?' -- answer the request, and a friend's
    follow-up about the trip is still fair game; Orion's background threads are not."""
    frame = build_attention_frame(
        ctx={
            "user_message": "traveling for work this week, any packing tips?",
            "current_turn_llm_read": _READ_DIRECT,
            "current_turn_llm_signals": [
                {"phrase": "work trip this week", "type": "plan", "natural_question": "Where's the trip taking you?"}
            ],
        },
        inputs=_inputs(),
        detectors=_detectors_with_background(salience=1.0, confidence=1.0),
    )
    assert frame.selected_action.action_type == "ask"
    assert frame.selected_action.question_text == "Where's the trip taking you?"
    background_loop = next(loop for loop in frame.open_loops if "Biometrics" in loop.description)
    assert background_loop.askability <= 0.25, "direct turn must make Orion's background thread unaskable"
    background_action = next(a for a in frame.candidate_actions if a.open_loop_id == background_loop.id)
    assert background_action.action_type != "ask"
    assert "at most one selected ask" not in background_action.rationale, (
        "background loop must be held back by the direct turn, not merely out-competed for the one ask slot"
    )


def test_question_mark_alone_no_longer_suppresses_curiosity() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "had the best ramen of my life today, ever been obsessed with a food?", "current_turn_llm_read": _READ_SHARING},
        inputs=_inputs(),
    )
    assert not any(s.reason == "user_needs_direct_answer" for s in frame.suppressions)


def test_unavailable_turn_read_fails_closed_for_background_curiosity() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "so far so good", "current_turn_llm_read": {"ok": False, "wants_direct_answer": None}},
        inputs=_inputs(),
        detectors=_detectors_with_background(salience=1.0, confidence=1.0),
    )
    assert any(
        s.reason == "user_needs_direct_answer" and s.target_ref == "turn_read_unavailable" for s in frame.suppressions
    )
    assert frame.selected_action.action_type != "ask"


def test_no_user_text_adds_no_turn_read_suppression() -> None:
    frame = build_attention_frame(ctx={"user_message": ""}, inputs=_inputs(), detectors=[_FakeDetector()])
    assert not any(s.reason == "user_needs_direct_answer" for s in frame.suppressions)


def test_already_known_fact_suppresses_redundant_question() -> None:
    frame = build_attention_frame(
        ctx={
            "user_message": "Tell me about Project Silver Loom",
            "memory_digest": "Project Silver Loom is already known.",
            "current_turn_llm_signals": [{"phrase": "Project Silver Loom", "type": "concept"}],
        },
        inputs=_inputs(),
    )
    assert any(loop.already_known for loop in frame.open_loops)
    assert any(s.reason == "already_known" for s in frame.suppressions)


def test_open_loop_without_autonomy_boost_stays_below_ask_threshold() -> None:
    # Renamed from test_high_value_open_loop_selects_single_ask (2026-07-30,
    # chore/delete-orion-drives): the old assertion relied on
    # AutonomySignalDetector emitting a salience-boosting signal from
    # drive-tension data to push this open loop over min_ask_score. That
    # detector (and its only inputs, AutonomyStateV2.attention_items /
    # AutonomySummaryV1.top_drives/active_tensions) is retired -- nothing
    # produces autonomy-boosted salience anymore, so this open loop now
    # genuinely scores below the ask threshold.
    #
    # Renamed again (kill-legacy-regex-attention-detector patch): with a
    # single LLM-judged candidate (the new detector's real shape -- see
    # CurrentTurnSignalDetector) rather than the old regex detector's 2-3
    # overlapping same-message candidates (activity/named/proper_phrase all
    # firing on one sentence), the Borda rank-aggregated score for a lone
    # loop lands just above the watch/defer line (0.48) instead of below
    # it -- see policy.py's own note on select_actions() that these cutoffs
    # are absolute thresholds against a fundamentally relative rank score,
    # not yet recalibrated. The material behavior this test protects --
    # never promoted to "ask" without an autonomy boost -- is unchanged.
    frame = build_attention_frame(
        ctx={
            "user_message": "I am planning next week's migration around Zephyr Bridge.",
            "current_turn_llm_signals": [{"phrase": "Zephyr Bridge", "type": "concept"}],
        },
        inputs=_inputs(),
    )
    asks = [a for a in frame.candidate_actions if a.action_type == "ask"]
    assert frame.selected_action is not None
    assert frame.selected_action.action_type in {"watch", "defer"}
    assert frame.selected_action.score < 0.65  # below min_ask
    assert len([a for a in asks if a.question_text]) <= 1


def test_low_value_open_loop_selects_non_ask() -> None:
    frame = build_attention_frame(
        ctx={"user_message": "Please implement this plan for Orion.", "current_turn_llm_read": _READ_DIRECT},
        inputs=_inputs(),
    )
    assert frame.selected_action is not None
    assert frame.selected_action.action_type in {"watch", "defer", "suppress", "none"}
    assert any(s.reason == "user_needs_direct_answer" for s in frame.suppressions)


def test_concept_and_autonomy_pressure_influence_ranking() -> None:
    signals_ctx = {
        "user_message": "I am exploring Blue Lattice.",
        "current_turn_llm_signals": [{"phrase": "Blue Lattice", "type": "concept"}],
    }
    low = build_attention_frame(ctx=dict(signals_ctx), inputs=_inputs())
    high = build_attention_frame(
        ctx=dict(signals_ctx),
        inputs=_inputs(
            concept_induction={"self": ["lattice coherence"], "relationship": [], "growth": [], "tension": ["fragmentation"]},
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


