from __future__ import annotations

from orion.social.scenario_replay import SocialScenarioReplayHarness, load_scenarios


def test_social_room_replay_harness_passes_fixture_pack() -> None:
    harness = SocialScenarioReplayHarness()

    report = harness.run(load_scenarios())

    assert report["summary"]["scenario_count"] >= 20
    assert not report["summary"]["failed_scenarios"]
    assert report["summary"]["passed_count"] == report["summary"]["scenario_count"]


def test_social_room_replay_harness_runs_real_seams_for_direct_peer_question() -> None:
    harness = SocialScenarioReplayHarness()
    fixture = load_scenarios(only=["direct-peer-question-multi-peer"])[0]

    result = harness.run_scenario(fixture)

    assert result.passed is True
    assert set(result.seams_exercised) >= {
        "social-memory-summary",
        "social-memory-inspection",
        "bridge-routing-policy",
        "hub-request-builder",
        "prompt-grounding",
    }
    assert result.observed_outcomes["routing_decision"] == "reply_to_peer"
    assert "THREAD ROUTING:" in result.observed_outcomes["rendered_prompt"]
    assert result.observed_outcomes["request_metadata"]["social_thread_routing"]["routing_decision"] == "reply_to_peer"


def test_social_room_replay_harness_keeps_safety_visible_in_results() -> None:
    harness = SocialScenarioReplayHarness()

    blocked = harness.run_scenario(load_scenarios(only=["blocked-private-material-stays-blocked"])[0])
    pending = harness.run_scenario(load_scenarios(only=["pending-artifact-stays-non-active"])[0])

    assert blocked.passed is True
    assert blocked.observed_outcomes["private_material_blocked"] is True
    assert any("blocked string remained absent" in item for item in blocked.safety_observations)
    assert pending.passed is True
    assert pending.observed_outcomes["pending_artifact_non_active"] is True
    assert any("pending artifact dialogue stayed non-active" in item for item in pending.safety_observations)


def test_social_room_replay_harness_surfaces_bounded_gif_metadata() -> None:
    harness = SocialScenarioReplayHarness()

    result = harness.run_scenario(load_scenarios(only=["light-affiliative-exchange-allows-bounded-gif"])[0])

    assert result.passed is True
    assert result.observed_outcomes["gif_decision_kind"] == "text_plus_gif"
    assert result.observed_outcomes["gif_allowed"] is True
    assert result.observed_outcomes["gif_transport_metadata"]["gif_intent"] == "laugh_with"


def test_social_room_replay_harness_harvests_gif_specific_regressions() -> None:
    harness = SocialScenarioReplayHarness()

    playful = harness.run_scenario(load_scenarios(only=["playful-room-text-only-streak-allows-bounded-gif"])[0])
    looped = harness.run_scenario(load_scenarios(only=["repeated-gif-intent-loop-stays-text-only"])[0])
    fallback = harness.run_scenario(load_scenarios(only=["gif-transport-fallback-stays-text-only"])[0])

    assert playful.passed is True
    assert playful.observed_outcomes["gif_decision_kind"] == "text_plus_gif"
    assert "fresh_room_ritual_supports_playfulness" in playful.observed_outcomes["gif_reasons"]
    assert playful.observed_outcomes["gif_media_hint_present"] is True

    assert looped.passed is True
    assert looped.observed_outcomes["gif_decision_kind"] == "text_only"
    assert "gif_intent_loop_detected" in looped.observed_outcomes["gif_reasons"]

    assert fallback.passed is True
    assert fallback.observed_outcomes["gif_transport_degraded"] == "true"
    assert fallback.observed_outcomes["gif_media_hint_present"] is False


def test_social_room_replay_harness_harvests_peer_gif_proxy_regressions() -> None:
    harness = SocialScenarioReplayHarness()

    clear = harness.run_scenario(load_scenarios(only=["peer-gif-clear-celebration-proxy"])[0])
    ambiguous = harness.run_scenario(load_scenarios(only=["peer-gif-ambiguous-metadata-stays-unknown"])[0])
    laugh = harness.run_scenario(load_scenarios(only=["peer-gif-text-disambiguates-laughter"])[0])
    contested = harness.run_scenario(load_scenarios(only=["peer-gif-contested-turn-stays-secondary"])[0])
    noisy = harness.run_scenario(load_scenarios(only=["peer-gif-misleading-tags-stays-uncertain"])[0])
    no_meta = harness.run_scenario(load_scenarios(only=["peer-gif-no-useful-metadata-unknown"])[0])

    assert clear.passed is True
    assert clear.observed_outcomes["peer_gif_present"] is True
    assert clear.observed_outcomes["peer_gif_reaction_class"] == "celebrate"

    assert ambiguous.passed is True
    assert ambiguous.observed_outcomes["peer_gif_reaction_class"] == "unknown"
    assert ambiguous.observed_outcomes["peer_gif_confidence"] == "low"

    assert laugh.passed is True
    assert laugh.observed_outcomes["peer_gif_reaction_class"] == "laugh_with"

    assert contested.passed is True
    assert contested.observed_outcomes["peer_gif_cue_disposition"] in {"softened", "ignored"}

    assert noisy.passed is True
    assert noisy.observed_outcomes["peer_gif_confidence"] == "low"
    assert noisy.observed_outcomes["peer_gif_ambiguity"] == "high"

    assert no_meta.passed is True
    assert no_meta.observed_outcomes["peer_gif_reaction_class"] == "unknown"


def test_social_room_replay_failures_are_inspectable() -> None:
    harness = SocialScenarioReplayHarness()
    fixture = load_scenarios(only=["repair-after-wrong-thread-response"])[0]
    bad_fixture = fixture.model_copy(
        update={
            "expectation": fixture.expectation.model_copy(
                update={
                    "routing_decision": "reply_to_peer",
                    "repair_decision": "repair",
                }
            )
        }
    )

    result = harness.run_scenario(bad_fixture)

    assert result.passed is False
    assert any("routing_decision" in item for item in result.mismatch_reasons)
    assert any("repair_decision" in item for item in result.mismatch_reasons)
    assert result.observed_outcomes["repair_type"] == "thread_mismatch"
