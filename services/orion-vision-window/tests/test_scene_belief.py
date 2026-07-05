from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scene_belief import SceneBeliefRegistry, SceneBeliefTracker


def test_belief_ignores_single_empty_observation() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door", "screen"}))
    tracker.observe(frozenset())
    result = tracker.observe(frozenset({"door", "screen"}))
    assert result.believed_labels == frozenset({"door", "screen"})
    assert result.added == frozenset()
    assert result.removed == frozenset()


def test_belief_requires_enter_votes() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door", "screen"}))
    tracker.observe(frozenset({"door", "screen"}))
    once = tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" not in once.believed_labels
    twice = tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" in twice.believed_labels
    assert "package" in twice.added


def test_belief_requires_exit_votes() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    for _ in range(3):
        tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" in tracker.believed_labels
    tracker.observe(frozenset({"door", "screen"}))
    result = tracker.observe(frozenset({"door", "screen"}))
    assert "package" not in result.believed_labels
    assert result.removed == frozenset({"package"})


def test_exit_votes_respected() -> None:
    strict = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=0)
    for _ in range(3):
        strict.observe(frozenset({"package", "door"}))
    strict.observe(frozenset({"door"}))
    strict.observe(frozenset({"door"}))
    assert "package" in strict.believed_labels

    lenient = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    for _ in range(3):
        lenient.observe(frozenset({"package", "door"}))
    lenient.observe(frozenset({"door"}))
    result = lenient.observe(frozenset({"door"}))
    assert "package" not in result.believed_labels


def test_enrich_evidence_includes_believed_tier() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door"}))
    tracker.observe(frozenset({"door"}))
    tracker.observe(frozenset({"door"}))
    evidence = {
        "hard_labels": ["door"],
        "soft_labels": [],
        "host_person_hits": 0,
        "edge_person_hits": 0,
        "caption_count": 0,
    }
    enriched = tracker.enrich_evidence(evidence)
    assert enriched["hard_labels"] == ["door"]
    assert enriched["believed_hard_labels"] == ["door"]
    assert enriched["belief"]["schema"] == "scene_belief.v1"
    assert enriched["belief"]["vote_n"] == 3
    assert enriched["belief"]["enter_votes"] == 2
    assert enriched["belief"]["exit_votes"] == 1
    assert enriched["belief"]["observation_count"] == 3


def test_registry_isolates_streams() -> None:
    registry = SceneBeliefRegistry(vote_n=3, enter_votes=2, exit_votes=1)
    registry.observe("cam0", frozenset({"door"}))
    registry.observe("cam0", frozenset({"door"}))
    registry.observe("cam1", frozenset({"chair"}))
    cam0 = registry.observe("cam0", frozenset({"door"}))
    cam1 = registry.observe("cam1", frozenset({"chair"}))
    assert cam0.believed_labels == frozenset({"door"})
    assert cam1.believed_labels == frozenset()
