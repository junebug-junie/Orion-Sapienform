"""Tests for curiosity_ctx adapter."""

from __future__ import annotations

import json

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.relational.adapters.curiosity_ctx import (
    map_curiosity_ctx_to_substrate,
)


def _make_signal(
    *,
    signal_id: str = "sig-1",
    signal_type: str = "ontology_sparse_region",
    signal_strength: float = 0.8,
    confidence: float = 0.7,
    evidence_summary: str = "Some evidence",
    focal_node_refs: list[str] | None = None,
) -> FrontierInvocationSignalV1:
    """Factory for test signals."""
    return FrontierInvocationSignalV1(
        signal_id=signal_id,
        signal_type=signal_type,
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="world_ontology",
        task_type_candidate="ontology_expand",
        signal_strength=signal_strength,
        confidence=confidence,
        evidence_summary=evidence_summary,
        focal_node_refs=focal_node_refs or [],
    )


def test_empty_signals_returns_none():
    """ctx without curiosity_signals → None."""
    record = map_curiosity_ctx_to_substrate({})
    assert record is None


def test_none_ctx_returns_none():
    """None ctx → None."""
    record = map_curiosity_ctx_to_substrate(None)
    assert record is None


def test_single_signal_emits_node():
    """1 curiosity signal → one curiosity:unresolved_gaps node."""
    signal = _make_signal()
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": [signal]})

    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1

    node = record.nodes[0]
    assert node.label == "curiosity:unresolved_gaps"
    assert node.anchor_scope == "orion"
    assert node.subject_ref == "entity:orion"
    assert node.signals.salience == 0.4
    assert node.signals.confidence == 0.7
    assert node.metadata["gap_count"] == 1


def test_top_3_signals_by_strength():
    """10 signals → only top 3 by signal_strength in output."""
    signals = [
        _make_signal(signal_id=f"sig-{i}", signal_strength=float(i) / 10.0)
        for i in range(10)
    ]
    # Unsorted: strengths are 0.0, 0.1, 0.2, ..., 0.9
    # Top 3 by strength should be 0.9, 0.8, 0.7 (indices 9, 8, 7)

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": signals})

    assert record is not None
    assert len(record.nodes) == 1

    node = record.nodes[0]
    assert node.metadata["gap_count"] == 3
    assert node.metadata["signal_types"] == [
        "ontology_sparse_region",
        "ontology_sparse_region",
        "ontology_sparse_region",
    ]


def test_caps_focal_refs_at_6():
    """signals with many focal_refs → capped at 6 total."""
    # Create 3 signals, each with 4 focal refs (total would be 12, cap at 6)
    signals = [
        _make_signal(
            signal_id=f"sig-{i}",
            focal_node_refs=[f"node-{i}-{j}" for j in range(4)],
        )
        for i in range(3)
    ]

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": signals})

    assert record is not None
    node = record.nodes[0]

    # Should have exactly 6 focal refs (capped)
    # From each signal we take up to 2, so first signal contributes 2,
    # second signal contributes 2, third signal contributes 2 = 6 total
    focal_refs = node.metadata["focal_node_refs"]
    assert len(focal_refs) == 6
    assert focal_refs == [
        "node-0-0",
        "node-0-1",
        "node-1-0",
        "node-1-1",
        "node-2-0",
        "node-2-1",
    ]


def test_salience_0_4_not_higher():
    """emitted node has salience exactly 0.4 (verify it, don't relax)."""
    signal = _make_signal(signal_strength=0.95)
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": [signal]})

    assert record is not None
    node = record.nodes[0]
    # Salience must be exactly 0.4, not derived from signal strength
    assert node.signals.salience == 0.4


def test_accepts_dict_version():
    """coerce dict version of FrontierInvocationSignalV1."""
    signal_dict = {
        "signal_id": "sig-1",
        "signal_type": "ontology_sparse_region",
        "anchor_scope": "orion",
        "subject_ref": "entity:orion",
        "target_zone": "world_ontology",
        "task_type_candidate": "ontology_expand",
        "signal_strength": 0.8,
        "confidence": 0.7,
        "evidence_summary": "Some evidence",
        "focal_node_refs": [],
    }

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": [signal_dict]})

    assert record is not None
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "curiosity:unresolved_gaps"


def test_accepts_json_string_version():
    """coerce JSON string version."""
    signal = _make_signal()
    json_str = signal.model_dump_json()

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": [json_str]})

    assert record is not None
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "curiosity:unresolved_gaps"


def test_accepts_mixed_types():
    """coerce mixed types: model, dict, JSON string."""
    signal_model = _make_signal(signal_id="sig-1", signal_strength=0.9)
    signal_dict = {
        "signal_id": "sig-2",
        "signal_type": "concept_instability",
        "anchor_scope": "orion",
        "subject_ref": "entity:orion",
        "target_zone": "concept_graph",
        "task_type_candidate": "concept_expand",
        "signal_strength": 0.8,
        "confidence": 0.6,
        "evidence_summary": "Dict evidence",
        "focal_node_refs": [],
    }
    signal_json = _make_signal(signal_id="sig-3", signal_strength=0.7).model_dump_json()

    record = map_curiosity_ctx_to_substrate(
        {"curiosity_signals": [signal_model, signal_dict, signal_json]}
    )

    assert record is not None
    node = record.nodes[0]
    assert node.metadata["gap_count"] == 3
    # Confidence should be average of 0.7, 0.6, 0.7
    expected_confidence = (0.7 + 0.6 + 0.7) / 3
    assert node.signals.confidence == expected_confidence


def test_degrades_on_invalid_json():
    """invalid JSON → None, no raise."""
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": "not json"})
    assert record is None


def test_degrades_on_empty_list():
    """empty list → None."""
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": []})
    assert record is None


def test_degrades_on_none_signals():
    """curiosity_signals=None → None."""
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": None})
    assert record is None


def test_degrades_on_garbage_input():
    """various garbage inputs → None, no raise."""
    assert map_curiosity_ctx_to_substrate({"curiosity_signals": "not json"}) is None
    assert (
        map_curiosity_ctx_to_substrate({"curiosity_signals": {"not": "a list"}})
        is None
    )
    assert (
        map_curiosity_ctx_to_substrate({"curiosity_signals": 42})
        is None
    )
    assert (
        map_curiosity_ctx_to_substrate({"curiosity_signals": 3.14})
        is None
    )


def test_confidences_averaged_correctly():
    """average confidence from top 3 signals."""
    signals = [
        _make_signal(signal_id="sig-1", signal_strength=0.9, confidence=0.5),
        _make_signal(signal_id="sig-2", signal_strength=0.8, confidence=0.6),
        _make_signal(signal_id="sig-3", signal_strength=0.7, confidence=0.8),
        _make_signal(signal_id="sig-4", signal_strength=0.6, confidence=0.9),  # should be ignored
    ]

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": signals})

    assert record is not None
    node = record.nodes[0]
    # Average of top 3: (0.5 + 0.6 + 0.8) / 3 = 0.633...
    expected_confidence = (0.5 + 0.6 + 0.8) / 3
    assert abs(node.signals.confidence - expected_confidence) < 0.001


def test_evidence_summaries_collected():
    """evidence_summaries from signals collected in metadata."""
    signals = [
        _make_signal(
            signal_id="sig-1",
            signal_strength=0.9,
            evidence_summary="Evidence A",
        ),
        _make_signal(
            signal_id="sig-2",
            signal_strength=0.8,
            evidence_summary="Evidence B",
        ),
        _make_signal(
            signal_id="sig-3",
            signal_strength=0.7,
            evidence_summary="",  # empty, should still be in signal but might not be in list
        ),
    ]

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": signals})

    assert record is not None
    node = record.nodes[0]
    evidence_summaries = node.metadata["evidence_summaries"]
    # Empty evidence should be filtered out
    assert "Evidence A" in evidence_summaries
    assert "Evidence B" in evidence_summaries
    assert len(evidence_summaries) == 2


def test_signal_types_preserved():
    """signal_types from top 3 signals preserved in metadata."""
    signals = [
        _make_signal(signal_id="sig-1", signal_strength=0.9, signal_type="ontology_sparse_region"),
        _make_signal(signal_id="sig-2", signal_strength=0.8, signal_type="concept_instability"),
        _make_signal(signal_id="sig-3", signal_strength=0.7, signal_type="contradiction_hotspot"),
    ]

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": signals})

    assert record is not None
    node = record.nodes[0]
    signal_types = node.metadata["signal_types"]
    assert signal_types == [
        "ontology_sparse_region",
        "concept_instability",
        "contradiction_hotspot",
    ]


def test_tier_rank_set_correctly():
    """provenance tier_rank should be 4 (snapshot_ephemeral)."""
    signal = _make_signal()
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": [signal]})

    assert record is not None
    node = record.nodes[0]
    assert node.provenance.tier_rank == 4
    assert node.provenance.source_kind == "curiosity"
    assert node.provenance.source_channel == "substrate.curiosity"
    assert node.provenance.producer == "curiosity_adapter"
    assert node.provenance.authority == "local_inferred"


def test_persisted_candidates_round_trip_through_adapter():
    """Signals dumped the way substrate-runtime persists them (model_dump
    mode="json" into candidates_json) hydrate the adapter and emit a node."""
    signals = [
        _make_signal(signal_id="sig-rt-1", signal_strength=0.9, evidence_summary="gap a"),
        _make_signal(signal_id="sig-rt-2", signal_strength=0.5, evidence_summary="gap b"),
    ]
    persisted = [sig.model_dump(mode="json") for sig in signals]

    record = map_curiosity_ctx_to_substrate({"curiosity_signals": persisted})

    assert record is not None
    node = record.nodes[0]
    assert node.label == "curiosity:unresolved_gaps"
    assert node.metadata["gap_count"] == 2
    assert node.metadata["evidence_summaries"] == ["gap a", "gap b"]
