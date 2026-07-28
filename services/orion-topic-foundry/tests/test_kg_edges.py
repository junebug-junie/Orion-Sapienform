"""Tests for kg_edges.py's typed-edge generation.

Regression coverage for the 2026-07-28 retirement of the
`orion:kg:edge:ingest.v1` bus publish (zero live consumers -- see
orion/bus/channels.yaml's former comment and
orion/substrate/adapters/topic_foundry.py's module docstring): edges are
still computed and persisted to Postgres/artifacts, they just no longer go
out on the bus. `app.services.bus_events`/`get_bus_publisher` are no longer
imported by this module at all -- these tests both exercise the surviving
compute/persist path and act as a regression guard against that import
reappearing.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from app.services import kg_edges


FAKE_RUN_ID = UUID("11111111-1111-1111-1111-111111111111")
FAKE_MODEL_ID = UUID("22222222-2222-2222-2222-222222222222")


def _run_row(run_id: UUID = FAKE_RUN_ID, model_id: UUID = FAKE_MODEL_ID) -> dict:
    # No "artifact_paths" key -- _write_edge_artifacts degrades to a no-op
    # (early "if not run_dir: return"), keeping this test filesystem-free.
    return {"run_id": str(run_id), "model_id": str(model_id)}


def _segment(segment_id, *, entities=None, questions=None, claims=None, next_steps=None) -> dict:
    return {
        "segment_id": str(segment_id),
        "meaning": {
            "entities": entities or [],
            "questions": questions or [],
            "claims": claims or [],
            "next_steps": next_steps or [],
        },
    }


def test_kg_edges_module_no_longer_imports_bus_publisher() -> None:
    """Direct regression guard for the retirement: the removed
    `from app.services.bus_events import get_bus_publisher` import (and the
    KgEdgeIngestV1/KgEdgeIngestItemV1 schema imports) must not reappear.
    """
    assert not hasattr(kg_edges, "get_bus_publisher")
    assert not hasattr(kg_edges, "KgEdgeIngestV1")
    assert not hasattr(kg_edges, "KgEdgeIngestItemV1")
    assert not hasattr(kg_edges, "_publish_edge_batch")


def test_generate_edges_for_run_returns_zero_for_missing_run(monkeypatch) -> None:
    monkeypatch.setattr(kg_edges, "fetch_run", lambda run_id: None)
    assert kg_edges.generate_edges_for_run(FAKE_RUN_ID) == 0


def test_generate_edges_for_run_computes_and_persists_edges(monkeypatch) -> None:
    segment_id = uuid4()
    persisted: dict = {}

    monkeypatch.setattr(kg_edges, "fetch_run", lambda run_id: _run_row())
    monkeypatch.setattr(kg_edges, "fetch_model", lambda model_id: {"name": "test-model"})
    monkeypatch.setattr(
        kg_edges,
        "fetch_segments",
        lambda run_id, has_enrichment=True: [
            _segment(segment_id, entities=["Juniper Feld", "Orion"], questions=["what next?"])
        ],
    )

    def fake_replace(*, run_id, edges):
        persisted["run_id"] = run_id
        persisted["edges"] = edges

    monkeypatch.setattr(kg_edges, "replace_edges_for_run", fake_replace)

    count = kg_edges.generate_edges_for_run(FAKE_RUN_ID)

    # 2 "mentions" (confidence 0.6) + 1 "asks_about" (confidence 0.5), all
    # at/above the default min_confidence=0.2 floor.
    assert count == 3
    assert persisted["run_id"] == FAKE_RUN_ID
    predicates = sorted(edge["predicate"] for edge in persisted["edges"])
    assert predicates == ["asks_about", "mentions", "mentions"]
    mention_objects = sorted(
        edge["object"] for edge in persisted["edges"] if edge["predicate"] == "mentions"
    )
    assert mention_objects == ["juniper feld", "orion"]


def test_generate_edges_for_run_filters_below_min_confidence(monkeypatch) -> None:
    segment_id = uuid4()

    monkeypatch.setattr(kg_edges, "fetch_run", lambda run_id: _run_row())
    monkeypatch.setattr(kg_edges, "fetch_model", lambda model_id: {"name": "test-model"})
    monkeypatch.setattr(
        kg_edges,
        "fetch_segments",
        lambda run_id, has_enrichment=True: [_segment(segment_id, questions=["asks_about is 0.5 confidence"])],
    )
    monkeypatch.setattr(kg_edges, "replace_edges_for_run", lambda *, run_id, edges: None)

    # min_confidence above asks_about's fixed 0.5 confidence -- must filter it out.
    count = kg_edges.generate_edges_for_run(FAKE_RUN_ID, min_confidence=0.6)
    assert count == 0


def test_generate_edges_for_run_missing_model_degrades_to_unknown(monkeypatch) -> None:
    segment_id = uuid4()
    persisted: dict = {}

    monkeypatch.setattr(kg_edges, "fetch_run", lambda run_id: _run_row())
    monkeypatch.setattr(kg_edges, "fetch_model", lambda model_id: None)
    monkeypatch.setattr(
        kg_edges,
        "fetch_segments",
        lambda run_id, has_enrichment=True: [_segment(segment_id, entities=["x"])],
    )
    monkeypatch.setattr(
        kg_edges, "replace_edges_for_run", lambda *, run_id, edges: persisted.update(edges=edges)
    )

    count = kg_edges.generate_edges_for_run(FAKE_RUN_ID)
    assert count == 1
    assert persisted["edges"][0]["subject"] == "unknown"
