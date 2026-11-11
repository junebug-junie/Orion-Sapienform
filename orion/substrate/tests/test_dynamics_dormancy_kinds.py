"""Dormancy is decided only for node kinds the store can persist it for.

Live bug (2026-11-11): every substrate dynamics tick logged
``dormancy_transitions`` ~130-170 while the dormant count stayed flat. The
tick flagged dormancy for every node kind, but the Falkor store saves
``dormant`` on concept nodes only. A non-concept node below threshold read
back ``dormant=False`` on the next tick and "flipped" again -- a fake
transition plus a wasted graph write, every tick, forever.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.store import InMemorySubstrateGraphStore

_NOW = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)


def _cold_node(node_id: str, node_kind: str) -> BaseSubstrateNodeV1:
    """A node already below the dormancy threshold (activation and recency 0)."""
    return BaseSubstrateNodeV1(
        node_id=node_id,
        node_kind=node_kind,
        anchor_scope="orion",
        temporal={"observed_at": _NOW},
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=0.4,
            activation={
                "activation": 0.0,
                "recency_score": 0.0,
                "decay_half_life_seconds": None,
                "decay_floor": 0.0,
            },
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test",
        ),
        metadata={"dynamic_pressure": 0.0},
    )


class _FalkorLikeStore(InMemorySubstrateGraphStore):
    """Counts node writes and, like FalkorSubstrateStore, keeps ``dormant``
    on concept nodes only (literal on purpose: mirrors the durable store,
    independent of the shared constant under test)."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list = []

    def update_node_dynamics(self, write):  # noqa: ANN001
        self.writes.append(write)
        if write.node_kind != "concept":
            write = replace(write, dormant=None, dormancy_updated_at=None)
        return super().update_node_dynamics(write)


def _engine(store: InMemorySubstrateGraphStore, node_ids: list[str]) -> SubstrateDynamicsEngine:
    engine = SubstrateDynamicsEngine(store=store)
    activations: dict[str, float] = {}
    for node_id in node_ids:
        activations[node_id] = 0.0
        activations[f"{node_id}:recency"] = 0.0
    engine._compute_pressures = lambda nodes, outgoing, tick_at: ({n: 0.0 for n in node_ids}, {})  # type: ignore[method-assign]
    engine._compute_activations = lambda updated_nodes, outgoing, pressures, tick_at: activations  # type: ignore[method-assign]
    return engine


def test_cold_non_concept_node_never_transitions_or_writes() -> None:
    store = _FalkorLikeStore()
    for kind in ("entity", "evidence", "assertion"):
        store.upsert_node(identity_key=f"id-{kind}", node=_cold_node(f"n-{kind}", kind))
    engine = _engine(store, ["n-entity", "n-evidence", "n-assertion"])

    for tick in range(3):
        result = engine.tick(now=_NOW)
        assert result.dormancy_transitions == [], f"phantom transitions on tick {tick}"
    assert store.writes == []


def test_cold_concept_node_goes_dormant_once_and_stays() -> None:
    store = _FalkorLikeStore()
    store.upsert_node(identity_key="id-concept", node=_cold_node("n-concept", "concept"))
    engine = _engine(store, ["n-concept"])

    first = engine.tick(now=_NOW)
    assert [(t.node_id, t.to_state) for t in first.dormancy_transitions] == [("n-concept", "dormant")]
    assert len(store.writes) == 1 and store.writes[0].dormant is True

    second = engine.tick(now=_NOW)
    assert second.dormancy_transitions == []
    assert len(store.writes) == 1
