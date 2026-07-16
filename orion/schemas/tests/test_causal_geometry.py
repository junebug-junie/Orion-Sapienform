from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.causal_geometry import (
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)
from orion.schemas.field_state import FieldEdgeV1
from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY, resolve
from orion.substrate.mutation_contracts import CONTRACTS


def _snapshot() -> CausalGeometrySnapshotV1:
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    return CausalGeometrySnapshotV1(
        snapshot_id="snap-1",
        generated_at=now,
        window_start=now,
        window_end=now,
        edges=[
            CausalGeometryEdgeV1(
                source_id="node:atlas",
                target_id="capability:reasoning",
                lag_sec=60,
                strength=0.42,
                significance=0.01,
                n_samples=120,
                window_start=now,
                window_end=now,
            )
        ],
        designed_topology_version="topology-v3",
        divergence=[
            CausalGeometryDivergenceEntryV1(
                source_id="node:atlas",
                target_id="capability:reasoning",
                observed_strength=0.42,
                designed_weight=0.30,
                delta=0.12,
                status="both",
            )
        ],
        insufficient_data=False,
        notes=["seed"],
    )


def test_causal_geometry_snapshot_round_trip() -> None:
    snapshot = _snapshot()
    dumped = snapshot.model_dump(mode="json")
    restored = CausalGeometrySnapshotV1.model_validate(dumped)
    assert restored == snapshot
    assert restored.schema_version == "causal.geometry.snapshot.v1"


def test_causal_geometry_snapshot_registered() -> None:
    assert "CausalGeometrySnapshotV1" in _REGISTRY
    assert resolve("CausalGeometrySnapshotV1") is CausalGeometrySnapshotV1
    assert SCHEMA_REGISTRY["CausalGeometrySnapshotV1"].kind == "causal.geometry.snapshot.v1"
    assert SCHEMA_REGISTRY["CausalGeometrySnapshotV1"].model is CausalGeometrySnapshotV1


def test_field_edge_v1_defaults_without_provenance_fields() -> None:
    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:reasoning",
        edge_type="node_capability",
        weight=0.5,
    )
    assert edge.weight_source == "designed"
    assert edge.learned_at is None


def test_field_edge_v1_accepts_explicit_provenance_fields() -> None:
    learned_at = datetime(2026, 7, 16, tzinfo=timezone.utc)
    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:reasoning",
        edge_type="node_capability",
        weight=0.5,
        weight_source="learned",
        learned_at=learned_at,
    )
    assert edge.weight_source == "learned"
    assert edge.learned_at == learned_at
    dumped = edge.model_dump(mode="json")
    restored = FieldEdgeV1.model_validate(dumped)
    assert restored == edge


def test_field_topology_weight_patch_contract_registered() -> None:
    contract = CONTRACTS["field_topology_weight_patch"]
    assert contract.auto_promote_default is False
    assert set(contract.bounds.keys()) == set(contract.allowed_fields)
    assert contract.allowed_fields == ("edge_weight_delta",)
