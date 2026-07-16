from __future__ import annotations

from pathlib import Path

from orion.substrate.seed import (
    DEFAULT_SEED_CONCEPTS_PATH,
    load_seed_concept_nodes,
    load_seed_concepts_into_store,
)
from orion.substrate.store import InMemorySubstrateGraphStore


def test_load_seed_concepts_into_store_writes_three_canonical_concepts() -> None:
    store = InMemorySubstrateGraphStore()

    written = load_seed_concepts_into_store(store)
    assert written == 3

    result = store.query_concept_region(limit_nodes=32, limit_edges=64)
    assert result.query_kind == "concept_region"

    labels = {node.label for node in result.slice.nodes}
    assert labels == {"Orion", "Juniper", "Orion-Juniper relationship"}

    scopes = {node.anchor_scope for node in result.slice.nodes}
    assert scopes == {"orion", "juniper", "relationship"}

    assert len(result.slice.nodes) == 3
    for node in result.slice.nodes:
        assert node.promotion_state == "canonical"
        assert node.node_kind == "concept"
        assert node.definition


def test_load_seed_concepts_into_store_wires_relationship_edges() -> None:
    store = InMemorySubstrateGraphStore()
    load_seed_concepts_into_store(store)

    result = store.query_concept_region(limit_nodes=32, limit_edges=64)
    edge_pairs = {(edge.source.node_id, edge.target.node_id, edge.predicate) for edge in result.slice.edges}

    assert (
        "sub-concept-seed-orion_juniper_relationship",
        "sub-concept-seed-orion",
        "associated_with",
    ) in edge_pairs
    assert (
        "sub-concept-seed-orion_juniper_relationship",
        "sub-concept-seed-juniper",
        "associated_with",
    ) in edge_pairs


def test_load_seed_concept_nodes_missing_file_degrades_gracefully() -> None:
    nodes, edges = load_seed_concept_nodes(Path("/nonexistent/seed_concepts.yaml"))
    assert nodes == []
    assert edges == []


def test_load_seed_concept_nodes_malformed_yaml_degrades_gracefully(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("not_a_concepts_list: true\n", encoding="utf-8")

    nodes, edges = load_seed_concept_nodes(bad_file)
    assert nodes == []
    assert edges == []


def test_default_seed_concepts_path_exists() -> None:
    assert DEFAULT_SEED_CONCEPTS_PATH.exists()
