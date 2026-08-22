from __future__ import annotations

from pathlib import Path

from orion.substrate.seed import (
    DEFAULT_SEED_CONCEPTS_PATH,
    load_seed_concept_nodes,
    load_seed_concepts_into_store,
)
from orion.substrate.store import InMemorySubstrateGraphStore


def test_load_seed_concepts_into_store_writes_four_canonical_concepts() -> None:
    # Claude added 2026-08-20 as the fixture's 4th seed (see
    # docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md)
    # alongside Orion/Juniper/their relationship.
    store = InMemorySubstrateGraphStore()

    written = load_seed_concepts_into_store(store)
    assert written == 4

    result = store.query_concept_region(limit_nodes=32, limit_edges=64)
    assert result.query_kind == "concept_region"

    labels = {node.label for node in result.slice.nodes}
    assert labels == {"Orion", "Juniper", "Orion-Juniper relationship", "Claude"}

    scopes = {node.anchor_scope for node in result.slice.nodes}
    assert scopes == {"orion", "juniper", "claude", "relationship"}

    assert len(result.slice.nodes) == 4
    for node in result.slice.nodes:
        assert node.promotion_state == "canonical"
        assert node.node_kind == "concept"
        assert node.definition
        # Regression: golden concepts (highest-authority, longest-lived nodes
        # in the store) used to be permanently stuck at
        # activation=0.0/decay_half_life=None -- immune to Hub's live decay
        # scheduler forever, same bug as the organic-growth adapters. Fixed
        # in two layers: ConceptNodeV1's auto-seed validator gives every
        # concept a real half-life, and seed.py now gives golden concepts a
        # real salience (1.0 -- canonical/human_verified, no principled
        # reason to hedge lower) so their seeded activation is real too, not
        # just their half-life.
        assert node.signals.activation.decay_half_life_seconds is not None
        assert node.signals.activation.decay_half_life_seconds > 0
        assert node.signals.activation.activation == 1.0
        assert node.signals.salience == 1.0


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
    assert ("sub-concept-seed-claude", "sub-concept-seed-orion", "associated_with") in edge_pairs
    assert ("sub-concept-seed-claude", "sub-concept-seed-juniper", "associated_with") in edge_pairs


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


def test_golden_subject_anchor_node_ids_match_real_seed_fixture() -> None:
    # orion/substrate/adapters/concept_induction.py hardcodes
    # _GOLDEN_SUBJECT_ANCHOR_NODE_IDS (ConceptProfile.subject ->
    # golden node_id) rather than reading seed_concepts.yaml on every chat
    # turn. This test is the drift guard: if the fixture's keys/node_ids
    # ever change, this fails loudly instead of the adapter silently
    # emitting associated_with edges to node_ids that no longer exist.
    from orion.substrate.adapters.concept_induction import _GOLDEN_SUBJECT_ANCHOR_NODE_IDS

    nodes, _edges = load_seed_concept_nodes()
    real_node_ids = {n.node_id for n in nodes}

    for subject, expected_node_id in _GOLDEN_SUBJECT_ANCHOR_NODE_IDS.items():
        assert expected_node_id in real_node_ids, (
            f"_GOLDEN_SUBJECT_ANCHOR_NODE_IDS[{subject!r}] = {expected_node_id!r} "
            "does not match any node_id the real seed fixture produces"
        )
