"""Loader for the golden concept seed fixture (Phase 1, concept-atlas).

Reads `orion/substrate/seed_concepts.yaml` and writes a fixed set of
hand-authored, canonical `ConceptNodeV1` nodes (plus their cross-reference
edges) directly into a `SubstrateGraphStore`. This bypasses any
extraction/clustering pipeline entirely -- it exists to give Orion's
cognition an immediate, inspectable self/other/relationship floor
(Orion, Juniper, and the Orion-Juniper relationship) before organic
concept induction has produced anything.

This is intentionally a fixed 3-row fixture loader, not a general-purpose
seeding framework. Do not extend it into a plugin system for future seed
sets -- add a new fixture + loader when that need actually exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger(__name__)

DEFAULT_SEED_CONCEPTS_PATH = Path(__file__).with_name("seed_concepts.yaml")


def _seed_provenance(*, key: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="human_verified",
        source_kind="golden_seed_fixture",
        source_channel="orion:substrate:seed_concepts",
        producer="seed_concepts_loader",
        evidence_refs=[f"seed_concepts.yaml#{key}"],
    )


def _seed_temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def load_seed_concept_nodes(
    fixture_path: str | Path | None = None,
) -> tuple[list[ConceptNodeV1], list[SubstrateEdgeV1]]:
    """Parse the seed fixture into ConceptNodeV1/SubstrateEdgeV1 objects.

    Degrades gracefully: any missing file, unreadable YAML, or malformed
    entry is logged and skipped rather than raised, so callers never crash
    on a bad fixture. Returns (nodes, edges); either or both may be empty.
    """

    path = Path(fixture_path) if fixture_path is not None else DEFAULT_SEED_CONCEPTS_PATH

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("seed_concepts: could not read fixture at %s: %s", path, exc)
        return [], []

    try:
        parsed: Any = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        logger.warning("seed_concepts: could not parse YAML at %s: %s", path, exc)
        return [], []

    entries = parsed.get("concepts") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        logger.warning("seed_concepts: fixture at %s has no `concepts` list; skipping", path)
        return [], []

    nodes: list[ConceptNodeV1] = []
    node_id_by_key: dict[str, str] = {}
    pending_edges: list[tuple[str, list[str]]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning("seed_concepts: skipping non-mapping entry %r", entry)
            continue

        key = str(entry.get("key") or "").strip()
        label = str(entry.get("label") or "").strip()
        anchor_scope = entry.get("anchor_scope")
        if not key or not label or not anchor_scope:
            logger.warning("seed_concepts: skipping entry missing key/label/anchor_scope: %r", entry)
            continue

        node_id = f"sub-concept-seed-{key}"
        try:
            node = ConceptNodeV1(
                node_id=node_id,
                anchor_scope=anchor_scope,
                subject_ref=entry.get("subject_ref") or key,
                promotion_state="canonical",
                temporal=_seed_temporal(),
                provenance=_seed_provenance(key=key),
                label=label,
                definition=entry.get("definition"),
            )
        except Exception as exc:  # noqa: BLE001 - malformed fixture row must not crash callers
            logger.warning("seed_concepts: skipping invalid entry %r: %s", entry, exc)
            continue

        nodes.append(node)
        node_id_by_key[key] = node_id

        related_keys = entry.get("related_concept_keys") or []
        if isinstance(related_keys, list) and related_keys:
            pending_edges.append((key, [str(item) for item in related_keys]))

    edges: list[SubstrateEdgeV1] = []
    for source_key, related_keys in pending_edges:
        source_node_id = node_id_by_key.get(source_key)
        if not source_node_id:
            continue
        for related_key in related_keys:
            target_node_id = node_id_by_key.get(related_key)
            if not target_node_id:
                logger.warning(
                    "seed_concepts: skipping edge %s -> %s (unknown related key)",
                    source_key,
                    related_key,
                )
                continue
            edges.append(
                SubstrateEdgeV1(
                    edge_id=f"sub-edge-seed-{source_key}-{related_key}",
                    source=NodeRefV1(node_id=source_node_id, node_kind="concept"),
                    target=NodeRefV1(node_id=target_node_id, node_kind="concept"),
                    predicate="associated_with",
                    temporal=_seed_temporal(),
                    provenance=_seed_provenance(key=f"{source_key}-{related_key}"),
                )
            )

    return nodes, edges


def load_seed_concepts_into_store(
    store: SubstrateGraphStore,
    *,
    fixture_path: str | Path | None = None,
) -> int:
    """Write the golden seed concepts (+ edges) into `store`.

    Returns the number of concept nodes written. Never raises on a
    missing/malformed fixture -- logs and returns 0 instead.
    """

    nodes, edges = load_seed_concept_nodes(fixture_path)

    for node in nodes:
        store.upsert_node(
            identity_key=f"concept|{node.anchor_scope}|{node.subject_ref}|seed",
            node=node,
        )

    for edge in edges:
        store.upsert_edge(
            identity_key=f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}",
            edge=edge,
        )

    return len(nodes)
