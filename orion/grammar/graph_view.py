"""Layer/dimension summaries and Cytoscape-friendly graph payloads (spec §11.3)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from orion.grammar.constants import GRAMMAR_DIMENSIONS, GRAMMAR_LAYERS

_LAYER_INDEX = {layer: idx for idx, layer in enumerate(GRAMMAR_LAYERS)}


def layer_index(layer: str) -> int:
    """Vertical stack order; unknown layers sort after canonical layers."""
    return _LAYER_INDEX.get(layer, len(GRAMMAR_LAYERS))


def layer_y_position(layer: str, *, row_height: float = 80.0) -> float:
    """Y hint for layered layout (lower index = higher on screen)."""
    return float(layer_index(layer)) * row_height


def build_layer_summary(atoms: Iterable[Any]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for atom in atoms:
        layer = getattr(atom, "layer", None)
        if layer:
            counts[layer] += 1

    return [
        {
            "layer": layer,
            "atom_count": counts.get(layer, 0),
            "index": layer_index(layer),
        }
        for layer in GRAMMAR_LAYERS
        if counts.get(layer, 0) > 0
    ] + [
        {
            "layer": layer,
            "atom_count": count,
            "index": layer_index(layer),
        }
        for layer, count in sorted(counts.items())
        if layer not in GRAMMAR_LAYERS
    ]


def build_dimension_summary(atoms: Iterable[Any]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for atom in atoms:
        dims = getattr(atom, "dimensions", None) or []
        for dim in dims:
            counts[dim] += 1

    return [
        {
            "dimension": dimension,
            "atom_count": counts.get(dimension, 0),
        }
        for dimension in GRAMMAR_DIMENSIONS
        if counts.get(dimension, 0) > 0
    ] + [
        {
            "dimension": dimension,
            "atom_count": count,
        }
        for dimension, count in sorted(counts.items())
        if dimension not in GRAMMAR_DIMENSIONS
    ]


def atom_row_to_node(atom: Any, *, layout: str = "layered") -> dict[str, Any]:
    layer = getattr(atom, "layer", "")
    node: dict[str, Any] = {
        "id": atom.atom_id,
        "type": atom.atom_type,
        "label": atom.summary,
        "layer": layer,
        "dimensions": list(atom.dimensions or []),
        "confidence": atom.confidence,
        "salience": atom.salience,
        "semantic_role": atom.semantic_role,
    }
    if layout == "layered":
        node["y"] = layer_y_position(layer)
    return node


def edge_row_to_edge(edge: Any) -> dict[str, Any]:
    return {
        "id": edge.edge_id,
        "source": edge.from_atom_id,
        "target": edge.to_atom_id,
        "type": edge.relation_type,
        "confidence": edge.confidence,
        "salience": edge.salience,
        "layer_from": edge.layer_from,
        "layer_to": edge.layer_to,
        "temporal_relation": edge.temporal_relation,
    }


def build_layer_groups(atoms: Iterable[Any]) -> list[dict[str, Any]]:
    return build_layer_summary(atoms)


def build_dimension_groups(atoms: Iterable[Any]) -> list[dict[str, Any]]:
    return build_dimension_summary(atoms)
