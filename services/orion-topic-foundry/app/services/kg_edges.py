from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID, uuid4

from app.services.enrichment_contract import MEANING_EDGE_PREDICATES, coerce_meaning
from app.storage.repository import fetch_model, fetch_run, fetch_segments, replace_edges_for_run, utc_now


logger = logging.getLogger("topic-foundry.kg-edges")


def generate_edges_for_run(run_id: UUID, *, min_confidence: float = 0.2) -> int:
    run_row = fetch_run(run_id)
    if not run_row:
        return 0
    model_row = fetch_model(UUID(run_row["model_id"]))
    model_name = model_row["name"] if model_row else "unknown"

    segments = fetch_segments(run_id, has_enrichment=True)
    edges: List[Dict[str, Any]] = []
    for segment in segments:
        edges.extend(_edges_from_segment(segment, model_name=model_name, min_confidence=min_confidence))

    # A run that enriched real segments and produced zero edges is always a
    # defect, and until now it was completely silent -- topic_foundry_edges
    # held 0 rows for all time with nothing anywhere saying why. There are two
    # distinct causes and the per-segment warning below only covers one of
    # them: `_heuristic_enrich` hardcodes questions/claims/next_steps/entities
    # to [], so every heuristically-enriched run yields exactly 0 edges with a
    # perfectly well-formed `meaning` object and no `unstructured` marker.
    # This check catches both, and names the enricher so they are
    # distinguishable.
    if segments and not edges:
        enrichers = sorted(
            {str(seg.get("enrichment_version") or "unknown") for seg in segments}
        )
        logger.warning(
            "kg_edges_run_produced_no_edges run_id=%s enriched_segments=%s enrichment_versions=%s "
            "-- every enriched segment yielded zero edges; check the enricher's `meaning` output "
            "against app/services/enrichment_contract.py",
            run_id,
            len(segments),
            ",".join(enrichers),
        )

    replace_edges_for_run(run_id=run_id, edges=edges)
    _write_edge_artifacts(run_row, edges)
    # Bus publish (orion:kg:edge:ingest.v1) retired 2026-07-28 -- zero live
    # consumers (orion-rdf-writer never actually subscribed; orion-graphdb
    # never existed as a real service). These edges now reach a real
    # consumer via GET /kg/edges, pulled by
    # orion-hub/scripts/concept_atlas_routes.py into the live Falkor
    # substrate graph instead (see
    # orion/substrate/adapters/topic_foundry.py's module docstring). Rows
    # still persist to Postgres (replace_edges_for_run above) and remain
    # queryable via GET /edges and /kg/edges -- only the dead broadcast is
    # gone.
    return len(edges)


def _edges_from_segment(
    segment: Dict[str, Any],
    *,
    model_name: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    meaning = coerce_meaning(segment.get("meaning")) or {}
    edges: List[Dict[str, Any]] = []
    created_at = utc_now()
    segment_id = UUID(segment["segment_id"])

    # Derived from MEANING_EDGE_PREDICATES rather than four hardcoded
    # meaning.get(...) calls, so renaming a key moves the prompt, the
    # coercion and this builder together. Before, a rename would have left
    # this reading the dead names and emitting 0 edges silently -- the exact
    # failure class this file's fix exists to remove.
    for key, (predicate, confidence) in MEANING_EDGE_PREDICATES.items():
        edges.extend(
            _edges_from_list(
                segment_id,
                model_name,
                predicate,
                meaning.get(key, []),
                confidence=confidence,
                created_at=created_at,
            )
        )

    # The old code caught JSONDecodeError and silently substituted {}. That is
    # one of the two reasons topic_foundry_edges had 0 rows for all time (see
    # generate_edges_for_run for the other). Gated on `not edges` rather than
    # on the marker alone, so this never claims "no edges" about a segment
    # that carried prose in `summary` AND real entities alongside it.
    if meaning.get("unstructured") and not edges:
        logger.warning(
            "kg_edges_segment_meaning_unstructured segment_id=%s -- enricher returned prose, "
            "not the object shape declared in app/services/enrichment_contract.py; no edges from this segment",
            segment.get("segment_id"),
        )

    return [edge for edge in edges if edge["confidence"] >= min_confidence]


def _edges_from_list(
    segment_id: UUID,
    subject: str,
    predicate: str,
    values: Any,
    *,
    confidence: float,
    created_at,
) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    edges: List[Dict[str, Any]] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        edges.append(
            {
                "edge_id": uuid4(),
                "segment_id": segment_id,
                "subject": _normalize_text(subject) or "unknown",
                "predicate": predicate,
                "object": normalized,
                "confidence": confidence,
                "created_at": created_at,
            }
        )
    return edges


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text[:120]


def _write_edge_artifacts(run_row: Dict[str, Any], edges: List[Dict[str, Any]]) -> None:
    run_dir = (run_row.get("artifact_paths") or {}).get("run_dir")
    if not run_dir:
        return
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    edges_path = run_path / "kg_edges.jsonl"
    with edges_path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, default=str) + "\n")
