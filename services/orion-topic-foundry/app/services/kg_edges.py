from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID, uuid4

from app.services.enrichment_contract import coerce_meaning
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
    raw_meaning = segment.get("meaning")
    meaning = coerce_meaning(raw_meaning) or {}
    # The old code caught JSONDecodeError and silently substituted {}. That is
    # why topic_foundry_edges had 0 rows for all time despite 554 enriched
    # segments: every prose `meaning` produced an empty edge list and nothing
    # anywhere said so. coerce_meaning() preserves prose under `summary`
    # instead of discarding it, and the warning below makes a segment that
    # can never yield edges visible in logs rather than indistinguishable
    # from a segment that legitimately has none.
    if meaning.get("unstructured"):
        logger.warning(
            "kg_edges_segment_meaning_unstructured segment_id=%s -- enricher returned prose, "
            "not the object shape declared in app/services/enrichment_contract.py; no edges from this segment",
            segment.get("segment_id"),
        )
    edges: List[Dict[str, Any]] = []
    created_at = utc_now()
    segment_id = UUID(segment["segment_id"])

    edges.extend(
        _edges_from_list(
            segment_id,
            model_name,
            "mentions",
            meaning.get("entities", []),
            confidence=0.6,
            created_at=created_at,
        )
    )
    edges.extend(
        _edges_from_list(
            segment_id,
            model_name,
            "asks_about",
            meaning.get("questions", []),
            confidence=0.5,
            created_at=created_at,
        )
    )
    edges.extend(
        _edges_from_list(
            segment_id,
            model_name,
            "claims_about",
            meaning.get("claims", []),
            confidence=0.7,
            created_at=created_at,
        )
    )
    edges.extend(
        _edges_from_list(
            segment_id,
            model_name,
            "next_step",
            meaning.get("next_steps", []),
            confidence=0.8,
            created_at=created_at,
        )
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
