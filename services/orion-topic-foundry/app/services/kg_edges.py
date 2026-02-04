from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID, uuid4

from app.services.bus_events import get_bus_publisher
from app.storage.repository import fetch_model, fetch_run, fetch_segments, replace_edges_for_run, utc_now
from orion.schemas.topic_foundry import KgEdgeIngestItemV1, KgEdgeIngestV1


logger = logging.getLogger("topic-foundry.kg-edges")


def generate_edges_for_run(run_id: UUID, *, min_confidence: float = 0.2) -> int:
    run_row = fetch_run(run_id)
    if not run_row:
        return 0
    model_row = fetch_model(UUID(run_row["model_id"]))
    model_name = model_row["name"] if model_row else "unknown"
    model_id = UUID(run_row["model_id"])

    segments = fetch_segments(run_id, has_enrichment=True)
    edges: List[Dict[str, Any]] = []
    for segment in segments:
        edges.extend(_edges_from_segment(segment, model_name=model_name, min_confidence=min_confidence))

    replace_edges_for_run(run_id=run_id, edges=edges)
    _write_edge_artifacts(run_row, edges)
    _publish_edge_batch(edges, run_id=run_id, model_id=model_id, model_name=model_name)
    return len(edges)


def _edges_from_segment(
    segment: Dict[str, Any],
    *,
    model_name: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    meaning = segment.get("meaning") or {}
    if isinstance(meaning, str):
        try:
            meaning = json.loads(meaning)
        except json.JSONDecodeError:
            meaning = {}
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


def _publish_edge_batch(edges: List[Dict[str, Any]], *, run_id: UUID, model_id: UUID, model_name: str) -> None:
    if not edges:
        return
    payload = KgEdgeIngestV1(
        run_id=run_id,
        model_id=model_id,
        model_name=model_name,
        edges=[
            KgEdgeIngestItemV1(
                edge_id=edge["edge_id"],
                segment_id=edge["segment_id"],
                subject=edge["subject"],
                predicate=edge["predicate"],
                object=edge["object"],
                confidence=edge["confidence"],
                created_at=edge["created_at"],
            )
            for edge in edges
        ],
    )
    get_bus_publisher().publish_kg_edges(payload)
