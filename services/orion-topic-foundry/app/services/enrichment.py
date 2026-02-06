from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import requests

from app.models import EnrichmentSpec
from app.services.taxonomy import load_taxonomy
from app.services.bus_events import get_bus_publisher
from app.services.kg_edges import generate_edges_for_run
from app.settings import settings
from app.storage.repository import create_event, fetch_model, fetch_run, fetch_segments, update_run, update_segment_enrichment, utc_now
from orion.schemas.topic_foundry import TopicFoundryEnrichCompleteV1


logger = logging.getLogger("topic-foundry.enrichment")


def enqueue_enrichment(background_tasks, run_id: UUID, *, force: bool, enricher: Optional[str], limit: Optional[int]) -> None:
    background_tasks.add_task(_run_enrichment, run_id, force=force, enricher=enricher, limit=limit)


def run_enrichment_sync(run_id: UUID, *, force: bool, enricher: Optional[str], limit: Optional[int]) -> None:
    _run_enrichment(run_id, force=force, enricher=enricher, limit=limit)


def _run_enrichment(run_id: UUID, *, force: bool, enricher: Optional[str], limit: Optional[int]) -> None:
    run_row = fetch_run(run_id)
    if not run_row:
        return
    stats = run_row.get("stats") or {}
    started = utc_now()
    status = run_row.get("status") or "complete"
    if status == "failed":
        logger.warning("Skipping enrichment for failed run_id=%s", run_id)
        return

    update_run_payload = _build_run_record_for_update(run_row, stage="enriching", status="running")
    update_run(update_run_payload)

    spec = _load_enrichment_spec(run_row)
    taxonomy = load_taxonomy(spec.aspect_taxonomy)
    chosen_enricher = enricher or ("llm" if settings.topic_foundry_llm_enable else "heuristic")

    segments = fetch_segments(UUID(run_row["run_id"]), has_enrichment=None)
    if not force:
        segments = [seg for seg in segments if seg.get("enriched_at") is None]
    if limit:
        segments = segments[:limit]
    text_map = _load_segment_text_map(run_row)

    enriched_count = 0
    failed_count = 0
    enriched_payloads: List[Dict[str, Any]] = []
    for segment in segments:
        try:
            enrichment = _enrich_segment(segment, taxonomy, chosen_enricher, text_map)
            update_segment_enrichment(UUID(segment["segment_id"]), enrichment=enrichment, enrichment_version="v1")
            enriched_count += 1
            enriched_payloads.append({"segment_id": segment["segment_id"], "enrichment": enrichment})
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            logger.warning("Enrichment failed segment_id=%s error=%s", segment.get("segment_id"), exc)

    stats["segments_enriched"] = stats.get("segments_enriched", 0) + enriched_count
    stats["enrichment_failed"] = stats.get("enrichment_failed", 0) + failed_count
    stats["enrichment_secs"] = stats.get("enrichment_secs", 0) + _elapsed_secs(started)

    update_run_payload = _build_run_record_for_update(run_row, stage="enriched", status=status)
    update_run_payload.stats = stats
    update_run(update_run_payload)

    _write_enrichment_artifacts(run_row, enriched_payloads, taxonomy, chosen_enricher, enriched_count, failed_count)
    _publish_enrich_complete(run_row, enriched_count, failed_count)
    _generate_edges(run_row)


def _publish_enrich_complete(run_row: Dict[str, Any], enriched_count: int, failed_count: int) -> None:
    model_row = fetch_model(UUID(run_row["model_id"]))
    model_name = model_row["name"] if model_row else "unknown"
    model_version = model_row["version"] if model_row else "unknown"
    payload = TopicFoundryEnrichCompleteV1(
        run_id=UUID(run_row["run_id"]),
        model_id=UUID(run_row["model_id"]),
        dataset_id=UUID(run_row["dataset_id"]),
        model_name=model_name,
        model_version=model_version,
        status=run_row.get("status", "complete"),
        enriched_count=enriched_count,
        failed_count=failed_count,
        completed_at=run_row.get("completed_at"),
    )
    get_bus_publisher().publish_enrich_complete(payload)
    create_event(
        event_id=uuid4(),
        kind="enrich.complete",
        run_id=UUID(run_row["run_id"]),
        model_id=UUID(run_row["model_id"]),
        drift_id=None,
        payload=payload.model_dump(mode="json"),
        bus_status="queued" if settings.orion_bus_enabled else "disabled",
        bus_error=None,
        created_at=utc_now(),
    )


def _generate_edges(run_row: Dict[str, Any]) -> None:
    try:
        generate_edges_for_run(UUID(run_row["run_id"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("KG edge generation failed run_id=%s error=%s", run_row.get("run_id"), exc)


def _build_run_record_for_update(run_row: Dict[str, Any], *, stage: str, status: str):
    from app.models import RunRecord, RunSpecSnapshot, EnrichmentSpec, ModelSpec, WindowingSpec, DatasetSpec

    specs = run_row.get("specs") or {}
    run = RunRecord(
        run_id=UUID(run_row["run_id"]),
        model_id=UUID(run_row["model_id"]),
        dataset_id=UUID(run_row["dataset_id"]),
        specs=RunSpecSnapshot(
            dataset=DatasetSpec(**specs["dataset"]),
            windowing=WindowingSpec(**specs["windowing"]),
            model=ModelSpec(**specs["model"]),
            enrichment=EnrichmentSpec(**specs.get("enrichment", {})),
        ),
        spec_hash=run_row.get("spec_hash"),
        status=status,
        stage=stage,
        stats=run_row.get("stats") or {},
        artifact_paths=run_row.get("artifact_paths") or {},
        created_at=run_row.get("created_at"),
        started_at=run_row.get("started_at"),
        completed_at=run_row.get("completed_at"),
        error=run_row.get("error"),
    )
    return run


def _load_enrichment_spec(run_row: Dict[str, Any]) -> EnrichmentSpec:
    specs = run_row.get("specs") or {}
    enrichment = specs.get("enrichment") or {}
    return EnrichmentSpec(**enrichment)


def _enrich_segment(
    segment: Dict[str, Any], taxonomy: List[str], enricher: str, text_map: Dict[str, str]
) -> Dict[str, Any]:
    text = _segment_text(segment, text_map)
    if enricher == "llm" and settings.topic_foundry_llm_enable:
        result = _llm_enrich(text, taxonomy)
        if result is not None:
            return _finalize_enrichment(result)
    return _finalize_enrichment(_heuristic_enrich(text, taxonomy))


def _segment_text(segment: Dict[str, Any], text_map: Dict[str, str]) -> str:
    segment_id = str(segment.get("segment_id"))
    return text_map.get(segment_id, "")


def _heuristic_enrich(text: str, taxonomy: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    aspects = [aspect for aspect in taxonomy if aspect.lower() in lower][:5]
    title = text.split(".")[0][:60].strip() if text else "untitled"
    sentiment = {
        "valence": -0.2 if "error" in lower or "fail" in lower else 0.2,
        "arousal": 0.4 if "!" in text else 0.2,
        "stance": 0.0,
        "uncertainty": 0.6 if any(word in lower for word in ["maybe", "guess", "unsure"]) else 0.2,
        "friction": 0.7 if any(word in lower for word in ["stuck", "blocked", "angry", "frustrated"]) else 0.1,
    }
    meaning = {
        "intent": "debug" if "error" in lower else "explore",
        "outcome": "unknown",
        "questions": [],
        "claims": [],
        "next_steps": [],
        "entities": [],
    }
    return {
        "title": title,
        "aspects": aspects,
        "aspect_scores": {aspect: 0.6 for aspect in aspects},
        "sentiment": sentiment,
        "meaning": meaning,
        "evidence_spans": _extract_evidence(text),
    }


def _llm_enrich(text: str, taxonomy: List[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    url = settings.topic_foundry_llm_base_url.rstrip("/") + "/chat/completions"
    if settings.topic_foundry_llm_route:
        url = settings.topic_foundry_llm_base_url.rstrip("/") + settings.topic_foundry_llm_route
    payload = {
        "model": settings.topic_foundry_llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an analyst. Return STRICT JSON only.",
            },
            {
                "role": "user",
                "content": _llm_prompt(text, taxonomy),
            },
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(url, json=payload, timeout=settings.topic_foundry_llm_timeout_secs)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM enrichment failed; falling back to heuristic error=%s", exc)
        return None


def _llm_prompt(text: str, taxonomy: List[str]) -> str:
    return (
        "Enrich this segment. Provide JSON with keys: title, aspects, aspect_scores, sentiment, meaning, evidence_spans."
        f"\nTaxonomy: {taxonomy}\nText:\n{text}\n"
    )


def _extract_evidence(text: str) -> List[str]:
    words = text.split()
    if not words:
        return []
    span = " ".join(words[:20])
    return [span]


def _finalize_enrichment(raw: Dict[str, Any]) -> Dict[str, Any]:
    result = raw.copy()
    result.setdefault("title", "untitled")
    result.setdefault("aspects", [])
    result.setdefault("aspect_scores", {})
    result.setdefault("sentiment", {})
    result.setdefault("meaning", {})
    result.setdefault("evidence_spans", [])
    return result


def _write_enrichment_artifacts(
    run_row: Dict[str, Any],
    enriched_payloads: List[Dict[str, Any]],
    taxonomy: List[str],
    enricher: str,
    enriched_count: int,
    failed_count: int,
) -> None:
    run_dir = (run_row.get("artifact_paths") or {}).get("run_dir")
    if not run_dir:
        return
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    enriched_path = run_path / "segments_enriched.jsonl"
    with enriched_path.open("w", encoding="utf-8") as handle:
        for payload in enriched_payloads:
            handle.write(json.dumps(payload) + "\n")

    meta = {
        "enricher": enricher,
        "taxonomy": taxonomy,
        "segments_enriched": enriched_count,
        "failed": failed_count,
        "generated_at": utc_now().isoformat(),
    }
    (run_path / "enrichment_meta.json").write_text(json.dumps(meta, indent=2))


def _load_segment_text_map(run_row: Dict[str, Any]) -> Dict[str, str]:
    run_dir = (run_row.get("artifact_paths") or {}).get("run_dir")
    if not run_dir:
        return {}
    docs_path = Path(run_dir) / "documents.jsonl"
    if not docs_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with docs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            segment_id = str(payload.get("segment_id") or payload.get("doc_id"))
            text = str(payload.get("text", ""))
            if segment_id and text:
                mapping[segment_id] = text
    return mapping


def _elapsed_secs(started: datetime) -> float:
    return (utc_now() - started).total_seconds()
