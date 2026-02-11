from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from bertopic import BERTopic
from joblib import dump

from app.models import DatasetSpec, EnrichmentSpec, ModelSpec, RunRecord, RunSpecSnapshot, RunTrainRequest, SegmentRecord, WindowingSpec
from app.services.bus_events import get_bus_publisher
from app.services.conversation_overrides import OverrideRecord, apply_overrides, build_conversations
from app.services.data_access import fetch_dataset_rows
from app.services.enrichment import run_enrichment_sync
from app.services.topic_engine import TopicEngineError, build_embedder, build_topic_model
from app.services.types import BoundaryContext, RowBlock
from app.services.windowing import build_segments_with_stats
from app.settings import settings
from app.storage.repository import (
    create_event,
    fetch_latest_completed_run_by_scope,
    fetch_run,
    fetch_topics,
    insert_segments,
    insert_topics,
    list_conversation_overrides,
    update_run,
    upsert_conversation_rollups,
    utc_now,
)
from orion.schemas.topic_foundry import TopicFoundryRunCompleteV1

logger = logging.getLogger("topic-foundry.training")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _timestamp_bounds(timestamps: List[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    parsed = [dt for ts in timestamps if (dt := _parse_timestamp(ts)) is not None]
    if not parsed:
        return None, None
    return min(parsed), max(parsed)


def _snippet(text: str, max_chars: int = 400) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def enqueue_training(background_tasks, run_id: UUID, payload: RunTrainRequest, model_row: Dict[str, Any], dataset: DatasetSpec, spec_hash: str) -> None:
    background_tasks.add_task(_run_training, run_id, payload, model_row, dataset, spec_hash)


def _validate_mode_inputs(topic_mode: str, params: Dict[str, Any], segments: List[RowBlock]) -> None:
    if topic_mode == "guided" and not params.get("seed_topic_list"):
        raise TopicEngineError("guided mode requires topic_mode_params.seed_topic_list")
    if topic_mode == "zeroshot" and not params.get("zeroshot_topic_list"):
        raise TopicEngineError("zeroshot mode requires topic_mode_params.zeroshot_topic_list")
    if topic_mode == "class_based":
        classes = params.get("classes")
        if not isinstance(classes, list) or len(classes) != len(segments):
            raise TopicEngineError("class_based mode requires topic_mode_params.classes aligned to docs")
    if topic_mode == "dynamic":
        timestamps = [_parse_timestamp((seg.timestamps or [None])[0]) for seg in segments]
        if not all(ts is not None for ts in timestamps):
            raise TopicEngineError("dynamic mode requires timestamps per doc")


def _run_training(run_id: UUID, payload: RunTrainRequest, model_row: Dict[str, Any], dataset: DatasetSpec, spec_hash: str) -> None:
    run = _build_run_record(run_id, payload, model_row, dataset, spec_hash)
    run.status = "running"
    run.stage = "training"
    run.started_at = utc_now()
    update_run(run)

    try:
        start = time.monotonic()
        segments, _ = _prepare_segments(run, payload)
        if not segments:
            raise RuntimeError("No documents available for training")

        docs = [seg.text for seg in segments]
        model_meta = dict(model_row.get("model_meta") or run.specs.model.model_meta or {})
        topic_mode = payload.topic_mode or str(model_meta.get("topic_mode") or "standard")
        topic_mode_params = dict(payload.topic_mode_params or model_meta.get("topic_mode_params") or {})
        _validate_mode_inputs(topic_mode, topic_mode_params, segments)

        embedder = build_embedder(model_meta, settings)
        embeddings = embedder(docs)
        topic_model: BERTopic = build_topic_model(model_meta, settings, topic_mode, topic_mode_params)

        fit_kwargs: Dict[str, Any] = {}
        embedding_backend = str(model_meta.get("embedding_backend") or settings.topic_foundry_embedding_backend)
        if embedding_backend == "vector_host":
            fit_kwargs["embeddings"] = embeddings

        classes = topic_mode_params.get("classes") if topic_mode == "class_based" else None
        topics, probs = topic_model.fit_transform(docs, y=classes, **fit_kwargs)

        topics_arr = np.asarray(topics, dtype=int)
        probs_arr = None
        if probs is not None:
            probs_np = np.asarray(probs)
            if probs_np.ndim == 2:
                probs_arr = probs_np.max(axis=1)
            else:
                probs_arr = probs_np

        info = topic_model.get_topic_info().to_dict(orient="records")
        keywords = {
            str(int(row["Topic"])): [word for word, _ in (topic_model.get_topic(int(row["Topic"])) or [])]
            for row in info
            if row.get("Topic") is not None
        }

        stats = _compute_stats(topics_arr)
        stats.update({
            "total_secs": time.monotonic() - start,
            "topic_mode": topic_mode,
            "embedding_backend": embedding_backend,
            "representation": model_meta.get("representation") or settings.topic_foundry_representation,
            "reducer": model_meta.get("reducer") or settings.topic_foundry_reducer,
            "clusterer": model_meta.get("clusterer") or settings.topic_foundry_clusterer,
        })

        artifacts = _write_artifacts(run, model_row, docs, topic_model, topics_arr, probs_arr, info, keywords, topic_mode)

        segment_records: List[SegmentRecord] = []
        rollups: Dict[str, Any] = {}
        for idx, seg in enumerate(segments):
            start_at, end_at = _timestamp_bounds(seg.timestamps)
            label = int(topics_arr[idx])
            prob = float(probs_arr[idx]) if probs_arr is not None and idx < len(probs_arr) else None
            segment_records.append(
                SegmentRecord(
                    segment_id=UUID(seg.doc_id),
                    run_id=run.run_id,
                    size=len(seg.row_ids),
                    provenance={"row_ids": seg.row_ids, "timestamps": seg.timestamps, "doc_ids": [seg.doc_id], "conversation_id": str(seg.conversation_id) if seg.conversation_id else None},
                    topic_id=label,
                    topic_prob=prob,
                    is_outlier=label == -1,
                    snippet=_snippet(seg.text or "") or None,
                    chars=len(seg.text or ""),
                    row_ids_count=len(seg.row_ids),
                    start_at=start_at,
                    end_at=end_at,
                    created_at=utc_now(),
                )
            )
            if seg.conversation_id:
                key = str(seg.conversation_id)
                convo = rollups.setdefault(key, {"conversation_id": key, "topic_counts": {}, "segment_count": 0})
                convo["segment_count"] += 1
                convo["topic_counts"][str(label)] = convo["topic_counts"].get(str(label), 0) + 1
        insert_segments(segment_records)
        if rollups:
            upsert_conversation_rollups(run.run_id, rollups)

        topic_payloads = _build_topic_payloads(topics_arr, embeddings, scope=run.run_scope or "macro")
        if run.run_scope == "micro":
            macro_run = fetch_latest_completed_run_by_scope(run.model_id, "macro")
            if macro_run:
                _map_micro_to_macro(topic_payloads, fetch_topics(UUID(macro_run["run_id"]), scope="macro"))
        insert_topics(run.run_id, topic_payloads)

        run.stage = "trained"
        if run.specs.enrichment.enable_enrichment:
            run.stage = "enriching"
            update_run(run)
            run_enrichment_sync(run.run_id)
            run.stage = "enriched"
            latest = fetch_run(run.run_id)
            if latest and latest.get("stats"):
                stats = latest["stats"]
        run.status = "complete"
        run.stats = stats
        run.artifact_paths = artifacts
        run.completed_at = utc_now()
        update_run(run)
        _publish_run_complete(run, model_row)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training run failed run_id=%s", run_id)
        run.status = "failed"
        run.stage = "failed"
        run.error = str(exc)
        run.completed_at = utc_now()
        update_run(run)


def _publish_run_complete(run: RunRecord, model_row: Dict[str, Any]) -> None:
    payload = TopicFoundryRunCompleteV1(
        run_id=run.run_id,
        model_id=run.model_id,
        dataset_id=run.dataset_id,
        model_name=model_row.get("name", "unknown"),
        model_version=model_row.get("version", "unknown"),
        status=run.status,
        stats=run.stats or {},
        completed_at=run.completed_at,
    )
    get_bus_publisher().publish_run_complete(payload)
    create_event(event_id=uuid4(), kind="run.complete", run_id=run.run_id, model_id=run.model_id, drift_id=None, payload=payload.model_dump(mode="json"), bus_status="queued" if settings.orion_bus_enabled else "disabled", bus_error=None, created_at=utc_now())


def _build_run_record(run_id: UUID, payload: RunTrainRequest, model_row: Dict[str, Any], dataset: DatasetSpec, spec_hash: str) -> RunRecord:
    specs = RunSpecSnapshot(
        dataset=dataset,
        windowing=WindowingSpec(**model_row["windowing_spec"]),
        model=ModelSpec(**model_row["model_spec"]),
        enrichment=EnrichmentSpec(**model_row["enrichment_spec"]) if model_row.get("enrichment_spec") else EnrichmentSpec(),
        run_scope=payload.run_scope,
    )
    if specs.run_scope is None:
        specs.run_scope = "micro" if specs.windowing.windowing_mode == "conversation_bound" else "macro"
    return RunRecord(run_id=run_id, model_id=UUID(model_row["model_id"]), dataset_id=dataset.dataset_id, specs=specs, spec_hash=spec_hash, status="queued", run_scope=specs.run_scope, stats={}, artifact_paths={}, created_at=utc_now())


def _prepare_segments(run: RunRecord, payload: RunTrainRequest) -> tuple[List[RowBlock], int]:
    rows = fetch_dataset_rows(dataset=run.specs.dataset, start_at=payload.start_at, end_at=payload.end_at, limit=10_000)
    conversations = build_conversations(rows, dataset_id=run.dataset_id, spec=run.specs.windowing, text_columns=run.specs.dataset.text_columns, time_column=run.specs.dataset.time_column, id_column=run.specs.dataset.id_column, boundary_column=run.specs.dataset.boundary_column)
    overrides = [OverrideRecord(override_id=UUID(row["override_id"]), kind=row["kind"], payload=row["payload"], created_at=row["created_at"]) for row in list_conversation_overrides(run.dataset_id)]
    if overrides:
        conversations = apply_overrides(conversations, overrides)
    boundary_context = BoundaryContext(run_id=run.run_id, spec_hash=run.spec_hash, dataset_id=run.dataset_id, model_id=run.model_id, run_dir=str(Path(settings.topic_foundry_model_dir) / "runs" / str(run.run_id)))
    return build_segments_with_stats(conversations, spec=run.specs.windowing, embedding_url=run.specs.model.embedding_source_url or settings.topic_foundry_embedding_url, boundary_context=boundary_context, run_id=run.run_id)


def _compute_stats(labels: np.ndarray) -> Dict[str, Any]:
    doc_count = len(labels)
    cluster_count = len([label for label in set(int(v) for v in labels) if label >= 0])
    outlier_count = int(np.sum(labels == -1))
    outlier_rate = float(outlier_count / doc_count) if doc_count else 0.0
    return {"doc_count": doc_count, "segment_count": doc_count, "docs_generated": doc_count, "segments_generated": doc_count, "cluster_count": cluster_count, "outlier_count": outlier_count, "outlier_rate": outlier_rate, "outlier_pct": outlier_rate}


def _build_topic_payloads(labels: np.ndarray, embeddings: np.ndarray, *, scope: str) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    if embeddings.size == 0:
        return payloads
    for label in sorted(set(int(v) for v in labels)):
        mask = labels == label
        if not mask.any():
            continue
        payloads.append({"topic_id": label, "scope": scope, "parent_topic_id": None, "centroid": embeddings[mask].mean(axis=0).tolist(), "count": int(mask.sum()), "label": None})
    return payloads


def _map_micro_to_macro(micro_topics: List[Dict[str, Any]], macro_topics: List[Dict[str, Any]]) -> None:
    if not macro_topics:
        return
    macro_centroids = {int(topic["topic_id"]): np.array(topic["centroid"], dtype=np.float32) for topic in macro_topics if topic.get("centroid") is not None}
    if not macro_centroids:
        return
    ids = list(macro_centroids.keys())
    matrix = np.stack([macro_centroids[i] for i in ids], axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    for topic in micro_topics:
        vec = np.array(topic.get("centroid") or [], dtype=np.float32)
        if vec.size == 0:
            continue
        denom = np.linalg.norm(vec) * norms
        scores = (matrix @ vec) / np.where(denom == 0, 1.0, denom)
        topic["parent_topic_id"] = int(ids[int(np.argmax(scores))])


def _write_artifacts(run: RunRecord, model_row: Dict[str, Any], docs: List[str], topic_model: BERTopic, labels: np.ndarray, probabilities: Optional[np.ndarray], topic_info: List[Dict[str, Any]], topics_keywords: Dict[str, List[str]], topic_mode: str) -> Dict[str, Any]:
    base_dir = Path(settings.topic_foundry_model_dir)
    model_name = model_row["name"]
    model_version = model_row["version"]
    registry_dir = base_dir / "registry" / model_name / "versions" / model_version
    model_dir = registry_dir / "model"
    run_dir = base_dir / "runs" / str(run.run_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dump(topic_model, model_dir / "bertopic.joblib")
    (registry_dir / "settings.json").write_text(json.dumps(run.specs.model_dump(mode="json"), indent=2))
    (registry_dir / "model_meta.json").write_text(json.dumps(model_row.get("model_meta") or {}, indent=2))

    docs_path = run_dir / "documents.jsonl"
    with docs_path.open("w", encoding="utf-8") as fh:
        for idx, text in enumerate(docs):
            payload = {"doc_id": idx, "text": text, "topic_id": int(labels[idx]), "topic_prob": float(probabilities[idx]) if probabilities is not None and idx < len(probabilities) else None}
            fh.write(json.dumps(payload) + "\n")

    topics_summary = []
    for row in topic_info:
        topic_id = int(row.get("Topic", -1))
        topics_summary.append({"topic_id": topic_id, "count": int(row.get("Count", 0)), "label": row.get("Name")})

    (run_dir / "topics_summary.json").write_text(json.dumps(topics_summary, indent=2))
    (run_dir / "topics_keywords.json").write_text(json.dumps(topics_keywords, indent=2))

    artifacts: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "documents": str(docs_path),
        "topics_summary": str(run_dir / "topics_summary.json"),
        "topics_keywords": str(run_dir / "topics_keywords.json"),
    }

    if topic_mode == "hierarchical":
        hierarchy = topic_model.hierarchical_topics(docs).to_dict(orient="records")
        path = run_dir / "hierarchical_topics.json"
        path.write_text(json.dumps(hierarchy, indent=2))
        artifacts["hierarchical_topics"] = str(path)
    if topic_mode == "dynamic":
        timestamps = [_parse_timestamp((row.timestamps or [None])[0]) for row in run.specs.dataset.text_columns]  # placeholder safe
        _ = timestamps
    return artifacts
