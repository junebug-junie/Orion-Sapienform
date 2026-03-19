from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from hdbscan import HDBSCAN
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models import DatasetSpec, EnrichmentSpec, ModelSpec, RunRecord, RunSpecSnapshot, RunTrainRequest, SegmentRecord, WindowingSpec
from app.services.data_access import fetch_dataset_rows
from app.services.conversation_overrides import OverrideRecord, apply_overrides, build_conversations
from app.services.embedding_client import VectorHostEmbeddingProvider
from app.services.bus_events import get_bus_publisher
from app.services.enrichment import run_enrichment_sync
from app.services.types import BoundaryContext, RowBlock
from app.services.windowing import build_segments_with_stats
from app.settings import settings
from app.storage.repository import create_event, fetch_run, insert_segments, list_conversation_overrides, update_run, utc_now
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
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _compute_topic_artifacts(
    segments: List[RowBlock],
    labels: np.ndarray,
    *,
    max_keywords: int = 12,
) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    summary_counts: Dict[int, int] = {}
    texts_by_topic: Dict[int, List[str]] = {}
    for seg, label in zip(segments, labels, strict=False):
        topic_id = int(label)
        summary_counts[topic_id] = summary_counts.get(topic_id, 0) + 1
        if seg.text:
            texts_by_topic.setdefault(topic_id, []).append(seg.text)

    summary = []
    for topic_id, count in sorted(summary_counts.items(), key=lambda x: (-x[1], x[0])):
        outlier_pct = 1.0 if topic_id == -1 else 0.0
        summary.append({"topic_id": topic_id, "count": count, "outlier_pct": outlier_pct, "label": None})

    keywords: Dict[str, List[str]] = {}
    topic_texts = [(tid, texts) for tid, texts in texts_by_topic.items() if tid != -1 and texts]
    if topic_texts:
        all_texts = [text for _, texts in topic_texts for text in texts]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(all_texts)
        vocab = np.array(vectorizer.get_feature_names_out())
        offset = 0
        for topic_id, texts in topic_texts:
            count = len(texts)
            topic_matrix = tfidf[offset : offset + count]
            offset += count
            scores = np.asarray(topic_matrix.mean(axis=0)).ravel()
            top_idx = scores.argsort()[::-1][:max_keywords]
            keywords[str(topic_id)] = [vocab[i] for i in top_idx if scores[i] > 0]

    return summary, keywords


def enqueue_training(
    background_tasks,
    run_id: UUID,
    payload: RunTrainRequest,
    model_row: Dict[str, Any],
    dataset: DatasetSpec,
    spec_hash: str,
) -> None:
    background_tasks.add_task(_run_training, run_id, payload, model_row, dataset, spec_hash)


def _run_training(
    run_id: UUID,
    payload: RunTrainRequest,
    model_row: Dict[str, Any],
    dataset: DatasetSpec,
    spec_hash: str,
) -> None:
    run = _build_run_record(run_id, payload, model_row, dataset, spec_hash)
    run.status = "running"
    run.stage = "training"
    run.started_at = utc_now()
    update_run(run)

    try:
        start = time.monotonic()
        segments, blocks_generated = _prepare_segments(run, payload)
        if not segments:
            raise RuntimeError("No documents available for training")
        doc_texts = [seg.text for seg in segments]
        embedder = VectorHostEmbeddingProvider(run.specs.model.embedding_source_url)
        embed_start = time.monotonic()
        embeddings = np.array(embedder.embed_texts(doc_texts), dtype=np.float32)
        embed_secs = time.monotonic() - embed_start

        cluster_start = time.monotonic()
        clusterer = _build_clusterer(run.specs.model)
        labels = clusterer.fit_predict(embeddings)
        probabilities = getattr(clusterer, "probabilities_", None)
        cluster_secs = time.monotonic() - cluster_start

        stats = _compute_stats(labels, segments)
        stats["blocks_generated"] = blocks_generated
        stats.update(
            {
                "embed_secs": embed_secs,
                "cluster_secs": cluster_secs,
                "total_secs": time.monotonic() - start,
            }
        )

        artifact_paths = _write_artifacts(
            run,
            model_row,
            segments,
            labels,
            probabilities,
            clusterer,
            embedder,
            stats,
        )

        segment_records = []
        for idx, seg in enumerate(segments):
            start_at, end_at = _timestamp_bounds(seg.timestamps)
            snippet = _snippet(seg.text or "")
            label = int(labels[idx])
            topic_prob = None
            if probabilities is not None and idx < len(probabilities):
                topic_prob = float(probabilities[idx])
            is_outlier = label == -1
            segment_records.append(
                SegmentRecord(
                    segment_id=UUID(seg.doc_id),
                    run_id=run.run_id,
                    size=len(seg.row_ids),
                    provenance={
                        "row_ids": seg.row_ids,
                        "timestamps": seg.timestamps,
                        "doc_ids": [seg.doc_id],
                    },
                    topic_id=label,
                    topic_prob=topic_prob,
                    is_outlier=is_outlier,
                    snippet=snippet or None,
                    chars=len(seg.text or ""),
                    row_ids_count=len(seg.row_ids),
                    start_at=start_at,
                    end_at=end_at,
                    created_at=utc_now(),
                )
            )
        insert_segments(segment_records)

        run.stage = "trained"
        if run.specs.enrichment.enable_enrichment:
            run.stage = "enriching"
            update_run(run)
            run_enrichment_sync(run.run_id, force=False, enricher=run.specs.enrichment.enricher, limit=None)
            run.stage = "enriched"
            latest = fetch_run(run.run_id)
            if latest and latest.get("stats"):
                stats = latest["stats"]
        run.status = "complete"
        run.stats = stats
        run.artifact_paths = artifact_paths
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
    model_name = model_row.get("name", "unknown")
    model_version = model_row.get("version", "unknown")
    payload = TopicFoundryRunCompleteV1(
        run_id=run.run_id,
        model_id=run.model_id,
        dataset_id=run.dataset_id,
        model_name=model_name,
        model_version=model_version,
        status=run.status,
        stats=run.stats or {},
        completed_at=run.completed_at,
    )
    get_bus_publisher().publish_run_complete(payload)
    create_event(
        event_id=uuid4(),
        kind="run.complete",
        run_id=run.run_id,
        model_id=run.model_id,
        drift_id=None,
        payload=payload.model_dump(mode="json"),
        bus_status="queued" if settings.orion_bus_enabled else "disabled",
        bus_error=None,
        created_at=utc_now(),
    )


def _build_run_record(
    run_id: UUID, payload: RunTrainRequest, model_row: Dict[str, Any], dataset: DatasetSpec, spec_hash: str
) -> RunRecord:
    specs = RunSpecSnapshot(
        dataset=dataset,
        windowing=WindowingSpec(**model_row["windowing_spec"]),
        model=ModelSpec(**model_row["model_spec"]),
        enrichment=EnrichmentSpec(**model_row["enrichment_spec"]) if model_row.get("enrichment_spec") else EnrichmentSpec(),
    )
    return RunRecord(
        run_id=run_id,
        model_id=UUID(model_row["model_id"]),
        dataset_id=dataset.dataset_id,
        specs=specs,
        spec_hash=spec_hash,
        status="queued",
        stats={},
        artifact_paths={},
        created_at=utc_now(),
    )


def _prepare_segments(run: RunRecord, payload: RunTrainRequest) -> tuple[List[RowBlock], int]:
    rows = fetch_dataset_rows(
        dataset=run.specs.dataset,
        start_at=payload.start_at,
        end_at=payload.end_at,
        limit=10_000,
    )
    conversations = build_conversations(
        rows,
        dataset_id=run.dataset_id,
        spec=run.specs.windowing,
        text_columns=run.specs.dataset.text_columns,
        time_column=run.specs.dataset.time_column,
        id_column=run.specs.dataset.id_column,
    )
    overrides = [
        OverrideRecord(
            override_id=UUID(row["override_id"]),
            kind=row["kind"],
            payload=row["payload"],
            created_at=row["created_at"],
        )
        for row in list_conversation_overrides(run.dataset_id)
    ]
    if overrides:
        conversations = apply_overrides(conversations, overrides)
    run_dir = str(Path(settings.topic_foundry_model_dir) / "runs" / str(run.run_id))
    boundary_context = BoundaryContext(
        run_id=run.run_id,
        spec_hash=run.spec_hash,
        dataset_id=run.dataset_id,
        model_id=run.model_id,
        run_dir=run_dir,
    )
    segments, blocks_generated = build_segments_with_stats(
        conversations,
        spec=run.specs.windowing,
        embedding_url=run.specs.model.embedding_source_url,
        boundary_context=boundary_context,
    )
    return segments, blocks_generated


def _build_clusterer(spec: ModelSpec) -> HDBSCAN:
    params = {"min_cluster_size": spec.min_cluster_size, "metric": spec.metric}
    params.update(spec.params)
    params.setdefault("prediction_data", True)
    return HDBSCAN(**params)


def _compute_stats(labels: np.ndarray, segments: List[RowBlock]) -> Dict[str, Any]:
    doc_count = len(segments)
    unique_labels = set(int(label) for label in labels)
    cluster_count = len([label for label in unique_labels if label >= 0])
    outlier_count = sum(1 for label in labels if int(label) < 0)
    outlier_pct = float(outlier_count / doc_count) if doc_count else 0.0
    return {
        "doc_count": doc_count,
        "segment_count": doc_count,
        "docs_generated": doc_count,
        "segments_generated": doc_count,
        "cluster_count": cluster_count,
        "outlier_pct": outlier_pct,
    }


def _write_artifacts(
    run: RunRecord,
    model_row: Dict[str, Any],
    segments: List[RowBlock],
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
    clusterer: HDBSCAN,
    embedder: VectorHostEmbeddingProvider,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    base_dir = Path(settings.topic_foundry_model_dir)
    model_name = model_row["name"]
    model_version = model_row["version"]

    registry_dir = base_dir / "registry" / model_name / "versions" / model_version
    model_dir = registry_dir / "model"
    run_dir = base_dir / "runs" / str(run.run_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dump(clusterer, model_dir / "clusterer.joblib")

    model_meta = {
        "model_id": model_row["model_id"],
        "model_name": model_name,
        "model_version": model_version,
        "run_id": str(run.run_id),
        "dataset_id": str(run.dataset_id),
        "algorithm": run.specs.model.algorithm,
        "params": run.specs.model.params,
        "embedding_source_url": run.specs.model.embedding_source_url,
        "embedding_model": embedder.embedding_model,
        "embedding_dim": embedder.embedding_dim,
        "trained_at": utc_now().isoformat(),
        "stats": stats,
    }

    (registry_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2))
    (registry_dir / "settings.json").write_text(json.dumps(run.specs.model_dump(mode="json"), indent=2))
    (registry_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embedding_model": embedder.embedding_model,
                "embedding_dim": embedder.embedding_dim,
                "embedding_endpoint_url": run.specs.model.embedding_source_url,
            },
            indent=2,
        )
    )

    documents_path = run_dir / "documents.jsonl"
    segments_path = run_dir / "segments.jsonl"
    with documents_path.open("w", encoding="utf-8") as doc_file, segments_path.open(
        "w", encoding="utf-8"
    ) as seg_file:
        for idx, (seg, label) in enumerate(zip(segments, labels, strict=False)):
            prob = None
            if probabilities is not None and idx < len(probabilities):
                prob = float(probabilities[idx])
            doc_payload = {
                "doc_id": seg.doc_id,
                "text": seg.text,
                "row_ids": seg.row_ids,
                "timestamps": seg.timestamps,
                "segment_id": seg.doc_id,
                "cluster_id": int(label),
                "cluster_prob": prob,
            }
            doc_file.write(json.dumps(doc_payload) + "\n")
            seg_file.write(
                json.dumps(
                    {
                        "segment_id": seg.doc_id,
                        "row_ids": seg.row_ids,
                        "timestamps": seg.timestamps,
                        "doc_ids": [seg.doc_id],
                        "cluster_id": int(label),
                        "cluster_prob": prob,
                    }
                )
                + "\n"
            )

    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (run_dir / "run_record.json").write_text(json.dumps(run.model_dump(mode="json"), indent=2))

    topics_summary, topics_keywords = _compute_topic_artifacts(segments, labels)
    topics_summary_path = run_dir / "topics_summary.json"
    topics_keywords_path = run_dir / "topics_keywords.json"
    topics_summary_path.write_text(json.dumps(topics_summary, indent=2))
    topics_keywords_path.write_text(json.dumps(topics_keywords, indent=2))

    return {
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "documents": str(documents_path),
        "segments": str(segments_path),
        "model_meta": str(registry_dir / "model_meta.json"),
        "topics_summary": str(topics_summary_path),
        "topics_keywords": str(topics_keywords_path),
    }
