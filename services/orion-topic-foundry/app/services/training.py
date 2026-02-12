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

from app.models import DatasetSpec, EnrichmentSpec, ModelSpec, RunRecord, RunSpecSnapshot, RunTrainRequest, SegmentRecord, WindowingSpec
from app.services.bus_events import get_bus_publisher
from app.services.conversation_overrides import OverrideRecord, apply_overrides, build_conversations
from app.services.data_access import fetch_dataset_rows
from app.services.enrichment import run_enrichment_sync
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
from app.topic_engine import build_topic_engine
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


def _run_training(run_id: UUID, payload: RunTrainRequest, model_row: Dict[str, Any], dataset: DatasetSpec, spec_hash: str) -> None:
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
        docs = [seg.text for seg in segments]

        mode_params = dict(payload.topic_mode_params or {})
        if payload.topic_mode == "guided":
            mode_params.setdefault("seed_topic_list", mode_params.get("seed_topic_list") or mode_params.get("seeds"))
        if payload.topic_mode == "zeroshot":
            mode_params.setdefault("zeroshot_topic_list", mode_params.get("zeroshot_topic_list") or mode_params.get("topics"))

        model_meta = dict(model_row.get("model_meta") or {})
        model_meta.update(run.specs.model.model_meta or {})
        model_meta.update(mode_params)

        engine = build_topic_engine(model_meta)
        topic_model = BERTopic(
            embedding_model=engine.embedding_model,
            umap_model=engine.reducer,
            hdbscan_model=engine.clusterer,
            representation_model=engine.representation_model,
            **engine.bertopic_kwargs,
        )

        labels, probs = topic_model.fit_transform(docs)
        labels_np = np.array(labels, dtype=int)
        info = topic_model.get_topic_info().to_dict(orient="records")

        doc_count = len(docs)
        cluster_count = len([v for v in sorted(set(labels_np.tolist())) if v >= 0])
        outlier_count = int(np.sum(labels_np < 0))
        outlier_rate = float(outlier_count / doc_count) if doc_count else 0.0

        stats = {
            "doc_count": doc_count,
            "segment_count": doc_count,
            "docs_generated": doc_count,
            "segments_generated": doc_count,
            "blocks_generated": blocks_generated,
            "cluster_count": cluster_count,
            "outlier_pct": outlier_rate,
            "outlier_rate": outlier_rate,
            "topic_mode": payload.topic_mode,
            "topic_mode_params": mode_params,
            "training_secs": time.monotonic() - start,
            "topic_info_count": len(info),
            **engine.backend_names,
        }

        artifacts = _write_artifacts(run, topic_model, info, docs, labels, probs, stats, payload.topic_mode, mode_params, model_meta)

        seg_records=[]
        rollups: Dict[str, Any] = {}
        for idx, seg in enumerate(segments):
            start_at, end_at = _timestamp_bounds(seg.timestamps)
            label = int(labels[idx])
            prob = None
            if isinstance(probs, list) and idx < len(probs):
                try:
                    prob = float(np.max(np.asarray(probs[idx])))
                except Exception:
                    prob = None
            seg_records.append(SegmentRecord(
                segment_id=UUID(seg.doc_id), run_id=run.run_id, size=len(seg.row_ids),
                provenance={"row_ids": seg.row_ids, "timestamps": seg.timestamps, "doc_ids": [seg.doc_id], "conversation_id": str(seg.conversation_id) if seg.conversation_id else None},
                topic_id=label, topic_prob=prob, is_outlier=label==-1, snippet=_snippet(seg.text or "") or None,
                chars=len(seg.text or ""), row_ids_count=len(seg.row_ids), start_at=start_at, end_at=end_at, created_at=utc_now()
            ))
            if seg.conversation_id:
                ck = str(seg.conversation_id)
                convo = rollups.setdefault(ck, {"conversation_id": ck, "topic_counts": {}, "segment_count": 0})
                convo["segment_count"] += 1
                tc = convo["topic_counts"]
                tc[str(label)] = tc.get(str(label), 0) + 1
        insert_segments(seg_records)
        if rollups:
            for rv in rollups.values():
                counts = rv.get("topic_counts", {})
                rv["top_topics"] = sorted([{"topic_id": int(k), "count": v} for k, v in counts.items()], key=lambda x: -x["count"])[:5]
            upsert_conversation_rollups(run.run_id, rollups)

        topic_payloads = _build_topic_payloads(labels_np, topic_model, scope=run.run_scope or "macro")
        if run.run_scope == "micro":
            macro_run = fetch_latest_completed_run_by_scope(run.model_id, "macro")
            if macro_run:
                macro_topics = fetch_topics(UUID(macro_run["run_id"]), scope="macro")
                _map_micro_to_macro(topic_payloads, macro_topics)
        insert_topics(run.run_id, topic_payloads)

        run.status = "complete"
        run.stage = "trained"
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
    specs = RunSpecSnapshot(dataset=dataset, windowing=WindowingSpec(**model_row["windowing_spec"]), model=ModelSpec(**model_row["model_spec"]), enrichment=EnrichmentSpec(**model_row["enrichment_spec"]) if model_row.get("enrichment_spec") else EnrichmentSpec(), run_scope=payload.run_scope)
    if specs.run_scope is None:
        specs.run_scope = "micro" if specs.windowing.windowing_mode.startswith("conversation") else "macro"
    return RunRecord(run_id=run_id, model_id=UUID(model_row["model_id"]), dataset_id=dataset.dataset_id, specs=specs, spec_hash=spec_hash, status="queued", run_scope=specs.run_scope, stats={}, artifact_paths={}, created_at=utc_now())


def _prepare_segments(run: RunRecord, payload: RunTrainRequest) -> tuple[List[RowBlock], int]:
    rows = fetch_dataset_rows(dataset=run.specs.dataset, start_at=payload.start_at, end_at=payload.end_at, limit=10_000)
    conversations = build_conversations(rows, dataset_id=run.dataset_id, spec=run.specs.windowing, text_columns=run.specs.dataset.text_columns, time_column=run.specs.dataset.time_column, id_column=run.specs.dataset.id_column, boundary_column=run.specs.dataset.boundary_column)
    overrides = [OverrideRecord(override_id=UUID(row["override_id"]), kind=row["kind"], payload=row["payload"], created_at=row["created_at"]) for row in list_conversation_overrides(run.dataset_id)]
    if overrides:
        conversations = apply_overrides(conversations, overrides)
    run_dir = str(Path(settings.topic_foundry_model_dir) / "runs" / str(run.run_id))
    boundary_context = BoundaryContext(run_id=run.run_id, spec_hash=run.spec_hash, dataset_id=run.dataset_id, model_id=run.model_id, run_dir=run_dir)
    return build_segments_with_stats(conversations, spec=run.specs.windowing, embedding_url=run.specs.model.embedding_source_url or settings.topic_foundry_embedding_url, boundary_context=boundary_context, run_id=run.run_id)


def _build_topic_payloads(labels: np.ndarray, topic_model: BERTopic, *, scope: str) -> List[Dict[str, Any]]:
    payloads=[]
    for topic_id in sorted(set(int(v) for v in labels.tolist())):
        topic_docs = [idx for idx, val in enumerate(labels.tolist()) if int(val) == topic_id]
        if not topic_docs:
            continue
        centroid = topic_model.topic_embeddings_[topic_id] if hasattr(topic_model, "topic_embeddings_") and topic_id >= 0 else None
        payloads.append({"topic_id": topic_id, "scope": scope, "parent_topic_id": None, "centroid": centroid.tolist() if centroid is not None else None, "count": len(topic_docs), "label": None})
    return payloads


def _map_micro_to_macro(micro_topics: List[Dict[str, Any]], macro_topics: List[Dict[str, Any]]) -> None:
    if not macro_topics:
        return
    macro_centroids = {int(t["topic_id"]): np.array(t["centroid"], dtype=np.float32) for t in macro_topics if t.get("centroid") is not None}
    if not macro_centroids:
        return
    macro_ids = list(macro_centroids.keys())
    macro_matrix = np.stack([macro_centroids[mid] for mid in macro_ids], axis=0)
    macro_norms = np.linalg.norm(macro_matrix, axis=1)
    for topic in micro_topics:
        if topic.get("centroid") is None:
            continue
        vec = np.array(topic["centroid"], dtype=np.float32)
        denom = np.linalg.norm(vec) * macro_norms
        scores = (macro_matrix @ vec) / np.where(denom == 0, 1.0, denom)
        topic["parent_topic_id"] = int(macro_ids[int(np.argmax(scores))])


def _write_artifacts(run: RunRecord, topic_model: BERTopic, topic_info: List[Dict[str, Any]], docs: List[str], labels: List[int], probs: Any, stats: Dict[str, Any], topic_mode: str, topic_mode_params: Dict[str, Any], model_meta: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(settings.topic_foundry_model_dir)
    run_dir = base_dir / "runs" / str(run.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    topic_info_json = run_dir / "topic_info.json"
    topic_info_csv = run_dir / "topic_info.csv"
    top_words_json = run_dir / "top_words.json"
    run_meta_json = run_dir / "run_metadata.json"

    topic_info_json.write_text(json.dumps(topic_info, indent=2))
    headers = sorted({k for row in topic_info for k in row.keys()})
    topic_info_csv.write_text(",".join(headers) + "\n" + "\n".join([",".join([json.dumps(row.get(h, "")) for h in headers]) for row in topic_info]))

    top_words: Dict[str, Any] = {}
    for row in topic_info:
        tid = row.get("Topic")
        if tid is None:
            continue
        words = topic_model.get_topic(int(tid)) or []
        top_words[str(tid)] = [w for w, _ in words]
    top_words_json.write_text(json.dumps(top_words, indent=2))

    run_meta = {
        "run_id": str(run.run_id),
        "model_id": str(run.model_id),
        "dataset_id": str(run.dataset_id),
        "topic_mode": topic_mode,
        "topic_mode_params": topic_mode_params,
        "model_meta_used": model_meta,
        "stats": stats,
    }
    run_meta_json.write_text(json.dumps(run_meta, indent=2))

    return {
        "run_dir": str(run_dir),
        "topic_info_json": str(topic_info_json),
        "topic_info_csv": str(topic_info_csv),
        "top_words_json": str(top_words_json),
        "run_metadata_json": str(run_meta_json),
        "artifacts_summary": {"topic_count": len(topic_info), "doc_count": len(docs), "labels_count": len(labels), "has_probabilities": probs is not None},
    }
