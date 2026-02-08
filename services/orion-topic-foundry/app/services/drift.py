from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from hdbscan import approximate_predict
from joblib import load as joblib_load
from scipy.spatial.distance import jensenshannon

from app.models import DriftRecord, DatasetSpec, ModelSpec, WindowingSpec
from app.services.conversation_overrides import OverrideRecord, apply_overrides, build_conversations
from app.services.data_access import fetch_dataset_rows
from app.services.embedding_client import VectorHostEmbeddingProvider
from app.services.windowing import build_segments_with_stats
from app.settings import settings
from app.storage.repository import (
    fetch_active_model_by_name,
    fetch_latest_completed_run,
    list_conversation_overrides,
    create_event,
    insert_drift_record,
    list_drift_records,
    utc_now,
)
from app.services.bus_events import get_bus_publisher
from orion.schemas.topic_foundry import TopicFoundryDriftAlertV1


logger = logging.getLogger("topic-foundry.drift")


def run_drift_check(
    *,
    model_name: str,
    window_days: Optional[int],
    window_hours: Optional[int],
    threshold_js: Optional[float],
    threshold_outlier: Optional[float],
) -> Tuple[UUID, str]:
    model_row = fetch_active_model_by_name(model_name)
    if not model_row:
        raise ValueError("Active model not found")
    model_id = UUID(model_row["model_id"])
    model_version = model_row["version"]

    run_row = fetch_latest_completed_run(model_id)
    if not run_row:
        return uuid4(), "skipped"

    window_end = utc_now()
    window_start = _window_start(window_end, window_days, window_hours)

    baseline_shares, baseline_outlier_pct, baseline_top_share = _load_baseline_distribution(run_row)
    if not baseline_shares:
        return uuid4(), "skipped"

    current_shares, current_outlier_pct, current_top_share = _compute_current_distribution(
        run_row,
        window_start,
        window_end,
        model_name=model_name,
        model_version=model_version,
    )
    if not current_shares:
        return uuid4(), "skipped"

    js_divergence = _js_divergence(baseline_shares, current_shares)
    outlier_delta = current_outlier_pct - baseline_outlier_pct
    top_share_delta = current_top_share - baseline_top_share

    drift_id = uuid4()
    topic_shares = {
        "baseline": baseline_shares,
        "current": current_shares,
        "baseline_top_share": baseline_top_share,
        "current_top_share": current_top_share,
        "outlier_pct_delta": outlier_delta,
        "top_topic_share_delta": top_share_delta,
    }
    insert_drift_record(
        drift_id=drift_id,
        model_id=model_id,
        window_start=window_start,
        window_end=window_end,
        js_divergence=js_divergence,
        outlier_pct=current_outlier_pct,
        threshold_js=threshold_js,
        threshold_outlier=threshold_outlier,
        topic_shares=topic_shares,
        created_at=utc_now(),
    )

    if _is_drift_alert(js_divergence, outlier_delta, threshold_js, threshold_outlier):
        payload = TopicFoundryDriftAlertV1(
            drift_id=drift_id,
            model_id=model_id,
            model_name=model_name,
            window_start=window_start,
            window_end=window_end,
            js_divergence=js_divergence,
            outlier_pct_delta=outlier_delta,
            top_topic_share_delta=top_share_delta,
            threshold_js=threshold_js,
            threshold_outlier=threshold_outlier,
            created_at=utc_now(),
        )
        get_bus_publisher().publish_drift_alert(payload)
        create_event(
            event_id=uuid4(),
            kind="drift.alert",
            run_id=None,
            model_id=model_id,
            drift_id=drift_id,
            payload=payload.model_dump(mode="json"),
            bus_status="queued" if settings.orion_bus_enabled else "disabled",
            bus_error=None,
            created_at=utc_now(),
        )

    return drift_id, "complete"


def fetch_drift_records(model_name: str, *, limit: int) -> List[DriftRecord]:
    rows = list_drift_records(model_name, limit=limit)
    return [
        DriftRecord(
            drift_id=UUID(row["drift_id"]),
            model_id=UUID(row["model_id"]),
            window_start=row["window_start"],
            window_end=row["window_end"],
            js_divergence=row["js_divergence"],
            outlier_pct=row["outlier_pct"],
            threshold_js=row.get("threshold_js"),
            threshold_outlier=row.get("threshold_outlier"),
            outlier_pct_delta=(row.get("topic_shares") or {}).get("outlier_pct_delta"),
            top_topic_share_delta=(row.get("topic_shares") or {}).get("top_topic_share_delta"),
            topic_shares=row["topic_shares"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def drift_daemon_loop() -> None:
    from app.storage.repository import list_models

    while True:
        try:
            models = list_models()
            active_names = [row["name"] for row in models if row.get("stage") == "active"]
            for name in active_names:
                try:
                    run_drift_check(
                        model_name=name,
                        window_days=None,
                        window_hours=settings.topic_foundry_drift_window_hours,
                        threshold_js=None,
                        threshold_outlier=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Drift check failed model=%s error=%s", name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Drift daemon iteration failed error=%s", exc)
        await _sleep(settings.topic_foundry_drift_poll_seconds)


async def _sleep(seconds: int) -> None:
    import asyncio

    await asyncio.sleep(max(10, seconds))


def _window_start(end: datetime, window_days: Optional[int], window_hours: Optional[int]) -> datetime:
    days = window_days if window_days is not None else 0
    hours = window_hours if window_hours is not None else settings.topic_foundry_drift_window_hours
    return end - timedelta(days=days, hours=hours)


def _load_baseline_distribution(run_row: Dict[str, Any]) -> Tuple[Dict[str, float], float, float]:
    run_dir = (run_row.get("artifact_paths") or {}).get("run_dir")
    if not run_dir:
        return {}, 0.0, 0.0
    segments_path = Path(run_dir) / "segments.jsonl"
    if not segments_path.exists():
        return {}, 0.0, 0.0
    counts: Dict[str, int] = {}
    total = 0
    with segments_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            label = str(data.get("cluster_id"))
            counts[label] = counts.get(label, 0) + 1
            total += 1
    shares = _normalize_counts(counts, total)
    outlier_pct = shares.get("-1", 0.0)
    top_share = _top_topic_share(shares)
    return shares, outlier_pct, top_share


def _compute_current_distribution(
    run_row: Dict[str, Any],
    window_start: datetime,
    window_end: datetime,
    *,
    model_name: str,
    model_version: str,
) -> Tuple[Dict[str, float], float, float]:
    specs = run_row.get("specs") or {}
    dataset_spec = DatasetSpec(**(specs.get("dataset") or {}))
    windowing_spec = WindowingSpec(**(specs.get("windowing") or {}))
    model_spec = ModelSpec(**(specs.get("model") or {}))
    rows = fetch_dataset_rows(
        dataset=dataset_spec,
        start_at=window_start,
        end_at=window_end,
        limit=10_000,
    )
    if not rows:
        return {}, 0.0, 0.0

    conversations = build_conversations(
        rows,
        dataset_id=dataset_spec.dataset_id,
        spec=windowing_spec,
        text_columns=dataset_spec.text_columns,
        time_column=dataset_spec.time_column,
        id_column=dataset_spec.id_column,
        boundary_column=dataset_spec.boundary_column,
    )
    overrides = [
        OverrideRecord(
            override_id=UUID(row["override_id"]),
            kind=row["kind"],
            payload=row["payload"],
            created_at=row["created_at"],
        )
        for row in list_conversation_overrides(dataset_spec.dataset_id)
    ]
    if overrides:
        conversations = apply_overrides(conversations, overrides)

    segments, _ = build_segments_with_stats(
        conversations,
        spec=windowing_spec,
        embedding_url=model_spec.embedding_source_url,
        run_id=None,
    )
    if not segments:
        return {}, 0.0, 0.0

    embedder = VectorHostEmbeddingProvider(model_spec.embedding_source_url)
    embeddings = np.array(embedder.embed_texts([segment.text for segment in segments]), dtype=np.float32)
    clusterer = _load_clusterer(model_name, model_version)
    if clusterer is None:
        return {}, 0.0, 0.0
    try:
        labels, _ = approximate_predict(clusterer, embeddings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Approximate predict failed for model=%s error=%s", model_name, exc)
        return {}, 0.0, 0.0
    counts: Dict[str, int] = {}
    total = 0
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
        total += 1
    shares = _normalize_counts(counts, total)
    outlier_pct = shares.get("-1", 0.0)
    top_share = _top_topic_share(shares)
    return shares, outlier_pct, top_share


def _load_clusterer(model_name: str, model_version: Optional[str]):
    if not model_version:
        return None
    model_dir = Path(settings.topic_foundry_model_dir)
    clusterer_path = model_dir / "registry" / model_name / "versions" / model_version / "model" / "clusterer.joblib"
    if not clusterer_path.exists():
        logger.warning("Clusterer not found at %s", clusterer_path)
        return None
    return joblib_load(clusterer_path)


def _normalize_counts(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _top_topic_share(shares: Dict[str, float]) -> float:
    non_outlier = {key: val for key, val in shares.items() if key != "-1"}
    if not non_outlier:
        return 0.0
    return max(non_outlier.values())


def _js_divergence(baseline: Dict[str, float], current: Dict[str, float]) -> float:
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    if not keys:
        return 0.0
    p = np.array([baseline.get(key, 0.0) for key in keys], dtype=np.float64)
    q = np.array([current.get(key, 0.0) for key in keys], dtype=np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    return float(jensenshannon(p, q, base=2.0) ** 2)


def _is_drift_alert(
    js_divergence: float,
    outlier_delta: float,
    threshold_js: Optional[float],
    threshold_outlier: Optional[float],
) -> bool:
    exceeds_js = threshold_js is not None and js_divergence >= threshold_js
    exceeds_outlier = threshold_outlier is not None and abs(outlier_delta) >= threshold_outlier
    return exceeds_js or exceeds_outlier
