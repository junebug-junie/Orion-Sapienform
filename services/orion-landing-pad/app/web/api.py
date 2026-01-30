from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ..settings import Settings
from ..store.redis_store import PadStore


def _index_path() -> str:
    return "app/static/landing_pad/index.html"


def _parse_time(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _bucketize(points: Iterable[Tuple[int, float]], bucket_ms: int, agg: str) -> List[Tuple[int, float]]:
    if bucket_ms <= 0:
        return list(points)
    buckets: Dict[int, List[float]] = defaultdict(list)
    for ts, val in points:
        bucket = (ts // bucket_ms) * bucket_ms
        buckets[bucket].append(val)
    out: List[Tuple[int, float]] = []
    for bucket_ts in sorted(buckets):
        values = buckets[bucket_ts]
        if not values:
            continue
        if agg == "max":
            result = max(values)
        elif agg == "sum":
            result = sum(values)
        elif agg == "p95":
            result = _percentile(values, 0.95) or 0.0
        else:
            result = sum(values) / len(values)
        out.append((bucket_ts, result))
    return out


def _extract_dimension(event: Dict[str, Any], key: str) -> Optional[str]:
    if key in event:
        value = event.get(key)
        return str(value) if value is not None else None
    payload = event.get("payload") or {}
    if key in payload:
        value = payload.get(key)
        return str(value) if value is not None else None
    aliases = {
        "node": ["node", "source_node", "host"],
        "service": ["source_service", "service"],
    }
    for alias_key in aliases.get(key, []):
        if alias_key in payload:
            value = payload.get(alias_key)
            return str(value) if value is not None else None
    return None


def _event_value(event: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "pad.event.salience":
        return event.get("salience")
    if metric == "pad.event.novelty":
        return event.get("novelty")
    if metric == "pad.event.confidence":
        return event.get("confidence")
    if metric == "pad.event.count":
        return 1.0
    return None


def _frame_value(frame: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "pad.frame.salient_count":
        return float(len(frame.get("salient_event_ids") or []))
    if metric == "pad.frame.window_ms":
        return frame.get("window_ms")
    if metric == "pad.frame.tensor_dim":
        tensor = frame.get("tensor") or {}
        return tensor.get("dim")
    return None


def _auto_step_seconds(start_ms: int, end_ms: int) -> int:
    span_ms = max(end_ms - start_ms, 1)
    target_buckets = 90
    step_ms = max(int(span_ms / target_buckets), 1000)
    return max(int(step_ms / 1000), 1)


def _metric_catalog(event_samples: List[dict], frame_samples: List[dict]) -> List[Dict[str, str]]:
    metrics = [
        {"name": "pad.event.salience", "unit": "score", "description": "PadEventV1 salience values."},
        {"name": "pad.event.novelty", "unit": "score", "description": "PadEventV1 novelty values."},
        {"name": "pad.event.confidence", "unit": "score", "description": "PadEventV1 confidence values."},
        {"name": "pad.event.count", "unit": "count", "description": "Number of PadEventV1 entries."},
    ]
    if frame_samples:
        metrics.extend(
            [
                {"name": "pad.frame.salient_count", "unit": "count", "description": "Count of salient event ids per frame."},
                {"name": "pad.frame.window_ms", "unit": "ms", "description": "Frame window size in milliseconds."},
                {"name": "pad.frame.tensor_dim", "unit": "dim", "description": "Tensor dimension encoded with the frame."},
            ]
        )
    return metrics


def _collect_dimension_values(events: List[dict]) -> Dict[str, Dict[str, int]]:
    dimensions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        for key in ("source_service", "source_channel", "type", "subject"):
            value = event.get(key)
            if value is not None:
                dimensions[key][str(value)] += 1
        payload = event.get("payload") or {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                dimensions[key][str(value)] += 1
    return {key: dict(values) for key, values in dimensions.items()}


def build_router(*, store: PadStore, settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/ui")
    async def ui() -> FileResponse:
        return FileResponse(_index_path())

    @router.get("/")
    async def root() -> FileResponse:
        """Serve the UI at / as well.

        Important for reverse proxies (e.g. tailscale serve) that may strip the configured
        base prefix before forwarding. In that mode, external /landing-pad becomes internal /.
        We must NOT redirect to /ui (which would escape the proxied prefix).
        """
        return FileResponse(_index_path())

    @router.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/api/topics/summary")
    async def topics_summary(
        window_minutes: int = Query(1440),
        model_version: Optional[str] = Query(None),
        max_topics: int = Query(20),
    ) -> Dict[str, Any]:
        from ..reducers.topic_rail import get_topic_summary

        return await get_topic_summary(
            window_minutes=window_minutes,
            model_version=model_version,
            max_topics=max_topics,
            dsn=settings.postgres_uri,
        )

    @router.get("/api/topics/drift")
    async def topics_drift(
        window_minutes: int = Query(1440),
        model_version: Optional[str] = Query(None),
        min_turns: int = Query(10),
        max_sessions: int = Query(50),
    ) -> Dict[str, Any]:
        from ..reducers.topic_rail import get_topic_drift

        return await get_topic_drift(
            window_minutes=window_minutes,
            model_version=model_version,
            min_turns=min_turns,
            max_sessions=max_sessions,
            dsn=settings.postgres_uri,
        )

    @router.get("/api/metrics")
    async def metrics() -> Dict[str, Any]:
        event_samples = await store.get_event_payloads(limit=settings.ui_sample_limit)
        frame_samples = await store.get_frame_payloads(limit=settings.ui_sample_limit)
        return {"metrics": _metric_catalog(event_samples, frame_samples)}

    @router.get("/api/dimensions")
    async def dimensions() -> Dict[str, Any]:
        event_samples = await store.get_event_payloads(limit=settings.ui_sample_limit)
        return {"dimensions": _collect_dimension_values(event_samples)}

    @router.get("/api/query")
    async def query(
        metric: str = Query(...),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        lookback_minutes: Optional[int] = Query(None),
        group_by: Optional[str] = Query(None),
        agg: str = Query("avg"),
        step_seconds: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        end_ms = _parse_time(end) or now_ms

        if start is None:
            lookback = lookback_minutes or settings.ui_default_lookback_minutes
            start_ms = end_ms - (lookback * 60 * 1000)
        else:
            start_ms = _parse_time(start) or end_ms - (settings.ui_default_lookback_minutes * 60 * 1000)

        if step_seconds is None or step_seconds == "auto":
            step = _auto_step_seconds(start_ms, end_ms)
        else:
            step = max(int(step_seconds), 1)
        bucket_ms = step * 1000

        agg = agg.lower()
        group_by = (group_by or "").strip() or None
        if metric == "pad.event.count":
            agg = "sum"

        series_map: Dict[str, Dict[str, Any]] = {}
        values_for_summary: List[float] = []

        if metric.startswith("pad.event"):
            events = await store.range_event_payloads(start_ms, end_ms, limit=settings.ui_query_limit)
            for event in events:
                ts_ms = event.get("ts_ms")
                if not ts_ms:
                    continue
                value = _event_value(event, metric)
                if value is None:
                    continue

                label_value = _extract_dimension(event, group_by) if group_by else "all"
                label_value = label_value or "unknown"

                entry = series_map.setdefault(
                    label_value,
                    {"name": label_value, "labels": {group_by: label_value} if group_by else {}, "points": []},
                )
                entry["points"].append((int(ts_ms), float(value)))
                values_for_summary.append(float(value))

        elif metric.startswith("pad.frame"):
            frames = await store.range_frame_payloads(start_ms, end_ms, limit=settings.ui_query_limit)
            for frame in frames:
                ts_ms = frame.get("ts_ms")
                if not ts_ms:
                    continue
                value = _frame_value(frame, metric)
                if value is None:
                    continue
                entry = series_map.setdefault("frames", {"name": "frames", "labels": {}, "points": []})
                entry["points"].append((int(ts_ms), float(value)))
                values_for_summary.append(float(value))

        else:
            return {"series": [], "summary": {"count": 0}}

        series = []
        for entry in series_map.values():
            entry["points"] = _bucketize(entry["points"], bucket_ms, agg)
            series.append(entry)

        summary = {
            "count": len(values_for_summary),
            "min": min(values_for_summary) if values_for_summary else None,
            "p50": _percentile(values_for_summary, 0.5),
            "p95": _percentile(values_for_summary, 0.95),
            "max": max(values_for_summary) if values_for_summary else None,
            "last_ts": max((point[0] for s in series for point in s["points"]), default=None),
        }

        return {"series": series, "summary": summary}

    return router
