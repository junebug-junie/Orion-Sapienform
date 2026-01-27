#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from redis.asyncio import Redis

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.settings import Settings
from app.store.redis_store import PadStore


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > 120:
            return value[:117] + "..."
        return value
    if isinstance(value, list):
        return value[:5]
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in list(value.items())[:8]}
    return value


def _redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _redact_value(value) for key, value in list(payload.items())[:12]}


def _format_ts(ts_ms: int | None) -> str:
    if not ts_ms:
        return "n/a"
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.isoformat()


def _collect_dimensions(events: Iterable[Dict[str, Any]]) -> Dict[str, Counter]:
    dims: Dict[str, Counter] = defaultdict(Counter)
    for event in events:
        payload = event.get("payload") or {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                dims[key][str(value)] += 1
        for key in ("source_service", "source_channel", "type", "subject"):
            value = event.get(key)
            if value is not None:
                dims[key][str(value)] += 1
    return dims


def _env_kind_from_event(event: Dict[str, Any]) -> str | None:
    payload = event.get("payload") or {}
    if "kind" in payload:
        return str(payload.get("kind"))
    raw = payload.get("raw") or {}
    if isinstance(raw, dict) and raw.get("kind"):
        return str(raw.get("kind"))
    return None


def _metric_name_from_event(event: Dict[str, Any]) -> str | None:
    payload = event.get("payload") or {}
    name = payload.get("metric") or payload.get("name")
    if name:
        return str(name)
    return None


async def main() -> None:
    settings = Settings()
    redis_url = settings.merged_redis_url()
    limit = int(os.getenv("PAD_INVENTORY_LIMIT", "500"))

    redis = Redis.from_url(redis_url, decode_responses=False)
    store = PadStore(
        redis=redis,
        events_stream_key=settings.pad_events_stream_key,
        frames_stream_key=settings.pad_frames_stream_key,
        stream_maxlen=settings.pad_stream_maxlen,
        event_ttl=settings.pad_event_ttl_sec,
        frame_ttl=settings.pad_frame_ttl_sec,
    )

    events = await store.get_event_payloads(limit=limit)
    frames = await store.get_frame_payloads(limit=limit)

    event_types = Counter(event.get("type") for event in events if event.get("type"))
    env_kinds = Counter(filter(None, (_env_kind_from_event(event) for event in events)))
    metric_names = Counter(filter(None, (_metric_name_from_event(event) for event in events)))

    dimensions = _collect_dimensions(events)

    last_event_ts = max((event.get("ts_ms") for event in events if event.get("ts_ms")), default=None)
    last_frame_ts = max((frame.get("ts_ms") for frame in frames if frame.get("ts_ms")), default=None)

    print("Landing Pad Inventory Report")
    print("============================")
    print(f"Redis URL: {redis_url}")
    print(f"Events stream: {settings.pad_events_stream_key} (sampled {len(events)})")
    print(f"Frames stream: {settings.pad_frames_stream_key} (sampled {len(frames)})")
    print(f"Last event ts: {_format_ts(last_event_ts)}")
    print(f"Last frame ts: {_format_ts(last_frame_ts)}")

    print("\nEvent types seen:")
    for key, count in event_types.most_common():
        print(f"  - {key}: {count}")

    print("\nEnv kinds seen (if captured in payload):")
    for key, count in env_kinds.most_common():
        print(f"  - {key}: {count}")

    print("\nMetric names seen:")
    for key, count in metric_names.most_common():
        print(f"  - {key}: {count}")

    print("\nDimension keys (cardinality):")
    for key, counter in sorted(dimensions.items(), key=lambda item: len(item[1]), reverse=True):
        print(f"  - {key}: {len(counter)}")

    print("\nSample event payloads (redacted):")
    for event in events[:3]:
        payload = event.get("payload") or {}
        print(f"  - {_redact_payload(payload)}")

    print("\nSample frame payloads (redacted):")
    for frame in frames[:2]:
        print(f"  - {_redact_payload(frame)}")

    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())
