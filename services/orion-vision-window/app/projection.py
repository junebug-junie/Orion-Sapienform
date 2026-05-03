"""
Vision window projection helpers: stream keys, summaries, payload build.
Aligned with docs/superpowers/specs/2026-05-02-orion-vision-window-projection-design.md §9.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Tuple

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionArtifactPayload, VisionWindowPayload

SNAPSHOT_SCHEMA_V1 = "vision_window_snapshot.v1"
MAX_URIS_PER_ENVELOPE = 32


def stream_key_from_artifact(art: VisionArtifactPayload) -> str:
    """Best-effort stream / source key from artifact inputs (§15 open: heuristic)."""
    inp = art.inputs or {}
    for key in ("stream_id", "camera_id", "clip_id"):
        v = inp.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    if art.device and str(art.device).strip():
        return str(art.device).strip()
    return "default"


def camera_id_from_artifact(art: VisionArtifactPayload) -> str | None:
    inp = art.inputs or {}
    v = inp.get("camera_id")
    if v is not None and str(v).strip():
        return str(v).strip()
    return None


def artifact_uris_from_artifact(art: VisionArtifactPayload) -> List[str]:
    """Lightweight URI/path pointers only; no frame bytes (spec §3, §9)."""
    out: List[str] = []
    seen: set[str] = set()

    def push(s: str) -> None:
        s = str(s).strip()
        if not s or len(s) > 2048 or s in seen:
            return
        seen.add(s)
        out.append(s)

    inp = art.inputs or {}
    for k, v in inp.items():
        if not isinstance(v, str):
            continue
        if v.startswith(("http://", "https://", "s3://")):
            push(v)
        elif k.endswith("_uri") or k.endswith("_url") or k in ("image_path", "video_path", "path"):
            push(v)
    refs = art.debug_refs or {}
    for v in refs.values():
        if isinstance(v, str):
            push(v)
    return out[:MAX_URIS_PER_ENVELOPE]


def summarize_items(items: List[Tuple[VisionArtifactPayload, float]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    captions: List[str] = []
    for art, _ts in items:
        if art.outputs.objects:
            for obj in art.outputs.objects:
                counts[obj.label] = counts.get(obj.label, 0) + 1
        if art.outputs.caption and art.outputs.caption.text:
            captions.append(art.outputs.caption.text)
    top_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "object_counts": counts,
        "top_labels": top_labels,
        "item_count": len(items),
        "captions": captions,
        "label_counts": counts,
        "detection_count": sum(counts.values()) if counts else 0,
    }


def upstream_ids_from_envelopes(envs: List[BaseEnvelope]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for env in envs:
        cid = env.correlation_id
        if cid is None:
            continue
        s = str(cid)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_window_payload(
    *,
    stream_id: str,
    items: List[Tuple[VisionArtifactPayload, float]],
    envs: List[BaseEnvelope],
    window_start: float,
    window_end: float,
    cursor: str,
    stale_after_ms: int,
) -> VisionWindowPayload:
    if not items:
        raise ValueError("build_window_payload requires at least one item")
    arts = [t[0] for t in items]
    artifact_ids = [a.artifact_id for a in arts]
    summary = summarize_items(items)
    uris: List[str] = []
    seen_u: set[str] = set()
    for a in arts:
        for u in artifact_uris_from_artifact(a):
            if u not in seen_u:
                seen_u.add(u)
                uris.append(u)
            if len(uris) >= MAX_URIS_PER_ENVELOPE:
                break
    first = arts[0]
    age_ms = int(max(0.0, (time.time() - window_end) * 1000))
    freshness: Dict[str, Any] = {
        "age_ms": age_ms,
        "stale_after_ms": stale_after_ms,
        "expires_at": None,
    }
    return VisionWindowPayload(
        window_id=str(uuid.uuid4()),
        start_ts=window_start,
        end_ts=window_end,
        summary=summary,
        artifact_ids=artifact_ids,
        schema_version=SNAPSHOT_SCHEMA_V1,
        stream_id=stream_id if stream_id != "default" else None,
        source_node=None,
        camera_id=camera_id_from_artifact(first),
        cursor=cursor,
        upstream_event_ids=upstream_ids_from_envelopes(envs),
        artifact_uris=uris,
        freshness=freshness,
        meta={"projection": "orion-vision-window"},
    )


def envelope_to_http_dict(payload: VisionWindowPayload, *, source: str) -> Dict[str, Any]:
    """HTTP read model: status + envelope fields (§6)."""
    d = payload.model_dump(mode="json")
    end = float(payload.end_ts)
    age_ms = int(max(0.0, (time.time() - end) * 1000))
    stale_after = int((payload.freshness or {}).get("stale_after_ms") or 120_000)
    status = "ok"
    if age_ms > stale_after:
        status = "stale"
    return {
        "status": status,
        "source": source,
        "snapshot_id": payload.window_id,
        "stream_id": payload.stream_id,
        "generated_at": end,
        "cursor": payload.cursor,
        "age_ms": age_ms,
        "envelope": d,
    }
