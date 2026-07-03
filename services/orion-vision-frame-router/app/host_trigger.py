from __future__ import annotations

from orion.schemas.vision import VisionTaskResultPayload

HOST_TRIGGER_SCORE_THRESHOLD = 0.25


def stream_id_from_host_result(
    result: VisionTaskResultPayload,
    *,
    fallback_stream_id: str | None,
) -> str | None:
    if result.artifact and result.artifact.inputs:
        sid = result.artifact.inputs.get("stream_id")
        if sid is not None and str(sid).strip():
            return str(sid).strip()
    if fallback_stream_id and str(fallback_stream_id).strip():
        return str(fallback_stream_id).strip()
    return None


def extract_host_trigger_labels(
    result: VisionTaskResultPayload,
    *,
    allowed: set[str],
    score_threshold: float = HOST_TRIGGER_SCORE_THRESHOLD,
) -> list[str]:
    if not result.ok or result.artifact is None:
        return []
    objects = result.artifact.outputs.objects or []
    seen: set[str] = set()
    out: list[str] = []
    allowed_lower = {a.lower() for a in allowed}
    for obj in objects:
        label = str(obj.label or "").strip().lower()
        if not label or label not in allowed_lower:
            continue
        if float(obj.score) < score_threshold:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out
