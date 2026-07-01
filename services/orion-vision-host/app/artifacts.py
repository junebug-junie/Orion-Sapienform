from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from orion.schemas.vision import (
    VisionArtifactPayload,
    VisionArtifactOutputs,
    VisionObject,
    VisionCaption,
    VisionEmbedding,
)

from .models import VisionResult

_RESERVED_OUTPUT_KEYS = {
    "objects",
    "caption",
    "embedding",
    "configured",
    "implemented",
    "kind",
    "device",
    "model_id",
    "_fingerprints",
}


def merge_result_inputs(
    request: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine task request + meta into the artifact provenance inputs dict.

    On key collision, frame/routing meta wins over request.
    """
    return {**(request or {}), **(meta or {})}


def build_artifact_payload(res: VisionResult) -> Optional[VisionArtifactPayload]:
    """Build VisionArtifactPayload from a successful VisionResult, or None if no artifacts."""
    if not res.artifacts:
        return None

    artifacts = res.artifacts
    artifact_id = str(uuid.uuid4())

    objects = None
    caption = None
    embedding = None

    if "objects" in artifacts and isinstance(artifacts["objects"], list):
        objects = []
        for obj in artifacts["objects"]:
            objects.append(VisionObject(
                label=str(obj.get("label", "unknown")),
                score=float(obj.get("score", 0.0)),
                box_xyxy=obj.get("box_xyxy", [0, 0, 0, 0])
            ))

    if "caption" in artifacts and isinstance(artifacts["caption"], dict):
        caption = VisionCaption(
            text=artifacts["caption"].get("text", ""),
            confidence=artifacts["caption"].get("confidence")
        )

    if "embedding" in artifacts and isinstance(artifacts["embedding"], dict):
        embedding = VisionEmbedding(
            ref=artifacts["embedding"].get("ref", ""),
            path=artifacts["embedding"].get("path", ""),
            dim=artifacts["embedding"].get("dim", 0)
        )

    outputs = VisionArtifactOutputs(
        objects=objects,
        caption=caption,
        embedding=embedding
    )

    for k, v in artifacts.items():
        if k not in _RESERVED_OUTPUT_KEYS and not hasattr(outputs, k):
            setattr(outputs, k, v)

    fingerprints = artifacts.get("_fingerprints")
    if isinstance(fingerprints, dict) and fingerprints:
        model_fingerprints = {str(k): str(v) for k, v in fingerprints.items()}
    else:
        model_fingerprints = {res.task_type: str(artifacts.get("model_id", "unknown"))}

    return VisionArtifactPayload(
        artifact_id=artifact_id,
        correlation_id=res.corr_id,
        task_type=res.task_type,
        device=res.device or "unknown",
        inputs=dict(res.inputs or {}),
        outputs=outputs,
        timing={"latency_s": res.meta.get("latency_s", 0.0)},
        model_fingerprints=model_fingerprints,
    )
