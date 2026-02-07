from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from app.models import EnrichmentSpec, ModelSpec, WindowingSpec


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    return ts.isoformat()


def compute_spec_hash(
    *,
    dataset_id: UUID,
    model_id: UUID,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    windowing_spec: WindowingSpec,
    model_spec: ModelSpec,
    enrichment_spec: EnrichmentSpec,
    run_scope: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "model_id": str(model_id),
        "start_at": _iso(start_at),
        "end_at": _iso(end_at),
        "windowing_spec": windowing_spec.model_dump(mode="json"),
        "model_spec": model_spec.model_dump(mode="json"),
        "enrichment_spec": enrichment_spec.model_dump(mode="json"),
        "run_scope": run_scope,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
