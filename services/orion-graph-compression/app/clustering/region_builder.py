from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Set

from orion.schemas.graph_compression import CompressionRegionV1


def stable_region_id(scope: str, kind: str, nodes: frozenset) -> str:
    """Deterministic region ID from scope + kind + sorted node URIs."""
    content = f"{scope}:{kind}:" + ":".join(sorted(nodes))
    digest = hashlib.sha256(content.encode()).hexdigest()[:24]
    return f"urn:orion:compression:region:{digest}"


def build_region(
    *,
    nodes: Set[str],
    scope: str,
    kind: str,
    summary: str,
    summary_kind: str,
    salience: float,
    trust_tier: str,
    compression_version: str,
) -> CompressionRegionV1:
    node_list = sorted(nodes)
    exemplar_ids = node_list[:5] if node_list else []
    if not exemplar_ids:
        exemplar_ids = [f"urn:orion:compression:empty:{scope}"]
    derived_from = node_list[:20] if node_list else [f"urn:orion:compression:empty:{scope}"]

    return CompressionRegionV1(
        region_id=stable_region_id(scope, kind, frozenset(nodes)),
        scope=scope,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        summary=summary,
        summary_kind=summary_kind,  # type: ignore[arg-type]
        salience=salience,
        trust_tier=trust_tier,
        exemplar_ids=exemplar_ids,
        derived_from=derived_from,
        generated_at=datetime.now(timezone.utc),
        compression_version=compression_version,
    )
