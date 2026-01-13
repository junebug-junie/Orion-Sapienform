from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Tuple

from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, MemoryItemV1

try:
    from .render import render_items
except ImportError:  # pragma: no cover - fallback when not in package context
    from render import render_items  # type: ignore


def _norm_score(score: Any) -> float:
    try:
        f = float(score)
    except Exception:
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(0.0, min(1.0, f))


def _key_for(item: Dict[str, Any]) -> str:
    uri = item.get("uri") or ""
    src_id = item.get("id") or ""
    text = item.get("text") or item.get("snippet") or ""
    key_src = uri or src_id or hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
    return key_src


def fuse_candidates(
    *,
    candidates: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    latency_ms: int = 0,
) -> MemoryBundleV1:
    max_per_source = int(profile.get("max_per_source", 3))
    max_total = int(profile.get("max_total_items", 12))
    render_budget = int(profile.get("render_budget_tokens", 256))

    seen_keys: set[str] = set()
    per_source: Dict[str, int] = {}
    items: List[MemoryItemV1] = []
    backend_counts: Dict[str, int] = {}

    for cand in candidates:
        source = str(cand.get("source") or "unknown")
        backend_counts[source] = backend_counts.get(source, 0) + 1
        key = _key_for(cand)
        if key in seen_keys:
            continue
        if per_source.get(source, 0) >= max_per_source:
            continue

        score = _norm_score(cand.get("score"))
        snippet = cand.get("text") or cand.get("snippet") or ""

        item = MemoryItemV1(
            id=str(cand.get("id") or key),
            source=source,
            source_ref=cand.get("source_ref"),
            uri=cand.get("uri"),
            score=score,
            ts=cand.get("ts"),
            title=cand.get("title"),
            snippet=str(snippet)[:800],
            tags=[str(t) for t in (cand.get("tags") or []) if t],
        )

        seen_keys.add(key)
        per_source[source] = per_source.get(source, 0) + 1
        items.append(item)
        if len(items) >= max_total:
            break

    rendered = render_items(items, render_budget)
    stats = MemoryBundleStatsV1(
        backend_counts=backend_counts,
        latency_ms=latency_ms,
        profile=profile.get("profile"),
    )
    return MemoryBundleV1(rendered=rendered, items=items, stats=stats)
