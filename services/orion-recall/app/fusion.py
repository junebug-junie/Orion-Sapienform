from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, MemoryItemV1

try:
    from .render import render_items
except ImportError:  # pragma: no cover - fallback when not in package context
    from render import render_items  # type: ignore

DEFAULT_BACKEND_WEIGHTS = {
    "vector": 1.0,
    "sql_timeline": 0.9,
    "sql_chat": 0.6,
    "rdf_chat": 0.4,
    "rdf": 0.3,
}
OVERLAP_WEIGHT = 0.15
EXACT_MATCH_WEIGHT = 0.45
RECENCY_WEIGHT = 0.2


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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", (text or "").lower())


def _rare_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", text or "")
    return [tok for tok in tokens if len(tok) >= 6 or any(ch.isdigit() for ch in tok)]


def _overlap_count(query_text: str, snippet: str) -> int:
    if not query_text:
        return 0
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return 0
    snippet_tokens = set(_tokenize(snippet))
    return len(query_tokens.intersection(snippet_tokens))


def _exact_match_boost(query_text: str, snippet: str) -> float:
    for token in _rare_tokens(query_text):
        if token.lower() in (snippet or "").lower():
            return 1.0
    return 0.0


def _recency_score(ts: Any, *, half_life_hours: float) -> float:
    if ts is None or half_life_hours <= 0:
        return 0.0
    try:
        ts_val = float(ts)
    except Exception:
        return 0.0
    age_hours = max(0.0, (time.time() - ts_val) / 3600.0)
    return max(0.0, min(1.0, 0.5 ** (age_hours / half_life_hours)))


def _backend_weights(profile: Dict[str, Any]) -> Dict[str, float]:
    weights = dict(DEFAULT_BACKEND_WEIGHTS)
    raw = profile.get("backend_weights")
    if raw is None:
        relevance = profile.get("relevance")
        if isinstance(relevance, dict):
            raw = relevance.get("backend_weights")
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                weights[str(key)] = float(value)
            except Exception:
                continue
    return weights


def _tag_prefix_boost(tags: List[str], profile: Dict[str, Any]) -> float:
    filters = profile.get("filters")
    if not isinstance(filters, dict):
        return 0.0
    prefixes = filters.get("prefer_tags")
    if not isinstance(prefixes, list):
        return 0.0
    boost = 0.0
    per_match = float(filters.get("prefer_tags_boost", 0.15))
    for tag in tags:
        if any(str(tag).startswith(str(prefix)) for prefix in prefixes):
            boost = max(boost, per_match)
    return boost


def _turn_effect_boost(cand: Dict[str, Any], profile: Dict[str, Any]) -> float:
    sql_cfg = profile.get("sql")
    if not isinstance(sql_cfg, dict):
        return 0.0
    if not sql_cfg.get("enable_turn_effect_boost"):
        return 0.0
    try:
        delta = float(cand.get("turn_effect_delta") or 0.0)
    except Exception:
        return 0.0
    min_delta = float(sql_cfg.get("turn_effect_min_abs_delta", 0.2))
    if abs(delta) < min_delta:
        return 0.0
    weight = float(sql_cfg.get("turn_effect_boost_weight", 0.35))
    return weight * min(1.0, abs(delta))


def _denial_patterns() -> List[re.Pattern[str]]:
    return [
        re.compile(r"i\\s+don['’]t\\s+have\\s+(a|any)\\s+(specific\\s+)?memory", re.I),
        re.compile(r"i\\s+don['’]t\\s+recall", re.I),
        re.compile(r"could\\s+you\\s+provide\\s+a\\s+keyword", re.I),
    ]


def fuse_candidates(
    *,
    candidates: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    latency_ms: int = 0,
    query_text: str | None = None,
    session_id: str | None = None,
    diagnostic: bool = False,
    browse_mode: bool = False,
) -> Tuple[MemoryBundleV1, List[Dict[str, Any]]]:
    max_per_source = int(profile.get("max_per_source", 3))
    max_total = int(profile.get("max_total_items", 12))
    render_budget = int(profile.get("render_budget_tokens", 256))
    half_life_hours = float(profile.get("time_decay_half_life_hours", 72) or 72)
    weights = _backend_weights(profile)
    query_text = query_text or ""

    seen_keys: Dict[str, Dict[str, Any]] = {}
    per_source: Dict[str, int] = {}
    items: List[MemoryItemV1] = []
    backend_counts: Dict[str, int] = {}
    ranking_debug: List[Dict[str, Any]] = []

    denial_patterns = _denial_patterns()
    for cand in candidates:
        source = str(cand.get("source") or "unknown")
        backend_counts[source] = backend_counts.get(source, 0) + 1
        key = _key_for(cand)
        snippet = str(cand.get("text") or cand.get("snippet") or "")
        exact_boost = _exact_match_boost(query_text, snippet) if query_text else 0.0
        if query_text and not browse_mode:
            denial_hit = any(pattern.search(snippet) for pattern in denial_patterns)
            if denial_hit and exact_boost <= 0.0:
                continue

        base_score = _norm_score(cand.get("score"))
        overlap = _overlap_count(query_text, snippet)
        recency = _recency_score(cand.get("ts"), half_life_hours=half_life_hours)
        tags = [str(t) for t in (cand.get("tags") or []) if t]
        tag_boost = _tag_prefix_boost(tags, profile)
        turn_effect_boost = _turn_effect_boost(cand, profile)
        backend_weight = float(weights.get(source, 0.5))
        if browse_mode:
            composite = base_score
        else:
            composite = backend_weight * (
                base_score
                + (OVERLAP_WEIGHT * overlap)
                + (EXACT_MATCH_WEIGHT * exact_boost)
                + (RECENCY_WEIGHT * recency)
            ) + tag_boost + turn_effect_boost

        ranked = dict(cand)
        ranked["_relevance"] = {
            "key": key,
            "base_score": base_score,
            "overlap": overlap,
            "exact_boost": exact_boost,
            "recency": recency,
            "backend_weight": backend_weight,
            "tag_boost": tag_boost,
            "turn_effect_boost": turn_effect_boost,
            "composite_score": composite,
        }

        existing = seen_keys.get(key)
        if existing is None or ranked["_relevance"]["composite_score"] > existing["_relevance"]["composite_score"]:
            seen_keys[key] = ranked

    ranked_candidates = list(seen_keys.values())
    if browse_mode:
        ranked_candidates.sort(
            key=lambda item: (
                -float(item.get("ts") or 0.0),
                str(item.get("source") or ""),
                str(item.get("id") or item["_relevance"]["key"]),
            )
        )
    else:
        ranked_candidates.sort(
            key=lambda item: (
                -float(item["_relevance"]["composite_score"]),
                -float(item["_relevance"]["base_score"]),
                -float(item["_relevance"]["recency"]),
                str(item.get("source") or ""),
                str(item.get("id") or item["_relevance"]["key"]),
            )
        )

    for idx, cand in enumerate(ranked_candidates, start=1):
        source = str(cand.get("source") or "unknown")
        if per_source.get(source, 0) >= max_per_source:
            selected = False
        else:
            selected = len(items) < max_total

        if selected:
            snippet = cand.get("text") or cand.get("snippet") or ""
            composite_score = float(cand["_relevance"]["composite_score"])
            rel = cand["_relevance"]
            tags = [str(t) for t in (cand.get("tags") or []) if t]
            tags.extend(
                [
                    f"rel:{rel['composite_score']:.3f}",
                    f"ov:{rel['overlap']}",
                    f"ex:{rel['exact_boost']:.1f}",
                    f"rc:{rel['recency']:.2f}",
                    f"bw:{rel['backend_weight']:.2f}",
                    f"bs:{rel['base_score']:.2f}",
                ]
            )
            item = MemoryItemV1(
                id=str(cand.get("id") or cand["_relevance"]["key"]),
                source=source,
                source_ref=cand.get("source_ref"),
                uri=cand.get("uri"),
                score=max(0.0, min(1.0, composite_score)),
                ts=cand.get("ts"),
                title=cand.get("title"),
                snippet=str(snippet)[:800],
                tags=tags,
            )
            per_source[source] = per_source.get(source, 0) + 1
            items.append(item)

        if diagnostic:
            ranking_debug.append(
                {
                    "id": str(cand.get("id") or cand["_relevance"]["key"]),
                    "source": source,
                    "rank": idx,
                    "selected": selected,
                    "composite_score": cand["_relevance"]["composite_score"],
                    "backend_weight": cand["_relevance"]["backend_weight"],
                    "overlap": cand["_relevance"]["overlap"],
                    "exact_boost": cand["_relevance"]["exact_boost"],
                    "recency": cand["_relevance"]["recency"],
                }
            )

    profile_name = profile.get("profile")
    rendered = render_items(items, render_budget, profile_name=profile_name)
    is_graphtri = bool(profile_name) and (
        str(profile_name) == "graphtri.v1" or str(profile_name).startswith("graphtri")
    )
    if is_graphtri:
        job_offer_terms = ("job offer", "AI/ML", "Architect")
        bundle_has_terms = any(
            any(term in (item.snippet or "") for term in job_offer_terms) for item in items
        )
        top_vector = next(
            (item for item in items if str(item.source or "") == "vector" and item.snippet), None
        )
        top_vector_head = (top_vector.snippet or "")[:80] if top_vector else ""
        digest_has_terms = any(term in rendered for term in job_offer_terms)
        logger = logging.getLogger("orion-recall.render")
        logger.info(
            "graphtri_render_summary top_vector_snippet_head=%r bundle_has_job_offer_terms=%s digest_has_job_offer_terms=%s rendered_len_chars=%s",
            top_vector_head,
            bundle_has_terms,
            digest_has_terms,
            len(rendered),
        )
    stats = MemoryBundleStatsV1(
        backend_counts=backend_counts,
        latency_ms=latency_ms,
        profile=profile.get("profile"),
    )
    return MemoryBundleV1(rendered=rendered, items=items, stats=stats), ranking_debug
