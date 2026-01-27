from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
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
    "rdf_chat": 0.5,
    "rdf": 0.4,
}
DEFAULT_SCORE_WEIGHT = 0.7
DEFAULT_TEXT_SIM_WEIGHT = 0.2
DEFAULT_RECENCY_WEIGHT = 0.1
DEFAULT_SESSION_BOOST = 0.1


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


def _text_tokens(text: str) -> List[str]:
    if not text:
        return []
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(token) >= 3:
            tokens.append(token)
    return tokens


def _text_similarity(query: str, snippet: str) -> float:
    query_terms = set(_text_tokens(query))
    if not query_terms:
        return 0.0
    snippet_terms = set(_text_tokens(snippet))
    if not snippet_terms:
        return 0.0
    overlap = query_terms.intersection(snippet_terms)
    return min(1.0, len(overlap) / max(1, len(query_terms)))


def _extract_session_id(item: Dict[str, Any]) -> str | None:
    sid = item.get("session_id")
    if isinstance(sid, str) and sid:
        return sid
    meta = item.get("meta")
    if isinstance(meta, dict):
        meta_sid = meta.get("session_id")
        if isinstance(meta_sid, str) and meta_sid:
            return meta_sid
    tags = item.get("tags") or []
    if isinstance(tags, (list, tuple)):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("session_id:"):
                return tag.split("session_id:", 1)[-1]
    return None


def _recency_score(ts: Any, *, now: float, half_life_hours: float) -> float:
    if not ts or half_life_hours <= 0:
        return 0.0
    try:
        ts_val = float(ts)
    except Exception:
        return 0.0
    age_seconds = max(0.0, now - ts_val)
    age_hours = age_seconds / 3600.0
    return max(0.0, min(1.0, 0.5 ** (age_hours / half_life_hours)))


def _relevance_config(profile: Dict[str, Any]) -> Dict[str, Any]:
    relevance = profile.get("relevance")
    if not isinstance(relevance, dict):
        relevance = {}
    backend_weights = relevance.get("backend_weights")
    if not isinstance(backend_weights, dict):
        backend_weights = profile.get("backend_weights")
    if not isinstance(backend_weights, dict):
        backend_weights = DEFAULT_BACKEND_WEIGHTS

    def _to_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    return {
        "backend_weights": {str(k): _to_float(v, 0.0) for k, v in backend_weights.items()},
        "score_weight": _to_float(relevance.get("score_weight", DEFAULT_SCORE_WEIGHT), DEFAULT_SCORE_WEIGHT),
        "text_similarity_weight": _to_float(
            relevance.get("text_similarity_weight", DEFAULT_TEXT_SIM_WEIGHT), DEFAULT_TEXT_SIM_WEIGHT
        ),
        "recency_weight": _to_float(relevance.get("recency_weight", DEFAULT_RECENCY_WEIGHT), DEFAULT_RECENCY_WEIGHT),
        "enable_recency": bool(relevance.get("enable_recency", False)),
        "recency_half_life_hours": _to_float(
            relevance.get("recency_half_life_hours", profile.get("time_decay_half_life_hours", 72)),
            72.0,
        ),
        "session_boost": _to_float(relevance.get("session_boost", DEFAULT_SESSION_BOOST), DEFAULT_SESSION_BOOST),
    }


def fuse_candidates(
    *,
    candidates: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    latency_ms: int = 0,
    query_text: str | None = None,
    session_id: str | None = None,
    diagnostic: bool = False,
) -> Tuple[MemoryBundleV1, List[Dict[str, Any]]]:
    max_per_source = int(profile.get("max_per_source", 3))
    max_total = int(profile.get("max_total_items", 12))
    render_budget = int(profile.get("render_budget_tokens", 256))

    relevance_cfg = _relevance_config(profile)
    now = datetime.now(timezone.utc).timestamp()
    seen_keys: Dict[str, Dict[str, Any]] = {}
    per_source: Dict[str, int] = {}
    items: List[MemoryItemV1] = []
    backend_counts: Dict[str, int] = {}
    ranking_debug: List[Dict[str, Any]] = []

    for cand in candidates:
        source = str(cand.get("source") or "unknown")
        backend_counts[source] = backend_counts.get(source, 0) + 1
        key = _key_for(cand)
        base_score = _norm_score(cand.get("score"))
        snippet = str(cand.get("text") or cand.get("snippet") or "")
        text_similarity = (
            _text_similarity(query_text or "", snippet) if relevance_cfg["text_similarity_weight"] > 0 else 0.0
        )
        recency = (
            _recency_score(cand.get("ts"), now=now, half_life_hours=relevance_cfg["recency_half_life_hours"])
            if relevance_cfg["enable_recency"]
            else 0.0
        )
        backend_weight = relevance_cfg["backend_weights"].get(source, 0.5)
        session_match = _extract_session_id(cand)
        session_boost = relevance_cfg["session_boost"] if session_id and session_match == session_id else 0.0
        composite = (
            backend_weight
            * (
                relevance_cfg["score_weight"] * base_score
                + relevance_cfg["text_similarity_weight"] * text_similarity
                + relevance_cfg["recency_weight"] * recency
            )
            + session_boost
        )

        ranked = dict(cand)
        ranked["_relevance"] = {
            "key": key,
            "base_score": base_score,
            "text_similarity": text_similarity,
            "recency": recency,
            "backend_weight": backend_weight,
            "session_boost": session_boost,
            "composite_score": composite,
        }

        existing = seen_keys.get(key)
        if existing is None or ranked["_relevance"]["composite_score"] > existing["_relevance"]["composite_score"]:
            seen_keys[key] = ranked

    ranked_candidates = list(seen_keys.values())
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
            item = MemoryItemV1(
                id=str(cand.get("id") or cand["_relevance"]["key"]),
                source=source,
                source_ref=cand.get("source_ref"),
                uri=cand.get("uri"),
                score=max(0.0, min(1.0, composite_score)),
                ts=cand.get("ts"),
                title=cand.get("title"),
                snippet=str(snippet)[:800],
                tags=[str(t) for t in (cand.get("tags") or []) if t],
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
                    "recency_contribution": relevance_cfg["recency_weight"] * cand["_relevance"]["recency"],
                    "session_boost": cand["_relevance"]["session_boost"],
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
