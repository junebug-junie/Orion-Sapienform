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
TRANSCRIPT_SOURCES = {"sql_chat", "sql_timeline"}


def _norm_score(score: Any) -> float:
    try:
        f = float(score)
    except Exception:
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(0.0, min(1.0, f))


def _normalize_whitespace(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _extract_transcript_parts(text: str) -> tuple[str, str]:
    compact = _normalize_whitespace(text)
    exact = re.search(r'ExactUserText:\s*"(.*?)"\s*OrionResponse:\s*"(.*?)"', compact, flags=re.I)
    if exact:
        return exact.group(1).strip(), exact.group(2).strip()
    user = ""
    assistant = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("user:"):
            user = line.split(":", 1)[1].strip()
        elif lowered.startswith("orion:"):
            assistant = line.split(":", 1)[1].strip()
    return user, assistant


def _transcript_fingerprint(text: str) -> str:
    user, assistant = _extract_transcript_parts(text)
    if user or assistant:
        canonical = f"user:{_normalize_whitespace(user).lower()}|orion:{_normalize_whitespace(assistant).lower()}"
    else:
        canonical = _normalize_whitespace(text).lower()
    return hashlib.md5(canonical.encode("utf-8", "ignore")).hexdigest()


def _is_transcript_like(item: Dict[str, Any], snippet: str) -> bool:
    source = str(item.get("source") or "").strip().lower()
    if source in TRANSCRIPT_SOURCES:
        return True
    tags = [str(tag).lower() for tag in (item.get("tags") or []) if tag]
    if any(tag.startswith("chat_timeline") for tag in tags):
        return True
    if source == "vector":
        if "exactusertext:" in snippet.lower() or "orionresponse:" in snippet.lower():
            return True
        if "user:" in snippet.lower() and "orion:" in snippet.lower():
            return True
    return False


def _strong_turn_id(item: Dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    for key in ("correlation_id", "chat_turn_id", "message_id", "trace_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return ""


def _key_for(item: Dict[str, Any]) -> tuple[str, str]:
    uri = str(item.get("uri") or "").strip()
    src_id = str(item.get("id") or "").strip()
    snippet = str(item.get("text") or item.get("snippet") or "")
    transcript_like = _is_transcript_like(item, snippet)
    if transcript_like:
        strong_id = _strong_turn_id(item)
        if strong_id:
            return f"turn:{strong_id}", "transcript_strong_id"
        return f"transcript:{_transcript_fingerprint(snippet)}", "transcript_fingerprint"
    key_src = uri or src_id or hashlib.md5(snippet.encode("utf-8", "ignore")).hexdigest()
    return f"default:{key_src}", "default"


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]{3,}", text.lower()) if tok}


def _materially_same_transcript(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    ta = _token_set(a)
    tb = _token_set(b)
    if len(ta) < 6 or len(tb) < 6:
        return False
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    if union == 0:
        return False
    return (inter / union) >= 0.92


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


def _relevance_cfg(profile: Dict[str, Any]) -> Dict[str, Any]:
    relevance = profile.get("relevance")
    cfg = relevance if isinstance(relevance, dict) else {}
    return {
        "score_weight": max(0.0, float(cfg.get("score_weight", 0.7))),
        "text_similarity_weight": max(0.0, float(cfg.get("text_similarity_weight", OVERLAP_WEIGHT + EXACT_MATCH_WEIGHT))),
        "recency_weight": max(0.0, float(cfg.get("recency_weight", RECENCY_WEIGHT))),
        "enable_recency": bool(cfg.get("enable_recency", True)),
    }


def _text_similarity_signal(query_text: str, *, overlap: int, exact_boost: float) -> float:
    if not query_text:
        return max(0.0, min(1.0, exact_boost))
    q_tokens = set(_tokenize(query_text))
    if not q_tokens:
        return max(0.0, min(1.0, exact_boost))
    overlap_norm = min(1.0, overlap / float(len(q_tokens)))
    return max(0.0, min(1.0, max(overlap_norm, exact_boost)))


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


def _candidate_allowed(cand: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    filters = profile.get("filters")
    if not isinstance(filters, dict):
        return True

    source = str(cand.get("source") or "unknown")
    tags = [str(tag) for tag in (cand.get("tags") or []) if tag]

    allowed_sources = filters.get("allowed_sources")
    if isinstance(allowed_sources, list) and allowed_sources:
        if source not in {str(item) for item in allowed_sources}:
            return False

    excluded_prefixes = filters.get("exclude_tags_prefixes")
    if isinstance(excluded_prefixes, list):
        for tag in tags:
            if any(tag.startswith(str(prefix)) for prefix in excluded_prefixes):
                return False

    required_any = filters.get("required_tags_any")
    if isinstance(required_any, list) and required_any:
        if not any(tag in {str(item) for item in required_any} for tag in tags):
            return False

    return True


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
    rel_cfg = _relevance_cfg(profile)
    query_text = query_text or ""

    seen_keys: Dict[str, Dict[str, Any]] = {}
    transcript_dedupe_count = 0
    per_source: Dict[str, int] = {}
    items: List[MemoryItemV1] = []
    backend_counts: Dict[str, int] = {}
    ranking_debug: List[Dict[str, Any]] = []
    drop_counts: Dict[str, int] = {}
    selected_counts: Dict[str, int] = {}
    novelty_drop_count = 0

    denial_patterns = _denial_patterns()
    for cand in candidates:
        if not _candidate_allowed(cand, profile):
            continue
        source = str(cand.get("source") or "unknown")
        backend_counts[source] = backend_counts.get(source, 0) + 1
        key, key_type = _key_for(cand)
        snippet = str(cand.get("text") or cand.get("snippet") or "")
        exact_boost = _exact_match_boost(query_text, snippet) if query_text else 0.0
        if query_text and not browse_mode:
            denial_hit = any(pattern.search(snippet) for pattern in denial_patterns)
            if denial_hit and exact_boost <= 0.0:
                continue

        base_score = _norm_score(cand.get("score"))
        overlap = _overlap_count(query_text, snippet)
        recency = _recency_score(cand.get("ts"), half_life_hours=half_life_hours)
        text_similarity = _text_similarity_signal(query_text, overlap=overlap, exact_boost=exact_boost)
        tags = [str(t) for t in (cand.get("tags") or []) if t]
        tag_boost = _tag_prefix_boost(tags, profile)
        turn_effect_boost = _turn_effect_boost(cand, profile)
        backend_weight = float(weights.get(source, 0.5))
        if browse_mode:
            composite = base_score
        else:
            recency_component = recency if rel_cfg["enable_recency"] else 0.0
            composite = backend_weight * (
                (rel_cfg["score_weight"] * base_score)
                + (rel_cfg["text_similarity_weight"] * text_similarity)
                + (rel_cfg["recency_weight"] * recency_component)
            ) + tag_boost + turn_effect_boost

        ranked = dict(cand)
        ranked["_relevance"] = {
            "key": key,
            "key_type": key_type,
            "base_score": base_score,
            "overlap": overlap,
            "exact_boost": exact_boost,
            "text_similarity": text_similarity,
            "recency": recency,
            "recency_enabled": rel_cfg["enable_recency"],
            "backend_weight": backend_weight,
            "tag_boost": tag_boost,
            "turn_effect_boost": turn_effect_boost,
            "composite_score": composite,
        }

        existing = seen_keys.get(key)
        if existing is None or ranked["_relevance"]["composite_score"] > existing["_relevance"]["composite_score"]:
            if existing is not None and str(key_type).startswith("transcript"):
                transcript_dedupe_count += 1
            seen_keys[key] = ranked
        elif str(key_type).startswith("transcript"):
            transcript_dedupe_count += 1

    ranked_candidates = list(seen_keys.values())
    if transcript_dedupe_count:
        logging.getLogger("orion-recall.fusion").info(
            "fusion transcript dedupe collapsed=%s",
            transcript_dedupe_count,
        )
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

    selected_transcripts: List[str] = []
    for idx, cand in enumerate(ranked_candidates, start=1):
        source = str(cand.get("source") or "unknown")
        drop_reason = ""
        if per_source.get(source, 0) >= max_per_source:
            selected = False
            drop_reason = "max_per_source"
        else:
            selected = len(items) < max_total
            if not selected:
                drop_reason = "max_total_items"

        snippet = str(cand.get("text") or cand.get("snippet") or "")
        transcript_like = _is_transcript_like(cand, snippet)
        transcript_norm = ""
        if selected and transcript_like:
            user, assistant = _extract_transcript_parts(snippet)
            transcript_norm = _normalize_whitespace(
                f"user:{user.lower()}|orion:{assistant.lower()}" if (user or assistant) else snippet.lower()
            )
            duplicate_transcript = any(_materially_same_transcript(transcript_norm, prev) for prev in selected_transcripts)
            if duplicate_transcript:
                selected = False
                drop_reason = "transcript_novelty"
                novelty_drop_count += 1

        if selected:
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
            selected_counts[source] = selected_counts.get(source, 0) + 1
            items.append(item)
            if transcript_like and transcript_norm:
                selected_transcripts.append(transcript_norm)
        elif drop_reason:
            drop_counts[drop_reason] = drop_counts.get(drop_reason, 0) + 1

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
                    "text_similarity": cand["_relevance"]["text_similarity"],
                    "recency": cand["_relevance"]["recency"],
                    "drop_reason": drop_reason or None,
                    "key_type": cand["_relevance"].get("key_type"),
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
        diagnostic=(
            {
                "transcript_dedupe_collapsed": transcript_dedupe_count,
                "novelty_drop_count": novelty_drop_count,
                "drop_counts": drop_counts,
                "source_candidate_counts": backend_counts,
                "source_selected_counts": selected_counts,
            }
            if diagnostic
            else None
        ),
    )
    return MemoryBundleV1(rendered=rendered, items=items, stats=stats), ranking_debug
