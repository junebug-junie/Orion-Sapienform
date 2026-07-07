from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional dependency in test env
    psycopg2 = None
from pydantic import ValidationError
from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import (
    MemoryBundleV1,
    RecallDecisionV1,
    RecallQueryV1,
)

try:
    from .fusion import fuse_candidates, pcr_fuse_belief_candidates, render_continuity_bundle
    from .pcr_collectors import apply_collector_plan, collectors_for_intent
    from .collectors.active_packet import fetch_active_packet_fragments
    from .profiles import get_profile
    from .settings import settings
    from .source_policy import build_vector_policy
    from .storage.rdf_adapter import (
        fetch_rdf_fragments,
        fetch_rdf_expansion_terms,
        fetch_rdf_connected_chatturns,
        fetch_graphtri_anchors,
        fetch_rdf_graphtri_anchor_terms,
        fetch_rdf_graphtri_fragments,
        fetch_rdf_chatturn_fragments,
        fetch_rdf_chatturn_exact_matches,
    )
    from .sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments
    from .sql_chat import fetch_chat_history_pairs, fetch_chat_messages, fetch_chat_turn_timestamps
    from .cards_adapter import fetch_card_fragments_guarded
    try:
        from .storage.graph_compression_adapter import fetch_graph_compression_fragments
    except ImportError:
        fetch_graph_compression_fragments = None  # type: ignore

except ImportError as _e:  # pragma: no cover - fallback for runtime pathing
    _IMPORT_ERROR = _e
    try:
        # Container/package-safe absolute imports
        from app.fusion import fuse_candidates, pcr_fuse_belief_candidates, render_continuity_bundle  # type: ignore
        from app.pcr_collectors import apply_collector_plan, collectors_for_intent  # type: ignore
        from app.collectors.active_packet import fetch_active_packet_fragments  # type: ignore
        from app.profiles import get_profile  # type: ignore
        from app.settings import settings  # type: ignore
        from app.source_policy import build_vector_policy  # type: ignore
        from app.storage.rdf_adapter import (  # type: ignore
            fetch_rdf_fragments,
            fetch_rdf_expansion_terms,
            fetch_rdf_connected_chatturns,
            fetch_graphtri_anchors,
            fetch_rdf_graphtri_anchor_terms,
            fetch_rdf_graphtri_fragments,
            fetch_rdf_chatturn_fragments,
            fetch_rdf_chatturn_exact_matches,
        )
        from app.sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments  # type: ignore
        from app.sql_chat import fetch_chat_history_pairs, fetch_chat_messages, fetch_chat_turn_timestamps  # type: ignore
        from app.cards_adapter import fetch_card_fragments_guarded  # type: ignore
        try:
            from app.storage.graph_compression_adapter import fetch_graph_compression_fragments  # type: ignore
        except ImportError:
            fetch_graph_compression_fragments = None  # type: ignore
    except ImportError:
        # IMPORTANT: raise the real root cause, not the fallback failure
        raise _IMPORT_ERROR

logger = logging.getLogger("orion-recall.worker")

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None

_recall_pg_pool: Any = None


def set_recall_pg_pool(pool: Any) -> None:
    global _recall_pg_pool
    _recall_pg_pool = pool


RECALL_REQUEST_KIND = "recall.query.v1"
RECALL_REPLY_KIND = "recall.reply.v1"
RECALL_TELEMETRY_KIND = "recall.decision.v1"
def _build_recall_pressure_events(
    *,
    q: RecallQueryV1,
    decision: RecallDecisionV1,
    bundle: MemoryBundleV1,
    compare_summary: Dict[str, Any] | None = None,
    anchor_plan: Dict[str, Any] | None = None,
    selected_evidence_cards: list[Dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    events: list[MutationPressureEvidenceV1] = []
    snippets = [item.snippet.lower() for item in bundle.items]
    query_text = (q.fragment or "").lower()
    selected_any = bool(bundle.items)
    exact_anchor_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\d+\b|\b[A-Fa-f0-9]{7,40}\b", str(q.fragment or ""))[:8]
    exact_anchor_hit = any(token.lower() in " ".join(snippets) for token in exact_anchor_tokens) if exact_anchor_tokens else True
    top_source = bundle.items[0].source if bundle.items else ""
    stale_selected = False
    if bundle.items and bundle.items[0].ts is not None:
        try:
            age_hours = max(0.0, (time.time() - float(bundle.items[0].ts)) / 3600.0)
            stale_selected = age_hours > 24.0 * 30.0
        except Exception:
            stale_selected = False

    shared_metadata: Dict[str, Any] = {"recall_evidence_kind": "live_shadow"}
    if isinstance(compare_summary, dict) and compare_summary:
        shared_metadata["v1_v2_compare"] = compare_summary
    if isinstance(anchor_plan, dict) and anchor_plan:
        shared_metadata["anchor_plan"] = anchor_plan
    if isinstance(selected_evidence_cards, list) and selected_evidence_cards:
        shared_metadata["selected_evidence_cards"] = selected_evidence_cards[:8]

    if not selected_any:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="recall_miss_or_dissatisfaction",
                confidence=0.9,
                evidence_refs=[f"recall_decision:{decision.id}", f"query:{q.fragment[:120]}"],
                metadata={"reason": "no_selected_items", "profile": decision.profile, **shared_metadata},
            )
        )
    if selected_any and query_text and not any(tok in " ".join(snippets) for tok in _extract_keywords(q.fragment)):
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="unsupported_memory_claim",
                confidence=0.72,
                evidence_refs=[f"recall_decision:{decision.id}", f"selected_ids:{','.join(decision.selected_ids[:6])}"],
                metadata={"reason": "selected_without_query_support", **shared_metadata},
            )
        )
    if selected_any and top_source == "vector" and any("vector" in str(tag).lower() for item in bundle.items for tag in item.tags):
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="irrelevant_semantic_neighbor",
                confidence=0.55,
                evidence_refs=[f"recall_decision:{decision.id}", f"top_source:{top_source}"],
                metadata={"reason": "vector_top_hit_requires_anchor_validation", **shared_metadata},
            )
        )
    if not exact_anchor_hit:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="missing_exact_anchor",
                confidence=0.81,
                evidence_refs=[f"recall_decision:{decision.id}", f"anchor_tokens:{','.join(exact_anchor_tokens[:6])}"],
                metadata={"reason": "exact_anchor_not_in_selected", **shared_metadata},
            )
        )
    if stale_selected:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="stale_memory_selected",
                confidence=0.76,
                evidence_refs=[f"recall_decision:{decision.id}", f"top_id:{bundle.items[0].id}"],
                metadata={"reason": "top_item_stale", "source": bundle.items[0].source, **shared_metadata},
            )
        )
    # keep bounded and first-class serialized
    return [item.model_dump(mode="json") for item in events[:5]]



def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )


def _extract_entities(text: str) -> List[str]:
    ents = set()
    ents.update(re.findall(r"[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*", text))
    ents.update(re.findall(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", text, flags=re.I))
    ents.update(re.findall(r"[A-Za-z0-9_]+\\.[A-Za-z0-9_\\.]+", text))
    return [e.strip() for e in ents if e.strip()]


def _extract_keywords(text: str, *, max_keywords: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", (text or "").lower())
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _anchor_tokens(text: str, *, max_tokens: int = 3) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"\\b[A-Za-z][A-Za-z0-9]*\\d+\\b", text)
    seen = set()
    anchors: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        anchors.append(token)
        if len(anchors) >= max_tokens:
            break
    return anchors


def _is_memory_browse(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return bool(
        re.search(r"\\b(fetch|show|list|browse|recall)\\b", lowered)
        and re.search(r"\\b(memory|memories|recent|context)\\b", lowered)
    )


_SOCIAL_TOKENS = {
    "hi",
    "hey",
    "hello",
    "thanks",
    "thank",
    "good",
    "great",
    "cool",
    "nice",
    "fine",
    "well",
    "friend",
    "orion",
    "juniper",
    "morning",
    "afternoon",
    "evening",
    "awesome",
    "okay",
    "ok",
}

_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "have",
    "been",
    "just",
    "your",
    "about",
    "into",
    "over",
    "mostly",
    "very",
    "really",
    "still",
    "will",
    "would",
    "should",
    "could",
}


def _strip_recall_instruction_tail(text: str) -> Tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "", False
    parts = [part.strip(" -:\t") for part in re.split(r"[\n;]+", raw) if part.strip()]
    if len(parts) < 2:
        return raw, False
    tail = parts[-1].lower()
    instruction_markers = (
        "use recall",
        "based on recall",
        "remain based on recall",
        "stay based on recall",
        "process injection",
    )
    if any(marker in tail for marker in instruction_markers):
        return " ".join(parts[:-1]).strip(), True
    return raw, False


def _social_clause(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True
    if re.match(r"^(hi|hey|hello|yo|how are you|how's it going)[!.? ]*$", normalized):
        return True
    tokens = re.findall(r"[a-z']{2,}", normalized)
    if len(tokens) <= 7 and tokens and all(token in _SOCIAL_TOKENS for token in tokens):
        return True
    return False


def _informative_score(text: str) -> int:
    tokens = re.findall(r"[a-z0-9']{3,}", text.lower())
    informative = [tok for tok in tokens if tok not in _STOPWORDS]
    long_terms = [tok for tok in informative if len(tok) >= 6 or any(ch.isdigit() for ch in tok)]
    return len(informative) + len(long_terms)


def _derive_chat_general_query(fragment: str, *, verb: str | None, profile_name: str) -> Dict[str, Any]:
    raw = str(fragment or "").strip()
    applies = str(verb or "") == "chat_general" or profile_name.startswith("chat.general")
    if not applies:
        return {
            "query_fragment": raw,
            "tail_stripped": False,
            "query_changed": False,
            "turn_type": "default",
            "dropped_clauses": 0,
        }
    trimmed, tail_stripped = _strip_recall_instruction_tail(raw)
    clauses = [c.strip() for c in re.split(r"[.!?\n]+", trimmed) if c.strip()]
    substantive = [c for c in clauses if not _social_clause(c)]
    dropped = max(0, len(clauses) - len(substantive))
    ranked = sorted(substantive, key=_informative_score, reverse=True)
    query_fragment = ". ".join(ranked[:2]).strip() if ranked else trimmed
    if not query_fragment:
        query_fragment = raw
    turn_type = "substantive" if substantive else "social"
    return {
        "query_fragment": query_fragment,
        "tail_stripped": tail_stripped,
        "query_changed": query_fragment != raw,
        "turn_type": turn_type,
        "dropped_clauses": dropped,
    }


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _parse_exclusion(q: RecallQueryV1) -> Dict[str, Any]:
    raw = q.exclude if isinstance(q.exclude, dict) else {}
    active_turn_text = str(raw.get("active_turn_text") or q.fragment or "").strip()
    active_turn_ids: List[str] = []
    for value in raw.get("active_turn_ids", []):
        text = str(value or "").strip()
        if text and text not in active_turn_ids:
            active_turn_ids.append(text)
    try:
        active_turn_ts = float(raw.get("active_turn_ts")) if raw.get("active_turn_ts") is not None else None
    except Exception:
        active_turn_ts = None
    return {
        "active_turn_text": active_turn_text,
        "active_turn_ids": active_turn_ids,
        "active_turn_ts": active_turn_ts,
    }


def _suppress_self_hits(
    candidates: List[Dict[str, Any]],
    *,
    active_turn_text: str,
    active_turn_ids: List[str],
    active_turn_ts: float | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not candidates:
        return [], {}
    normalized_active = _normalize_text(active_turn_text)
    id_set = {str(v).strip() for v in active_turn_ids if str(v).strip()}
    suppression_counts: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    for cand in candidates:
        source = str(cand.get("source") or "unknown")
        cand_id = str(cand.get("id") or "").strip()
        cand_text = _normalize_text(cand.get("text") or cand.get("snippet") or "")
        remove = False
        if cand_id and cand_id in id_set:
            remove = True
        elif normalized_active and normalized_active in cand_text:
            age_ok = True
            if active_turn_ts is not None and cand.get("ts") is not None:
                try:
                    age_ok = abs(float(cand.get("ts")) - active_turn_ts) <= 180.0
                except Exception:
                    age_ok = True
            if age_ok:
                remove = True
        if remove:
            suppression_counts[source] = suppression_counts.get(source, 0) + 1
            continue
        filtered.append(cand)
    return filtered, suppression_counts

def _anchor_overlap(text: str, anchors: List[str]) -> int:
    if not text or not anchors:
        return 0
    lowered = text.lower()
    return sum(1 for term in anchors if term and term.lower() in lowered)


def _artifact_density(text: str) -> int:
    if not text:
        return 0
    patterns = [
        r"/[A-Za-z0-9_\-./]+",
        r"orion-[a-z0-9\-]+",
        r"orion:[a-z0-9:._-]+",
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _anchor_from_uri(uri: str) -> str:
    if not uri:
        return ""
    tail = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return tail.replace("_", " ").strip()


def _extract_anchor_terms(rdf_items: List[Dict[str, Any]], *, max_items: int = 12) -> List[str]:
    terms: List[str] = []
    seen = set()

    def _add(term: str) -> None:
        cleaned = (term or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        terms.append(cleaned)

    for item in rdf_items:
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        for candidate in (meta.get("subject"), item.get("uri"), item.get("id")):
            if isinstance(candidate, str):
                _add(_anchor_from_uri(candidate))
        text = item.get("text")
        if isinstance(text, str):
            for token in re.findall(r"[A-Za-z0-9_]{3,}", text):
                _add(token)
        if len(terms) >= max_items:
            break

    return terms[:max_items]


def _expand_query(fragment: str, *, verb: str | None, intent: str | None, enable: bool) -> List[str]:
    if not enable:
        return [fragment]
    signals = [fragment]
    hints = []
    if verb:
        hints.append(verb)
    if intent:
        hints.append(intent)
    entities = _extract_entities(fragment)
    signals.extend(hints)
    signals.extend(entities)
    return [s for s in signals if s]


def _build_anchor_set(
    *,
    query_text: str,
    session_id: str | None,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    anchors = {
        "entities_terms": [],
        "tags_terms": [],
        "claim_objs": [],
        "related_terms": [],
    }
    if not _rdf_enabled(profile) or not settings.RECALL_RDF_ENDPOINT_URL:
        return anchors
    query_terms = _extract_keywords(query_text) if query_text else []
    try:
        anchors = fetch_graphtri_anchors(
            session_id=session_id,
            query_terms=query_terms,
            max_terms=12,
        )
    except Exception as exc:
        logger.debug(f"graphtri anchor fetch skipped: {exc}")
    return anchors


def _rdf_enabled(profile: Dict[str, Any]) -> bool:
    profile_name = str(profile.get("profile") or "")
    profile_enable_rdf = bool(profile.get("enable_rdf", False))
    return (
        profile_enable_rdf
        or profile_name.startswith("deep.graph")
        or profile_name.startswith("graphtri")
        or settings.RECALL_ENABLE_RDF
    ) and int(profile.get("rdf_top_k", 0)) > 0


def _rdf_expansion_enabled(profile: Dict[str, Any]) -> bool:
    if not _rdf_enabled(profile):
        return False
    return bool(profile.get("enable_rdf_expansion", False))


def _rdf_graphtri_mode(profile: Dict[str, Any]) -> bool:
    profile_name = str(profile.get("profile") or "")
    return (
        bool(profile.get("rdf_graphtri_mode", False))
        or profile_name.startswith("graphtri")
        or profile_name.startswith("deep.graph")
    )


def _sql_timeline_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_SQL_TIMELINE:
        return False
    return bool(profile.get("enable_sql_timeline", True))


def _sql_chat_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_SQL_CHAT:
        return False
    profile_name = str(profile.get("profile") or "")
    default_enabled = not profile_name.startswith("chat.general")
    return bool(profile.get("enable_sql_chat", default_enabled))


_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_rdf_chatturn(frag: Dict[str, Any]) -> bool:
    if "chatturn" in {str(t).lower() for t in (frag.get("tags") or [])}:
        return True
    ref = str(frag.get("uri") or frag.get("id") or "")
    return "/chatTurn/" in ref


def _chatturn_id_from_fragment(frag: Dict[str, Any]) -> Optional[str]:
    """Recover the chat_history_log id from an RDF chat-turn fragment.

    RDF turn IRIs look like ``.../chatTurn/<uuid-with-underscores>`` because the writer
    sanitizes ``-`` to ``_``. Reverse that and validate the UUID shape so we only join
    ids we can trust.
    """
    ref = str(frag.get("uri") or frag.get("id") or "")
    if "/chatTurn/" not in ref:
        return None
    tail = ref.rsplit("/chatTurn/", 1)[-1].strip()
    if not tail:
        return None
    candidate = tail.replace("_", "-")
    return candidate if _UUID_RE.match(candidate) else None


async def _window_rdf_chatturn_candidates(
    candidates: List[Dict[str, Any]],
    *,
    since_minutes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop RDF chat-turn candidates older than ``since_minutes``, stamping real timestamps.

    The graph stores no usable ChatTurn timestamp, so these rails otherwise leak months-old
    turns into reflective recall regardless of the profile window. We resolve each turn's
    created_at from chat_history_log and keep only those inside the window. Non chat-turn
    candidates (SQL, cards, RDF claims, etc.) are returned untouched. Memory cards are never
    touched here.
    """
    if since_minutes <= 0:
        return candidates, 0
    turn_ids: Dict[int, str] = {}
    for idx, frag in enumerate(candidates):
        if not _is_rdf_chatturn(frag):
            continue
        cid = _chatturn_id_from_fragment(frag)
        if cid is not None:
            turn_ids[idx] = cid
    if not turn_ids:
        return candidates, 0

    try:
        ts_map = await fetch_chat_turn_timestamps(list(set(turn_ids.values())), since_minutes)
    except Exception as exc:  # pragma: no cover - defensive; never fail recall on this
        logger.debug("rdf chat-turn windowing skipped: %s", exc)
        return candidates, 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for idx, frag in enumerate(candidates):
        if idx not in turn_ids:
            kept.append(frag)
            continue
        ts = ts_map.get(turn_ids[idx])
        if ts is None:
            # Outside the window or not resolvable to a chat row → drop from reflective recall.
            dropped += 1
            continue
        frag = dict(frag)
        frag["ts"] = ts
        kept.append(frag)
    return kept, dropped


_SQL_CHAT_SOURCES = frozenset({"sql_timeline", "sql_chat"})


def _sql_chat_row_id(candidate: Dict[str, Any]) -> Optional[str]:
    raw = str(candidate.get("id") or "").strip()
    if not raw:
        return None
    if _UUID_RE.match(raw):
        return raw
    if raw.startswith("chat_"):
        return None
    # chat_history_log rows sometimes use correlation_id-shaped ids
    if _UUID_RE.match(raw.replace("_", "-")):
        return raw.replace("_", "-")
    return None


async def _window_sql_chat_candidates(
    candidates: List[Dict[str, Any]],
    *,
    since_minutes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop SQL chat/timeline candidates outside ``since_minutes``.

    Mirrors ``_window_rdf_chatturn_candidates``: anchor-exact SQL rails used
    ``fetch_exact_fragments`` without a temporal filter, so old turns could surface
    into journal/metacog recall when expansion tokens matched generic words.
    Memory cards and non-SQL sources are untouched.
    """
    if since_minutes <= 0:
        return candidates, 0

    cutoff = time.time() - (int(since_minutes) * 60)
    sql_indices: Dict[int, str] = {}
    for idx, frag in enumerate(candidates):
        if str(frag.get("source") or "") not in _SQL_CHAT_SOURCES:
            continue
        row_id = _sql_chat_row_id(frag)
        if row_id is not None:
            sql_indices[idx] = row_id

    ts_map: Dict[str, float] = {}
    if sql_indices:
        try:
            ts_map = await fetch_chat_turn_timestamps(list(set(sql_indices.values())), since_minutes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sql chat windowing skipped: %s", exc)
            return candidates, 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for idx, frag in enumerate(candidates):
        source = str(frag.get("source") or "")
        if source not in _SQL_CHAT_SOURCES:
            kept.append(frag)
            continue

        row_id = sql_indices.get(idx)
        if row_id is not None:
            ts = ts_map.get(row_id)
            if ts is None:
                dropped += 1
                continue
            frag = dict(frag)
            frag["ts"] = ts
            kept.append(frag)
            continue

        ts_val = frag.get("ts")
        try:
            ts_float = float(ts_val) if ts_val is not None else 0.0
        except Exception:
            ts_float = 0.0
        if ts_float <= 0.0 or ts_float < cutoff:
            dropped += 1
            continue
        kept.append(frag)

    return kept, dropped


async def _fetch_anchor_candidates(
    *,
    query_text: str,
    session_id: str | None,
    node_id: str | None,
    profile: Dict[str, Any],
    diagnostic: bool = False,
    exclusion: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    tokens = _anchor_tokens(query_text)
    if not tokens:
        return [], {}

    candidates: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    limit = max(3, min(10, int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K))))
    exclusion = exclusion or {}
    since_minutes = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))

    try:
        sql_items = await fetch_exact_fragments(
            tokens=tokens,
            session_id=session_id,
            node_id=node_id,
            limit=limit,
            since_minutes=since_minutes,
            exclude_ids=exclusion.get("active_turn_ids"),
            exclude_text=exclusion.get("active_turn_text"),
        )
        counts["sql_timeline_anchor"] = len(sql_items)
        for item in sql_items:
            tags = list(item.tags or [])
            tags.append("anchor_exact")
            candidates.append(
                {
                    "id": item.id,
                    "source": "sql_timeline",
                    "source_ref": item.source_ref,
                    "text": item.text,
                    "ts": item.ts,
                    "tags": tags,
                    "score": 0.95,
                }
            )
    except Exception as exc:
        logger.debug(f"sql anchor fetch skipped: {exc}")

    counts["vector_anchor"] = 0

    if _rdf_enabled(profile) and settings.RECALL_RDF_ENDPOINT_URL:
        try:
            rdf = fetch_rdf_chatturn_exact_matches(
                tokens=tokens,
                session_id=session_id,
                max_items=limit,
            )
            counts["rdf_chat_anchor"] = len(rdf)
            for item in rdf:
                item = dict(item)
                item["tags"] = list(item.get("tags") or []) + ["anchor_exact"]
                item["score"] = max(0.9, float(item.get("score") or 0.0))
                candidates.append(item)
        except Exception as exc:
            logger.debug(f"rdf anchor fetch skipped: {exc}")

    if diagnostic:
        logger.info(
            "anchor rail tokens=%s counts=%s",
            tokens,
            counts,
        )

    return candidates, counts


def _cards_fetch_enabled(profile: Dict[str, Any]) -> bool:
    if not bool(getattr(settings, "RECALL_ENABLE_CARDS", False)):
        return False
    w = profile.get("backend_weights")
    if not isinstance(w, dict):
        w = {}
    rel = profile.get("relevance")
    if isinstance(rel, dict) and isinstance(rel.get("backend_weights"), dict):
        w = {**w, **rel["backend_weights"]}
    try:
        wt = float(w.get("cards", 0.0) or 0.0)
    except Exception:
        wt = 0.0
    topk = int(profile.get("cards_top_k", 0) or 0)
    return topk > 0 or wt > 0.0


async def _query_backends(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
    diagnostic: bool = False,
    exclusion: Dict[str, Any] | None = None,
    lane: str | None = None,
    include_cards: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    exclusion = exclusion or {}

    rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
    rdf_top_k = int(profile.get("rdf_top_k", 0))
    expansion_terms: List[str] = []
    anchor_plan: Dict[str, Any] = {
        "entities_terms": [],
        "tags_terms": [],
        "claim_objs": [],
        "related_terms": [],
    }
    if rdf_enabled:
        try:
            if _rdf_expansion_enabled(profile):
                try:
                    query_terms = _extract_keywords(fragment) if fragment else []
                    anchor_plan = fetch_graphtri_anchors(
                        session_id=session_id,
                        query_terms=query_terms,
                        max_terms=int(profile.get("rdf_expansion_top_k", 8)),
                    )
                    expansion_terms = list(anchor_plan.get("related_terms") or [])
                    backend_counts["rdf_anchor_terms"] = len(expansion_terms)
                except Exception as exc:
                    logger.debug(f"rdf expansion anchor fetch skipped: {exc}")
                    backend_counts["rdf_anchor_terms"] = 0
            else:
                backend_counts["rdf_anchor_terms"] = 0

            # 0) Pull raw ChatTurns (prompt/response) from GRAPH <orion:chat>.
            rdf_chat: List[Dict[str, Any]] = []
            rdf_chat = fetch_rdf_chatturn_fragments(
                query_text=fragment,
                session_id=session_id,
                max_items=max(rdf_top_k, 6),
            )
            backend_counts["rdf_chat"] = len(rdf_chat)
            candidates.extend(rdf_chat)

            if expansion_terms:
                try:
                    connected = fetch_rdf_connected_chatturns(
                        terms=expansion_terms,
                        max_items=int(
                            profile.get("rdf_connected_chat_top_k", rdf_top_k)
                        ),
                    )
                    backend_counts["rdf_connected_chat"] = len(connected)
                    candidates.extend(connected)
                except Exception as exc:
                    logger.debug(f"rdf connected chat fetch skipped: {exc}")
                    backend_counts["rdf_connected_chat"] = 0
            else:
                backend_counts["rdf_connected_chat"] = 0

            # Claims / neighborhood fragments (graphtri lane for brain.recall.v1).
            rdf: List[Dict[str, Any]] = []
            if _rdf_graphtri_mode(profile):
                rdf = fetch_rdf_graphtri_fragments(
                    query_text=fragment,
                    session_id=session_id,
                    max_items=rdf_top_k,
                )
                if not rdf:
                    rdf = fetch_rdf_fragments(
                        query_text=fragment,
                        max_items=rdf_top_k,
                    )
            else:
                rdf = fetch_rdf_fragments(
                    query_text=fragment,
                    max_items=rdf_top_k,
                )

            backend_counts["rdf"] = len(rdf)
            candidates.extend(rdf)
        except Exception as exc:
            logger.debug(f"rdf backend skipped: {exc}")


    backend_counts["vector"] = 0

    if diagnostic:
        logger.info(
            "recall rdf_enabled=%s rdf_top_k=%s rdf_candidates=%s expansion_terms=%s",
            rdf_enabled,
            rdf_top_k,
            backend_counts.get("rdf", 0),
            expansion_terms,
        )
        if rdf_enabled:
            logger.info(
                "recall rdf_expansion profile=%s enabled=%s graphtri_mode=%s anchor_terms=%s anchor_plan=%s connected_chat=%s",
                profile.get("profile"),
                _rdf_expansion_enabled(profile),
                _rdf_graphtri_mode(profile),
                expansion_terms,
                anchor_plan,
                backend_counts.get("rdf_connected_chat", 0),
            )

    if _sql_chat_enabled_for_profile(profile):
        try:
            chat_pairs = await fetch_chat_history_pairs(
                limit=int(profile.get("sql_chat_top_k", settings.RECALL_SQL_TOP_K)),
                since_minutes=int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                exclude_text=exclusion.get("active_turn_text"),
                exclude_ids=exclusion.get("active_turn_ids"),
            )
            backend_counts["sql_chat_pairs"] = len(chat_pairs)
            for item in chat_pairs:
                candidates.append(
                    {
                        "id": item.id,
                        "source": "sql_chat",
                        "source_ref": item.source_ref,
                        "text": item.text,
                        "ts": item.ts,
                        "tags": ["sql", "chat", "pairs"],
                        "score": 0.75,
                    }
                )

            chat_msgs = await fetch_chat_messages(
                limit=int(profile.get("sql_chat_top_k", settings.RECALL_SQL_TOP_K)),
                since_minutes=int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                exclude_text=exclusion.get("active_turn_text"),
                exclude_ids=exclusion.get("active_turn_ids"),
            )
            backend_counts["sql_chat_msgs"] = len(chat_msgs)
            for item in chat_msgs:
                candidates.append(
                    {
                        "id": item.id,
                        "source": "sql_chat",
                        "source_ref": item.source_ref,
                        "text": item.text,
                        "ts": item.ts,
                        "tags": ["sql", "chat", "messages"],
                        "score": 0.75,
                    }
                )
        except Exception as exc:
            logger.debug(f"sql chat backend skipped: {exc}")
    elif diagnostic:
        logger.info(
            "recall sql_chat skipped profile=%s enable_sql_chat=%s global_sql_chat_enabled=%s",
            profile.get("profile"),
            profile.get("enable_sql_chat"),
            settings.RECALL_ENABLE_SQL_CHAT,
        )

    if _sql_timeline_enabled_for_profile(profile):
        try:

            since_minutes_effective = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))
            since_hours_effective = int(profile.get("sql_since_hours", max(1, since_minutes_effective // 60)))
            sql_top_k = int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K))

            recent_items = await fetch_recent_fragments(
                session_id,
                node_id,
                since_minutes_effective,
                sql_top_k,
                exclude_ids=exclusion.get("active_turn_ids"),
                exclude_text=exclusion.get("active_turn_text"),
            )
            related_items = await fetch_related_by_entities(
                entities,
                since_hours_effective,
                sql_top_k,
                session_id=session_id,
                exclude_ids=exclusion.get("active_turn_ids"),
                exclude_text=exclusion.get("active_turn_text"),
            )

            recent_items = list(recent_items) + list(related_items)
            backend_counts["sql_timeline"] = len(recent_items)
            for item in recent_items:
                candidates.append(
                    {
                        "id": item.id,
                        "source": "sql_timeline",
                        "source_ref": item.source_ref,
                        "text": item.text,
                        "ts": item.ts,
                        "session_id": item.session_id,
                        "tags": item.tags,
                        "turn_effect_delta": item.turn_effect_delta,
                        "score": 0.7,
                    }
                )
        except Exception as exc:
            logger.debug(f"sql timeline backend skipped: {exc}")
    elif diagnostic:
        logger.info(
            "recall sql_timeline skipped profile=%s enable_sql_timeline=%s global_sql_timeline_enabled=%s",
            profile.get("profile"),
            profile.get("enable_sql_timeline"),
            settings.RECALL_ENABLE_SQL_TIMELINE,
        )

    if include_cards and _cards_fetch_enabled(profile):
        pool = _recall_pg_pool
        if pool is not None and asyncpg is not None:
            try:
                card_frags = await fetch_card_fragments_guarded(
                    pool,
                    fragment,
                    profile,
                    lane=lane,
                    timeout_sec=float(getattr(settings, "RECALL_CARDS_TIMEOUT_SEC", 0.25) or 0.25),
                    max_neighbors=int(getattr(settings, "RECALL_CARDS_MAX_NEIGHBORS", 6) or 6),
                )
                backend_counts["cards"] = len(card_frags)
                candidates.extend(card_frags)
            except Exception as exc:
                logger.warning("cards fetch skipped: %s", exc)
        elif diagnostic:
            logger.info("recall cards skipped pool_asyncpg_available=%s", pool is not None)

    # ── Graph Compression backend ─────────────────────────────────────────────
    compression_enabled = (
        bool(profile.get("enable_graph_compression"))
        and bool(getattr(settings, "RECALL_COMPRESSION_ENABLED", False))
        and bool(getattr(settings, "RECALL_COMPRESSION_PG_DSN", None))
        and fetch_graph_compression_fragments is not None
    )
    if compression_enabled:
        try:
            # Run the blocking Postgres + Fuseki I/O off the event loop so it does
            # not stall the recall hot path (mirrors the memory_graph_sparql path).
            compression_frags = await asyncio.to_thread(
                fetch_graph_compression_fragments,
                query_text=fragment,
                mode=str(profile.get("compression_mode") or "unified"),
                max_global=int(profile.get("compression_global_top_k") or 5),
                max_local=int(profile.get("compression_local_top_k") or 5),
                scopes=list(profile.get("compression_scopes") or ["episodic", "substrate", "self_study"]),
                pg_dsn=settings.RECALL_COMPRESSION_PG_DSN,
                rdf_query_url=getattr(settings, "RECALL_COMPRESSION_RDF_QUERY_URL", None),
                rdf_user=getattr(settings, "RECALL_COMPRESSION_RDF_USER", "admin"),
                rdf_pass=getattr(settings, "RECALL_COMPRESSION_RDF_PASS", "orion"),
                timeout_sec=float(getattr(settings, "RECALL_COMPRESSION_TIMEOUT_SEC", 3.0)),
            )
            backend_counts["graph_compression"] = len(compression_frags)
            candidates.extend(compression_frags)
        except Exception as exc:
            logger.debug("graph_compression_backend_skipped reason=%s", exc)
            backend_counts["graph_compression"] = 0
    else:
        backend_counts["graph_compression"] = 0

    return candidates, backend_counts


def _persist_decision(decision: RecallDecisionV1) -> None:
    """
    Durable log to Postgres if available. Best-effort.
    """
    dsn = settings.RECALL_PG_DSN
    if not dsn:
        return
    if psycopg2 is None:
        return
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
    except Exception as exc:
        logger.debug(f"recall telemetry pg connect failed: {exc}")
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recall_telemetry (
                    id uuid primary key,
                    corr_id text,
                    session_id text,
                    node_id text,
                    verb text,
                    profile text,
                    query text,
                    selected_ids jsonb,
                    backend_counts jsonb,
                    latency_ms integer,
                    created_at timestamptz default now()
                )
                """
            )
            cur.execute(
                """
                INSERT INTO recall_telemetry
                (id, corr_id, session_id, node_id, verb, profile, query, selected_ids, backend_counts, latency_ms)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    decision.id,
                    decision.corr_id,
                    decision.session_id,
                    decision.node_id,
                    decision.verb,
                    decision.profile,
                    decision.query,
                    decision.selected_ids,
                    decision.backend_counts,
                    decision.latency_ms,
                ),
            )
    except Exception as exc:
        logger.debug(f"recall telemetry persist failed: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _log_debug_dump(
    *,
    corr_id: str,
    profile: Dict[str, Any],
    backend_counts: Dict[str, int],
    items: List[Any],
) -> None:
    top_n = int(getattr(settings, "RECALL_DEBUG_DUMP_TOP_N", 0) or 0)
    if top_n <= 0:
        return
    logger.info(
        "REC_TAPE RECALL corr_id=%s profile=%s backend_counts=%s selected_count=%s",
        corr_id,
        profile.get("profile"),
        backend_counts,
        len(items),
    )
    for idx, item in enumerate(items[:top_n]):
        source = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else None)
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        score = (
            getattr(item, "score", None)
            if hasattr(item, "score")
            else (item.get("score") if isinstance(item, dict) else None)
        )
        source_ref = getattr(item, "source_ref", None) or (item.get("source_ref") if isinstance(item, dict) else None)
        snippet = getattr(item, "snippet", None) or (item.get("text") if isinstance(item, dict) else None)
        snippet_head = str(snippet or "")[:160].replace("\n", " ")
        logger.info(
            "REC_TAPE RECALL item idx=%s source=%s id=%s score=%s source_ref=%s snippet_head=%r",
            idx,
            source,
            item_id,
            score,
            source_ref,
            snippet_head,
        )


def _bounded_selected_summary(items: List[Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in items[: max(1, limit)]:
        source = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else None)
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        score = getattr(item, "score", None) if hasattr(item, "score") else (item.get("score") if isinstance(item, dict) else None)
        source_ref = getattr(item, "source_ref", None) or (item.get("source_ref") if isinstance(item, dict) else None)
        summary.append(
            {
                "id": str(item_id or ""),
                "source": str(source or "unknown"),
                "score": float(score or 0.0),
                "source_ref": str(source_ref or "")[:120] or None,
            }
        )
    return summary


async def process_recall(
    q: RecallQueryV1,
    *,
    corr_id: str,
    diagnostic: bool = False,
) -> Tuple[MemoryBundleV1, RecallDecisionV1]:
    recall_phase = getattr(q, "recall_phase", None)
    selected_profile = q.profile
    intent_payload: Dict[str, Any] | None = None
    if bool(getattr(settings, "RECALL_INTENT_ROUTING_ENABLED", True)):
        try:
            from .intent import classify_intent_v1, intent_telemetry_payload, resolve_profile_for_intent

            profile_explicit = bool(getattr(q, "profile_explicit", False)) or recall_phase in {
                "continuity",
                "purposeful",
            }
            ic = classify_intent_v1(str(q.fragment or ""))
            if profile_explicit:
                selected_profile = q.profile
            else:
                selected_profile = resolve_profile_for_intent(ic.intent, fallback_profile=q.profile)
            intent_payload = intent_telemetry_payload(
                query_text=str(q.fragment or ""),
                intent=ic.intent,
                profile=selected_profile,
                override=profile_explicit,
            )
        except Exception as exc:
            logger.debug("intent routing skipped: %s", exc)
            selected_profile = q.profile
            intent_payload = None

    profile = get_profile(selected_profile)
    profile_name = str(profile.get("profile") or "")
    retrieval_intent = getattr(q, "retrieval_intent", None) or "semantic"
    pcr_backend_plan: dict[str, bool] = {}
    if settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        pcr_backend_plan = collectors_for_intent(retrieval_intent)
        profile = apply_collector_plan(profile, pcr_backend_plan)
        profile_name = str(profile.get("profile") or profile_name)
    query_targeting = _derive_chat_general_query(q.fragment, verb=q.verb, profile_name=profile_name)
    query_fragment = str(query_targeting.get("query_fragment") or q.fragment or "")
    if diagnostic and query_targeting.get("query_changed"):
        logger.info(
            "recall query_targeting adjusted profile=%s verb=%s raw=%r targeted=%r turn_type=%s tail_stripped=%s",
            profile_name,
            q.verb,
            (q.fragment or "")[:220],
            query_fragment[:220],
            query_targeting.get("turn_type"),
            query_targeting.get("tail_stripped"),
        )
    enable_qe = bool(profile.get("enable_query_expansion", True))
    signals = _expand_query(query_fragment, verb=q.verb, intent=q.intent, enable=enable_qe)
    ignored_session_id = q.session_id
    effective_session_id: str | None = None
    exclusion = _parse_exclusion(q)
    source_gating: Dict[str, str] = {}
    vector_policy = build_vector_policy(profile, settings)
    source_gating["vector"] = "removed_from_orion_recall"
    source_gating["sql_timeline"] = "enabled" if _sql_timeline_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["sql_chat"] = "enabled" if _sql_chat_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["rdf"] = "enabled" if _rdf_enabled(profile) else "disabled_by_profile_or_global"

    t0 = time.time()
    timing_breakdown_ms: Dict[str, int] = {}
    candidates: List[Dict[str, Any]] = []
    backend_counts_total: Dict[str, int] = {}

    if _is_memory_browse(query_fragment):
        since_minutes_effective = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))
        browse_limit = max(10, min(20, int(profile.get("max_total_items", 12))))
        if _sql_timeline_enabled_for_profile(profile):
            source_gating["sql_timeline"] = "enabled"
            try:
                recent_items = await fetch_recent_fragments(
                    effective_session_id,
                    q.node_id,
                    since_minutes_effective,
                    browse_limit,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )
            except Exception as exc:
                logger.debug(f"browse timeline fetch skipped: {exc}")
                recent_items = []
        else:
            source_gating["sql_timeline"] = "disabled_by_profile_or_global"
            logger.info(
                "browse sql_timeline skipped profile=%s enable_sql_timeline=%s global_sql_timeline_enabled=%s",
                profile.get("profile"),
                profile.get("enable_sql_timeline"),
                settings.RECALL_ENABLE_SQL_TIMELINE,
            )
            recent_items = []
        backend_counts_total["sql_timeline"] = len(recent_items)
        for item in recent_items:
            candidates.append(
                {
                    "id": item.id,
                    "source": "sql_timeline",
                    "source_ref": item.source_ref,
                    "text": item.text,
                    "ts": item.ts,
                    "tags": list(item.tags or []) + ["memory_browse"],
                    "score": 0.6,
                }
            )
        latency_ms = int((time.time() - t0) * 1000)
        bundle, ranking_debug = fuse_candidates(
            candidates=candidates,
            profile=profile,
            latency_ms=latency_ms,
            query_text=None,
            session_id=None,
            diagnostic=diagnostic,
            browse_mode=True,
        )
        _log_debug_dump(
            corr_id=corr_id,
            profile=profile,
            backend_counts=backend_counts_total or bundle.stats.backend_counts,
            items=list(bundle.items),
        )
        decision = RecallDecisionV1(
            corr_id=corr_id or str(uuid4()),
            session_id=ignored_session_id,
            node_id=q.node_id,
            verb=q.verb,
            profile=str(profile.get("profile") or q.profile),
            query=q.fragment,
            selected_ids=[i.id for i in bundle.items],
            backend_counts=backend_counts_total or bundle.stats.backend_counts,
            latency_ms=latency_ms,
            dropped=dict((bundle.stats.diagnostic or {}).get("drop_counts") or {}),
            ranking_debug=ranking_debug if diagnostic else [],
            recall_debug=(
                {
                    "profile_selected": str(profile.get("profile") or q.profile),
                    "profile_requested": q.profile,
                    **({"recall_intent": intent_payload} if intent_payload else {}),
                    "query_expansion_enabled": enable_qe,
                    "query_targeting": {
                        **query_targeting,
                        "raw_fragment": q.fragment,
                    },
                    "source_gating": source_gating,
                    "vector_policy": vector_policy,
                    "active_turn": {
                        "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                        "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                        "ts_present": exclusion.get("active_turn_ts") is not None,
                        "self_hit_suppressed": 0,
                    },
                    "fusion": bundle.stats.diagnostic or {},
                    "latency_breakdown_ms": {"total": latency_ms},
                    "selected_summary": _bounded_selected_summary(list(bundle.items)),
                }
                if diagnostic
                else {}
            ),
        )
        pressure_events = _build_recall_pressure_events(q=q, decision=decision, bundle=bundle)
        if pressure_events:
            merged_debug = dict(decision.recall_debug or {})
            merged_debug["pressure_events"] = pressure_events
            decision = decision.model_copy(update={"recall_debug": merged_debug})
        _log_debug_dump(
            corr_id=decision.corr_id,
            profile=profile,
            backend_counts=decision.backend_counts or {},
            items=list(bundle.items),
        )
        return bundle, decision

    fetch_started = time.time()
    anchor_candidates: List[Dict[str, Any]] = []
    anchor_counts: Dict[str, int] = {}
    if bool(profile.get("enable_anchor_candidates", True)):
        anchor_candidates, anchor_counts = await _fetch_anchor_candidates(
            query_text=query_fragment,
            session_id=effective_session_id,
            node_id=q.node_id,
            profile=profile,
            diagnostic=diagnostic,
            exclusion=exclusion,
        )
    elif diagnostic:
        logger.info(
            "recall anchor rail skipped profile=%s enable_anchor_candidates=%s",
            profile.get("profile"),
            profile.get("enable_anchor_candidates"),
        )
    timing_breakdown_ms["anchor_fetch"] = int((time.time() - fetch_started) * 1000)
    if anchor_candidates:
        candidates.extend(anchor_candidates)
        for key, value in anchor_counts.items():
            backend_counts_total[key] = value

    if profile_name.startswith("graphtri"):
        anchor_set = _build_anchor_set(
            query_text=q.fragment,
            session_id=effective_session_id,
            profile=profile,
        )
        anchor_terms = anchor_set.get("related_terms") or []
        vector_queries = [query_fragment]
        vector_queries.extend([f"{query_fragment} {term}" for term in anchor_terms[:8]])
        sql_filters = anchor_terms[:6]
        query_plan = {
            "vector_queries": vector_queries,
            "sql_filters": sql_filters,
            "sql_session_id": effective_session_id,
        }

        rdf_top_k = int(profile.get("rdf_top_k", 0))
        rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
        rdf_items: List[Dict[str, Any]] = []
        rdf_connected: List[Dict[str, Any]] = []
        if rdf_enabled:
            try:
                rdf_items = fetch_rdf_graphtri_fragments(
                    query_text=query_fragment,
                    session_id=effective_session_id,
                    max_items=rdf_top_k,
                )
                if not rdf_items:
                    rdf_items = fetch_rdf_fragments(query_text=query_fragment, max_items=rdf_top_k)
                backend_counts_total["rdf"] = len(rdf_items)
                candidates.extend(rdf_items)

                if anchor_terms:
                    rdf_connected = fetch_rdf_connected_chatturns(
                        terms=anchor_terms[:10],
                        max_items=max(6, min(12, rdf_top_k or 12)),
                    )
                    backend_counts_total["rdf_chat_connected"] = len(rdf_connected)
                    candidates.extend(rdf_connected)
            except Exception as exc:
                logger.debug(f"rdf backend skipped: {exc}")
        source_gating["vector"] = "removed_from_orion_recall"
        backend_counts_total["vector"] = 0

        sql_attempted = False
        sql_session_id = effective_session_id
        sql_filters_used: List[str] = []
        if _sql_timeline_enabled_for_profile(profile):
            source_gating["sql_timeline"] = "enabled"
            try:
                sql_attempted = True
                sql_filters_used = sql_filters

                since_minutes_effective = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))
                since_hours_effective = int(profile.get("sql_since_hours", max(1, since_minutes_effective // 60)))
                sql_top_k = int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K))

                recent_items = await fetch_recent_fragments(
                    effective_session_id,
                    q.node_id,
                    since_minutes_effective,
                    sql_top_k,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )
                related_items = await fetch_related_by_entities(
                    sql_filters_used or [],
                    since_hours_effective,
                    sql_top_k,
                    session_id=effective_session_id,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )

                recent_items = list(recent_items) + list(related_items)
                backend_counts_total["sql_timeline"] = len(recent_items)
                for item in recent_items:
                    candidates.append(
                        {
                            "id": item.id,
                            "source": "sql_timeline",
                            "source_ref": item.source_ref,
                            "text": item.text,
                            "ts": item.ts,
                            "session_id": item.session_id,
                            "tags": item.tags,
                            "score": 0.7,
                        }
                    )
            except Exception as exc:
                logger.debug(f"sql timeline backend skipped: {exc}")
        elif diagnostic:
            source_gating["sql_timeline"] = "disabled_by_profile_or_global"
            logger.info(
                "graphtri sql_timeline skipped profile=%s enable_sql_timeline=%s global_sql_timeline_enabled=%s",
                profile.get("profile"),
                profile.get("enable_sql_timeline"),
                settings.RECALL_ENABLE_SQL_TIMELINE,
            )

        if diagnostic:
            logger.info(
                "graphtri anchors=%s plan=%s rdf_count=%s vector_count=%s sql_count=%s",
                anchor_set,
                query_plan,
                backend_counts_total.get("rdf", 0),
                backend_counts_total.get("vector", 0),
                backend_counts_total.get("sql_timeline", 0),
            )
            logger.info(
                "graphtri sql attempted=%s session_id=%s filters=%s count=%s",
                sql_attempted,
                sql_session_id,
                sql_filters_used,
                backend_counts_total.get("sql_timeline", 0),
            )
    else:
        backend_start = time.time()
        for sig_i, sig in enumerate(signals):
            cand, counts = await _query_backends(
                sig,
                profile,
                session_id=effective_session_id,
                node_id=q.node_id,
                entities=_extract_entities(query_fragment),
                diagnostic=diagnostic,
                exclusion=exclusion,
                lane=getattr(q, "lane", None),
                include_cards=(sig_i == 0),
            )
            candidates.extend(cand)
            for k, v in counts.items():
                backend_counts_total[k] = backend_counts_total.get(k, 0) + v
        timing_breakdown_ms["backend_fetch"] = int((time.time() - backend_start) * 1000)

    if getattr(settings, "RECALL_MEMORY_GRAPH_SPARQL_ENABLED", False):
        try:
            from .memory_graph_sparql import fetch_memory_graph_sparql_candidates

            mg_extra = await asyncio.to_thread(fetch_memory_graph_sparql_candidates, query_fragment, settings)
            candidates.extend(mg_extra)
            backend_counts_total["memory_graph_sparql"] = len(mg_extra)
        except Exception as exc:
            logger.debug("memory_graph_sparql augment skipped: %s", exc)

    if settings.RECALL_RDF_CHAT_WINDOW_ENABLED:
        rdf_chat_window_min = int(
            profile.get("rdf_chat_since_minutes")
            or profile.get("sql_since_minutes")
            or settings.RECALL_SQL_SINCE_MINUTES
        )
        candidates, rdf_chat_dropped = await _window_rdf_chatturn_candidates(
            candidates, since_minutes=rdf_chat_window_min
        )
        if rdf_chat_dropped:
            backend_counts_total["rdf_chat_out_of_window_dropped"] = rdf_chat_dropped
            logger.info(
                "rdf chat-turn windowing profile=%s window_min=%s dropped=%s",
                profile.get("profile"),
                rdf_chat_window_min,
                rdf_chat_dropped,
            )

    sql_chat_window_min = int(
        profile.get("sql_chat_since_minutes")
        or profile.get("sql_since_minutes")
        or settings.RECALL_SQL_SINCE_MINUTES
    )
    candidates, sql_chat_dropped = await _window_sql_chat_candidates(
        candidates, since_minutes=sql_chat_window_min
    )
    if sql_chat_dropped:
        backend_counts_total["sql_chat_out_of_window_dropped"] = sql_chat_dropped
        logger.info(
            "sql chat windowing profile=%s window_min=%s dropped=%s",
            profile.get("profile"),
            sql_chat_window_min,
            sql_chat_dropped,
        )

    suppression_start = time.time()
    candidates, suppressed = _suppress_self_hits(
        candidates,
        active_turn_text=str(exclusion.get("active_turn_text") or ""),
        active_turn_ids=list(exclusion.get("active_turn_ids") or []),
        active_turn_ts=exclusion.get("active_turn_ts"),
    )
    timing_breakdown_ms["self_hit_suppression"] = int((time.time() - suppression_start) * 1000)
    if suppressed:
        logger.info(
            "recall self-hit suppression active_turn_ids=%s suppressed=%s",
            exclusion.get("active_turn_ids"),
            suppressed,
        )

    if settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        if pcr_backend_plan.get("active_packet") and settings.RECALL_ACTIVE_PACKET_ENABLED:
            try:
                ap_frags = await fetch_active_packet_fragments(q, pool=_recall_pg_pool, settings=settings)
                candidates.extend(ap_frags)
                backend_counts_total["active_packet"] = len(ap_frags)
            except Exception as exc:
                logger.debug("active_packet collector skipped: %s", exc)

    latency_ms = int((time.time() - t0) * 1000)
    fuse_started = time.time()
    if settings.RECALL_PCR_ENABLED and recall_phase == "continuity":
        profile["sql_since_minutes"] = settings.RECALL_CONTINUITY_SQL_MINUTES
        profile["render_budget_tokens"] = settings.RECALL_CONTINUITY_RENDER_BUDGET
        bundle, ranking_debug = render_continuity_bundle(
            candidates=candidates,
            profile=profile,
            query_text=query_fragment,
            latency_ms=latency_ms,
            session_id=effective_session_id,
        )
    elif settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        belief_budget = q.belief_digest_max_tokens or settings.RECALL_BELIEF_RENDER_BUDGET
        profile["render_budget_tokens"] = belief_budget
        bundle, ranking_debug = pcr_fuse_belief_candidates(
            candidates=candidates,
            profile=profile,
            retrieval_intent=str(retrieval_intent),
            query_text=query_fragment,
            latency_ms=latency_ms,
        )
    else:
        bundle, ranking_debug = fuse_candidates(
            candidates=candidates,
            profile=profile,
            latency_ms=latency_ms,
            query_text=query_fragment,
            session_id=effective_session_id,
            diagnostic=diagnostic,
            substantive_query=str(query_targeting.get("turn_type")) == "substantive",
        )
    timing_breakdown_ms["fusion"] = int((time.time() - fuse_started) * 1000)
    timing_breakdown_ms["total"] = latency_ms
    pcr_debug: Dict[str, Any] | None = None
    if settings.RECALL_PCR_ENABLED and recall_phase in {"continuity", "purposeful"}:
        pcr_debug = {
            "enabled": settings.RECALL_PCR_ENABLED,
            "phase": recall_phase,
            "retrieval_intent": retrieval_intent,
            "intent_rule_id": (q.task_hints or {}).get("rule_id") if isinstance(q.task_hints, dict) else None,
            "backend_plan": list(pcr_backend_plan.keys()) if pcr_backend_plan else [],
        }
    decision = RecallDecisionV1(
        corr_id=corr_id or str(uuid4()),
        session_id=ignored_session_id,
        node_id=q.node_id,
        verb=q.verb,
        profile=str(profile.get("profile") or q.profile),
        query=query_fragment,
        selected_ids=[i.id for i in bundle.items],
        backend_counts=backend_counts_total or bundle.stats.backend_counts,
        latency_ms=latency_ms,
        dropped=dict((bundle.stats.diagnostic or {}).get("drop_counts") or {}),
        ranking_debug=ranking_debug if diagnostic else [],
        recall_debug=(
            {
                "latency_breakdown_ms": timing_breakdown_ms,
                "profile_selected": str(profile.get("profile") or q.profile),
                "profile_requested": q.profile,
                **({"recall_intent": intent_payload} if intent_payload else {}),
                "query_expansion_enabled": enable_qe,
                "query_targeting": {
                    **query_targeting,
                    "raw_fragment": q.fragment,
                },
                "source_gating": source_gating,
                "vector_policy": vector_policy,
                "active_turn": {
                    "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                    "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                    "ts_present": exclusion.get("active_turn_ts") is not None,
                    "self_hit_suppressed": suppressed,
                },
                "fusion": bundle.stats.diagnostic or {},
                "selected_summary": _bounded_selected_summary(list(bundle.items)),
                **({"pcr": pcr_debug} if pcr_debug else {}),
            }
            if diagnostic
            else (
                {"latency_breakdown_ms": timing_breakdown_ms, **({"pcr": pcr_debug} if pcr_debug else {})}
            )
        ),
    )
    compare_summary: Dict[str, Any] = {}
    anchor_plan_summary: Dict[str, Any] = {}
    selected_cards: list[Dict[str, Any]] = []
    should_shadow_compare = bool(
        not bundle.items
        or any(str(item.source or "") == "vector" for item in bundle.items[:2])
        or bool(_anchor_tokens(q.fragment, max_tokens=6))
    )
    if should_shadow_compare:
        try:
            from .recall_v2 import run_recall_v2_shadow

            shadow_bundle, shadow_debug = await run_recall_v2_shadow(q, profile=profile)
            compare_summary = {
                "v1_latency_ms": decision.latency_ms,
                "v2_latency_ms": int(shadow_debug.get("latency_ms") or 0),
                "selected_count_delta": len(shadow_bundle.items) - len(bundle.items),
                "v1_selected_count": len(bundle.items),
                "v2_selected_count": len(shadow_bundle.items),
            }
            anchor_plan_summary = dict(shadow_debug.get("plan") or {})
            selected_cards = list(shadow_debug.get("ranked_cards") or [])[:6]
        except Exception as exc:
            logger.debug(f"recall shadow compare skipped: {exc}")

    pressure_events = _build_recall_pressure_events(
        q=q,
        decision=decision,
        bundle=bundle,
        compare_summary=compare_summary,
        anchor_plan=anchor_plan_summary,
        selected_evidence_cards=selected_cards,
    )
    if pressure_events:
        merged_debug = dict(decision.recall_debug or {})
        merged_debug["pressure_events"] = pressure_events
        if compare_summary:
            merged_debug["compare_summary"] = compare_summary
        if anchor_plan_summary:
            merged_debug["anchor_plan_summary"] = anchor_plan_summary
        if selected_cards:
            merged_debug["selected_evidence_cards"] = selected_cards
        decision = decision.model_copy(update={"recall_debug": merged_debug})
    if diagnostic:
        logger.info(
            "recall_diagnostic_summary corr_id=%s profile=%s requested_profile=%s gating=%s drop_counts=%s selected_counts=%s suppressed=%s latency_breakdown_ms=%s selected=%s",
            decision.corr_id,
            profile.get("profile"),
            q.profile,
            source_gating,
            decision.dropped,
            (bundle.stats.diagnostic or {}).get("source_selected_counts", {}),
            suppressed,
            timing_breakdown_ms,
            decision.selected_ids[:8],
        )
    _log_debug_dump(
        corr_id=decision.corr_id,
        profile=profile,
        backend_counts=decision.backend_counts or {},
        items=list(bundle.items),
    )
    return bundle, decision


def build_reply_envelope(bundle: MemoryBundleV1, env: BaseEnvelope, *, debug: Dict[str, Any] | None = None) -> BaseEnvelope:
    payload: Dict[str, Any] = {"bundle": bundle.model_dump(mode="json")}
    if debug:
        payload["debug"] = debug
    return BaseEnvelope(
        kind=RECALL_REPLY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=payload,
        reply_to=None,
    )


def telemetry_envelope(decision: RecallDecisionV1, env: BaseEnvelope) -> BaseEnvelope:
    return BaseEnvelope(
        kind=RECALL_TELEMETRY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=decision.model_dump(mode="json"),
    )


async def handle_recall(env: BaseEnvelope, *, bus) -> BaseEnvelope:
    if env.kind not in {RECALL_REQUEST_KIND, "recall.query.request"}:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    raw_payload: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    diagnostic = bool((raw_payload.get("options") or {}).get("diagnostic"))
    payload_obj = dict(raw_payload)
    payload_obj.pop("options", None)
    try:
        q = RecallQueryV1.model_validate(payload_obj)
    except ValidationError as ve:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    corr = str(env.correlation_id)
    logger.info(
        "recall_bus_request_begin corr_id=%s verb=%s profile=%s session_id=%s node_id=%s",
        corr,
        q.verb,
        q.profile,
        q.session_id,
        q.node_id,
    )
    wall_t0 = time.perf_counter()
    bundle, decision = await process_recall(q, corr_id=corr, diagnostic=diagnostic)
    wall_ms = int((time.perf_counter() - wall_t0) * 1000)
    logger.info(
        "recall_bus_request_complete corr_id=%s wall_ms=%s process_latency_ms=%s profile=%s verb=%s backend_counts=%s",
        corr,
        wall_ms,
        decision.latency_ms,
        decision.profile,
        decision.verb,
        decision.backend_counts,
    )
    if wall_ms >= 30000:
        logger.warning(
            "recall_bus_request_slow corr_id=%s wall_ms=%s latency_breakdown_ms=%s",
            corr,
            wall_ms,
            (decision.recall_debug or {}).get("latency_breakdown_ms"),
        )

    # emit telemetry (fire and forget)
    try:
        await bus.publish(settings.RECALL_BUS_TELEMETRY, telemetry_envelope(decision, env))
    except Exception as exc:
        logger.debug(f"telemetry publish failed: {exc}")

    _persist_decision(decision)

    debug_payload: Dict[str, Any] | None = None
    if diagnostic:
        debug_payload = {
            "decision": {
                "corr_id": decision.corr_id,
                "profile": decision.profile,
                "selected_ids": decision.selected_ids[:8],
                "backend_counts": decision.backend_counts,
                "dropped": decision.dropped,
                "recall_debug": decision.recall_debug,
            }
        }
    return build_reply_envelope(bundle, env, debug=debug_payload)
