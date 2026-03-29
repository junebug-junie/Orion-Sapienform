from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional dependency in test env
    psycopg2 = None
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import (
    MemoryBundleV1,
    RecallDecisionV1,
    RecallQueryV1,
)

try:
    from .fusion import fuse_candidates
    from .profiles import get_profile
    from .settings import settings
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
    from .storage.vector_adapter import fetch_vector_fragments, fetch_vector_exact_matches
    from .sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments
    from .sql_chat import fetch_chat_history_pairs, fetch_chat_messages

except ImportError as _e:  # pragma: no cover - fallback for runtime pathing
    _IMPORT_ERROR = _e
    try:
        # Container/package-safe absolute imports
        from app.fusion import fuse_candidates  # type: ignore
        from app.profiles import get_profile  # type: ignore
        from app.settings import settings  # type: ignore
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
        from app.storage.vector_adapter import fetch_vector_fragments, fetch_vector_exact_matches  # type: ignore
        from app.sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments  # type: ignore
        from app.sql_chat import fetch_chat_history_pairs, fetch_chat_messages  # type: ignore
    except ImportError:
        # IMPORTANT: raise the real root cause, not the fallback failure
        raise _IMPORT_ERROR

logger = logging.getLogger("orion-recall.worker")

RECALL_REQUEST_KIND = "recall.query.v1"
RECALL_REPLY_KIND = "recall.reply.v1"
RECALL_TELEMETRY_KIND = "recall.decision.v1"


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
    return (
        profile_name.startswith("deep.graph")
        or profile_name.startswith("graphtri")
        or settings.RECALL_ENABLE_RDF
    ) and int(profile.get("rdf_top_k", 0)) > 0


def _vector_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_VECTOR:
        return False
    try:
        return int(profile.get("vector_top_k", settings.RECALL_DEFAULT_MAX_ITEMS)) > 0
    except Exception:
        return False


def _sql_timeline_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_SQL_TIMELINE:
        return False
    return bool(profile.get("enable_sql_timeline", True))


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

    try:
        sql_items = await fetch_exact_fragments(
            tokens=tokens,
            session_id=session_id,
            node_id=node_id,
            limit=limit,
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

    if _vector_enabled_for_profile(profile):
        try:
            vec = fetch_vector_exact_matches(
                tokens=tokens,
                max_items=limit,
                session_id=session_id,
                profile_name=profile.get("profile"),
                node_id=node_id,
                exclude_ids=exclusion.get("active_turn_ids"),
                exclude_text=exclusion.get("active_turn_text"),
            )
            counts["vector_anchor"] = len(vec)
            for item in vec:
                item = dict(item)
                item["tags"] = list(item.get("tags") or []) + ["anchor_exact"]
                item["score"] = max(0.95, float(item.get("score") or 0.0))
                candidates.append(item)
        except Exception as exc:
            logger.debug(f"vector anchor fetch skipped: {exc}")

    elif diagnostic:
        logger.info(
            "anchor rail vector skipped profile=%s vector_top_k=%s global_vector_enabled=%s",
            profile.get("profile"),
            profile.get("vector_top_k"),
            settings.RECALL_ENABLE_VECTOR,
        )

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


async def _query_backends(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
    diagnostic: bool = False,
    exclusion: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    exclusion = exclusion or {}

    rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
    rdf_top_k = int(profile.get("rdf_top_k", 0))
    expansion_terms: List[str] = []
    if rdf_enabled:
        try:

            profile_name = str(profile.get("profile") or "")

            # 0) Pull raw ChatTurns (prompt/response) from GRAPH <orion:chat>.
            # This is the only place your "exact text I used" lives.
            rdf_chat: List[Dict[str, Any]] = []
            rdf_chat = fetch_rdf_chatturn_fragments(
                query_text=fragment,
                session_id=session_id,
                max_items=max(rdf_top_k, 6),
            )
            backend_counts["rdf_chat"] = len(rdf_chat)
            candidates.extend(rdf_chat)

            # 1) Keep existing RDF paths (claims / neighborhood) as additional context.
            rdf: List[Dict[str, Any]] = []
            if profile_name.startswith("graphtri"):
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


    if _vector_enabled_for_profile(profile):
        seeds = [fragment, *entities]
        expansions = expansion_terms[:6]
        vector_queries: List[str] = []
        seen = set()
        for term in [*seeds, *expansions]:
            cleaned = (term or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            vector_queries.append(cleaned)
            if len(vector_queries) >= 4:
                break
        if not vector_queries:
            vector_queries = [fragment]
        per_query = max(1, int(profile.get("vector_top_k", settings.RECALL_DEFAULT_MAX_ITEMS)) // len(vector_queries))
        vec_count = 0
        raw_vector_filters = profile.get("vector_meta_filters")
        vector_filters = raw_vector_filters if isinstance(raw_vector_filters, dict) else None
        for term in vector_queries:
            try:
                vec = fetch_vector_fragments(
                    query_text=term,
                    time_window_days=settings.RECALL_DEFAULT_TIME_WINDOW_DAYS,
                    max_items=per_query,
                    session_id=session_id,
                    profile_name=profile.get("profile"),
                    node_id=node_id,
                    metadata_filters=vector_filters,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )
                vec_count += len(vec)
                candidates.extend(vec)
            except Exception as exc:
                logger.debug(f"vector backend skipped: {exc}")
        backend_counts["vector"] = vec_count
        if diagnostic:
            logger.info(
                "recall expansions profile=%s session_id=%s seeds=%s expansions=%s vector_queries=%s vector_count=%s",
                profile.get("profile"),
                session_id,
                seeds,
                expansions,
                vector_queries,
                vec_count,
            )

    elif diagnostic:
        logger.info(
            "recall vector skipped profile=%s vector_top_k=%s global_vector_enabled=%s",
            profile.get("profile"),
            profile.get("vector_top_k"),
            settings.RECALL_ENABLE_VECTOR,
        )

    if diagnostic:
        logger.info(
            "recall rdf_enabled=%s rdf_top_k=%s rdf_candidates=%s expansion_terms=%s",
            rdf_enabled,
            rdf_top_k,
            backend_counts.get("rdf", 0),
            expansion_terms,
        )

    if settings.RECALL_ENABLE_SQL_CHAT:
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
    profile = get_profile(q.profile)
    profile_name = str(profile.get("profile") or "")
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
    source_gating["vector"] = "enabled" if _vector_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["sql_timeline"] = "enabled" if _sql_timeline_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["rdf"] = "enabled" if _rdf_enabled(profile) else "disabled_by_profile_or_global"

    t0 = time.time()
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
                    "query_expansion_enabled": enable_qe,
                    "query_targeting": {
                        **query_targeting,
                        "raw_fragment": q.fragment,
                    },
                    "source_gating": source_gating,
                    "active_turn": {
                        "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                        "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                        "ts_present": exclusion.get("active_turn_ts") is not None,
                        "self_hit_suppressed": 0,
                    },
                    "fusion": bundle.stats.diagnostic or {},
                    "selected_summary": _bounded_selected_summary(list(bundle.items)),
                }
                if diagnostic
                else {}
            ),
        )
        _log_debug_dump(
            corr_id=decision.corr_id,
            profile=profile,
            backend_counts=decision.backend_counts or {},
            items=list(bundle.items),
        )
        return bundle, decision

    anchor_candidates, anchor_counts = await _fetch_anchor_candidates(
        query_text=query_fragment,
        session_id=effective_session_id,
        node_id=q.node_id,
        profile=profile,
        diagnostic=diagnostic,
        exclusion=exclusion,
    )
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
        if _vector_enabled_for_profile(profile):
            source_gating["vector"] = "enabled"
            vector_top_k = int(profile.get("vector_top_k", settings.RECALL_DEFAULT_MAX_ITEMS))
            per_query = max(1, vector_top_k // max(1, len(vector_queries)))
            seen_vec = set()
            vec_count = 0
            vector_candidates: List[Dict[str, Any]] = []
            raw_vector_filters = profile.get("vector_meta_filters")
            vector_filters = raw_vector_filters if isinstance(raw_vector_filters, dict) else None
            for term in vector_queries:
                try:
                    vec = fetch_vector_fragments(
                        query_text=term,
                        time_window_days=settings.RECALL_DEFAULT_TIME_WINDOW_DAYS,
                        max_items=per_query,
                        session_id=effective_session_id,
                        profile_name=profile.get("profile"),
                        node_id=q.node_id,
                        metadata_filters=vector_filters,
                        exclude_ids=exclusion.get("active_turn_ids"),
                        exclude_text=exclusion.get("active_turn_text"),
                    )
                except Exception as exc:
                    logger.debug(f"vector backend skipped: {exc}")
                    continue
                for item in vec:
                    item = dict(item)
                    item["meta"] = dict(item.get("meta") or {})
                    item["meta"]["vector_query"] = term
                    vector_candidates.append(item)

            vector_candidates_total = len(vector_candidates)
            rerank_weights = {"anchor_overlap": 0.15, "artifact_density": 0.10}
            reranked: List[Dict[str, Any]] = []
            for item in vector_candidates:
                item_id = item.get("id") or item.get("uri")
                if item_id in seen_vec:
                    continue
                seen_vec.add(item_id)
                text = str(item.get("text") or item.get("snippet") or "")
                anchor_count = _anchor_overlap(text, anchor_terms)
                density = _artifact_density(text)
                base = float(item.get("score") or 0.0)
                item["meta"]["anchor_overlap"] = anchor_count
                item["meta"]["artifact_density"] = density
                item["score"] = base + rerank_weights["anchor_overlap"] * anchor_count + rerank_weights["artifact_density"] * density
                reranked.append(item)

            reranked.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            for item in reranked[:vector_top_k]:
                candidates.append(item)
                vec_count += 1

            backend_counts_total["vector"] = vec_count
            if diagnostic:
                logger.info(
                    "graphtri vector rerank queries_used=%s total=%s deduped=%s top_anchors=%s weights=%s",
                    vector_queries,
                    vector_candidates_total,
                    len(reranked),
                    anchor_terms[:6],
                    rerank_weights,
                )

        elif diagnostic:
            source_gating["vector"] = "disabled_by_profile_or_global"
            logger.info(
                "graphtri vector skipped profile=%s vector_top_k=%s global_vector_enabled=%s",
                profile.get("profile"),
                profile.get("vector_top_k"),
                settings.RECALL_ENABLE_VECTOR,
            )

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
        for sig in signals:
            cand, counts = await _query_backends(
                sig,
                profile,
                session_id=effective_session_id,
                node_id=q.node_id,
                entities=_extract_entities(query_fragment),
                diagnostic=diagnostic,
                exclusion=exclusion,
            )
            candidates.extend(cand)
            for k, v in counts.items():
                backend_counts_total[k] = backend_counts_total.get(k, 0) + v

    candidates, suppressed = _suppress_self_hits(
        candidates,
        active_turn_text=str(exclusion.get("active_turn_text") or ""),
        active_turn_ids=list(exclusion.get("active_turn_ids") or []),
        active_turn_ts=exclusion.get("active_turn_ts"),
    )
    if suppressed:
        logger.info(
            "recall self-hit suppression active_turn_ids=%s suppressed=%s",
            exclusion.get("active_turn_ids"),
            suppressed,
        )
    latency_ms = int((time.time() - t0) * 1000)
    bundle, ranking_debug = fuse_candidates(
        candidates=candidates,
        profile=profile,
        latency_ms=latency_ms,
        query_text=query_fragment,
        session_id=effective_session_id,
        diagnostic=diagnostic,
        substantive_query=str(query_targeting.get("turn_type")) == "substantive",
    )
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
                "profile_selected": str(profile.get("profile") or q.profile),
                "profile_requested": q.profile,
                "query_expansion_enabled": enable_qe,
                "query_targeting": {
                    **query_targeting,
                    "raw_fragment": q.fragment,
                },
                "source_gating": source_gating,
                "active_turn": {
                    "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                    "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                    "ts_present": exclusion.get("active_turn_ts") is not None,
                    "self_hit_suppressed": suppressed,
                },
                "fusion": bundle.stats.diagnostic or {},
                "selected_summary": _bounded_selected_summary(list(bundle.items)),
            }
            if diagnostic
            else {}
        ),
    )
    if diagnostic:
        logger.info(
            "recall_diagnostic_summary corr_id=%s profile=%s requested_profile=%s gating=%s drop_counts=%s selected_counts=%s suppressed=%s selected=%s",
            decision.corr_id,
            profile.get("profile"),
            q.profile,
            source_gating,
            decision.dropped,
            (bundle.stats.diagnostic or {}).get("source_selected_counts", {}),
            suppressed,
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

    bundle, decision = await process_recall(q, corr_id=str(env.correlation_id), diagnostic=diagnostic)

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
