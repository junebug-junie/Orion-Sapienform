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
        fetch_graphtri_anchors,
        fetch_rdf_graphtri_anchor_terms,
        fetch_rdf_graphtri_fragments,
        fetch_rdf_chatturn_fragments,
    )
    from .storage.vector_adapter import fetch_vector_fragments
    from .sql_timeline import fetch_recent_fragments, fetch_related_by_entities
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
            fetch_graphtri_anchors,
            fetch_rdf_graphtri_anchor_terms,
            fetch_rdf_graphtri_fragments,
            fetch_rdf_chatturn_fragments,
        )
        from app.storage.vector_adapter import fetch_vector_fragments  # type: ignore
        from app.sql_timeline import fetch_recent_fragments, fetch_related_by_entities  # type: ignore
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
    if not session_id:
        return anchors
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


async def _query_backends(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
    diagnostic: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}

    rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
    rdf_top_k = int(profile.get("rdf_top_k", 0))
    expansion_terms: List[str] = []
    if rdf_enabled:
        try:

            profile_name = str(profile.get("profile") or "")

            # 0) Pull raw ChatTurns (prompt/response) from GRAPH <orion:chat>.
            # This is the only place your "exact text I used" lives.
            rdf_chat: List[Dict[str, Any]] = []
            if session_id:
                rdf_chat = fetch_rdf_chatturn_fragments(
                    query_text=fragment,
                    session_id=session_id,
                    max_items=max(rdf_top_k, 6),
                )
                backend_counts["rdf_chat"] = len(rdf_chat)
                candidates.extend(rdf_chat)

            # 1) Keep existing RDF paths (claims / neighborhood) as additional context.
            rdf: List[Dict[str, Any]] = []
            if profile_name.startswith("graphtri") and session_id:
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


    if settings.RECALL_ENABLE_VECTOR:
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
                    node_id=node_id,
                    metadata_filters=vector_filters,
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

    if settings.RECALL_ENABLE_SQL_TIMELINE or profile.get("enable_sql_timeline"):
        try:
            recent_items = await fetch_recent_fragments(
                session_id,
                node_id,
                int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K)),
            )
            related_items = await fetch_related_by_entities(
                entities,
                int(profile.get("sql_since_hours", max(1, settings.RECALL_SQL_SINCE_MINUTES // 60))),
                int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K)),
                session_id=session_id,
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
                        "tags": item.tags,
                        "score": 0.7,
                    }
                )
        except Exception as exc:
            logger.debug(f"sql timeline backend skipped: {exc}")

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


async def process_recall(
    q: RecallQueryV1,
    *,
    corr_id: str,
    diagnostic: bool = False,
) -> Tuple[MemoryBundleV1, RecallDecisionV1]:
    profile = get_profile(q.profile)
    enable_qe = bool(profile.get("enable_query_expansion", True))
    signals = _expand_query(q.fragment, verb=q.verb, intent=q.intent, enable=enable_qe)
    profile_name = str(profile.get("profile") or "")

    t0 = time.time()
    candidates: List[Dict[str, Any]] = []
    backend_counts_total: Dict[str, int] = {}
    if profile_name.startswith("graphtri"):
        anchor_set = _build_anchor_set(
            query_text=q.fragment,
            session_id=q.session_id,
            profile=profile,
        )
        anchor_terms = anchor_set.get("related_terms") or []
        vector_queries = [q.fragment]
        vector_queries.extend([f"{q.fragment} {term}" for term in anchor_terms[:8]])
        sql_filters = anchor_terms[:6]
        query_plan = {
            "vector_queries": vector_queries,
            "sql_filters": sql_filters,
            "sql_session_id": q.session_id,
        }

        rdf_top_k = int(profile.get("rdf_top_k", 0))
        rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
        rdf_items: List[Dict[str, Any]] = []
        if rdf_enabled:
            try:
                if q.session_id:
                    rdf_items = fetch_rdf_graphtri_fragments(
                        query_text=q.fragment,
                        session_id=q.session_id,
                        max_items=rdf_top_k,
                    )
                    if not rdf_items:
                        rdf_items = fetch_rdf_fragments(query_text=q.fragment, max_items=rdf_top_k)
                else:
                    rdf_items = fetch_rdf_fragments(query_text=q.fragment, max_items=rdf_top_k)
                backend_counts_total["rdf"] = len(rdf_items)
                candidates.extend(rdf_items)
            except Exception as exc:
                logger.debug(f"rdf backend skipped: {exc}")
        if settings.RECALL_ENABLE_VECTOR:
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
                        session_id=q.session_id,
                        node_id=q.node_id,
                        metadata_filters=vector_filters,
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

        sql_attempted = False
        sql_session_id = q.session_id
        sql_filters_used: List[str] = []
        if settings.RECALL_ENABLE_SQL_TIMELINE or profile.get("enable_sql_timeline"):
            try:
                sql_attempted = True
                sql_filters_used = sql_filters if q.session_id else []
                recent_items = await fetch_recent_fragments(
                    q.session_id,
                    q.node_id,
                    int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                    int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K)),
                )
                related_items = await fetch_related_by_entities(
                    sql_filters_used or [],
                    int(profile.get("sql_since_hours", max(1, settings.RECALL_SQL_SINCE_MINUTES // 60))),
                    int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K)),
                    session_id=q.session_id,
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
                            "tags": item.tags,
                            "score": 0.7,
                        }
                    )
            except Exception as exc:
                logger.debug(f"sql timeline backend skipped: {exc}")

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
                session_id=q.session_id,
                node_id=q.node_id,
                entities=_extract_entities(q.fragment),
                diagnostic=diagnostic,
            )
            candidates.extend(cand)
            for k, v in counts.items():
                backend_counts_total[k] = backend_counts_total.get(k, 0) + v

    latency_ms = int((time.time() - t0) * 1000)
    bundle = fuse_candidates(candidates=candidates, profile=profile, latency_ms=latency_ms)

    decision = RecallDecisionV1(
        corr_id=corr_id or str(uuid4()),
        session_id=q.session_id,
        node_id=q.node_id,
        verb=q.verb,
        profile=q.profile,
        query=q.fragment,
        selected_ids=[i.id for i in bundle.items],
        backend_counts=backend_counts_total or bundle.stats.backend_counts,
        latency_ms=latency_ms,
        dropped={},  # placeholder for future detailed drop reasons
    )
    return bundle, decision


def build_reply_envelope(bundle: MemoryBundleV1, env: BaseEnvelope) -> BaseEnvelope:
    return BaseEnvelope(
        kind=RECALL_REPLY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload={"bundle": bundle.model_dump(mode="json")},
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

    return build_reply_envelope(bundle, env)
