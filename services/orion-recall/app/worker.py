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
    from .storage.rdf_adapter import fetch_rdf_fragments
    from .storage.vector_adapter import fetch_vector_fragments
    from .sql_timeline import fetch_recent_fragments, fetch_related_by_entities
except ImportError:  # pragma: no cover - fallback for test harness pathing
    from fusion import fuse_candidates  # type: ignore
    from profiles import get_profile  # type: ignore
    from settings import settings  # type: ignore
    try:
        from rdf_adapter import fetch_rdf_fragments  # type: ignore
        from vector_adapter import fetch_vector_fragments  # type: ignore
        from sql_timeline import fetch_recent_fragments, fetch_related_by_entities  # type: ignore
    except ImportError:
        from storage.rdf_adapter import fetch_rdf_fragments  # type: ignore
        from storage.vector_adapter import fetch_vector_fragments  # type: ignore
        from sql_timeline import fetch_recent_fragments, fetch_related_by_entities  # type: ignore

logger = logging.getLogger("orion-recall.worker")


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


async def _query_backends(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}

    if settings.RECALL_ENABLE_VECTOR:
        try:
            vec = fetch_vector_fragments(
                query_text=fragment,
                time_window_days=settings.RECALL_DEFAULT_TIME_WINDOW_DAYS,
                max_items=int(profile.get("vector_top_k", settings.RECALL_DEFAULT_MAX_ITEMS)),
            )
            backend_counts["vector"] = len(vec)
            candidates.extend(vec)
        except Exception as exc:
            logger.debug(f"vector backend skipped: {exc}")

    if settings.RECALL_ENABLE_RDF and settings.RECALL_RDF_ENDPOINT_URL:
        try:
            rdf = fetch_rdf_fragments(
                query_text=fragment,
                max_items=int(profile.get("rdf_top_k", 0)),
            )
            backend_counts["rdf"] = len(rdf)
            candidates.extend(rdf)
        except Exception as exc:
            logger.debug(f"rdf backend skipped: {exc}")

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


async def process_recall(q: RecallQueryV1, *, corr_id: str) -> Tuple[MemoryBundleV1, RecallDecisionV1]:
    profile = get_profile(q.profile)
    enable_qe = bool(profile.get("enable_query_expansion", True))
    signals = _expand_query(q.fragment, verb=q.verb, intent=q.intent, enable=enable_qe)

    t0 = time.time()
    candidates: List[Dict[str, Any]] = []
    backend_counts_total: Dict[str, int] = {}
    for sig in signals:
        cand, counts = await _query_backends(
            sig,
            profile,
            session_id=q.session_id,
            node_id=q.node_id,
            entities=_extract_entities(q.fragment),
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
        kind=settings.RECALL_BUS_REPLY_DEFAULT,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload={"bundle": bundle.model_dump(mode="json")},
        reply_to=None,
    )


def telemetry_envelope(decision: RecallDecisionV1, env: BaseEnvelope) -> BaseEnvelope:
    return BaseEnvelope(
        kind=settings.RECALL_BUS_TELEMETRY,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=decision.model_dump(mode="json"),
    )


async def handle_recall(env: BaseEnvelope, *, bus) -> BaseEnvelope:
    if env.kind not in {settings.RECALL_BUS_INTAKE, "recall.query.request", "recall.query.v1"}:
        return BaseEnvelope(
            kind=settings.RECALL_BUS_REPLY_DEFAULT,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    try:
        q = RecallQueryV1.model_validate(payload_obj)
    except ValidationError as ve:
        return BaseEnvelope(
            kind=settings.RECALL_BUS_REPLY_DEFAULT,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    bundle, decision = await process_recall(q, corr_id=str(env.correlation_id))

    # emit telemetry (fire and forget)
    try:
        await bus.publish(settings.RECALL_BUS_TELEMETRY, telemetry_envelope(decision, env))
    except Exception as exc:
        logger.debug(f"telemetry publish failed: {exc}")

    _persist_decision(decision)

    return build_reply_envelope(bundle, env)
