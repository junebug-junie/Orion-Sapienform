"""Self Atlas's second `self_concept_history` producer (self-model rebuild
arc, Patch 3 follow-up, 2026-09-05): feeds topic-foundry's real per-cluster
LLM labels for the self-facts dataset into the same append-only, evidence-
linked history that Layer 3's reflection already writes to
(`orion/schemas/self_concept_history.py`'s `produced_by="self_atlas_cluster"`
value, reserved in Patch 3 but unused until now).

Reachable via `publish_self_atlas_cluster_history()`, a zero-arg, plain sync
function -- same calling convention as this service's other Self Atlas
scheduler steps (`concept_atlas_routes.py`'s `trigger_topic_foundry_self_*`/
`concept_atlas_ingest_topic_foundry_self`), so `main.py`'s scheduler tick
calls it identically: `await asyncio.to_thread(publish_self_atlas_cluster_history)`.

Investigated live 2026-09-05 (not guessed) before writing this:

- Cluster labels only ever exist as `GET /topics`' `TopicSummaryItem.label`
  (topic-foundry has no DB column for it, see
  `services/orion-topic-foundry/app/routers/topics.py`'s own comment) --
  `fetch_run_topics_and_keywords()`/`fetch_segments_for_run()`
  (`topic_foundry_client.py`) are the real, already-existing way to reach it.
- A missing label already has an established fallback in this exact repo:
  `orion/substrate/adapters/topic_foundry.py` returns `f"topic_{topic_id}"`
  when no label exists -- mirrored here rather than inventing a new one.
- Evidence is real: each `SegmentRecord.provenance["row_ids"]` is a list of
  `self_knowledge_items.item_id` values (the self dataset's own `id_column`).
- No completion event exists to react to (the two registered topic-foundry
  bus channels for this have zero consumers anywhere) -- this is pull-based,
  same as every other Self Atlas step, called once per scheduler tick.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("orion-hub.self_atlas_cluster_history")

CHANNEL_SELF_CONCEPT_HISTORY_WRITE = "orion:self_concept:history:write"

# HDBSCAN's noise/outlier bucket -- never a real topic. Matches
# concept_atlas_routes.py's own _OUTLIER_TOPIC_ID / topic_foundry_client.py's
# OUTLIER_TOPIC_ID (not imported from either -- this module is deliberately
# import-light, same rationale concept_atlas_routes.py itself gives for not
# importing the substrate adapter's copy of this same constant).
_OUTLIER_TOPIC_ID = -1

# Bounds the evidence_refs list per cluster -- a large/dominant cluster could
# otherwise carry thousands of item_ids into a single bus envelope. Matches
# this service's existing per-call fan-out caps (topic_foundry_client.py's
# MAX_TOPICS_FOR_KEYWORDS, MAX_KG_EDGES_LIMIT).
_MAX_EVIDENCE_REFS_PER_CLUSTER = 50

# How many of a cluster's keywords to fold into `content` -- a label alone
# is 2-6 words (confirmed live against real topics_summary.json output),
# too thin on its own for a self-concept description.
_MAX_KEYWORDS_IN_CONTENT = 10


def _slugify(label: str) -> str:
    """Same small inline pattern as ``orion/core/storage/memory_cards.py``'s
    own slug helper -- recreated locally rather than cross-service imported
    (service-boundary rule, CLAUDE.md section 5)."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower().strip())[:80].strip("-")


def _self_atlas_concept_id(topic_id: int, label: Optional[str]) -> str:
    """Stable-ish identity for a cluster: derived from its label when one
    exists (so a retrain that reproduces the same label keeps writing to the
    same concept_id lineage), falling back to the topic_id when it doesn't
    -- mirrors ``orion/substrate/adapters/topic_foundry.py``'s own
    ``f"topic_{topic_id}"`` fallback for a missing label.

    Known, disclosed limitation (not fixed here): topic-foundry itself has
    no stable cross-run cluster identity. A retrain that meaningfully
    relabels a cluster starts a "new" concept_id lineage under this scheme
    -- the underlying instability, not something this function can paper
    over without inventing cross-run cluster matching that doesn't exist
    anywhere in topic-foundry today.
    """
    slug = _slugify(label) if label else ""
    if slug:
        return f"self-atlas-cluster-{slug}"
    return f"self-atlas-cluster-topic-{topic_id}"


def _self_atlas_content(label: Optional[str], topic_id: int, keywords: list[str]) -> str:
    display_label = label or f"topic_{topic_id}"
    trimmed_keywords = [str(k).strip() for k in keywords if str(k).strip()][:_MAX_KEYWORDS_IN_CONTENT]
    if trimmed_keywords:
        return f"Self Atlas cluster '{display_label}': related terms -- {', '.join(trimmed_keywords)}."
    return f"Self Atlas cluster '{display_label}' (no keywords available)."


def _build_self_atlas_cluster_events(
    *,
    topics: list[dict[str, Any]],
    keywords_by_topic: dict[int, list[str]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pure, no I/O: groups already-fetched segments by topic_id (skipping
    the outlier bucket) to collect each cluster's evidence, then returns one
    ``{concept_id, content, evidence_refs}`` dict per real topic. Testable
    with fixture data, no network or DB -- same shape as
    ``orion.substrate.adapters.topic_foundry.map_topic_foundry_run_to_substrate``'s
    own pure-conversion contract.
    """
    row_ids_by_topic: dict[int, set[str]] = {}
    for seg in segments:
        topic_id = seg.get("topic_id")
        try:
            topic_id_int = int(topic_id)
        except (TypeError, ValueError):
            continue
        if topic_id_int == _OUTLIER_TOPIC_ID:
            continue
        provenance = seg.get("provenance")
        row_ids = provenance.get("row_ids") if isinstance(provenance, dict) else None
        if not isinstance(row_ids, list):
            continue
        bucket = row_ids_by_topic.setdefault(topic_id_int, set())
        for row_id in row_ids:
            if row_id is not None:
                bucket.add(str(row_id))

    events: list[dict[str, Any]] = []
    for item in topics:
        raw_topic_id = item.get("topic_id")
        try:
            topic_id = int(raw_topic_id)
        except (TypeError, ValueError):
            continue
        if topic_id == _OUTLIER_TOPIC_ID:
            continue
        label = item.get("label")
        label = str(label).strip() if label else None
        keywords = keywords_by_topic.get(topic_id, [])
        evidence_refs = sorted(row_ids_by_topic.get(topic_id, set()))[:_MAX_EVIDENCE_REFS_PER_CLUSTER]
        events.append(
            {
                "concept_id": _self_atlas_concept_id(topic_id, label),
                "content": _self_atlas_content(label, topic_id, keywords),
                "evidence_refs": evidence_refs,
            }
        )
    return events


def _database_url() -> str:
    """Same fallback chain as this service's other direct-Postgres readers
    (``attention_loops_store.py``'s ``_database_url()``)."""
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


_ENGINE: Any = None


def _get_engine() -> Any:
    """Lazy, cached SQLAlchemy engine -- ``attention_loops_store.py``'s exact
    idiom, reused rather than re-invented. Returns ``None`` (never raises) if
    SQLAlchemy import or engine construction fails, so a missing driver/DSN
    degrades this producer to "always write version 1" rather than crashing
    the scheduler tick."""
    global _ENGINE
    if _ENGINE is None:
        try:
            from sqlalchemy import create_engine

            _ENGINE = create_engine(_database_url(), pool_pre_ping=True)
        except Exception as exc:  # pragma: no cover - defensive, mirrors sibling readers
            logger.debug("self_atlas_cluster_history_engine_unavailable error=%s", exc)
            return None
    return _ENGINE


def _latest_self_concept_history_row(concept_id: str) -> Optional[tuple[int, str]]:
    """Real ``(version, content)`` of the newest row for ``concept_id``, or
    ``None`` if there isn't one yet or the lookup fails for any reason.
    Fails soft (same convention as cortex-exec's
    ``_next_self_concept_version``) -- an unreachable DB must not block this
    producer from at least attempting a version-1 write.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        if engine is None:
            return None
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT version, content FROM self_concept_history "
                        "WHERE concept_id = :concept_id ORDER BY version DESC LIMIT 1"
                    ),
                    {"concept_id": concept_id},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return int(row["version"]), str(row["content"])
    except Exception as exc:
        logger.debug("self_atlas_cluster_history_lookup_failed concept_id=%s error=%s", concept_id, exc)
        return None


def _unavailable(reason: str, error: Optional[str] = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False, "reason": reason}
    if error:
        payload["error"] = error
    payload.update(extra)
    return payload


def publish_self_atlas_cluster_history() -> dict[str, Any]:
    """Fetch the self-facts dataset's latest completed topic-foundry run,
    build one candidate ``self_concept_history`` row per real cluster, skip
    any whose content is unchanged since its last published version (so an
    unchanged cluster produces zero rows on every scheduler tick -- this
    replaces any need for in-process "have I seen this run_id" tracking, and
    stays correct across a service restart, which an in-memory flag would
    not), and publish the rest.

    Never raises -- every failure mode degrades to an ``_unavailable(...)``
    dict, same contract as this service's other Self Atlas scheduler steps.
    Deliberately a plain sync ``def``: the underlying HTTP + DB calls are all
    blocking, and the caller (``main.py``'s scheduler tick) already runs
    every sibling step the same way, via ``asyncio.to_thread`` -- an
    ``async def`` here would just move the blocking-event-loop problem
    inside this function instead of solving it.
    """
    from scripts.settings import settings

    base_url = str(getattr(settings, "TOPIC_FOUNDRY_BASE_URL", "") or "").strip()
    if not base_url:
        return _unavailable("topic_foundry_base_url_not_configured", published_count=0)

    from scripts.concept_atlas_routes import _TOPIC_FOUNDRY_SELF_MODEL_NAME
    from scripts.topic_foundry_client import (
        TopicFoundryClientError,
        fetch_run_topics_and_keywords,
        fetch_segments_for_run,
    )

    try:
        fetched = fetch_run_topics_and_keywords(base_url, model_name=_TOPIC_FOUNDRY_SELF_MODEL_NAME)
    except TopicFoundryClientError as exc:
        logger.warning("self_atlas_cluster_history_fetch_failed error=%s", exc)
        return _unavailable("topic_foundry_fetch_failed", str(exc), published_count=0)
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug crash the tick
        logger.warning("self_atlas_cluster_history_unexpected_fetch_error error=%s", exc)
        return _unavailable("topic_foundry_unexpected_error", str(exc), published_count=0)

    run_id = fetched["run_id"]
    topics = fetched["topics"]
    keywords_by_topic = fetched["keywords_by_topic"]

    try:
        segments = fetch_segments_for_run(base_url, run_id)
    except TopicFoundryClientError as exc:
        logger.warning("self_atlas_cluster_history_segments_fetch_failed run_id=%s error=%s", run_id, exc)
        segments = []  # degrade to empty evidence_refs, not an aborted tick
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("self_atlas_cluster_history_segments_unexpected_error run_id=%s error=%s", run_id, exc)
        segments = []

    candidates = _build_self_atlas_cluster_events(topics=topics, keywords_by_topic=keywords_by_topic, segments=segments)
    if not candidates:
        return _unavailable("no_usable_topics", run_id=run_id, topics_fetched=len(topics), published_count=0)

    to_publish: list[dict[str, Any]] = []
    skipped_unchanged = 0
    for candidate in candidates:
        existing = _latest_self_concept_history_row(candidate["concept_id"])
        if existing is not None and existing[1] == candidate["content"]:
            skipped_unchanged += 1
            continue
        version = existing[0] + 1 if existing is not None else 1
        to_publish.append({**candidate, "version": version})

    if not to_publish:
        return {
            "available": True,
            "run_id": run_id,
            "topics_seen": len(candidates),
            "published_count": 0,
            "skipped_unchanged_count": skipped_unchanged,
            "concept_ids": [],
        }

    published_count, failed_count = _publish_events(to_publish, correlation_id=run_id)
    return {
        "available": True,
        "run_id": run_id,
        "topics_seen": len(candidates),
        "published_count": published_count,
        "failed_count": failed_count,
        "skipped_unchanged_count": skipped_unchanged,
        "concept_ids": [c["concept_id"] for c in to_publish],
    }


def _publish_events(events: list[dict[str, Any]], *, correlation_id: str) -> tuple[int, int]:
    """Publish each event over the bus, one short-lived connection for the
    whole batch -- ``bus_publish.py``'s ``publish_attention_loop_outcome()``
    idiom (``OrionBusAsync`` + ``anyio.run``), reused rather than inventing a
    new bus-lifecycle pattern. Returns ``(published_count, failed_count)``;
    never raises -- a bus-connect failure marks every event failed rather
    than crashing the scheduler tick."""
    import anyio

    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.self_concept_history import SelfConceptHistoryV1
    from scripts.settings import settings

    def _source_ref() -> ServiceRef:
        return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)

    published = 0
    failed = 0

    async def _run() -> None:
        nonlocal published, failed
        bus = OrionBusAsync(str(settings.ORION_BUS_URL))
        await bus.connect()
        try:
            for event in events:
                try:
                    payload = SelfConceptHistoryV1(
                        concept_id=event["concept_id"],
                        version=event["version"],
                        content=event["content"],
                        evidence_refs=event["evidence_refs"],
                        produced_by="self_atlas_cluster",
                    )
                    envelope = BaseEnvelope(
                        kind="self_concept.history.write.v1",
                        source=_source_ref(),
                        correlation_id=correlation_id,
                        payload=payload.model_dump(mode="json"),
                    )
                    await bus.publish(CHANNEL_SELF_CONCEPT_HISTORY_WRITE, envelope)
                    published += 1
                except Exception as exc:
                    logger.warning(
                        "self_atlas_cluster_history_publish_one_failed concept_id=%s error=%s",
                        event.get("concept_id"),
                        exc,
                    )
                    failed += 1
        finally:
            await bus.close()

    try:
        anyio.run(_run)
    except Exception as exc:  # pragma: no cover - defensive, e.g. bus connect itself failing
        logger.warning("self_atlas_cluster_history_bus_unavailable error=%s", exc)
        return 0, len(events)

    return published, failed
