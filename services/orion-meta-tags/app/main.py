# services/orion-meta-tags/app/main.py
import asyncio
import logging
import traceback
import uuid
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

import spacy
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, Rabbit
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.graph.falkor_client import RedisGraphQueryClient
from orion.schemas.collapse_mirror import should_route_to_triage

from .settings import settings
from .models import EventIn, Enrichment
from .falkor_recall_writer import write_chat_turn_tags_to_falkor

# Setup Logging
logging.basicConfig(level=logging.getLevelName(settings.LOG_LEVEL))
logger = logging.getLogger(settings.SERVICE_NAME)

# Initialize NLP
try:
    nlp = spacy.load(settings.SPA_MODEL)
except OSError:
    from spacy.cli import download

    logger.info("Downloading spaCy model %s...", settings.SPA_MODEL)
    download(settings.SPA_MODEL)
    nlp = spacy.load(settings.SPA_MODEL)

# Global chassis references
meta_tagger: Hunter | None = None
meta_tagger_rpc: Rabbit | None = None

# Lazily constructed: only ever built if RECALL_FALKOR_TAG_ENTITY_ENABLED is
# true, so a disabled flag never attempts a FalkorDB connection.
_falkor_client: RedisGraphQueryClient | None = None


def _get_falkor_client() -> RedisGraphQueryClient:
    global _falkor_client
    if _falkor_client is None:
        _falkor_client = RedisGraphQueryClient(
            uri=settings.FALKORDB_URI, graph_name=settings.FALKORDB_RECALL_GRAPH
        )
    return _falkor_client


async def _write_chat_turn_tags_to_falkor_async(
    *, turn_id: str, session_id: str | None, ts: str, correlation_id: str | None,
    tags: list[str], entities: list[str],
) -> None:
    if not settings.RECALL_FALKOR_TAG_ENTITY_ENABLED:
        return
    try:
        # Falkor client is a sync redis client; keep GRAPH.QUERY off the
        # event loop so triage intake doesn't stall (same pattern as
        # orion/spark/concept_induction/bus_worker.py's Falkor write).
        result = await asyncio.to_thread(
            write_chat_turn_tags_to_falkor,
            _get_falkor_client(),
            turn_id=turn_id,
            session_id=session_id,
            ts=ts,
            correlation_id=correlation_id,
            tags=tags,
            entities=entities,
        )
        logger.info("falkor_recall_write_committed turn_id=%s result=%s", turn_id, result)
    except Exception as exc:
        # Additive/dark path: never let a Falkor write failure break the
        # existing tags.enriched publish it rides alongside.
        logger.warning("falkor_recall_write_failed turn_id=%s error=%s", turn_id, exc)


def _svc_ref() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )


def _normalize_observer(observer: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(observer or "")).encode("ascii", "ignore").decode("ascii")
    return normalized.strip().lower()


def _is_juniper(observer: Any) -> bool:
    normalized = _normalize_observer(observer)
    return normalized == "juniper" or normalized.startswith("juniper")


def _is_orion(observer: Any) -> bool:
    normalized = _normalize_observer(observer)
    return normalized == "orion" or normalized.startswith("orion")


async def handle_meta_tags_rpc(env: BaseEnvelope) -> BaseEnvelope:
    """
    RPC handler for MetaTagsService (used as a Cortex-Exec plan step).

    IMPORTANT:
    - Must always return a stable non-null 'id' in payload so caller can join.
    - Do NOT assume upstream provides id.
    """
    raw = env.payload if isinstance(env.payload, dict) else {}

    # Determine a stable id:
    req_id = (
        raw.get("id")
        or raw.get("event_id")
        or (str(env.correlation_id) if env.correlation_id else None)
    )
    if not req_id:
        req_id = str(uuid.uuid4())

    # If EventIn requires id, ensure it exists.
    if "id" not in raw or not raw.get("id"):
        raw = dict(raw)
        raw["id"] = req_id

    # Normalize text through EventIn (your existing pattern)
    in_payload = EventIn(**raw)

    doc = nlp(in_payload.text or "")
    tags = [ent.text for ent in doc.ents]
    entities = [ent.label_ for ent in doc.ents]

    out = {
        "id": req_id,
        "tags": tags,
        "entities": entities,
    }

    return BaseEnvelope(
        kind="meta_tags.result.v1",
        source=_svc_ref(),
        correlation_id=env.correlation_id,
        payload=out,
    )


async def handle_triage_event(envelope: BaseEnvelope) -> None:
    """
    Handler for 'orion:collapse:triage' streaming path.
    Keeps your existing behavior: emits Enrichment -> tags.enriched.
    """
    global meta_tagger

    try:
        raw_payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        if not should_route_to_triage(raw_payload):
            logger.info(
                "skip meta-tags: non_strict kind=%s id=%s observer=%s source_service=%s",
                envelope.kind,
                raw_payload.get("id") or raw_payload.get("event_id"),
                raw_payload.get("observer"),
                raw_payload.get("source_service"),
            )
            return
        if envelope.kind in {"chat.history", "social.turn.stored.v1"}:
            turn_id = raw_payload.get("turn_id") or raw_payload.get("correlation_id")
            if not turn_id and envelope.correlation_id:
                turn_id = str(envelope.correlation_id)
            if not turn_id:
                turn_id = uuid.uuid4().hex
            if not raw_payload.get("id"):
                raw_payload = dict(raw_payload)
                raw_payload["id"] = turn_id

            in_payload = EventIn(**raw_payload)
            logger.info("📨 Processing chat turn %s (Text len: %d)", in_payload.id, len(in_payload.text or ""))

            doc = nlp(in_payload.text or "")
            entities = [ent.text for ent in doc.ents]
            tags = list(entities)

            sentiment_tag = "sentiment:neutral"
            positive_keywords = {"triumphant", "relief", "capable", "good", "success"}
            negative_keywords = {"anxious", "fear", "fail", "bad", "panic"}

            tokens = set((in_payload.text or "").lower().split())
            if tokens & positive_keywords:
                sentiment_tag = "sentiment:positive"
            elif tokens & negative_keywords:
                sentiment_tag = "sentiment:negative"

            tags.append(sentiment_tag)

            # Phase 2 recall Falkor write (dark by default): gated to
            # chat.history only, not social.turn.stored.v1 -- that's Phase 6,
            # deliberately not touched here even though it shares this code
            # path. session_id lives on the raw chat.history payload
            # (ChatHistoryTurnV1) but is not part of MetaTagsPayload/Enrichment
            # below, so it's read directly from raw_payload here, not from
            # the enrichment object. ts is the envelope's own Titanium
            # created_at (real event time), not Enrichment.ts below (which is
            # meta-tags' own processing time -- a different, later value).
            if envelope.kind == "chat.history":
                await _write_chat_turn_tags_to_falkor_async(
                    turn_id=str(turn_id),
                    session_id=raw_payload.get("session_id"),
                    ts=envelope.created_at.isoformat(),
                    correlation_id=str(envelope.correlation_id) if envelope.correlation_id else None,
                    tags=tags,
                    entities=entities,
                )

            enrichment = Enrichment(
                id=turn_id,
                collapse_id=turn_id,
                service_name=settings.SERVICE_NAME,
                service_version=settings.SERVICE_VERSION,
                enrichment_type="chat_tagging",
                tags=tags,
                entities=entities,
                salience=0.0,
                ts=datetime.now(timezone.utc).isoformat(),
                node=settings.NODE_NAME,
                correlation_id=str(envelope.correlation_id) if envelope.correlation_id else str(turn_id),
                source_message_id=str(envelope.id) if envelope.id else None,
            )

            out_env = envelope.derive_child(
                kind="tags.enriched",
                source=_svc_ref(),
                payload=enrichment,
            )

            await meta_tagger.bus.publish(settings.CHANNEL_EVENTS_TAGGED_CHAT, out_env)
            logger.info("✅ Published chat tags for %s -> %s", turn_id, settings.CHANNEL_EVENTS_TAGGED_CHAT)
            return

        # VALIDATION & TEXT EXTRACTION
        raw_id = raw_payload.get("id") or raw_payload.get("event_id")
        if not raw_id and envelope.correlation_id:
            raw_id = str(envelope.correlation_id)
        if not raw_id:
            raw_id = uuid.uuid4().hex
        if not raw_payload.get("id"):
            raw_payload = dict(raw_payload)
            raw_payload["id"] = raw_id

        observer = _normalize_observer(raw_payload.get("observer"))
        if _is_orion(observer):
            logger.info("skip meta-tags: observer=orion collapse_id=%s", raw_id)
            return
        if not _is_juniper(observer):
            logger.info("skip meta-tags: observer unknown=%s collapse_id=%s", observer or "unknown", raw_id)
            return

        in_payload = EventIn(**raw_payload)
        logger.info("📨 Processing %s (Text len: %d)", in_payload.id, len(in_payload.text or ""))

        # NLP PROCESSING
        doc = nlp(in_payload.text or "")

        # Extract entities (as tags)
        tags = [ent.text for ent in doc.ents]
        entities = [ent.text for ent in doc.ents]

        # Basic sentiment heuristic
        sentiment_tag = "sentiment:neutral"
        positive_keywords = {"triumphant", "relief", "capable", "good", "success"}
        negative_keywords = {"anxious", "fear", "fail", "bad", "panic"}

        tokens = set((in_payload.text or "").lower().split())
        if tokens & positive_keywords:
            sentiment_tag = "sentiment:positive"
        elif tokens & negative_keywords:
            sentiment_tag = "sentiment:negative"

        tags.append(sentiment_tag)

        # Ensure collapse_id is never None (prevents downstream NOT NULL surprises)
        target_collapse_id = in_payload.collapse_id or in_payload.id
        if not target_collapse_id:
            logger.warning("⚠️  Missing collapse_id for event %s, sql-writer may fail.", in_payload.id)

        enrichment = Enrichment(
            id=in_payload.id,
            collapse_id=target_collapse_id,
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            enrichment_type="tagging",
            tags=tags,
            entities=entities,
            salience=0.0,
            ts=datetime.now(timezone.utc).isoformat(),
            node=settings.NODE_NAME,
            correlation_id=str(envelope.correlation_id) if envelope.correlation_id else str(in_payload.id),
            source_message_id=str(envelope.id) if envelope.id else None,
        )

        # Wrap in standard envelope
        out_env = envelope.derive_child(
            kind="tags.enriched",
            source=_svc_ref(),
            payload=enrichment,
        )

        await meta_tagger.bus.publish(settings.CHANNEL_EVENTS_TAGGED, out_env)
        logger.info(
            "✅ Published tags for %s observer=%s -> %s",
            in_payload.id,
            observer or "unknown",
            settings.CHANNEL_EVENTS_TAGGED,
        )

    except Exception as e:
        logger.error("❌ Error processing event: %s", e)
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_tagger, meta_tagger_rpc

    # Configure Chassis
    config = ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
    )

    # Streaming triage consumer (keep)
    meta_tagger = Hunter(
        cfg=config,
        handler=handle_triage_event,
        patterns=[settings.CHANNEL_EVENTS_TRIAGE, settings.CHANNEL_EVENTS_CHAT_TURN, "orion:chat:social:stored"],
    )

    # RPC step-service (MetaTagsService)
    meta_tagger_rpc = Rabbit(
        config,
        request_channel="orion:exec:request:MetaTagsService",
        handler=handle_meta_tags_rpc,
    )

    await meta_tagger_rpc.start_background()
    await meta_tagger.start_background()
    logger.info("Meta-Tags starting... triage=%s rpc=%s", settings.CHANNEL_EVENTS_TRIAGE, "orion:exec:request:MetaTagsService")

    yield

    logger.info("Shutting down...")
    try:
        await meta_tagger.stop()
    finally:
        await meta_tagger_rpc.stop()


app = FastAPI(
    title="Orion Meta-Tags",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    status = "disconnected"
    if meta_tagger and meta_tagger.bus and meta_tagger.bus.redis:
        status = "connected"
    return {"status": "ok", "bus": status, "model": settings.SPA_MODEL}
