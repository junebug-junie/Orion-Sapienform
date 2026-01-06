import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI
import chromadb
from chromadb.config import Settings as ChromaSettings

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vector.schemas import VectorDocumentUpsertV1, VectorWriteRequest

from app.settings import settings

# Setup Logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)

# --- Global Resources ---
chroma_client: Optional[chromadb.HttpClient] = None
hunter: Optional[Hunter] = None

# --- Configuration ---
def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
    )

def _setup_resources():
    """Initialize ChromaDB connection."""
    global chroma_client
    
    logger.info(f"🔌 Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}...")
    try:
        chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
        )
        # Test connection
        chroma_client.heartbeat()
        logger.info("✅ ChromaDB Connected.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to ChromaDB: {e}")
        # We don't raise here to allow the service to start, but writes will fail.
        chroma_client = None

def normalize_to_request(env: BaseEnvelope) -> Optional[VectorWriteRequest]:
    """
    Adapts various incoming kinds to a unified VectorWriteRequest.
    """
    kind = env.kind
    payload = env.payload

    # Handle Pydantic models (if decoded by codec)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    if not isinstance(payload, dict):
        return None

    # If it's already a direct vector write request
    if kind == "vector.write.request":
        try:
            return VectorWriteRequest.model_validate(payload)
        except Exception:
            return None

    # Normalization Map
    content = ""
    # Ensure metadata values are primitive types (strings), specifically ID and timestamp
    meta = {
        "source_node": env.source.node or "unknown",
        "kind": kind,
        "timestamp": str(env.created_at) if env.created_at else "",
        "id": str(env.id) # Cast UUID to string for Chroma metadata
    }
    
    # [FIX] Use the default collection from .env (via settings)
    collection = settings.CHROMA_COLLECTION_DEFAULT

    # [FIX] Check for "collapse.mirror" OR "collapse.mirror.entry"
    # Specific logic overrides the default collection
    if kind in ("collapse.mirror", "collapse.mirror.entry"):
        collection = "orion_collapse"
        content = f"{payload.get('summary', '')} {payload.get('mantra', '')} {payload.get('trigger', '')}"
        meta.update({
            "observer": payload.get("observer", "unknown"),
            "type": payload.get("type", "unknown")
        })
    elif kind in ("chat.message", "chat.history"):
        collection = "orion_chat"
        content = payload.get("content") or payload.get("message", "")
        meta.update({
            "role": payload.get("role", "unknown"),
            "session_id": payload.get("session_id", "")
        })
    elif kind == "rag.document":
        collection = "orion_knowledge"
        content = payload.get("text") or payload.get("content", "")
        meta.update({
            "filename": payload.get("filename", ""),
            "doc_id": payload.get("id", "")
        })
    elif kind == "cognition.trace":
        collection = "orion_cognition"
        # CognitionTracePayload logic
        # We index the final text.
        # Payload is a dict here (already dumped) or object? Env payload is usually dict.

        # We need to handle potential Pydantic model in payload if not serialized?
        # BaseEnvelope.payload is usually dict after decoding.

        # Extract fields
        final_text = payload.get("final_text") or ""
        verb = payload.get("verb", "unknown")
        mode = payload.get("mode", "unknown")
        correlation_id = payload.get("correlation_id", "")

        content = final_text
        if not content:
             # Fallback to description of what happened
             content = f"Cognition trace for {verb} in {mode} mode."

        meta.update({
            "correlation_id": str(correlation_id),
            "verb": verb,
            "mode": mode,
            "source": "cognition.trace"
        })
    else:
        # Fallback for generic text/event
        content = payload.get("text") or payload.get("summary") or ""

    if not content.strip():
        return None

    # Cast env.id (UUID) to str to satisfy VectorWriteRequest schema
    return VectorWriteRequest(
        id=str(env.id),
        kind=kind,
        content=content,
        metadata=meta,
        collection_name=collection
    )


def _to_upsert(req: VectorWriteRequest, env: BaseEnvelope) -> VectorDocumentUpsertV1:
    """
    Normalize legacy VectorWriteRequest to the new VectorDocumentUpsertV1.
    """
    vector = req.vector or []
    return VectorDocumentUpsertV1(
        doc_id=req.id,
        kind=req.kind,
        text=req.content,
        metadata=req.metadata,
        collection=req.collection_name,
        embedding=vector,
        embedding_dim=len(vector) if vector else None,
        embedding_model=None,
    )

# --- Bus Handler ---
async def handle_envelope(env: BaseEnvelope) -> None:
    """
    Receives upsert envelopes and writes them to ChromaDB using provided embeddings.
    """
    if not chroma_client:
        logger.warning("Skipping vector write: Resources not initialized.")
        return

    try:
        req: Optional[VectorDocumentUpsertV1]
        if env.kind == "memory.vector.upsert.v1":
            try:
                payload_dict = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
                req = VectorDocumentUpsertV1.model_validate(payload_dict)
            except Exception as e:
                logger.warning(f"Invalid upsert payload: {e}")
                return
        else:
            legacy_req = normalize_to_request(env)
            if not legacy_req:
                return
            req = _to_upsert(legacy_req, env)

        if not req.embedding:
            logger.warning(f"Skipping {req.kind}: no embedding supplied for id={req.doc_id}")
            return

        meta = dict(req.metadata)
        if req.embedding_model:
            meta["embedding_model"] = req.embedding_model
        if req.embedding_dim is not None:
            meta["embedding_dim"] = req.embedding_dim
        if req.latent_ref:
            meta["latent_ref"] = req.latent_ref
        if req.latent_summary:
            meta["latent_summary"] = req.latent_summary

        collection_name = req.collection or settings.CHROMA_COLLECTION_DEFAULT
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        await asyncio.to_thread(
            collection.upsert,
            ids=[req.doc_id],
            embeddings=[req.embedding],
            documents=[req.text],
            metadatas=[meta]
        )
        
        logger.info(f"✨ Stored {req.kind} -> {collection_name} (id={req.doc_id})")

    except Exception as e:
        logger.exception(f"Error processing {env.kind}: {e}")


# --- Lifecycle & App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    _setup_resources()
    
    # Start Bus Listener
    config = _cfg()
    channels = settings.SUBSCRIBE_CHANNELS

    logger.info(f"🏹 Hunter starting. Subscribing to: {channels}")
    hunter = Hunter(config, patterns=channels, handler=handle_envelope)
    await hunter.start_background()
    
    yield
    
    logger.info("🛑 Stopping Hunter...")
    if hunter:
        await hunter.stop()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

@app.get("/health")
def health():
    bus_connected = bool(hunter and hunter.bus and getattr(hunter.bus, "_redis", None))
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "chroma_connected": chroma_client is not None,
        "bus_connected": bus_connected,
    }
