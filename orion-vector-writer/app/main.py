import logging
import threading
import json
from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

from app.settings import settings
from app.models import CollapseTriageEvent, TagsEnrichedEvent
from orion.core.bus.service import OrionBus

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# --- Global Objects (initialized on startup) ---
embedder: SentenceTransformer | None = None
chroma_client: chromadb.HttpClient | None = None
collection: chromadb.Collection | None = None

def listener_worker():
    """
    A single worker that creates its own bus connection, subscribes to all
    relevant channels, and processes messages into the vector store.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error("Bus connection failed. Vector-Writer listener thread exiting.")
        return

    channels = settings.get_subscribe_channels()
    channel_map = {
        settings.CHANNEL_COLLAPSE_TRIAGE: CollapseTriageEvent,
        settings.CHANNEL_TAGS_ENRICHED: TagsEnrichedEvent,
    }

    logger.info(f"👂 Subscribing to channels: {channels}")
    for message in bus.subscribe(*channels):
        source_channel = message.get("channel")
        data = message.get("data")
        schema = channel_map.get(source_channel)

        if not all([source_channel, data, schema]):
            logger.warning(f"Skipping message from unhandled channel: {source_channel}")
            continue

        try:
            # --- 1. Validate and Transform ---
            validated_data = schema.model_validate(data)
            
            # The document ID should be the original collapse event ID
            doc_id = getattr(validated_data, 'collapse_id', validated_data.id)
            
            # The text to be embedded is the summary
            text_to_embed = getattr(validated_data, 'summary', '')
            if not text_to_embed:
                logger.warning(f"Message {doc_id} from {source_channel} has no 'summary' to embed.")
                continue

            # --- 2. Generate Embedding ---
            if embedder and collection:
                embedding = embedder.encode(text_to_embed).tolist()
                
                # Metadata includes everything from the event
                metadata = validated_data.model_dump()
                
                # --- 3. Upsert into ChromaDB ---
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text_to_embed],
                    metadatas=[metadata]
                )
                logger.info(f"✅ Upserted document {doc_id} from channel {source_channel}.")
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {source_channel}: {e}", exc_info=True)


@app.on_event("startup")
def startup_event():
    """
    Initializes the embedding model, ChromaDB client, and starts the listener.
    """
    global embedder, chroma_client, collection
    
    try:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        logger.info(f"Connecting to ChromaDB at: {settings.VECTOR_DB_URL}")
        chroma_client = chromadb.HttpClient(url=settings.VECTOR_DB_URL)
        chroma_client.heartbeat() # Throws exception if connection fails
        
        collection = chroma_client.get_or_create_collection(name=settings.VECTOR_DB_COLLECTION)
        logger.info(f"✅ Connected to ChromaDB and got collection '{settings.VECTOR_DB_COLLECTION}'.")

    except Exception as e:
        logger.critical(f"🚨 Failed to initialize vector store connection: {e}", exc_info=True)
        # In a real-world scenario, you might want the service to exit if it can't connect.
        return

    if settings.ORION_BUS_ENABLED:
        logger.info("🚀 Starting listener thread...")
        threading.Thread(target=listener_worker, daemon=True).start()
    else:
        logger.warning("⚠️ Bus is disabled; writer will be idle.")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "channels": settings.get_subscribe_channels(),
    }
