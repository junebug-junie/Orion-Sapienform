import logging
import threading
import json

import chromadb
import chromadb.utils.embedding_functions as ef
import time
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict, Any

from .settings import settings
from app.models import CollapseTriageEvent, ChatMessageEvent, RAGDocumentEvent
from orion.core.bus.service import OrionBus

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# A thread-safe queue for batching documents
_doc_queue: List[Dict[str, Any]] = []
_queue_lock = threading.Lock()

def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all values in a metadata dictionary are simple types (str, int, float, bool)
    by JSON-encoding any complex types (lists, dicts).
    """
    flat_meta = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat_meta[key] = value
        elif value is not None:
            # Convert lists, dicts, etc., into a JSON string
            flat_meta[key] = json.dumps(value)
    return flat_meta


def batch_upsert_worker(chroma_client, embedding_function, bus):
    """
    Periodically checks the queue and upserts a batch of documents to ChromaDB.
    """
    logger.info("⚙️ Batch upsert worker started. Batch size: %d", settings.BATCH_SIZE)

    # Initialize collection
    collection = None
    for attempt in range(10):
        try:
            if getattr(settings, "VECTOR_DB_CREATE_IF_MISSING", True):
                collection = chroma_client.get_or_create_collection(
                    name=settings.VECTOR_DB_COLLECTION,
                    embedding_function=embedding_function,
                )
                logger.info(f"🧠 Connected to Chroma → collection '{settings.VECTOR_DB_COLLECTION}' (created if missing).")
            else:
                collection = chroma_client.get_collection(name=settings.VECTOR_DB_COLLECTION)
                logger.info(f"🧠 Connected to existing Chroma collection '{settings.VECTOR_DB_COLLECTION}'.")
            break
        except Exception as e:
            logger.warning(f"⏳ Waiting for ChromaDB... attempt {attempt+1}/10 ({e})")
            time.sleep(5)
    else:
        logger.critical("🚨 Failed to connect to ChromaDB after multiple attempts.")
        return

    # --- Upsert loop ---
    while True:
        time.sleep(2)
        with _queue_lock:
            if not _doc_queue:
                continue
            batch = _doc_queue[:settings.BATCH_SIZE]
            del _doc_queue[:settings.BATCH_SIZE]

        try:
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [ _flatten_metadata(doc["metadata"]) for doc in batch ]

            collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            logger.info(f"✅ Upserted {len(batch)} documents into Chroma collection '{settings.VECTOR_DB_COLLECTION}'")

            # Publish confirmation
            for d in batch:
                bus.publish(settings.PUBLISH_CHANNEL_VECTOR_CONFIRM, {"id": d["id"], "status": "stored"})

        except Exception as e:
            logger.error(f"❌ Failed to upsert batch to ChromaDB: {e}", exc_info=True)

def listener_worker(bus: OrionBus):
    """
    Listens for messages, validates them, and adds them to a queue for batch processing.
    """
    channel_to_model_map = {
        settings.SUBSCRIBE_CHANNEL_COLLAPSE: CollapseTriageEvent,
        settings.SUBSCRIBE_CHANNEL_CHAT: ChatMessageEvent,
        settings.SUBSCRIBE_CHANNEL_RAG_DOC: RAGDocumentEvent,
    }

    channels = list(channel_to_model_map.keys())
    logger.info(f"👂 Subscribing to channels: {channels}")

    for message in bus.subscribe(*channels):
        channel = message.get("channel")
        data = message.get("data")
        model = channel_to_model_map.get(channel)

        if not all([channel, data, model]):
            continue

        try:
            validated_event = model.model_validate(data)
            doc_to_queue = validated_event.to_document()

            with _queue_lock:
                _doc_queue.append(doc_to_queue)

            logger.debug(f"📥 Queued document {doc_to_queue['id']} from channel {channel}")

        except Exception as e:
            logger.warning(f"Skipping invalid message on channel {channel}: {e}")


@app.on_event("startup")
def startup_event():
    """
    Initializes all dependencies and starts the background worker threads,
    passing the dependencies to them directly.
    """
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    try:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)

        logger.info(f"Connecting to ChromaDB server at: {settings.VECTOR_DB_HOST}:{settings.VECTOR_DB_PORT}")
        chroma_client = chromadb.HttpClient(
            host=settings.VECTOR_DB_HOST, 
            port=settings.VECTOR_DB_PORT
        )
        chroma_client.heartbeat()

        logger.info("✅ Initialized ChromaDB client and embedding model.")
    except Exception as e:
        logger.critical(f"🚨 Failed to initialize a required service: {e}", exc_info=True)
        return

    if settings.ORION_BUS_ENABLED:
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
        logger.info("Starting listener and batch worker threads...")
        # The listener needs its own bus instance for thread-safety
        listen_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)

        threading.Thread(target=listener_worker, args=(listen_bus,), daemon=True).start()
        threading.Thread(target=batch_upsert_worker, args=(chroma_client, embedding_function, bus), daemon=True).start()
    else:
        logger.warning("Bus is disabled; vector writer will be idle.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "queue_size": len(_doc_queue),
    }

