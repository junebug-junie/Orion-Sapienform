# app/vector.py
import os, uuid, logging
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("RAG_COLLECTION", "docs")

_client: QdrantClient | None = None
_dim: int | None = None


def get_client() -> QdrantClient:
    """Singleton Qdrant client with lazy init."""
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        _client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=10.0)
    return _client


def ensure_collection(dim: int):
    """Ensure Qdrant collection exists with correct dim. Recreate if mismatch."""
    global _dim
    _dim = dim
    cli = get_client()

    try:
        collections = {c.name: c for c in cli.get_collections().collections}
        if COLLECTION not in collections:
            logger.info(f"Creating Qdrant collection '{COLLECTION}' with dim={dim}")
            cli.create_collection(
                collection_name=COLLECTION,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
        else:
            # Check existing dimension
            info = cli.get_collection(COLLECTION)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                logger.warning(
                    f"Collection '{COLLECTION}' has dim={existing_dim}, "
                    f"but embedder requires dim={dim}. Recreating..."
                )
                cli.delete_collection(COLLECTION)
                cli.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
            else:
                logger.debug(
                    f"Collection '{COLLECTION}' already exists with correct dim={dim}."
                )
    except UnexpectedResponse as e:
        logger.error(f"Failed to ensure collection '{COLLECTION}': {e}")
        raise


def upsert_embeddings(embeddings: List[List[float]], payloads: List[Dict]):
    """Insert or update points into Qdrant."""
    cli = get_client()
    points = [
        qm.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pl)
        for vec, pl in zip(embeddings, payloads)
    ]
    logger.debug(f"Upserting {len(points)} points to '{COLLECTION}'...")
    try:
        cli.upsert(collection_name=COLLECTION, points=points, wait=True)
    except UnexpectedResponse as e:
        logger.error(f"Upsert failed: {e}")
        raise


def search(query_emb: List[float], k: int = 4, filter_: qm.Filter | None = None) -> List[Tuple[float, Dict]]:
    """Search collection and return (score, payload) tuples."""
    cli = get_client()
    try:
        results = cli.search(
            collection_name=COLLECTION,
            query_vector=query_emb,
            limit=k,
            query_filter=filter_,
        )
        logger.debug(f"Qdrant returned {len(results)} hits.")
        return [(r.score, r.payload) for r in results]
    except UnexpectedResponse as e:
        logger.error(f"Search failed: {e}")
        return []
