import logging
import chromadb
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse

from .settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)


class VectorStore:
    """
    Connects to Orion's ChromaDB instance using the 0.4.x API (host + port only).
    """
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.db_url = settings.VECTOR_DB_URL               # example: http://orion-athena-vector-db:8000
        self.collection_name = settings.VECTOR_DB_COLLECTION
        self.embedder = None
        self.client = None
        self.collection = None

    def initialize(self):
        try:
            # -------------------------------
            # Load embedding model
            # -------------------------------
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)

            # -------------------------------
            # Parse host/port from URL
            # -------------------------------
            logger.info(f"Connecting to ChromaDB server at: {self.db_url}")

            parsed = urlparse(self.db_url)

            host = parsed.hostname
            port = parsed.port or 8000

            if not host:
                raise ValueError(f"Invalid VECTOR_DB_URL: {self.db_url}")

            # -------------------------------
            # Connect using LEGACY 0.4.x API
            # -------------------------------
            self.client = chromadb.HttpClient(host=host, port=port)

            # -------------------------------
            # Get or connect to collection
            # -------------------------------
            logger.info(f"Connecting to ChromaDB collection '{self.collection_name}'")

            self.collection = self.client.get_collection(name=self.collection_name)

            logger.info(f"✅ Vector store initialized. Connected to '{self.collection_name}'.")

        except Exception as e:
            logger.critical(f"🚨 Failed to initialize vector store client: {e}", exc_info=True)
            raise

    def search(self, query: str, n_results: int = 3):
        if not self.collection:
            logger.error("Vector store not initialized.")
            return []

        try:
            logger.info(f"Searching for: {query[:60]}…")

            embedding = self.embedder.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )

            return results.get("documents", [[]])[0]

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []


vector_store = VectorStore()
