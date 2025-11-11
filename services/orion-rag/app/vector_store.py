import logging
import os
from urllib.parse import urlparse
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)


class VectorStore:
    """
    Connects to the standalone Orion Vector DB (Chroma) over HTTP.
    Embeds queries locally and does similarity search on the server.
    """

    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.db_url = settings.VECTOR_DB_URL.strip()  # e.g. http://100.92.216.81:8000
        self.collection_name = settings.VECTOR_DB_COLLECTION.strip()
        # Optional flags (safe defaults)
        self.create_if_missing = (
            str(getattr(settings, "VECTOR_DB_CREATE_IF_MISSING", "true")).lower()
            == "true"
        )
        # Optional tenant (leave empty for single-tenant server)
        self.db_tenant = str(getattr(settings, "VECTOR_DB_TENANT", "")).strip()

        self.embedder = None
        self.client = None
        self.collection = None

    def _parse_url(self):
        """
        Turn VECTOR_DB_URL into host/port/ssl the Chroma HttpClient expects.
        """
        u = urlparse(self.db_url)
        if not u.scheme:
            # Allow raw host or host:port
            # Treat as http and re-parse
            u = urlparse(f"http://{self.db_url}")

        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 8000)
        ssl = u.scheme == "https"
        return host, port, ssl

    def initialize(self):
        """
        Load embedding model and initialize Chroma HTTP client.
        Crucially, we set tenant=None if you're running a single-tenant server.
        """
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)

            host, port, ssl = self._parse_url()
            logger.info(
                f"Connecting to ChromaDB server at: {host}:{port} "
                f"(ssl={'on' if ssl else 'off'})"
            )

            # IMPORTANT:
            # Chroma client defaults to tenant='default_tenant' which 404s on single-tenant servers.
            # We explicitly neutralize it by passing tenant=None unless you provided VECTOR_DB_TENANT.
            client_kwargs = dict(host=host, port=port, ssl=ssl)
            if self.db_tenant:
                client_kwargs["tenant"] = self.db_tenant
                logger.info(f"Using Chroma tenant: {self.db_tenant}")
            else:
                client_kwargs["tenant"] = None
                logger.info("Single-tenant mode: no tenant will be used")

            self.client = chromadb.HttpClient(**client_kwargs)

            # Get (or lazily create) the collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(
                    f"✅ Connected to collection '{self.collection_name}'."
                )
            except Exception:
                if self.create_if_missing:
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name, metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(
                        f"🆕 Created and connected to collection '{self.collection_name}'."
                    )
                else:
                    raise

        except Exception as e:
            logger.critical(f"🚨 Failed to initialize vector store client: {e}", exc_info=True)
            raise

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """
        Embed a query and perform similarity search on the remote vector store.
        Returns the list of matched documents' text.
        """
        if not self.collection or not self.embedder:
            logger.error("Vector store client is not initialized. Cannot perform search.")
            return []

        try:
            logger.info(f"Performing vector search for query: '{query[:50]}...'")
            query_embedding = self.embedder.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
            documents = results.get("documents", [[]])[0]
            logger.info(f"Found {len(documents)} relevant documents.")
            return documents

        except Exception as e:
            logger.error(f"Failed to perform search: {e}", exc_info=True)
            return []


# Global instance
vector_store = VectorStore()
