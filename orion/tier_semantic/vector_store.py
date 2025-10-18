import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

class VectorStore:
    """
    A client that connects to the standalone Orion Vector DB service.
    This writer client handles embedding and adding documents.
    """
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.db_host = settings.VECTOR_DB_HOST
        self.db_port = settings.VECTOR_DB_PORT
        self.collection_name = settings.VECTOR_DB_COLLECTION
        self.embedder = None
        self.client = None
        self.collection = None

    def initialize(self):
        """
        Loads the embedding model and initializes the ChromaDB HTTP client
        to connect to the standalone vector database service.
        """
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            # The model is pre-loaded in the Dockerfile, so this should be fast
            self.embedder = SentenceTransformer(self.embedding_model_name)
            self.client = chromadb.HttpClient(host=self.db_host, port=self.db_port)
            logger.info(f"Connecting to ChromaDB server at: {self.db_host}:{self.db_port}")

            # Get or create the collection on the server.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Using cosine distance
            )

            logger.info(f"✅ Vector store client initialized and connected to collection '{self.collection_name}'.")

        except Exception as e:
            logger.critical(f"🚨 Failed to initialize vector store client: {e}", exc_info=True)
            raise

    def add_documents(self, documents: List[str]):
        """
        Embeds a list of document texts and adds them to the collection.
        """
        if not self.collection or not self.embedder:
            logger.error("Vector store is not initialized. Cannot add documents.")
            return

        try:
            # Generate embeddings
            embeddings = self.embedder.encode(documents).tolist()
 
            # Generate unique IDs for each chunk
            # NOTE: This is a simple incremental ID. For production, you'd
            # want a more robust or content-based hashing strategy.
            base_id = self.collection.count()
            ids = [f"chunk_{base_id + i}" for i in range(len(documents))]
            
            # Add to the collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids
            )
            logger.info(f"Successfully added {len(documents)} new documents to the collection.")
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}", exc_info=True)

# Create a single, global instance that can be imported
vector_store = VectorStore()
