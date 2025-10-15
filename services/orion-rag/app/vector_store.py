import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from .settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

class VectorStore:
    """
    A client that connects to the standalone Orion Vector DB service.
    It handles embedding queries and performing similarity searches.
    """
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.db_url = settings.VECTOR_DB_URL
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
            # The RAG service still needs the model to embed the user's query
            self.embedder = SentenceTransformer(self.embedding_model_name)
            
            logger.info(f"Connecting to ChromaDB server at: {self.db_url}")
            # Use HttpClient to connect to the remote ChromaDB instance
            self.client = chromadb.HttpClient(host=self.db_url, port=8000) # Assuming default Chroma port
            
            # Get a handle to the existing collection on the server.
            self.collection = self.client.get_collection(name=self.collection_name)
            
            logger.info(f"✅ Vector store client initialized and connected to collection '{self.collection_name}'.")
            
        except Exception as e:
            logger.critical(f"🚨 Failed to initialize vector store client: {e}", exc_info=True)
            raise

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """
        Embeds a query and performs a similarity search against the remote
        vector store, returning the text of the most relevant documents.
        """
        if not all([self.collection, self.embedder]):
            logger.error("Vector store client is not initialized. Cannot perform search.")
            return []

        try:
            logger.info(f"Performing vector search for query: '{query[:50]}...'")
            query_embedding = self.embedder.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            documents = results.get("documents", [[]])[0]
            logger.info(f"Found {len(documents)} relevant documents.")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to perform search: {e}", exc_info=True)
            return []

# Create a single, global instance that can be imported by other modules.
vector_store = VectorStore()

