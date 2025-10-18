import argparse
import logging
from app.settings import settings
from orion.tier_semantic.vector_store import vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_script")

def ingest_file(source_path: str):
    """
    Reads a text file, splits it into chunks, and ingests them into the vector store.
    """
    collection_name = settings.VECTOR_DB_COLLECTION
    logger.info(f"📚 Starting ingestion for file: {source_path}")
    logger.info(f"Target collection: '{collection_name}'")

    try:
        # 1. Load the document
        with open(source_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Successfully loaded {len(text)} characters.")

        # 2. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")

        if not chunks:
            logger.warning("No chunks created. Nothing to ingest.")
            return

        # 3. Initialize the vector store client
        vector_store.initialize()

        # 4. Add chunks to the vector store
        logger.info("Adding document chunks to the vector store... This may take a while.")
        vector_store.add_documents(chunks)

        count = vector_store.collection.count()
        logger.info(f"✅ Ingestion complete! Collection '{collection_name}' now contains {count} documents.")

    except FileNotFoundError:
        logger.critical(f"🚨 ERROR: Source file not found at: {source_path}")
    except Exception as e:
        logger.critical(f"🚨 An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a text document into the Orion Vector DB.")
    parser.add_argument("source_file", type=str, help="Path to the source text file inside the container.")
    args = parser.parse_args()
    ingest_file(args.source_file)
