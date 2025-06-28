# emergence/memory/tier2_semantic/chroma_reader.py
from chromadb.utils import embedding_functions
from emergence.memory.tier2_semantic.chroma_client import get_chroma_collection

DEFAULT_EMBED = embedding_functions.DefaultEmbeddingFunction()

def query_chroma(query: str, top_k: int = 5) -> list:
    """
    Semantic query against the Chroma memory collection.
    Returns list of matching memory entry dicts.
    """
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [
        {"summary": doc, **meta}
        for doc, meta in zip(docs, metas)
    ]

