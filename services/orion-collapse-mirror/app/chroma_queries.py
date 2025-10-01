from typing import Optional, Dict, Any
from app.chroma_db import collection

def query_collapse_memory(
    prompt: str,
    observer: Optional[str] = None,
    type: Optional[str] = None,
    emergent_entity: Optional[str] = None,
    mantra: Optional[str] = None,
    n_results: int = 3
) -> Dict[str, Any]:
    """
    Query the Chroma collection for collapse memories.
    Returns summaries + metadata.
    """
    filter_conditions = {}
    if observer:
        filter_conditions["observer"] = observer
    if type:
        filter_conditions["type"] = type
    if emergent_entity:
        filter_conditions["emergent_entity"] = emergent_entity
    if mantra:
        filter_conditions["mantra"] = mantra

    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
        where=filter_conditions or None
    )

    return {
        "query": prompt,
        "filter": filter_conditions or None,
        "results": [
            {"summary": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    }
