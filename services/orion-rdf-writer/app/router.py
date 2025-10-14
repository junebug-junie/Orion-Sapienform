from fastapi import APIRouter, HTTPException
from app.rdf_builder import build_triples
# Import the specific function we need from the refactored service.py
from app.service import _push_to_graphdb
from app.utils import logger

router = APIRouter(prefix="/rdf", tags=["rdf"])

@router.post("/ingest")
async def ingest_rdf(payload: dict):
    """
    Ingest a single event as RDF triples via an HTTP endpoint.
    This is useful for testing or manual data injection.
    """
    try:
        nt_data, graph_name = build_triples(payload)
        # Call the standalone push function instead of the old class method.
        _push_to_graphdb(nt_data, graph_name, payload)
        return {"status": "ok", "id": payload.get("id"), "graph": graph_name}
    except Exception as e:
        logger.exception("❌ HTTP ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
