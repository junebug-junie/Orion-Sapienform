from fastapi import APIRouter, HTTPException
from app.rdf_builder import build_triples
from app.service import OrionRDFWriterService
from app.utils import logger

router = APIRouter(prefix="/rdf", tags=["rdf"])
rdf_service = OrionRDFWriterService()

@router.post("/ingest")
async def ingest_rdf(payload: dict):
    """
    Ingest a single event as RDF triples.
    Accepts JSON payloads compatible with Collapse Mirror / bus events.
    """
    try:
        nt_data, graph_name = build_triples(payload)
        rdf_service._push_to_graphdb(nt_data, graph_name, payload)
        return {"status": "ok", "id": payload.get("id"), "graph": graph_name}
    except Exception as e:
        logger.exception("❌ HTTP ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
