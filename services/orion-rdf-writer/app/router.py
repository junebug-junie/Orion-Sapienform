import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.rdf_builder import build_triples_from_envelope
from app.service import _push_to_rdf_store
from app.utils import logger
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

router = APIRouter(prefix="/rdf", tags=["rdf"])


@router.post("/ingest")
async def ingest_rdf(payload: dict):
    try:
        kind = str(payload.get("kind") or "rdf.write.request")
        nt_data, graph_name = build_triples_from_envelope(kind, payload)
        if not nt_data:
            raise HTTPException(status_code=400, detail="no triples generated for payload")
        env = BaseEnvelope(
            kind=kind,
            source=ServiceRef(name="http-ingest", node=None, version=None),
            correlation_id=uuid4(),
            payload=payload if isinstance(payload, dict) else {},
        )
        await _push_to_rdf_store(nt_data, graph_name, env=env)
        return {"status": "ok", "id": payload.get("id"), "graph": graph_name}
    except HTTPException:
        raise
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="rdf write queue saturated") from None
    except Exception as e:
        logger.exception("HTTP ingest failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
