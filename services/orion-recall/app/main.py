# app/main.py
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from redis import asyncio as aioredis

from .settings import settings
from .types import PhiContext, RecallQuery
from .pipeline import run_recall_pipeline


logger = logging.getLogger("recall-app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class PhiPayload(BaseModel):
    valence: float = 0.0
    energy: float = 0.0
    coherence: float = 0.0
    novelty: float = 0.0


class RecallRequestBody(BaseModel):
    text: str = ""
    mode: str = Field(default_factory=lambda: settings.RECALL_DEFAULT_MODE)
    max_items: int = Field(default_factory=lambda: settings.RECALL_DEFAULT_MAX_ITEMS)
    time_window_days: int = Field(
        default_factory=lambda: settings.RECALL_DEFAULT_TIME_WINDOW_DAYS
    )
    tags: List[str] = Field(default_factory=list)
    phi: Optional[PhiPayload] = None
    trace_id: Optional[str] = None


class RecallResponseBody(BaseModel):
    fragments: List[Dict[str, Any]]
    debug: Dict[str, Any]


bus_task: Optional[asyncio.Task] = None


def _build_query_from_payload(payload: Dict[str, Any], source: str) -> RecallQuery:
    text = payload.get("text") or payload.get("query") or payload.get("query_text") or ""
    max_items = int(payload.get("max_items") or settings.RECALL_DEFAULT_MAX_ITEMS)
    time_window_days = int(
        payload.get("time_window_days") or settings.RECALL_DEFAULT_TIME_WINDOW_DAYS
    )
    mode = str(payload.get("mode") or settings.RECALL_DEFAULT_MODE)
    tags = payload.get("tags") or []

    phi_payload = payload.get("phi") or {}
    phi = None
    if isinstance(phi_payload, dict) and phi_payload:
        phi = PhiContext(
            valence=float(phi_payload.get("valence", 0.0) or 0.0),
            energy=float(phi_payload.get("energy", 0.0) or 0.0),
            coherence=float(phi_payload.get("coherence", 0.0) or 0.0),
            novelty=float(phi_payload.get("novelty", 0.0) or 0.0),
        )

    trace_id = payload.get("trace_id")

    return RecallQuery(
        text=text,
        max_items=max_items,
        time_window_days=time_window_days,
        mode=mode,
        tags=tags,
        phi=phi,
        trace_id=trace_id,
        source=source,
    )


def _fragments_to_dict(frags) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in frags:
        out.append(
            {
                "id": f.id,
                "kind": f.kind,
                "source": f.source,
                "text": f.text,
                "ts": f.ts,
                "tags": list(f.tags or []),
                "salience": f.salience,
                "valence": f.valence,
                "arousal": f.arousal,
                "meta": f.meta,
            }
        )
    return out


async def _bus_loop():
    """
    Listens on CHANNEL_RECALL_REQUEST and publishes replies to reply_channel.
    """
    if not settings.ORION_BUS_ENABLED:
        logger.info("Bus disabled; bus loop will not start.")
        return

    r = aioredis.from_url(settings.ORION_BUS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(settings.CHANNEL_RECALL_REQUEST)
    logger.info(f"Subscribed to recall channel: {settings.CHANNEL_RECALL_REQUEST}")

    async for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        try:
            data = msg["data"]
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", "ignore")

            payload = json.loads(data)
            logger.info(f"Recall bus request received: keys={list(payload.keys())}")

            q = _build_query_from_payload(payload, source="bus")
            result = run_recall_pipeline(q)

            trace_id = q.trace_id or payload.get("trace_id")
            reply_channel = payload.get("reply_channel") or (
                f"{settings.CHANNEL_RECALL_DEFAULT_REPLY_PREFIX}:"
                f"{trace_id or 'default'}"
            )

            reply = {
                "trace_id": trace_id,
                "source": "recall",
                "service": settings.SERVICE_NAME,
                "version": settings.SERVICE_VERSION,
                "query": {
                    "text": q.text,
                    "mode": q.mode,
                    "max_items": q.max_items,
                    "time_window_days": q.time_window_days,
                    "tags": q.tags,
                },
                "fragments": _fragments_to_dict(result.fragments),
                "debug": result.debug,
            }

            await r.publish(reply_channel, json.dumps(reply))
            logger.info(f"Recall reply published to {reply_channel} (n={len(result.fragments)})")
        except Exception as e:
            logger.exception(f"Error handling recall bus message: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bus_task
    logger.info("🧠 Orion Recall starting up…")

    if settings.ORION_BUS_ENABLED:
        bus_task = asyncio.create_task(_bus_loop())
    else:
        logger.info("Bus disabled; not starting listener.")

    yield

    logger.info("🧠 Orion Recall shutting down…")
    if bus_task:
        bus_task.cancel()
        try:
            await bus_task
        except Exception:
            pass


app = FastAPI(
    title="Orion Recall Service",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.post("/recall", response_model=RecallResponseBody)
async def recall_endpoint(body: RecallRequestBody):
    q = RecallQuery(
        text=body.text,
        max_items=body.max_items,
        time_window_days=body.time_window_days,
        mode=body.mode,
        tags=body.tags,
        phi=PhiContext(**body.phi.dict()) if body.phi else None,
        trace_id=body.trace_id,
        source="http",
    )
    result = run_recall_pipeline(q)
    return RecallResponseBody(
        fragments=_fragments_to_dict(result.fragments),
        debug=result.debug,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
