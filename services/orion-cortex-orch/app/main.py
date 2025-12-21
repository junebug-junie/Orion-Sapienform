import logging
import sys
import threading
from typing import Any, Dict

import orjson
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from uuid import uuid4

from orion.core.bus.service import OrionBus
from .orchestrator import (
    OrchestrateVerbRequest,
    OrchestrateVerbResponse,
    run_cortex_verb,
)
from .settings import get_settings


class ORJSONResponse(JSONResponse):
    """
    FastAPI response class using orjson for speed and nice indentation.
    """

    media_type = "application/json"

    def render(self, content: Dict) -> bytes:
        return orjson.dumps(
            content,
            option=orjson.OPT_INDENT_2
            | orjson.OPT_SORT_KEYS
            | orjson.OPT_NON_STR_KEYS,
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger("orion-cortex-orchestrator")

bus = OrionBus(
    url=settings.orion_bus_url,
    enabled=settings.orion_bus_enabled,
)

app = FastAPI(
    title="Orion Cortex Orchestrator",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "node_name": settings.node_name,
        "bus_enabled": bus.enabled,
        "exec_request_prefix": settings.exec_request_prefix,
        "exec_result_prefix": settings.exec_result_prefix,
        "cortex_orch_request_channel": settings.cortex_orch_request_channel,
        "cortex_orch_result_prefix": settings.cortex_orch_result_prefix,
    }


@app.post("/orchestrate", response_model=OrchestrateVerbResponse)
def orchestrate(req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
    """
    HTTP entrypoint for running a Cortex verb.

    This is synchronous and blocks until all configured services respond per step
    (or time out).
    """
    logger.info(
        "Orchestrating verb=%s from origin_node=%s with %d step(s)",
        req.verb_name,
        req.origin_node,
        len(req.steps) if req.steps else 0,
    )
    return run_cortex_verb(bus, req)


# -------------------------------
# Bus-driven orchestrator worker
# -------------------------------


# In services/orion-cortex-orch/app/main.py

def _cortex_orch_bus_worker() -> None:
    """
    Background worker that listens on CORTEX_ORCH_REQUEST_CHANNEL.
    Includes robust unpacking for nested 'payload' envelopes.
    """
    if not bus.enabled:
        logger.warning(
            "Bus is disabled; Cortex orchestrator bus worker will not start."
        )
        return

    channel = settings.cortex_orch_request_channel
    logger.info(
        "Starting Cortex orchestrator bus worker on channel '%s'", channel
    )

    for msg in bus.raw_subscribe(channel):
        envelope = msg.get("data") or {}

        # 1. Extract Metadata from Envelope
        trace_id = envelope.get("trace_id") or str(uuid4())
        result_channel = envelope.get("result_channel") or (
            f"{settings.cortex_orch_result_prefix}:{trace_id}"
        )

        logger.info(
            "Received bus orchestrate request on %s (trace_id=%s, result_channel=%s)",
            channel,
            trace_id,
            result_channel,
        )

        # 2. Smart Unpack: Handle nested "payload" vs top-level keys
        # If 'verb_name' is missing at top level but exists inside 'payload', flatten it.
        request_data = envelope
        if "verb_name" not in request_data and "payload" in request_data:
            possible_inner = request_data["payload"]
            if isinstance(possible_inner, dict) and "verb_name" in possible_inner:
                logger.info("Unpacking nested payload for trace_id=%s", trace_id)
                # Merge envelope keys (like origin_node) with inner payload keys
                request_data = {**request_data, **possible_inner}

        try:
            # 3. Validate against Pydantic Model
            req = OrchestrateVerbRequest(**request_data)
        except ValidationError as ve:
            logger.error(
                "Validation error in bus-driven orchestrate (trace_id=%s): %s",
                trace_id,
                ve,
            )
            bus.publish(
                result_channel,
                {
                    "trace_id": trace_id,
                    "ok": False,
                    "kind": "cortex_orchestrate_error",
                    "error_type": "validation_error",
                    "errors": ve.errors(),
                    "raw_payload": envelope, # Return original envelope for debug
                },
            )
            continue
        except Exception as e:
            logger.exception(
                "Unexpected error parsing orchestrate payload (trace_id=%s)", trace_id
            )
            bus.publish(
                result_channel,
                {
                    "trace_id": trace_id,
                    "ok": False,
                    "kind": "cortex_orchestrate_error",
                    "error_type": "parse_exception",
                    "message": str(e),
                },
            )
            continue

        # 4. Execute
        try:
            resp = run_cortex_verb(bus, req)
            bus.publish(
                result_channel,
                {
                    "trace_id": trace_id,
                    "ok": True,
                    "kind": "cortex_orchestrate_result",
                    **resp.model_dump(mode="json"),
                },
            )
        except Exception as e:
            logger.exception(
                "Unhandled error during bus-driven orchestrate (trace_id=%s)",
                trace_id,
            )
            bus.publish(
                result_channel,
                {
                    "trace_id": trace_id,
                    "ok": False,
                    "kind": "cortex_orchestrate_error",
                    "error_type": "execution_exception",
                    "message": str(e),
                },
            )



@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "Cortex Orchestrator starting; HTTP on %s:%s, bus channel=%s",
        settings.api_host,
        settings.api_port,
        settings.cortex_orch_request_channel,
    )

    # Spawn the bus worker in a daemon thread so it doesn't block the event loop.
    worker_thread = threading.Thread(
        target=_cortex_orch_bus_worker,
        name="cortex-orchestrator-bus-worker",
        daemon=True,
    )
    worker_thread.start()
