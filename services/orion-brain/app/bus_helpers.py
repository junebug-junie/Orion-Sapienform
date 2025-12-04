# app/bus_helpers.py
import json
import logging
import time
from typing import Any, Dict, Optional

from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL,
    ORION_BUS_ENABLED,
    CHANNEL_BRAIN_OUT,
    CHANNEL_BRAIN_STATUS,
    CHANNEL_BRAIN_STREAM,
    CHANNEL_CHAT_HISTORY_LOG,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Shared bus accessor
# ─────────────────────────────────────────────

def _get_bus() -> OrionBus:
    return OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)


# ─────────────────────────────────────────────
# Utility: coerce Redis payloads into dicts
# ─────────────────────────────────────────────

def _coerce_to_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize bus message data into a dict, if possible.

    Handles:
      - dict (pass-through)
      - JSON string
      - bytes -> UTF-8 -> JSON
    Returns None if it can't be parsed into an object.
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            logger.warning("Failed to decode bytes payload from bus")
            return None

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Received non-JSON string on bus: %r", raw[:200])
            return None

    logger.warning("Received unsupported payload type on bus: %r", type(raw))
    return None


# ─────────────────────────────────────────────
# Brain telemetry + chat history helpers
# ─────────────────────────────────────────────

def emit_brain_event(event_name: str, payload: Dict[str, Any]) -> None:
    """
    Fire-and-forget telemetry event for brain internals
    (routing decisions, backend selection, etc.).
    """
    bus = _get_bus()
    if not bus.enabled:
        return

    envelope = {
        "event": event_name,
        **payload,
    }

    try:
        bus.publish(CHANNEL_BRAIN_STREAM, envelope)
    except Exception as e:
        logger.error("Failed to emit brain event '%s': %s", event_name, e, exc_info=True)


def emit_brain_output(payload: Dict[str, Any]) -> None:
    """
    Optional helper if you want to fan brain outputs onto a generic channel.

    Not strictly required for Cortex/Spark, but kept for compatibility.
    """
    bus = _get_bus()
    if not bus.enabled:
        return

    try:
        bus.publish(CHANNEL_BRAIN_OUT, payload)
    except Exception as e:
        logger.error("Failed to emit brain output: %s", e, exc_info=True)


def emit_chat_history_log(payload: Dict[str, Any]) -> None:
    """
    Send a normalized chat history record to the SQL writer channel.
    """
    bus = _get_bus()
    if not bus.enabled:
        return

    try:
        bus.publish(CHANNEL_CHAT_HISTORY_LOG, payload)
    except Exception as e:
        logger.error("Failed to emit chat history log: %s", e, exc_info=True)


# ─────────────────────────────────────────────
# Cortex exec_step result helper
# ─────────────────────────────────────────────

def emit_cortex_step_result(
    *,
    service: str,
    correlation_id: str,
    reply_channel: str,
    result: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    status: str = "success",
    started_at: Optional[float] = None,
) -> None:
    """
    Standardized exec_step_result emitter for LLMGatewayService.

    This publishes onto `reply_channel` (e.g. `orion-exec:result:<uuid>`),
    and the Cortex orchestrator's `_wait_for_exec_results()` will:

      - filter by `trace_id`
      - read `service`, `ok`, `elapsed_ms`, and treat everything else as
        opaque `payload`.

    Payload shape:

      {
        "trace_id": "<correlation_id>",
        "service": "LLMGatewayService",
        "ok": true,
        "elapsed_ms": 123,
        "result": { ... },      # whatever brain produced
        "artifacts": { ... },   # optional
        "status": "success",    # or "error"
      }
    """
    bus = _get_bus()
    if not bus.enabled:
        return

    now = time.time()
    elapsed_ms = 0
    if started_at is not None:
        try:
            elapsed_ms = int((now - started_at) * 1000)
        except Exception:
            elapsed_ms = 0

    ok = status == "success"

    envelope = {
        "trace_id": correlation_id,
        "service": service,
        "ok": ok,
        "elapsed_ms": elapsed_ms,
        "result": result,
        "artifacts": artifacts or {},
        "status": status,
    }

    try:
        bus.publish(reply_channel, envelope)
        logger.info(
            "[%s] Emitted Cortex exec_step_result to %s (service=%s ok=%s elapsed_ms=%d)",
            correlation_id,
            reply_channel,
            service,
            ok,
            elapsed_ms,
        )
    except Exception as e:
        logger.error(
            "[%s] FAILED to publish Cortex exec_step_result to %s: %s",
            correlation_id,
            reply_channel,
            e,
            exc_info=True,
        )
