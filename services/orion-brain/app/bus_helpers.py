# app/bus_helpers.py
from typing import Dict, Any, Optional
import logging

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
# Core bus utilities
# ─────────────────────────────────────────────

def _get_bus() -> OrionBus:
    """
    Centralized OrionBus factory.
    """
    return OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)


def _publish(channel: str, payload: Dict[str, Any], debug_prefix: Optional[str] = None) -> None:
    """
    Single point of truth for publishing to the bus from the brain service.
    """
    bus = _get_bus()
    if not bus.enabled:
        logger.debug(f"[bus] Skipping publish to {channel}: bus disabled")
        return

    try:
        bus.publish(channel, payload)
        if debug_prefix:
            logger.debug(f"[bus] {debug_prefix} -> {channel}")
    except Exception as e:
        logger.error(f"[bus] publish to {channel} failed: {e}", exc_info=True)


# ─────────────────────────────────────────────
# Legacy brain emitters
# ─────────────────────────────────────────────

def emit_brain_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Publish a Brain status or internal event.

    event_type == "status" → CHANNEL_BRAIN_STATUS
    otherwise               → CHANNEL_BRAIN_STREAM
    """
    topic = CHANNEL_BRAIN_STATUS if event_type == "status" else CHANNEL_BRAIN_STREAM
    _publish(topic, data, debug_prefix=f"brain_event:{event_type}")


def emit_brain_output(data: Dict[str, Any]) -> None:
    """
    Publish a completed LLM inference result for the legacy chat path.
    """
    _publish(CHANNEL_BRAIN_OUT, data, debug_prefix="brain_output")


def emit_chat_history_log(data: Dict[str, Any]) -> None:
    """
    Publish a record of a prompt and its response to the chat history log.
    """
    _publish(CHANNEL_CHAT_HISTORY_LOG, data, debug_prefix="chat_history")


# ─────────────────────────────────────────────
# Cortex execution helpers (semantic-layer aware)
# ─────────────────────────────────────────────

def emit_cortex_step_result(
    *,
    service: str,
    correlation_id: str,
    reply_channel: str,
    result: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    status: str = "success",
) -> None:
    """
    Publish a standardized exec_step_result envelope back to the Cortex.
    """
    envelope: Dict[str, Any] = {
        "event": "exec_step_result",
        "status": status,
        "service": service,
        "correlation_id": correlation_id,
        "result": result or {},
        "artifacts": artifacts or {},
    }

    _publish(reply_channel, envelope, debug_prefix=f"cortex_step_result:{service}")
