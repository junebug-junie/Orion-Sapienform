# app/bus_helpers.py
import json
import logging
import os
from typing import Any, Dict, Optional

from orion.core.bus.service import OrionBus
from app.config import ORION_BUS_URL, ORION_BUS_ENABLED

logger = logging.getLogger("bus_helpers")


def _get_bus() -> OrionBus:
    """
    Create a fresh OrionBus instance.

    We keep this tiny so that helpers can be called from anywhere
    without worrying about circular imports.
    """
    return OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)


def emit_brain_event(kind: str, payload: Dict[str, Any]) -> None:
    """
    Fire an internal brain event on the bus.

    Channel is taken from env CHANNEL_BRAIN_EVENTS (if set),
    otherwise defaults to 'orion:brain:events'.
    """
    channel = os.getenv("CHANNEL_BRAIN_EVENTS", "orion:brain:events")
    bus = _get_bus()
    if not bus.enabled:
        logger.debug(
            "[emit_brain_event] Bus disabled; skipping event kind=%s payload=%r",
            kind,
            payload,
        )
        return

    message = {
        "kind": kind,
        **(payload or {}),
    }
    logger.debug("[emit_brain_event] -> %s: %s", channel, message)
    bus.publish(channel, message)


def emit_brain_output(payload: Dict[str, Any]) -> None:
    """
    Emit a final brain output message (e.g., for a UI or downstream service).

    Channel is taken from env CHANNEL_BRAIN_OUTPUT (if set),
    otherwise defaults to 'orion:brain:output'.
    """
    channel = os.getenv("CHANNEL_BRAIN_OUTPUT", "orion:brain:output")
    bus = _get_bus()
    if not bus.enabled:
        logger.debug(
            "[emit_brain_output] Bus disabled; skipping payload=%r",
            payload,
        )
        return

    logger.debug("[emit_brain_output] -> %s: %s", channel, payload)
    bus.publish(channel, payload or {})


def emit_chat_history_log(payload: Dict[str, Any]) -> None:
    """
    Emit a chat history log row for SQL writer.

    Channel is taken from env CHANNEL_CHAT_LOG (if set),
    otherwise defaults to 'orion:chat:history:log'.

    This is what your SQL writer is already listening on via BUS_TABLE_MAP.
    """
    channel = os.getenv("CHANNEL_CHAT_LOG", "orion:chat:history:log")
    bus = _get_bus()
    if not bus.enabled:
        logger.debug(
            "[emit_chat_history_log] Bus disabled; skipping payload=%r",
            payload,
        )
        return

    logger.info(
        "[emit_chat_history_log] Publishing chat history log to %s (keys=%s)",
        channel,
        list((payload or {}).keys()),
    )
    bus.publish(channel, payload or {})


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
    Emit a standardized exec_step_result for Cortex.

    This is called by brain after handling an exec_step so the Cortex
    orchestrator can fan-in results on the orion-exec:result:* channel.

    Args:
        service:       Name of the service producing this result (e.g. "BrainLLMService")
        correlation_id: Correlation/trace id for this step (per-orchestrator)
        reply_channel:  Full result channel (e.g. "orion-exec:result:<uuid>")
        result:         Main result payload (e.g. prompt + llm_output)
        artifacts:      Optional structured extras
        status:         "success" or "error"
    """
    bus = _get_bus()
    if not bus.enabled:
        logger.error(
            "[emit_cortex_step_result] Bus disabled; cannot publish result "
            "service=%s cid=%s reply_channel=%s",
            service,
            correlation_id,
            reply_channel,
        )
        return

    envelope: Dict[str, Any] = {
        "trace_id": correlation_id,
        "service": service,
        "ok": (status == "success"),
        "status": status,
        "result": result or {},
        "artifacts": artifacts or {},
    }

    logger.info(
        "[emit_cortex_step_result] -> %s (service=%s, cid=%s, ok=%s)",
        reply_channel,
        service,
        correlation_id,
        envelope["ok"],
    )
    bus.publish(reply_channel, envelope)
