"""Tell Juniper when a seat's controller cannot act (app/controller_health.py): a Hub Pending
Attention card through orion-notify ``/attention/request`` -- the same surface the mesh guardian's
stability checks use. An unacked card escalates to email after 60 min (notify rules.yaml
``chat_attention_default``).

One card when the seat turns degraded, one ack-free info card when it recovers. Delivery failures
are logged at ERROR, never raised: an alert path must not break the pool.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import httpx

from orion.schemas.notify import ChatAttentionRequest

logger = logging.getLogger("orion-gpu-pool.controller_alert")


def build_request(seat: str, state: str, view: dict[str, Any], *, source: str) -> ChatAttentionRequest:
    host = view.get("host", "unknown")
    if state == "degraded":
        reason = f"[GPU pool] {seat} can't be loaded: lane controller on {host} is refusing"
        message = "\n".join([
            view.get("advice", ""),
            "",
            f"refusals in a row: {view.get('refusals')} (first {view.get('first_seen')}, last {view.get('last_seen')})",
            f"reason: {view.get('reason')}",
            "check: python3 scripts/gpu_pool_actuator_probe.py (status -> config_unloadable)",
        ])
        severity, require_ack = "error", True
    else:
        reason = f"[GPU pool] {seat}: lane controller on {host} answering again"
        message = (f"The lane controller on {host} is acting on {seat} again (degraded since "
                   f"{view.get('degraded_since')}, recovered {view.get('recovered_at')}).")
        severity, require_ack = "info", False
    context = {"source_service": source, "event_kind": "orion.gpu_pool.controller_health.v1",
               "event": f"controller_{state}", "seat": seat, "reason": reason,
               **{k: v for k, v in view.items() if k != "advice"}}
    return ChatAttentionRequest(source_service=source, reason=reason, severity=severity, message=message,
                                context=context, require_ack=require_ack)


def make_notify_alert(base_url: str, token: str | None, *, source: str,
                      timeout: float = 10.0) -> Callable[[str, str, dict[str, Any]], Awaitable[bool]]:
    url = f"{base_url.rstrip('/')}/attention/request"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Orion-Notify-Token"] = token

    async def alert(seat: str, state: str, view: dict[str, Any]) -> bool:
        req = build_request(seat, state, view, source=source)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=req.model_dump(mode="json"), headers=headers)
                r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("gpu_pool_controller_alert NOT delivered seat=%s state=%s url=%s error=%s",
                         seat, state, url, exc)
            return False
        logger.info("gpu_pool_controller_alert delivered seat=%s state=%s", seat, state)
        return True

    return alert
