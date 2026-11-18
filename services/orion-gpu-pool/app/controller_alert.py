"""Tell Juniper when a seat's controller cannot act (orion/gpu_pool/controller_health.py): a Hub Pending
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


def build_fault_request(card: str, state: str, view: dict[str, Any], *, source: str) -> ChatAttentionRequest:
    """A card stuck in swap fault (runtime._check_fault_alerts). Critical: orion-notify emails
    critical at once (anything lower waits 60 min unacked). A faulted card gets no grants on any of
    its roles -- on gpu2 that is every painting and the agent-gpu2 seat."""
    seat = view.get("seat") or "unknown"
    if state == "faulted":
        reason = f"[GPU pool] {card} stuck in fault for {view.get('minutes')} min -- no work runs on it"
        message = "\n".join([
            f"The GPU pool has given {card} no work for {view.get('minutes')} minutes (since "
            f"{view.get('since')}). On gpu2 that means no paintings and no agent-gpu2 seat.",
            f"Swap seat: {seat}. Last action: {view.get('outcome')} -- {view.get('reason')}.",
            f"Containers seen: {view.get('observed')}.",
            "Self-repair already tried and could not decide which side holds the card.",
            "Check on circe: docker ps; nvidia-smi --query-compute-apps=pid,used_memory --format=csv",
            "Then stop the container that holds no GPU memory, or send the pool "
            f"GpuPoolControlV1 verb=clear_fault card={card} (Hub button on the card).",
        ])
        severity, require_ack = "critical", True
    else:
        reason = f"[GPU pool] {card} out of fault"
        message = f"{card} left fault at {view.get('cleared_at')} (seat {seat}); the pool schedules on it again."
        severity, require_ack = "info", False
    context = {"source_service": source, "event_kind": "orion.gpu_pool.card_fault.v1",
               "event": f"card_{state}", "card": card, "reason": reason,
               **{k: v for k, v in view.items() if k != "observed"}}
    return ChatAttentionRequest(source_service=source, reason=reason, severity=severity, message=message,
                                context=context, require_ack=require_ack)


def build_perception_request(name: str, state: str, view: dict[str, Any], *, source: str) -> ChatAttentionRequest:
    """Perception (vision-council) GPU load over its ceiling (runtime._check_perception_load). Perception is
    never shed by heat (body-economy design A0), so its own load is the thing watched."""
    if state == "high":
        reason = (f"[GPU pool] Orion's sight used {view.get('held_sec_last_hour')} GPU slot-seconds in the last "
                  f"hour (ceiling {view.get('ceiling_sec')})")
        message = "\n".join([
            "Perception leases (route metacog_perception: vision-council) are exempt from the heat shed.",
            "Their last-hour GPU hold is above 2x their normal p90. Something may be calling the council in a loop,",
            "or a new camera stream was added. Check: gpu_pool_events where priority='perception', by holder;",
            "vision-council logs for [COUNCIL] transition reasons.",
        ])
        # warning, not error: the 3360 ceiling was measured while vision-council was being shed by heat
        # (25,430 refusals in that week), so unshed load may run higher. A Hub card, no email, until the
        # ceiling is re-baselined on a week of perception-priority data.
        severity, require_ack = "warning", True
    else:
        reason = "[GPU pool] Orion's sight is back under its GPU ceiling"
        message = f"Perception's last-hour GPU hold is {view.get('held_sec_last_hour')} s (ceiling {view.get('ceiling_sec')})."
        severity, require_ack = "info", False
    context = {"source_service": source, "event_kind": "orion.gpu_pool.perception_load.v1",
               "event": f"perception_{state}", "reason": reason, **view}
    return ChatAttentionRequest(source_service=source, reason=reason, severity=severity, message=message,
                                context=context, require_ack=require_ack)


def make_notify_alert(base_url: str, token: str | None, *, source: str, timeout: float = 10.0,
                      builder: Callable[..., ChatAttentionRequest] = build_request,
                      ) -> Callable[[str, str, dict[str, Any]], Awaitable[bool]]:
    url = f"{base_url.rstrip('/')}/attention/request"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Orion-Notify-Token"] = token

    async def alert(seat: str, state: str, view: dict[str, Any]) -> bool:
        req = builder(seat, state, view, source=source)
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
