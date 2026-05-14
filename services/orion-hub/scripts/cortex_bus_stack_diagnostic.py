# services/orion-hub/scripts/cortex_bus_stack_diagnostic.py
"""Redis + Hub→Cortex gateway bus diagnostics (PING, PUBSUB NUMSUB, optional RPC probe)."""
from __future__ import annotations

import logging
import uuid
from time import perf_counter
from typing import Any, Dict, List, Optional

from redis import asyncio as aioredis

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult

logger = logging.getLogger("orion-hub.cortex_bus_stack_diagnostic")


def _decode_redis_val(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "replace")
    return x


async def _redis_pubsub_numsub(redis: aioredis.Redis, channels: List[str]) -> Dict[str, int]:
    if not channels:
        return {}
    raw = await redis.execute_command("PUBSUB", "NUMSUB", *channels)
    out: Dict[str, int] = {}
    if not isinstance(raw, (list, tuple)):
        return out
    for i in range(0, len(raw), 2):
        ch = str(_decode_redis_val(raw[i]))
        try:
            cnt = int(raw[i + 1])
        except (IndexError, TypeError, ValueError):
            cnt = -1
        out[ch] = cnt
    return out


def _rpc_result_summary(result: CortexChatResult) -> Dict[str, Any]:
    cr = result.cortex_result
    if cr is None:
        return {"cortex_result": None}
    err = cr.error if isinstance(cr.error, dict) else None
    return {
        "ok": bool(cr.ok),
        "status": cr.status,
        "verb": cr.verb,
        "mode": cr.mode,
        "final_text_len": len(str(cr.final_text or "")),
        "final_text_head": str(cr.final_text or "")[:200],
        "error_type": (err or {}).get("type") if err else None,
        "error_message_head": str((err or {}).get("message") or "")[:200] if err else None,
    }


async def run_cortex_bus_stack_diagnostic(
    *,
    redis_url: str,
    gateway_request_channel: str,
    gateway_result_prefix: str,
    orch_request_channel: str,
    orch_result_prefix: str,
    rpc_timeout_sec: float = 45.0,
    run_rpc: bool = True,
    verb: str = "skills.system.time_now.v1",
    prompt: str = "What time is it right now?",
    service_name: str = "hub-bus-diagnostic",
    service_version: str = "0.0.0",
    node_name: str = "diagnostic",
    own_bus: Optional[OrionBusAsync] = None,
    enforce_catalog: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    `own_bus`: when provided (e.g. Hub's connected global bus), reuse it for PING/NUMSUB/RPC so
    behavior matches production. When None, a dedicated short-lived client is used.
    """
    started = perf_counter()
    out: Dict[str, Any] = {
        "schema": "cortex_bus_stack_diagnostic.v1",
        "redis_url_redacted": _redact_redis_url(redis_url),
        "channels": {
            "gateway_request": gateway_request_channel,
            "gateway_result_prefix": gateway_result_prefix,
            "orch_request": orch_request_channel,
            "orch_result_prefix": orch_result_prefix,
        },
    }
    bus = own_bus
    created: Optional[OrionBusAsync] = None
    if bus is None:
        created = OrionBusAsync(redis_url, enforce_catalog=enforce_catalog)
        bus = created

    try:
        await bus.connect()
        r = bus.redis
        try:
            pong = await r.ping()
        except Exception as exc:
            out["redis_ping_ok"] = False
            out["redis_ping_error"] = str(exc)
            out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
            return out

        out["redis_ping_ok"] = pong is True or pong == b"PONG" or str(pong).upper() == "PONG"
        out["pubsub_subscribers"] = await _redis_pubsub_numsub(
            r,
            [gateway_request_channel, orch_request_channel],
        )

        gw_n = int(out["pubsub_subscribers"].get(gateway_request_channel, -1))
        orch_n = int(out["pubsub_subscribers"].get(orch_request_channel, -1))
        out["hints"] = []
        if gw_n == 0:
            out["hints"].append(
                "No Redis subscribers on the gateway request channel: orion-cortex-gateway "
                "consumer is likely down or pointed at a different Redis URL/DB index."
            )
        if orch_n == 0:
            out["hints"].append(
                "No Redis subscribers on the orchestrator request channel: cortex orchestrator "
                "is likely down or misconfigured; gateway would accept Hub traffic then stall on orch RPC."
            )

        if not run_rpc:
            out["rpc"] = {"skipped": True}
            out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
            return out

        corr = str(uuid.uuid4())
        reply_to = f"{gateway_result_prefix}:{corr}"
        chat_req = CortexChatRequest(
            prompt=prompt,
            messages=[{"role": "user", "content": prompt}],
            mode="brain",
            route_intent="none",
            verb=verb,
            packs=[],
            options={"force_agent_chain": False},
            recall={"enabled": False, "required": False},
            session_id="diagnostic-session",
            user_id="diagnostic-user",
            trace_id=f"bus-stack-{corr}",
            metadata={"source": "cortex_bus_stack_diagnostic"},
        )
        envelope = BaseEnvelope(
            kind="cortex.gateway.chat.request",
            source=ServiceRef(name=service_name, version=service_version, node=node_name),
            correlation_id=corr,
            reply_to=reply_to,
            payload=chat_req.model_dump(),
        )
        rpc_started = perf_counter()
        try:
            msg = await bus.rpc_request(
                gateway_request_channel,
                envelope,
                reply_channel=reply_to,
                timeout_sec=float(rpc_timeout_sec),
            )
        except TimeoutError as te:
            out["rpc"] = {
                "ok": False,
                "error": str(te),
                "correlation_id": corr,
                "reply_channel": reply_to,
                "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
            }
            out["hints"].append(
                f"RPC timed out after {rpc_timeout_sec}s waiting on the gateway result channel. "
                "If gateway subscriber count is >=1, the gateway is likely wedged or waiting on orch."
            )
            out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
            return out
        except Exception as exc:
            out["rpc"] = {
                "ok": False,
                "error": str(exc),
                "correlation_id": corr,
                "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
            }
            out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
            return out

        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            out["rpc"] = {
                "ok": False,
                "decode_ok": False,
                "decode_error": decoded.error,
                "correlation_id": corr,
                "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
            }
            out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
            return out

        payload = decoded.envelope.payload
        if not isinstance(payload, dict):
            out["rpc"] = {
                "ok": False,
                "error": f"unexpected_payload_type:{type(payload).__name__}",
                "correlation_id": corr,
                "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
            }
        else:
            try:
                parsed = CortexChatResult.model_validate(payload)
                out["rpc"] = {
                    "ok": True,
                    "correlation_id": corr,
                    "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
                    "result": _rpc_result_summary(parsed),
                }
            except Exception as ve:
                out["rpc"] = {
                    "ok": False,
                    "error": f"result_validation:{ve}",
                    "correlation_id": corr,
                    "elapsed_ms": round((perf_counter() - rpc_started) * 1000.0, 2),
                }

        out["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 2)
        return out
    finally:
        if created is not None:
            try:
                await created.close()
            except Exception:
                logger.debug("diagnostic_bus_close_failed", exc_info=True)


def _redact_redis_url(url: str) -> str:
    if "@" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        _hostpart = rest.split("@", 1)[1]
        return f"{scheme}://***@{_hostpart}"
    except Exception:
        return "redis://***"
