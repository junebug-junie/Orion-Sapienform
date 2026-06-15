from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict


def _decode_redis_val(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "replace")
    return x


class BusConsumerReadinessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    http_alive: Optional[bool] = None
    bus_consumer_ready: bool
    intake_channel: str
    subscriber_count: int
    heartbeat_fresh: Optional[bool] = None
    rpc_smoke_ok: Optional[bool] = None
    dependency_status: Literal["available", "unavailable"]
    error: Optional[str] = None


async def redis_pubsub_numsub(redis, channels: list[str]) -> dict[str, int]:
    if not channels:
        return {}
    raw = await redis.execute_command("PUBSUB", "NUMSUB", *channels)
    out: dict[str, int] = {}
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


def _parse_last_seen_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _heartbeat_matches_service(raw: dict[str, Any], *, service_name: str) -> bool:
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return False
    if str(payload.get("service") or "") != service_name:
        return False
    kind = raw.get("kind")
    if kind is not None and kind != "system.health.v1":
        return False
    return True


async def check_heartbeat_fresh(
    redis,
    *,
    service_name: str,
    health_channel: str,
    ttl_sec: float,
    listen_timeout_sec: float = 1.5,
) -> bool:
    pubsub = redis.pubsub()
    try:
        await pubsub.subscribe(health_channel)
        deadline = asyncio.get_running_loop().time() + float(listen_timeout_sec)
        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            msg = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=min(0.2, remaining),
            )
            if not msg or msg.get("type") != "message":
                continue
            data = msg.get("data")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", "replace")
            try:
                raw = json.loads(data) if isinstance(data, str) else {}
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict) or not _heartbeat_matches_service(raw, service_name=service_name):
                continue
            payload = raw.get("payload") or {}
            last_seen = _parse_last_seen_ts(payload.get("last_seen_ts") if isinstance(payload, dict) else None)
            if last_seen is None:
                continue
            age_sec = (datetime.now(timezone.utc) - last_seen).total_seconds()
            if age_sec <= float(ttl_sec):
                return True
        return False
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(health_channel)
        with contextlib.suppress(Exception):
            await pubsub.close()


async def check_bus_consumer_readiness(
    redis,
    *,
    intake_channel: str,
    service_name: str | None = None,
    health_channel: str = "orion:system:health",
    heartbeat_ttl_sec: float = 30.0,
    check_heartbeat: bool = True,
    rpc_smoke_fn: Callable[[], Awaitable[bool]] | None = None,
    rpc_timeout_sec: float = 2.0,
) -> BusConsumerReadinessResult:
    base = {
        "intake_channel": intake_channel,
        "subscriber_count": 0,
        "bus_consumer_ready": False,
        "dependency_status": "unavailable",
        "ok": False,
    }
    try:
        pong = await redis.ping()
        ping_ok = pong is True or pong == b"PONG" or str(pong).upper() == "PONG"
        if not ping_ok:
            return BusConsumerReadinessResult(
                **base,
                error="redis ping failed",
            )
    except Exception as exc:
        return BusConsumerReadinessResult(
            **base,
            error=str(exc),
        )

    try:
        numsub = await redis_pubsub_numsub(redis, [intake_channel])
    except Exception as exc:
        return BusConsumerReadinessResult(
            **base,
            error=f"pubsub numsub failed: {exc}",
        )

    subscriber_count = int(numsub.get(intake_channel, -1))
    bus_consumer_ready = subscriber_count > 0
    heartbeat_fresh: Optional[bool] = None
    rpc_smoke_ok: Optional[bool] = None
    error: Optional[str] = None

    if check_heartbeat and service_name:
        try:
            heartbeat_fresh = await check_heartbeat_fresh(
                redis,
                service_name=service_name,
                health_channel=health_channel,
                ttl_sec=heartbeat_ttl_sec,
            )
        except Exception as exc:
            heartbeat_fresh = False
            error = f"heartbeat check failed: {exc}"

    if rpc_smoke_fn is not None:
        try:
            rpc_smoke_ok = bool(
                await asyncio.wait_for(rpc_smoke_fn(), timeout=float(rpc_timeout_sec))
            )
        except Exception:
            rpc_smoke_ok = False

    ok = bus_consumer_ready
    if heartbeat_fresh is False:
        ok = False
    if rpc_smoke_ok is False:
        ok = False

    if not bus_consumer_ready and error is None:
        error = f"no subscribers on intake channel: {intake_channel}"

    return BusConsumerReadinessResult(
        ok=ok,
        bus_consumer_ready=bus_consumer_ready,
        intake_channel=intake_channel,
        subscriber_count=subscriber_count,
        heartbeat_fresh=heartbeat_fresh,
        rpc_smoke_ok=rpc_smoke_ok,
        dependency_status="available",
        error=error,
    )
