from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

from orion.bus.consumer_readiness import redis_pubsub_numsub

from .roster import ProbeConfig, ProbeMode


@dataclass(frozen=True)
class ProbeResult:
    status: Literal["probe_ok", "probe_bad"]
    reason: str | None = None
    subscriber_counts: dict[str, int] | None = None
    http_status: int | None = None


async def run_probe(*, redis, entry_probe: ProbeConfig) -> ProbeResult:
    mode = entry_probe.mode
    if mode in (ProbeMode.redis, ProbeMode.redis_and_http):
        try:
            await redis.ping()
        except Exception as exc:
            return ProbeResult(status="probe_bad", reason=f"redis_ping_failed:{exc}")

        if entry_probe.intake_channels:
            counts = await redis_pubsub_numsub(redis, entry_probe.intake_channels)
            for ch in entry_probe.intake_channels:
                if counts.get(ch, 0) <= 0:
                    return ProbeResult(
                        status="probe_bad",
                        reason=f"no_subscriber:{ch}",
                        subscriber_counts=counts,
                    )

    if mode in (ProbeMode.http, ProbeMode.redis_and_http):
        url = entry_probe.ready_url
        if not url:
            return ProbeResult(status="probe_bad", reason="missing_ready_url")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                return ProbeResult(status="probe_bad", reason=f"http_{resp.status_code}", http_status=resp.status_code)
            try:
                body = resp.json()
            except Exception:
                body = {}
            if isinstance(body, dict) and body.get("ok") is False:
                return ProbeResult(status="probe_bad", reason="http_ok_false", http_status=resp.status_code)
        except Exception as exc:
            return ProbeResult(status="probe_bad", reason=f"http_error:{exc}")

    return ProbeResult(status="probe_ok")
