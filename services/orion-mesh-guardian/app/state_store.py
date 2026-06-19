from __future__ import annotations

import json

from .state_machine import ServicePhase, ServiceState

STATE_HASH_KEY = "mesh-guardian:state"


def _state_to_dict(state: ServiceState) -> dict:
    return {
        "phase": state.phase.value,
        "consecutive_probe_fails": state.consecutive_probe_fails,
        "last_remediate_ts": state.last_remediate_ts,
        "attempts_this_hour": state.attempts_this_hour,
        "hour_window_start_ts": state.hour_window_start_ts,
        "post_grace_until_ts": state.post_grace_until_ts,
        "correlation_id": state.correlation_id,
        "pending_tier2": state.pending_tier2,
    }


def _state_from_dict(data: dict) -> ServiceState:
    return ServiceState(
        phase=ServicePhase(data.get("phase", ServicePhase.healthy.value)),
        consecutive_probe_fails=int(data.get("consecutive_probe_fails", 0)),
        last_remediate_ts=data.get("last_remediate_ts"),
        attempts_this_hour=int(data.get("attempts_this_hour", 0)),
        hour_window_start_ts=float(data.get("hour_window_start_ts", 0)),
        post_grace_until_ts=data.get("post_grace_until_ts"),
        correlation_id=data.get("correlation_id"),
        pending_tier2=bool(data.get("pending_tier2", False)),
    )


async def load_all(redis) -> dict[str, ServiceState]:
    raw = await redis.hgetall(STATE_HASH_KEY)
    out: dict[str, ServiceState] = {}
    for key, val in (raw or {}).items():
        k = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        text = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
        try:
            out[k] = _state_from_dict(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            continue
    return out


async def save_one(redis, service_id: str, state: ServiceState) -> None:
    await redis.hset(STATE_HASH_KEY, service_id, json.dumps(_state_to_dict(state)))
