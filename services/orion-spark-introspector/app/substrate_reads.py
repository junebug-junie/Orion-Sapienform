"""Fail-closed HTTP reads from orion-substrate-runtime (grammar truth + trajectory)
and orion-thought (reasoning activity)."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

_DARK_CLASSIFICATIONS = frozenset(
    {
        "dead_no_heartbeat",
        "blocked_on_event",
        "cursor_commit_failing",
    }
)


@dataclass(frozen=True)
class GrammarTruthSnapshot:
    degraded: bool
    degraded_reasons: list[str]
    enabled_reducers: dict[str, bool]
    reducer_health_by_name: dict[str, dict]


@dataclass(frozen=True)
class ExecutionTrajectorySnapshot:
    ok: bool
    projection: dict | None


@dataclass(frozen=True)
class ReasoningActivitySnapshot:
    ok: bool
    projection: dict | None


async def _response_json(resp: Any) -> dict[str, Any]:
    data = resp.json()
    if inspect.isawaitable(data):
        data = await data
    return dict(data) if isinstance(data, dict) else {}


def _grammar_http_error(exc: BaseException) -> GrammarTruthSnapshot:
    return GrammarTruthSnapshot(
        degraded=True,
        degraded_reasons=[f"http_error:{exc}"],
        enabled_reducers={},
        reducer_health_by_name={},
    )


async def fetch_grammar_truth(client: Any, url: str) -> GrammarTruthSnapshot:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = await _response_json(resp)
        return GrammarTruthSnapshot(
            degraded=bool(data.get("degraded")),
            degraded_reasons=list(data.get("degraded_reasons") or []),
            enabled_reducers={
                str(k): bool(v) for k, v in (data.get("enabled_reducers") or {}).items()
            },
            reducer_health_by_name=dict(data.get("reducer_health_by_name") or {}),
        )
    except Exception as exc:
        return _grammar_http_error(exc)


async def fetch_execution_trajectory(client: Any, url: str) -> ExecutionTrajectorySnapshot:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = await _response_json(resp)
        ok = bool(data.get("ok"))
        projection = data.get("projection")
        if projection is not None and not isinstance(projection, dict):
            projection = None
        return ExecutionTrajectorySnapshot(ok=ok, projection=projection)
    except Exception:
        return ExecutionTrajectorySnapshot(ok=False, projection=None)


async def fetch_reasoning_activity(client: Any, url: str) -> ReasoningActivitySnapshot:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = await _response_json(resp)
        ok = bool(data.get("ok"))
        projection = data.get("projection")
        if projection is not None and not isinstance(projection, dict):
            projection = None
        return ReasoningActivitySnapshot(ok=ok, projection=projection)
    except Exception:
        return ReasoningActivitySnapshot(ok=False, projection=None)


def cognitive_lane_dark(snapshot: GrammarTruthSnapshot) -> bool:
    if not snapshot.enabled_reducers.get("execution_trajectory"):
        return True
    health = snapshot.reducer_health_by_name.get("execution_trajectory") or {}
    return health.get("classification") in _DARK_CLASSIFICATIONS


class SubstrateReadCache:
    """Monotonic-TTL cache for substrate HTTP read payloads."""

    def __init__(self, ttl_sec: float) -> None:
        self._ttl_sec = ttl_sec
        self._grammar: dict | None = None
        self._grammar_at: float | None = None
        self._trajectory: dict | None = None
        self._trajectory_at: float | None = None
        self._reasoning_activity: dict | None = None
        self._reasoning_activity_at: float | None = None

    def put_grammar(self, value: dict) -> None:
        self._grammar = value
        self._grammar_at = time.monotonic()

    def get_grammar(self) -> dict | None:
        if self._grammar is None or self._grammar_at is None:
            return None
        if time.monotonic() - self._grammar_at > self._ttl_sec:
            return None
        return self._grammar

    def put_trajectory(self, value: dict) -> None:
        self._trajectory = value
        self._trajectory_at = time.monotonic()

    def get_trajectory(self) -> dict | None:
        if self._trajectory is None or self._trajectory_at is None:
            return None
        if time.monotonic() - self._trajectory_at > self._ttl_sec:
            return None
        return self._trajectory

    def put_reasoning_activity(self, value: dict) -> None:
        self._reasoning_activity = value
        self._reasoning_activity_at = time.monotonic()

    def get_reasoning_activity(self) -> dict | None:
        if self._reasoning_activity is None or self._reasoning_activity_at is None:
            return None
        if time.monotonic() - self._reasoning_activity_at > self._ttl_sec:
            return None
        return self._reasoning_activity
