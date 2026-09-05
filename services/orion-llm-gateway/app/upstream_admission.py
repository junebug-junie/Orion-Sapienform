"""Per-upstream in-flight caps for the bus chat path.

Incident 2026-09-05 (docs/superpowers/pr-reports/2026-09-05-stance-react-attention-
salience-gateway-starvation-incident.md): every bus request was dispatched with
`asyncio.to_thread(run_llm_chat, body)` onto the ONE default executor (32 threads
here), and each thread was held for the request's whole life -- the upstream HTTP
read included, up to READ_TIMEOUT_SEC (700s). When `quick`'s 4-slot llama.cpp
worker fell behind, `orion-topic-foundry` / memory-consolidation traffic to that
one upstream filled all 32 threads, and every later request -- including an
interactive `stance_react` turn bound for the idle `chat` worker, and `metacog`
calls bound for an idle `metacog` worker -- queued FIFO in the executor behind
them. Measured over 05:00-08:00Z: median 21 minutes from receipt to dispatch on
`quick`, 19 minutes on `metacog`, ~300 requests received that never completed
inside the window, and the GPU spent those minutes generating answers for callers
that had already timed out.

Two invariants, both enforced here and nowhere else:

1. Isolation. One upstream's saturation may consume only its own share of the
   thread pool. Each distinct upstream URL gets its own in-flight cap
   (`LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`), and main.py sizes the executor to the
   sum of those caps, so a request for an idle upstream always finds a thread.
2. No work for callers that already left. A request waits for its upstream's
   permit only as long as its own read-timeout budget allows. Past that, it is
   shed with a `gateway_overloaded` error instead of running a generation whose
   reply nobody is listening for -- which is what turned a busy lane into a
   20-minute backlog in the first place.

Waiting happens on an asyncio semaphore, off the thread pool, so a deep queue
costs no threads. `snapshot()` is exposed on `GET /admission` under "upstreams"
so the depth of every lane is visible live instead of inferred from thread
counts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

LEGACY_UPSTREAM = "legacy"  # no route table configured: one shared lane


@dataclass
class _LaneStats:
    max_inflight: int
    inflight: int = 0
    waiting: int = 0
    admitted: int = 0
    shed: int = 0
    longest_wait_s: float = 0.0
    last_shed_at: Optional[float] = None
    sem: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sem = asyncio.Semaphore(self.max_inflight)

    def public(self) -> Dict[str, Any]:
        return {
            "max_inflight": self.max_inflight,
            "inflight": self.inflight,
            "waiting": self.waiting,
            "admitted": self.admitted,
            "shed": self.shed,
            "longest_wait_s": round(self.longest_wait_s, 3),
            "last_shed_age_s": (
                None if self.last_shed_at is None else round(time.time() - self.last_shed_at, 1)
            ),
        }


class UpstreamAdmission:
    """One in-flight gate per upstream URL, created lazily, process lifetime."""

    def __init__(self, max_inflight: int) -> None:
        self.max_inflight = max(1, int(max_inflight))
        self._lanes: Dict[str, _LaneStats] = {}

    def lane(self, upstream: str) -> _LaneStats:
        key = upstream or LEGACY_UPSTREAM
        lane = self._lanes.get(key)
        if lane is None:
            lane = _LaneStats(max_inflight=self.max_inflight)
            self._lanes[key] = lane
        return lane

    def admit(self, upstream: str, *, max_wait_s: float) -> "_Admission":
        return _Admission(self.lane(upstream), max_wait_s=max_wait_s)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {key: lane.public() for key, lane in sorted(self._lanes.items())}

    def executor_workers(self, upstream_count: int, *, headroom: int = 4) -> int:
        """Threads the default executor needs so every lane can be full at once.

        `headroom` covers the chassis' own `to_thread` use (heartbeat details) so a
        fully loaded gateway still heartbeats.
        """
        return max(1, int(upstream_count)) * self.max_inflight + max(0, int(headroom))


class _Admission:
    """`async with gate.admit(url, max_wait_s=budget) as admitted:` -- `admitted` is
    False when the permit did not arrive inside the budget; the caller must then
    not dispatch."""

    def __init__(self, lane: _LaneStats, *, max_wait_s: float) -> None:
        self._lane = lane
        self._max_wait_s = max(0.0, float(max_wait_s))
        self._holding = False
        self.waited_s = 0.0
        # Exact, like priority_admission's queued flag: whether the acquire had to block.
        # A monotonic delta is never exactly 0, so "waited > 0" would log every request.
        self.queued = False

    async def __aenter__(self) -> bool:
        lane = self._lane
        started = time.monotonic()
        self.queued = lane.sem.locked()
        lane.waiting += 1
        try:
            try:
                async with asyncio.timeout(self._max_wait_s):
                    await lane.sem.acquire()
            except TimeoutError:
                self.waited_s = time.monotonic() - started
                lane.shed += 1
                lane.last_shed_at = time.time()
                return False
        finally:
            lane.waiting -= 1
        self.waited_s = time.monotonic() - started
        self._holding = True
        lane.inflight += 1
        lane.admitted += 1
        if self.queued and self.waited_s > lane.longest_wait_s:
            lane.longest_wait_s = self.waited_s
        return True

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._holding:
            self._holding = False
            self._lane.inflight -= 1
            self._lane.sem.release()


_gate: Optional[UpstreamAdmission] = None


def get_upstream_admission() -> UpstreamAdmission:
    """Process-wide gate, built from settings on first use."""
    global _gate
    if _gate is None:
        from .settings import settings

        _gate = UpstreamAdmission(int(settings.llm_gateway_upstream_max_inflight))
    return _gate


def reset_upstream_admission_for_tests() -> None:
    global _gate
    _gate = None
