"""Contention-based lane fallback for real (non-broker) gateway traffic.

Real metacog/quick/agent traffic never reaches `orion/durable_admission`'s
`decide_lane()` -- it's only ever consulted for curiosity/self-study runs
submitted through orion-durable-runs. Day-to-day bus/HTTP chat traffic goes
through `lane_routes.resolve_llm_lane_route()` instead, which is a pure
route-table lookup with no occupancy awareness at all: metacog/quick each
pick their own fixed route with no way to lean on the other.

This module adds exactly one thing on top of that: once a route-table key is
chosen, check whether its upstream is already at its real concurrency limit
and, if so, try a configured fallback partner instead. "Real" concurrency
limit is deliberately a separate, per-route number (`real_capacity_map`,
matching each backend's actual llama.cpp `--parallel` slot count) rather than
`upstream_admission.py`'s shared `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT` (8 by
default) -- that cap exists to protect the gateway's own thread pool across
every lane uniformly, not to describe any one backend's real slot count
(metacog runs `--parallel 1`; quick runs `--parallel 4`), so using it as the
swap trigger would rarely fire before 8 requests piled up on a 1-slot worker.

Reuses `upstream_admission.py`'s already-live, in-process, per-upstream
`inflight` gauge as the occupancy signal -- no new I/O, no new store, no new
pairwise mutex. `plan_llm_chat()` calls this synchronously (same "cheap, no
I/O" constraint as the rest of that function).

Deliberately excludes `agent` -> `agent-burst` (review-caught, 2026-09-24):
`agent-burst` is in `orion.llm.routes.BURST_LLM_ROUTES`, and
`capacity.py::CapacityPermit.acquire()` (line ~164) unconditionally rejects
any `BURST_LLM_ROUTES` lane that lacks a durable capacity lease --
`llm_gateway_capacity_enabled` does not matter, the check is
`self.lane in BURST_LLM_ROUTES and (not self.enabled or not self.lease)`.
Ordinary cortex-exec/orion-actions/Hub traffic never carries a lease, so
swapping it onto `agent-burst` would not queue it -- it would hard-fail every
time with `gateway_capacity_rejected`, turning "busy, please wait" into "your
request cannot be served." `_never_a_swap_target()` below is the standing
guard against this (and its `chat-burst` sibling) ever being reintroduced by
a future config change, not just an omission from the default JSON. Agent's
own burst-to-agent-burst behavior for real traffic needs a real lease-
acquisition path first -- that's separate, larger work, not done here.

Known limitation, not fixed here (review-caught, 2026-09-24): this is a soft,
best-effort steering decision, not a hard admission guarantee. The inflight
snapshot this reads is a point-in-time read on the event loop; the actual
increment happens later, in `upstream_admission.py`'s `_Admission.__aenter__`,
when the request is finally admitted for dispatch (`main.py`'s
`_dispatch_chat_unfenced`). Two concurrent requests for the same contended
lane can both see "not contended" here and both get swapped (or both stay)
before either has actually been admitted -- and the semaphore that gates real
dispatch is sized to the *shared* `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT` (8), not
this module's tighter per-route `real_capacity_map` (1 for metacog), so a
burst of concurrent requests can still all be admitted past a backend's real
1-slot capacity regardless of what this module decided. Fixing that properly
means making `upstream_admission.py` itself real-capacity-aware, which is a
change to the thread-pool-protection invariant it exists for (see its own
module docstring) -- out of scope for a routing-preference patch. What this
module guarantees: a *steering* decision made on the best information
available synchronously, nothing stronger.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

from orion.llm.routes import BURST_LLM_ROUTES

from .upstream_admission import UpstreamAdmission

logger = logging.getLogger("orion-llm-gateway.lane_contention")


def _never_a_swap_target(route_key: str) -> bool:
    """A BURST_LLM_ROUTES lane (agent-burst, chat-burst) requires a durable
    capacity lease that ordinary (unleased) gateway traffic never carries --
    swapping onto one always hard-fails at CapacityPermit.acquire(), it never
    queues. Checked regardless of what LLM_LANE_CONTENTION_FALLBACK_JSON says,
    so a future config change cannot silently reintroduce the bug this guards
    against."""
    return route_key in BURST_LLM_ROUTES


@lru_cache
def _parse_fallback_map(raw_json: str) -> Dict[str, Tuple[str, ...]]:
    if not raw_json:
        return {}
    try:
        raw = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("[LLM-GW] Invalid LLM_LANE_CONTENTION_FALLBACK_JSON: %s", exc)
        return {}
    if not isinstance(raw, dict):
        logger.error("[LLM-GW] LLM_LANE_CONTENTION_FALLBACK_JSON must be a dict, got %s", type(raw))
        return {}
    out: Dict[str, Tuple[str, ...]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            partners = (value,)
        elif isinstance(value, list):
            partners = tuple(str(v) for v in value if str(v).strip())
        else:
            logger.warning("[LLM-GW] Lane '%s' fallback ignored (invalid value type %s)", key, type(value))
            continue
        dropped = [p for p in partners if _never_a_swap_target(p)]
        if dropped:
            logger.error(
                "[LLM-GW] Lane '%s' fallback lists burst-lease-gated route(s) %s -- dropped, "
                "an unleased swap onto them always hard-fails (see lane_contention.py module docstring)",
                key, dropped,
            )
        kept = tuple(p for p in partners if not _never_a_swap_target(p))
        if kept:
            out[str(key)] = kept
    return out


@lru_cache
def _parse_capacity_map(raw_json: str) -> Dict[str, int]:
    if not raw_json:
        return {}
    try:
        raw = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("[LLM-GW] Invalid LLM_LANE_REAL_CAPACITY_JSON: %s", exc)
        return {}
    if not isinstance(raw, dict):
        logger.error("[LLM-GW] LLM_LANE_REAL_CAPACITY_JSON must be a dict, got %s", type(raw))
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            n = int(value)
        except (TypeError, ValueError):
            logger.warning("[LLM-GW] Lane '%s' real-capacity value ignored (not an int): %r", key, value)
            continue
        if n > 0:
            out[str(key)] = n
    return out


def _is_contended(
    route_key: str,
    url: Optional[str],
    *,
    real_capacity_map: Mapping[str, int],
    default_capacity: int,
    gate: UpstreamAdmission,
) -> bool:
    if not url:
        return False
    capacity = real_capacity_map.get(route_key, default_capacity)
    return gate.lane(url).inflight >= capacity


def resolve_contention_fallback(
    route_key: str,
    route_urls: Mapping[str, str],
    *,
    enabled: bool,
    fallback_map: Mapping[str, Tuple[str, ...]],
    real_capacity_map: Mapping[str, int],
    default_capacity: int,
    gate: UpstreamAdmission,
) -> Tuple[str, bool]:
    """Returns (chosen_route_key, swapped). Side-effect free except reading `gate`'s
    live counters -- never acquires/holds a permit itself, that still happens where
    it always did (upstream_admission.py, on dispatch).
    """
    if not enabled or not route_key:
        return route_key, False
    partners = fallback_map.get(route_key)
    if not partners:
        return route_key, False
    if not _is_contended(
        route_key, route_urls.get(route_key),
        real_capacity_map=real_capacity_map, default_capacity=default_capacity, gate=gate,
    ):
        return route_key, False
    for partner_key in partners:
        if _never_a_swap_target(partner_key):
            # Defense in depth: _parse_fallback_map() already drops these, but a
            # caller passing fallback_map directly (as tests do) must not be able
            # to route an unleased request onto a lease-gated lane.
            continue
        partner_url = route_urls.get(partner_key)
        if partner_url is None:
            continue
        if not _is_contended(
            partner_key, partner_url,
            real_capacity_map=real_capacity_map, default_capacity=default_capacity, gate=gate,
        ):
            return partner_key, True
    # Every candidate (preferred + all partners) is contended: keep the
    # preferred lane -- upstream_admission's own wait/shed handles the rest.
    return route_key, False
