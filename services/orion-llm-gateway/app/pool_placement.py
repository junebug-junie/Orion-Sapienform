"""Where an LLM call runs is orion-gpu-pool's decision, not the gateway's.

Every chat call (bus RPC, OpenAI passthrough, Anthropic passthrough) takes a lease from the pool
and is sent to the URL the pool granted. The gateway keeps exactly two routing jobs:

1. turn a caller's route name into a pool work class + priority (``config/gpu_pool.yaml``
   ``routes:``). A route the YAML does not list is refused -- never guessed onto a GPU;
2. run the call on a per-role thread pool, so a flood on one GPU cannot take the threads another
   GPU's calls need (one shared executor across lanes is a hidden FIFO).

Spec: docs/superpowers/specs/2026-09-24-gpu-pool-design.md, stage 3.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional

from fastapi.responses import StreamingResponse

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.gpu_pool.client import Lease, LeaseUnavailable
from orion.gpu_pool.client import gpu_lease as _client_gpu_lease
from orion.gpu_pool.config import PoolConfig, RouteSpec, load_pool_config
from orion.llm.routes import BACKGROUND_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, SYSTEM_LLM_ROUTES
from orion.schemas.gpu_pool import (
    GpuLeaseRefV1,
    GPU_POOL_STATE_REPLY_PREFIX,
    GPU_POOL_STATE_REQUEST_CHANNEL,
    GPU_POOL_STATE_REQUEST_KIND,
    GpuPoolStateRequestV1,
)

from .settings import settings

logger = logging.getLogger("orion-llm-gateway.pool")

# Indirection so tests can swap the pool for a fake without Redis.
gpu_lease = _client_gpu_lease

ROUTE_NOT_IN_POOL = "route_not_in_gpu_pool"
POOL_UNAVAILABLE = "gpu_pool_unavailable"
# raw.error / error.type when the pool took the lease back mid-call and the upstream was stopped.
POOL_RECALLED = "gpu_pool_recalled"
POOL_UNREACHABLE = "pool_unreachable"
# The pool answers at once with this when no role of the class has a per-slot context that big;
# the suffix is the largest known ctx_per_slot in the class.
MIN_CTX_EXCEEDS_PREFIX = "min_ctx_exceeds_class:"
# After an acquire RPC times out, later calls fail fast for this long instead of each waiting
# out its own RPC timeout against a pool that is not answering.
_UNREACHABLE_CACHE_SEC = 5.0
LLAMACPP_BACKEND = "llamacpp"  # every pool llm role is a llama.cpp server

HOLDER_OPENAI = "http:openai"      # AI Town NPC traffic; cortex-exec's first-person wait cue skips http:*
HOLDER_ANTHROPIC = "http:anthropic"

_STATE_CACHE_SEC = 10.0
_STATE_FAILURE_CACHE_SEC = 2.0
_STATE_RPC_TIMEOUT_SEC = 3.0
_UP_STATUSES = frozenset({"confirmed", "static"})

_bus: Any = None


def set_bus(bus: Any) -> None:
    """The RPC-capable bus the pool client uses (a fork of the chassis bus, set at startup)."""
    global _bus
    _bus = bus


def get_bus() -> Any:
    return _bus


# ── config ──────────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def pool_config() -> PoolConfig:
    return load_pool_config(settings.gpu_pool_config_path)


def reset_pool_config_cache() -> None:
    pool_config.cache_clear()


def pool_routes() -> Dict[str, RouteSpec]:
    return dict(pool_config().routes)


class RouteNotInPool(LookupError):
    def __init__(self, route: str):
        super().__init__(f"{ROUTE_NOT_IN_POOL}:{route}")
        self.route = route


def route_spec(route: str) -> RouteSpec:
    spec = pool_routes().get(str(route))
    if spec is None:
        raise RouteNotInPool(str(route))
    return spec


def route_not_in_pool_details(route: Optional[str]) -> Dict[str, Any]:
    return {"route": route, "available_routes": sorted(pool_routes())}


def wait_budget_sec(priority: str) -> float:
    if priority == "background":
        return float(settings.llm_gateway_pool_background_wait_sec)
    return float(settings.llm_gateway_pool_wait_sec)


def passthrough_wait_sec() -> float:
    """HTTP passthrough callers hold a socket open while they wait: never the 300/900s bus budgets."""
    return float(settings.llm_gateway_pool_passthrough_wait_sec)


def class_max_ctx(reason: Optional[str]) -> Optional[int]:
    """The class's largest ctx_per_slot from a ``min_ctx_exceeds_class:<n>`` refusal, else None."""
    text = str(reason or "")
    if not text.startswith(MIN_CTX_EXCEEDS_PREFIX):
        return None
    try:
        value = int(text[len(MIN_CTX_EXCEEDS_PREFIX):])
    except ValueError:
        return None
    return value if value > 0 else None


_unreachable_until = [0.0]


def reset_pool_unreachable() -> None:
    _unreachable_until[0] = 0.0


def pool_bus_ready() -> bool:
    return _bus is not None


# ── min_ctx estimate ────────────────────────────────────────────────────────────────────────


def _text_chars(value: Any) -> int:
    """Characters of prompt text in an OpenAI/Anthropic message content (str or block list)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_text_chars(v) for v in value)
    if isinstance(value, dict):
        total = 0
        for key in ("text", "content", "thinking"):
            total += _text_chars(value.get(key))
        if "input" in value:  # tool_use arguments are prompt tokens too
            total += len(json.dumps(value.get("input"), default=str))
        return total
    content = getattr(value, "content", None)
    if content is not None:
        return _text_chars(content)
    return 0


MAX_TOKENS_PLACEMENT_CAP = 1024


def estimate_min_ctx_tokens(messages: Iterable[Any], max_tokens: Any = None, *, extra: Any = None) -> int:
    """ceil(chars / 4) over every message content, plus room for the answer.

    A floor for placement, not a token count: the pool never places the call on a role whose
    per-slot context is smaller. A real overflow is still caught and re-leased once (main.py).
    The answer term is capped at MAX_TOKENS_PLACEMENT_CAP: callers routinely ask for far more
    output than they need (Claude Code / FCC send max_tokens=32000 on every call) and llama.cpp
    simply stops at the context edge, so counting the full request would refuse or force a spill
    on work that fits today.
    """
    chars = sum(_text_chars(m) for m in (messages or [])) + _text_chars(extra)
    tokens = math.ceil(chars / 4)
    try:
        mt = int(max_tokens) if max_tokens is not None and not isinstance(max_tokens, bool) else 0
    except (TypeError, ValueError):
        mt = 0
    return tokens + min(max(0, mt), MAX_TOKENS_PLACEMENT_CAP)


# ── leases ──────────────────────────────────────────────────────────────────────────────────


class PoolLease:
    """One pool lease whose release may happen in a different task (streaming responses)."""

    def __init__(self, *, route: str, spec: RouteSpec, holder: str, turn_correlation_id: Optional[str],
                 min_ctx_tokens: int, deadline_sec: float, hold: Optional[GpuLeaseRefV1] = None) -> None:
        self.route = route
        # Stage 4: the call runs under a durable run's hold -> ``attach`` (a child lease on the
        # hold's role), never a second lease of its own that would queue behind the run.
        self.hold = hold
        self.spec = spec
        self.holder = holder
        self.turn_correlation_id = turn_correlation_id
        self.min_ctx_tokens = int(min_ctx_tokens)
        self.deadline_sec = float(deadline_sec)
        self.lease: Optional[Lease] = None
        self._cm: Any = None
        self._released = False

    async def acquire(self) -> Lease:
        bus = get_bus()
        if bus is None:
            raise LeaseUnavailable("pool_bus_unavailable")
        if self.deadline_sec <= 0:
            raise LeaseUnavailable("deadline")
        if time.monotonic() < _unreachable_until[0]:
            raise LeaseUnavailable(POOL_UNREACHABLE)
        cm = gpu_lease(
            bus, work_class=self.spec.work_class, holder=self.holder, priority=self.spec.priority,
            kind="request", deadline_sec=self.deadline_sec, min_ctx_tokens=self.min_ctx_tokens,
            turn_correlation_id=self.turn_correlation_id, **({"hold": self.hold} if self.hold is not None else {}),
        )
        try:
            self.lease = await cm.__aenter__()
        except LeaseUnavailable:
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            # The acquire RPC went unanswered. Only a FULL-length timeout says the pool is down; one
            # shortened by this caller's tiny deadline must not fail everyone else (chat) for 5s.
            if getattr(exc, "full", True):
                _unreachable_until[0] = time.monotonic() + _UNREACHABLE_CACHE_SEC
            logger.warning("gpu_pool_unreachable route=%s class=%s holder=%s (acquire RPC timed out)",
                           self.route, self.spec.work_class, self.holder)
            raise LeaseUnavailable(POOL_UNREACHABLE) from exc
        except Exception as exc:  # noqa: BLE001 -- any other RPC failure: the pool is not usable
            logger.warning("gpu_pool_lease_rpc_failed route=%s error=%s: %s", self.route, type(exc).__name__, exc)
            raise LeaseUnavailable(POOL_UNREACHABLE) from exc
        self._cm = cm
        logger.info(
            "gpu_pool_lease_granted route=%s class=%s priority=%s holder=%s role=%s url=%s ctx_per_slot=%s "
            "min_ctx=%s corr=%s hold=%s",
            self.route, self.spec.work_class, self.spec.priority, self.holder, self.lease.grant.role,
            self.lease.grant.url, self.lease.grant.ctx_per_slot, self.min_ctx_tokens, self.turn_correlation_id,
            self.hold.lease_id if self.hold is not None else "-",
        )
        if self.hold is not None and self.lease.grant.role != self.hold.role:
            # The pool places children on the hold's role only; a mismatch is a pool bug worth seeing,
            # not a reason to refuse a call the pool itself granted.
            logger.warning("gpu_pool_attach_role_mismatch hold=%s hold_role=%s granted_role=%s corr=%s",
                           self.hold.lease_id, self.hold.role, self.lease.grant.role, self.turn_correlation_id)
        return self.lease

    async def release(self, error: Optional[BaseException] = None) -> None:
        """Idempotent. ``error`` set -> the pool records ``upstream_error``."""
        if self._cm is None or self._released:
            return
        self._released = True
        if error is None:
            await self._cm.__aexit__(None, None, None)
        else:
            await self._cm.__aexit__(type(error), error, error.__traceback__)


@contextlib.asynccontextmanager
async def lease_for_route(route: str, *, holder: str, turn_correlation_id: Optional[str],
                          min_ctx_tokens: int, deadline_sec: float,
                          hold: Optional[GpuLeaseRefV1] = None) -> AsyncIterator[Lease]:
    """``async with`` form: released ``ok`` on normal exit, ``upstream_error`` if the block raises."""
    handle = PoolLease(route=route, spec=route_spec(route), holder=holder,
                       turn_correlation_id=turn_correlation_id, min_ctx_tokens=min_ctx_tokens,
                       deadline_sec=deadline_sec, hold=hold)
    lease = await handle.acquire()
    try:
        yield lease
    except BaseException as exc:
        await handle.release(exc)
        raise
    await handle.release()


async def wait_lease_revoked(lease: Lease) -> str:
    """Return once the upstream call on ``lease`` must stop: ``"lost"`` as soon as the pool no longer
    holds the lease (expired/aborted), ``"recalled"`` once a recall's grace (``recall_by``, else the
    pool's clawback_grace_sec) has run out. A recalled borrower may finish inside the grace."""
    while True:
        if lease.lost.is_set():
            return "lost"
        if lease.recalled.is_set():
            if lease.recall_by is not None:
                remaining = (lease.recall_by - datetime.now(timezone.utc)).total_seconds()
            else:
                remaining = float(pool_config().defaults.clawback_grace_sec)
            if remaining <= 0:
                return "recalled"
            try:
                await asyncio.wait_for(lease.lost.wait(), timeout=remaining)
                return "lost"
            except asyncio.TimeoutError:
                return "recalled"
        lost = asyncio.ensure_future(lease.lost.wait())
        recalled = asyncio.ensure_future(lease.recalled.wait())
        try:
            await asyncio.wait({lost, recalled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            lost.cancel()
            recalled.cancel()


def mark_revoked(lease: Lease, reason: str) -> None:
    """The release reports ``cancelled``: the gateway stopped the call because the pool took the
    slot back, not because the upstream failed (so the pool's error accounting stays about GPUs)."""
    lease.release_outcome = "cancelled"
    lease.release_detail = f"{POOL_RECALLED}:{reason}"


def pool_unavailable_error(*, reason: str, route: Optional[str], work_class: Optional[str]) -> Dict[str, Any]:
    return {"type": POOL_UNAVAILABLE, "message": f"GPU pool could not place route '{route}': {reason}",
            "reason": reason, "route": route, "work_class": work_class}


def route_not_in_pool_error(route: Optional[str]) -> Dict[str, Any]:
    return {"type": ROUTE_NOT_IN_POOL, "message": f"route '{route}' is not in config/gpu_pool.yaml routes",
            **route_not_in_pool_details(route)}


# ── streaming: hold the lease for the whole stream ────────────────────────────────────────────

_supervisors: set = set()


def _retain(task: "asyncio.Task[Any]") -> None:
    _supervisors.add(task)

    def done(finished: "asyncio.Task[Any]") -> None:
        _supervisors.discard(finished)
        if not finished.cancelled():
            finished.exception()

    task.add_done_callback(done)


def stream_cleanup(upstream: Any, client: Any, handle: Optional[PoolLease]) -> Callable[..., Awaitable[None]]:
    """One retained cleanup task: survives Starlette's repeated disconnect cancellation, closes the
    upstream and client, then releases the pool lease (``upstream_error`` if the stream failed)."""
    task: Optional["asyncio.Task[None]"] = None
    failure: List[Optional[BaseException]] = [None]

    async def clean() -> None:
        try:
            await upstream.aclose()
        finally:
            try:
                await client.aclose()
            finally:
                if handle is not None:
                    await handle.release(failure[0])

    async def close(error: Optional[BaseException] = None) -> None:
        nonlocal task
        if error is not None and failure[0] is None:
            failure[0] = error
        if task is None:
            task = asyncio.create_task(clean(), name="pool-lease-stream-cleanup")
            _retain(task)
        await asyncio.shield(task)

    return close


class LeaseStreamingResponse(StreamingResponse):
    """Release the upstream and the lease even if ASGI fails before iterating the body."""

    def __init__(self, *args: Any, cleanup: Callable[..., Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


# ── per-role executors ──────────────────────────────────────────────────────────────────────

_executors: Dict[str, ThreadPoolExecutor] = {}
_executors_lock = threading.Lock()


def executor_for(url: str) -> ThreadPoolExecutor:
    """One thread pool per granted role URL, created on first use. The pool already bounds
    concurrency to real slots; this only keeps one GPU's calls from queueing behind another's."""
    key = str(url or "").rstrip("/")
    with _executors_lock:
        pool = _executors.get(key)
        if pool is None:
            label = key.rsplit(":", 1)[-1] or "role"
            pool = ThreadPoolExecutor(max_workers=int(settings.llm_gateway_executor_workers_per_role),
                                      thread_name_prefix=f"llm-gw-{label}")
            _executors[key] = pool
        return pool


def shutdown_executors() -> None:
    with _executors_lock:
        for pool in _executors.values():
            pool.shutdown(wait=False)
        _executors.clear()


# ── GET /routes compatibility view ──────────────────────────────────────────────────────────

_state_cache: Dict[str, Any] = {"at": 0.0, "ttl": 0.0, "state": None}


def reset_pool_state_cache() -> None:
    _state_cache.update(at=0.0, ttl=0.0, state=None)


async def _request_pool_state() -> Optional[Dict[str, Any]]:
    bus = get_bus()
    if bus is None:
        return None
    reply_channel = f"{GPU_POOL_STATE_REPLY_PREFIX}{uuid.uuid4().hex}"
    env = BaseEnvelope(
        kind=GPU_POOL_STATE_REQUEST_KIND,
        source=ServiceRef(name=settings.service_name, version=settings.service_version),
        correlation_id=uuid.uuid4(), reply_to=reply_channel,
        payload=GpuPoolStateRequestV1(include_leases=False).model_dump(mode="json", exclude_defaults=True),
    )
    raw = await bus.rpc_request(GPU_POOL_STATE_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                timeout_sec=_STATE_RPC_TIMEOUT_SEC, health_label="gpu_pool_state")
    decoded = bus.codec.decode(raw["data"])  # rpc_request returns the raw pubsub message
    payload = decoded.envelope.payload if getattr(decoded, "ok", True) else None
    return payload if isinstance(payload, dict) else None


async def fetch_pool_state() -> Optional[Dict[str, Any]]:
    """The pool's latest state, cached briefly. None when the pool cannot be reached."""
    now = time.monotonic()
    if _state_cache["at"] and now - _state_cache["at"] < _state_cache["ttl"]:
        return _state_cache["state"]
    try:
        state = await _request_pool_state()
    except Exception as exc:  # noqa: BLE001 -- unreachable pool is reported as unknown, not raised
        logger.warning("gpu_pool_state_unavailable error=%s", exc)
        state = None
    _state_cache.update(at=now, ttl=_STATE_CACHE_SEC if state else _STATE_FAILURE_CACHE_SEC, state=state)
    return state


def _definitional_priority(route_id: str) -> Optional[str]:
    """The Hub picker filters on this; it is what the route IS, not the pool's queue priority."""
    if route_id in BACKGROUND_LLM_ROUTES:
        return "background"
    if route_id in SYSTEM_LLM_ROUTES:
        return "system"
    return None


def _catalog_route_ids(cfg: PoolConfig) -> List[str]:
    ordered = [r for r in LLM_ROUTE_DISPLAY_ORDER if r in cfg.routes]
    return ordered + sorted(r for r in cfg.routes if r not in ordered)


def _served_by(cfg: PoolConfig, role: str) -> str:
    return f"{cfg.host.name}-worker-{role}"  # the same label the pool puts on a grant


def _role_serves_class(cfg: PoolConfig, work_class: str, role: str, cards: Dict[str, Dict[str, Any]]) -> bool:
    spec = cfg.roles.get(role)
    if spec is None or spec.operator_only:
        return False
    if cfg.owns(work_class, role):
        return True
    # A borrower only reaches a lendable card while the operator has it lent.
    return all(bool((cards.get(card) or {}).get("lent")) for card in cfg.lendable_cards(role))


def _entry(route_id: str, *, cfg: PoolConfig, role: str, status: str, discovered: Optional[Dict[str, Any]],
           checked_at: Optional[str], gate_open: Optional[bool] = None) -> Dict[str, Any]:
    live = status == "up" and discovered is not None
    return {
        "id": route_id,
        "served_by": _served_by(cfg, role),
        "backend": LLAMACPP_BACKEND,
        "status": status,
        "latency_ms": None,
        "last_checked_at": checked_at,
        # Full path when the pool knows it (durable-runs compares against a full activation path).
        "model": (discovered.get("model_path") or discovered.get("model_file")) if live else None,
        "vision": discovered.get("vision") if live else None,
        "n_ctx": discovered.get("ctx_per_slot") if live else None,
        "priority": _definitional_priority(route_id),
        "reserved_free_slots": None,
        "upstream": (discovered or {}).get("url") or cfg.url(role),
        "gate_open": gate_open,
    }


def build_routes_compat(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``GET /routes`` in its pre-pool shape, generated from pool state. Kept for durable-runs,
    fcc_motor, situational context, context-exec and the Hub until each reads pool state."""
    cfg = pool_config()
    routes: List[Dict[str, Any]] = []
    if not isinstance(state, dict):
        for route_id in _catalog_route_ids(cfg):
            first = cfg.classes[cfg.routes[route_id].work_class].roles[0]
            routes.append(_entry(route_id, cfg=cfg, role=first, status="unknown", discovered=None, checked_at=None))
        return {"default_route": str(settings.llm_route_default or "quick"), "routes": routes}

    roles = {r.get("role"): r for r in (state.get("roles") or []) if isinstance(r, dict)}
    cards = {c.get("card"): c for c in (state.get("cards") or []) if isinstance(c, dict)}
    generated_at = state.get("generated_at")

    def up(role: str) -> bool:
        return (roles.get(role) or {}).get("status") in _UP_STATUSES

    def checked(role: str) -> Optional[str]:
        value = (roles.get(role) or {}).get("checked_at") or generated_at
        return str(value) if value is not None else None

    for route_id in _catalog_route_ids(cfg):
        work_class = cfg.routes[route_id].work_class
        if route_id == "chat-burst":
            # Durable-runs reads this until stage 4: borrowable only while Juniper has gpu0 lent.
            lent = all(bool((cards.get(c) or {}).get("lent")) for c in cfg.roles["chat"].cards)
            status = ("up" if up("chat") else "down") if lent else "operator_closed"
            routes.append(_entry(route_id, cfg=cfg, role="chat", status=status, discovered=roles.get("chat"),
                                 checked_at=checked("chat"), gate_open=lent))
            continue
        if route_id == "agent-burst":
            status = "up" if up("agent-gpu2") else "down"
            routes.append(_entry(route_id, cfg=cfg, role="agent-gpu2", status=status,
                                 discovered=roles.get("agent-gpu2"), checked_at=checked("agent-gpu2")))
            continue
        candidates = cfg.classes[work_class].roles
        chosen = next((r for r in candidates if up(r) and _role_serves_class(cfg, work_class, r, cards)), None)
        if chosen is None:
            first = candidates[0]
            routes.append(_entry(route_id, cfg=cfg, role=first, status="down", discovered=roles.get(first),
                                 checked_at=checked(first)))
        else:
            routes.append(_entry(route_id, cfg=cfg, role=chosen, status="up", discovered=roles.get(chosen),
                                 checked_at=checked(chosen)))
    return {"default_route": str(settings.llm_route_default or "quick"), "routes": routes}


async def get_routes_payload() -> Dict[str, Any]:
    return build_routes_compat(await fetch_pool_state())
