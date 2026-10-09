"""stance_context_prepare: build stance_react's context while orion-mind runs.

Unified-turn latency L4 (docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md).

orion-thought sends ``stance_context_prepare`` at the same time as its orion-mind
call, on the prepare channel matching the exec lane it will send stance_react on
(``orion.schemas.stance_context_prepare.stance_context_prepare_channel``), so
both land on this container. The handler builds the brain reply context on a
ctx built exactly as the stance_react request will build it, and caches the ctx
changes the build made, keyed by correlation id.

When stance_react arrives with ``ctx["stance_prepare_requested"]``,
``prepare_brain_reply_context`` calls :func:`take_prepared_stance_context`:

* prepare done            -> apply the cached ctx changes, no build;
* prepare still building  -> await it (never start a second build: that would
  bring back the duplicate chat_stance_belief_log / cortex_turn rows and a
  second chat-lane probe call);
* prepare absent          -> wait up to 2 s for it to arrive, then build inline
  and mark the correlation id abandoned so a late prepare does not build too;
* prepare failed          -> build inline.

Each entry is used once and expires after 120 s. The cache is per process: a
container restart between the two requests means an inline build, nothing worse.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.stance_context_prepare import (
    STANCE_CONTEXT_PREPARE_RESULT_KIND,
    STANCE_PREPARE_REQUESTED_CTX_KEY,
    StanceContextPrepareRequestV1,
    StanceContextPrepareResultV1,
)

from .exec_ctx import build_exec_ctx
from .settings import settings

logger = logging.getLogger("orion.cortex.stance_prepare")

CACHE_TTL_SEC = 120.0
# How long stance_react waits for a prepare that has not reached this container.
ABSENT_WAIT_SEC = 2.0
# Upper bound on awaiting an in-flight build. A build that has not finished by
# then is hung on the single stance worker; an inline build would queue behind
# it anyway, so this only bounds how long stance_react sits before trying.
INFLIGHT_WAIT_SEC = 60.0

# ctx keys never carried from the prepare ctx to the stance ctx: per-request
# identity and the per-run grammar collector.
_NON_TRANSFERABLE_KEYS = frozenset(
    {
        "correlation_id",
        "trace_id",
        "parent_event_id",
        "trigger_correlation_id",
        "trigger_trace_id",
        "_cortex_exec_grammar_collector",
        STANCE_PREPARE_REQUESTED_CTX_KEY,
    }
)


# ---------------------------------------------------------------------------
# ctx delta: what the build changed
# ---------------------------------------------------------------------------


@dataclass
class StanceCtxDelta:
    changed: Dict[str, Any]
    removed: List[str]


def _snapshot(ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Deep copies of containers so in-place mutation by the build -- nested
    # too -- is visible. A value that cannot be deep-copied falls back to a
    # shallow copy (top-level mutation still visible).
    out: Dict[str, Any] = {}
    for k, v in ctx.items():
        if isinstance(v, (dict, list, set)):
            try:
                out[k] = copy.deepcopy(v)
            except Exception:  # noqa: BLE001
                out[k] = copy.copy(v)
        else:
            out[k] = v
    return out


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- uncomparable values count as changed
        return False


def compute_ctx_delta(before: Dict[str, Any], after: Dict[str, Any]) -> StanceCtxDelta:
    changed: Dict[str, Any] = {}
    for key, value in after.items():
        if key in _NON_TRANSFERABLE_KEYS:
            continue
        if key not in before:
            changed[key] = value
        elif not _same(value, before[key]):
            changed[key] = value
    removed = [k for k in before if k not in after and k not in _NON_TRANSFERABLE_KEYS]
    return StanceCtxDelta(changed=changed, removed=removed)


def apply_ctx_delta(ctx: Dict[str, Any], delta: StanceCtxDelta) -> None:
    for key in delta.removed:
        ctx.pop(key, None)
    for key, value in delta.changed.items():
        # A dict on both sides is merged, build's keys winning, so keys the
        # stance request's own router set before its hook (e.g. ctx["debug"]
        # recall_* entries) survive -- whether or not the prepare ctx had the
        # dict before the build.
        if isinstance(value, dict) and isinstance(ctx.get(key), dict):
            ctx[key] = {**ctx[key], **value}
        else:
            ctx[key] = value


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    future: "asyncio.Future[StanceCtxDelta]"
    created: float
    build_ms: Optional[float] = None


@dataclass
class TakeOutcome:
    """What stance_react got. ``delta`` is None when it must build inline."""

    outcome: str  # used | absent | failed | inflight_timeout | consumed
    delta: Optional[StanceCtxDelta] = None
    wait_ms: float = 0.0
    build_ms: Optional[float] = None


@dataclass
class StancePrepareCache:
    ttl_sec: float = CACHE_TTL_SEC
    clock: Callable[[], float] = time.monotonic
    _entries: Dict[str, _Entry] = field(default_factory=dict)
    _consumed: Dict[str, float] = field(default_factory=dict)
    _abandoned: Dict[str, float] = field(default_factory=dict)
    _arrivals: Dict[str, asyncio.Event] = field(default_factory=dict)

    def _purge(self) -> None:
        now = self.clock()
        for table in (self._consumed, self._abandoned):
            for corr in [c for c, t in table.items() if now - t > self.ttl_sec]:
                table.pop(corr, None)
        for corr in [c for c, e in self._entries.items() if now - e.created > self.ttl_sec]:
            entry = self._entries.pop(corr)
            if not entry.future.done():
                entry.future.cancel()
            logger.info("stance_prepare_expired corr=%s", corr)

    def __len__(self) -> int:
        return len(self._entries)

    def begin(self, corr: str) -> Tuple[Optional[_Entry], str]:
        """Register an in-flight prepare. Returns (entry, "ok") or (None, reason)."""
        self._purge()
        if corr in self._entries or corr in self._consumed:
            return None, "duplicate"
        if corr in self._abandoned:
            return None, "abandoned"
        entry = _Entry(future=asyncio.get_running_loop().create_future(), created=self.clock())
        self._entries[corr] = entry
        waiter = self._arrivals.pop(corr, None)
        if waiter is not None:
            waiter.set()
        return entry, "ok"

    def resolve(self, entry: _Entry, delta: StanceCtxDelta, build_ms: float) -> None:
        entry.build_ms = build_ms
        if not entry.future.done():
            entry.future.set_result(delta)

    def fail(self, entry: _Entry, exc: BaseException, build_ms: float) -> None:
        entry.build_ms = build_ms
        if not entry.future.done():
            entry.future.set_exception(exc)
            # Retrieve so an un-awaited failure does not log "exception never retrieved".
            entry.future.exception()

    async def take(
        self,
        corr: str,
        *,
        absent_wait_sec: float | None = None,
        inflight_wait_sec: float | None = None,
    ) -> TakeOutcome:
        absent_wait_sec = ABSENT_WAIT_SEC if absent_wait_sec is None else absent_wait_sec
        inflight_wait_sec = INFLIGHT_WAIT_SEC if inflight_wait_sec is None else inflight_wait_sec
        started = time.perf_counter()

        def _waited() -> float:
            return round((time.perf_counter() - started) * 1000.0, 1)

        self._purge()
        if corr in self._consumed:
            return TakeOutcome("consumed")
        entry = self._entries.get(corr)
        if entry is None:
            waiter = self._arrivals.setdefault(corr, asyncio.Event())
            try:
                await asyncio.wait_for(waiter.wait(), timeout=absent_wait_sec)
            except asyncio.TimeoutError:
                pass
            finally:
                if self._arrivals.get(corr) is waiter:
                    self._arrivals.pop(corr, None)
            entry = self._entries.get(corr)
            if entry is None:
                # A prepare arriving after this would build a second time.
                self._abandoned[corr] = self.clock()
                return TakeOutcome("absent", wait_ms=_waited())
        # Consume before awaiting: a second stance_react for this corr builds inline.
        self._entries.pop(corr, None)
        self._consumed[corr] = self.clock()
        try:
            delta = await asyncio.wait_for(asyncio.shield(entry.future), timeout=inflight_wait_sec)
        except asyncio.TimeoutError:
            return TakeOutcome("inflight_timeout", wait_ms=_waited(), build_ms=entry.build_ms)
        except asyncio.CancelledError:
            if entry.future.cancelled():
                return TakeOutcome("failed", wait_ms=_waited(), build_ms=entry.build_ms)
            raise
        except Exception:  # noqa: BLE001 -- the prepare's own failure
            return TakeOutcome("failed", wait_ms=_waited(), build_ms=entry.build_ms)
        return TakeOutcome("used", delta=delta, wait_ms=_waited(), build_ms=entry.build_ms)


_CACHE = StancePrepareCache()


def get_cache() -> StancePrepareCache:
    return _CACHE


# ---------------------------------------------------------------------------
# Consumer side: stance_react
# ---------------------------------------------------------------------------


async def take_prepared_stance_context(ctx: Dict[str, Any]) -> bool:
    """Apply a prepared context to ``ctx`` if one exists for this turn.

    Only runs when the stance request says a prepare was sent. Returns True when
    ctx now holds the prepared ``chat_stance_inputs``; False means build inline.
    """
    if not ctx.pop(STANCE_PREPARE_REQUESTED_CTX_KEY, None):
        return False
    corr = str(ctx.get("correlation_id") or "")
    if not corr:
        return False
    taken = await get_cache().take(corr)
    used = False
    if taken.delta is not None:
        apply_ctx_delta(ctx, taken.delta)
        used = isinstance(ctx.get("chat_stance_inputs"), dict)
        if not used:
            taken.outcome = "used_without_inputs"
    ctx["stance_prepare_overlap"] = {
        "outcome": taken.outcome,
        "build_ms": taken.build_ms,
        "wait_ms": taken.wait_ms,
        "lane": settings.exec_lane,
    }
    logger.info(
        "stance_prepare_overlap corr=%s side=cortex-exec outcome=%s build_ms=%s wait_ms=%s lane=%s",
        corr,
        taken.outcome,
        taken.build_ms,
        taken.wait_ms,
        settings.exec_lane,
    )
    return used


# ---------------------------------------------------------------------------
# Producer side: the stance_context_prepare RPC handler
# ---------------------------------------------------------------------------


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def build_prepare_ctx(env: BaseEnvelope, request: StanceContextPrepareRequestV1) -> Dict[str, Any]:
    """The ctx stance_react's router hook would build for this request."""
    raw = env.payload if isinstance(env.payload, dict) else {}
    raw_plan_request = raw.get("plan_request") if isinstance(raw.get("plan_request"), dict) else {}
    req = request.plan_request
    payload_context = raw_plan_request.get("context") or req.context or {}
    trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
    parent_event_id = (env.trace or {}).get("event_id") or (env.trace or {}).get("parent_event_id")
    ctx = build_exec_ctx(
        payload_context=dict(payload_context),
        req=req,
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        corr_id=str(env.correlation_id),
    )
    ctx.pop(STANCE_PREPARE_REQUESTED_CTX_KEY, None)
    # Same as PlanRunner.run_plan before its brain hook.
    extra = req.args.extra or {}
    options = extra.get("options") or ctx.get("options") or {}
    if isinstance(options, dict):
        ctx.setdefault("options", options)
        for key, val in options.items():
            ctx.setdefault(key, val)
    ctx["mode"] = extra.get("mode") or ctx.get("mode") or "brain"
    # The signal probe and the attention frame read ctx["verb"]
    # (current_turn_llm_signals.py, publish_attention_schema's leg).
    ctx["verb"] = req.plan.verb_name or "stance_react"
    return ctx


async def run_stance_context_prepare(
    env: BaseEnvelope, request: StanceContextPrepareRequestV1
) -> StanceContextPrepareResultV1:
    from .executor import brain_reply_context_skipped, prepare_brain_reply_context

    # Keyed by the envelope's normalized correlation id: stance_react looks up
    # ctx["correlation_id"] = str(env.correlation_id) (main.handle), so a
    # non-canonical spelling in the payload must not miss.
    corr = str(env.correlation_id)
    lane = settings.exec_lane
    cache = get_cache()
    entry, reason = cache.begin(corr)
    if entry is None:
        logger.info("stance_prepare_skipped corr=%s reason=%s lane=%s", corr, reason, lane)
        return StanceContextPrepareResultV1(correlation_id=corr, status="duplicate", lane=lane, error=reason)

    started = time.perf_counter()
    try:
        ctx = build_prepare_ctx(env, request)
        if str(ctx.get("mode") or "").strip().lower() != "brain":
            raise ValueError("not_brain_mode")
        if brain_reply_context_skipped(ctx.get("verb"), ctx):
            raise ValueError("brain_reply_context_skipped")
        before = _snapshot(ctx)
        inputs = await prepare_brain_reply_context(ctx)
        if not isinstance(inputs, dict):
            raise ValueError("no_chat_stance_inputs")
        delta = compute_ctx_delta(before, ctx)
    except Exception as exc:  # noqa: BLE001 -- stance_react falls back to an inline build
        build_ms = round((time.perf_counter() - started) * 1000.0, 1)
        cache.fail(entry, exc, build_ms)
        logger.warning(
            "stance_prepare_failed corr=%s lane=%s build_ms=%s err=%s: %s",
            corr,
            lane,
            build_ms,
            type(exc).__name__,
            exc,
        )
        return StanceContextPrepareResultV1(
            correlation_id=corr, status="failed", build_ms=build_ms, lane=lane, error=f"{type(exc).__name__}: {exc}"[:500]
        )
    build_ms = round((time.perf_counter() - started) * 1000.0, 1)
    cache.resolve(entry, delta, build_ms)
    logger.info(
        "stance_prepare_ready corr=%s lane=%s build_ms=%s keys=%s",
        corr,
        lane,
        build_ms,
        len(delta.changed),
    )
    return StanceContextPrepareResultV1(correlation_id=corr, status="ready", build_ms=build_ms, lane=lane)


async def handle_stance_context_prepare(env: BaseEnvelope) -> BaseEnvelope | None:
    """Rabbit handler for the stance_context_prepare channel."""
    try:
        request = StanceContextPrepareRequestV1.model_validate(env.payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("stance_prepare_invalid corr=%s err=%s", env.correlation_id, exc)
        result = StanceContextPrepareResultV1(
            correlation_id=str(env.correlation_id),
            status="failed",
            lane=settings.exec_lane,
            error="invalid_request",
        )
    else:
        result = await run_stance_context_prepare(env, request)
    if not env.reply_to:
        return None
    return BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_RESULT_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=list(env.causality_chain or []),
        payload=result.model_dump(mode="json"),
    )
