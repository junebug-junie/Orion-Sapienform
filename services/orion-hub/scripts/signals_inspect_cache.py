"""Hub in-memory cache for gateway-emitted OrionSignalV1 (Phase 2b inspect)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.signals.models import OrionSignalV1

logger = logging.getLogger("orion-hub.signals_inspect")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_trace_id(raw: str) -> str:
    t = (raw or "").strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    return t


def _chain_item(sig: OrionSignalV1) -> Dict[str, Any]:
    return {
        "organ_id": sig.organ_id,
        "signal_kind": sig.signal_kind,
        "signal_id": sig.signal_id,
        "observed_at": sig.observed_at.isoformat(),
        "dimensions": dict(sig.dimensions),
        "causal_parents": list(sig.causal_parents),
    }


def _stale_signal(sig: OrionSignalV1, now: datetime, window_sec: float) -> bool:
    ref = sig.emitted_at or sig.observed_at
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (now - ref).total_seconds() > window_sec


class SignalsInspectCache:
    """
    Subscribes to ``orion:signals:*`` (pattern), keeps latest ``OrionSignalV1`` per organ_id,
    and optionally a rolling trace cache keyed by ``otel_trace_id``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        subscribe_pattern: str,
        window_sec: float,
        trace_enabled: bool,
        trace_max_traces: int,
        trace_ttl_sec: float,
        trace_max_signals_per_trace: int,
    ) -> None:
        self.enabled = enabled
        self.subscribe_pattern = subscribe_pattern
        self.window_sec = float(window_sec)
        self.trace_enabled = trace_enabled
        self.trace_max_traces = int(trace_max_traces)
        self.trace_ttl_sec = float(trace_ttl_sec)
        self.trace_max_signals_per_trace = int(trace_max_signals_per_trace)

        self._lock = asyncio.Lock()
        self._latest_by_organ: Dict[str, OrionSignalV1] = {}
        self._chains: "OrderedDict[str, List[OrionSignalV1]]" = OrderedDict()
        self._trace_touch_mono: Dict[str, float] = {}
        self._trace_truncated: Dict[str, bool] = {}

        self._task: Optional[asyncio.Task] = None
        self._bus: Optional[OrionBusAsync] = None

    async def start(self, bus: OrionBusAsync) -> None:
        if not self.enabled:
            logger.info("signals_inspect_cache disabled via settings.")
            return
        if not bus or not bus.enabled:
            logger.warning("signals_inspect_cache not started (bus unavailable).")
            return
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-signals-inspect-cache")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to signal gateway output pattern: %s", self.subscribe_pattern)
        try:
            async with self._bus.subscribe(self.subscribe_pattern, patterns=True) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    await self._handle_message(msg)
        except asyncio.CancelledError:
            logger.info("signals_inspect_cache task cancelled.")
        except Exception as exc:
            logger.error("signals_inspect_cache loop failed: %s", exc, exc_info=True)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        if not self._bus:
            return
        mtype = msg.get("type")
        if mtype not in ("message", "pmessage"):
            return
        raw = msg.get("data")
        decoded = self._bus.codec.decode(raw)
        if not decoded.ok:
            return
        env = decoded.envelope
        kind = env.kind or ""
        if not kind.startswith("signal."):
            return
        payload = env.payload
        if not isinstance(payload, dict):
            return
        try:
            sig = OrionSignalV1.model_validate(payload)
        except Exception:
            logger.debug("signals_inspect_cache skip invalid OrionSignalV1 kind=%s", kind)
            return

        async with self._lock:
            self._latest_by_organ[sig.organ_id] = sig
            if self.trace_enabled and sig.otel_trace_id:
                tid = _normalize_trace_id(sig.otel_trace_id)
                if len(tid) == 32 and all(c in "0123456789abcdef" for c in tid):
                    lst = self._chains.setdefault(tid, [])
                    lst.append(sig)
                    lst.sort(key=lambda s: s.observed_at)
                    if len(lst) > self.trace_max_signals_per_trace:
                        self._trace_truncated[tid] = True
                        lst[:] = lst[-self.trace_max_signals_per_trace :]
                    self._chains.move_to_end(tid)
                    self._trace_touch_mono[tid] = time.monotonic()
                    while len(self._chains) > self.trace_max_traces:
                        old_tid, _ = self._chains.popitem(last=False)
                        self._trace_touch_mono.pop(old_tid, None)
                        self._trace_truncated.pop(old_tid, None)
                    self._evict_trace_ttl_locked()

    def _evict_trace_ttl_locked(self) -> None:
        cutoff = time.monotonic() - self.trace_ttl_sec
        for tid in list(self._chains.keys()):
            touched = self._trace_touch_mono.get(tid, 0.0)
            if touched < cutoff:
                self._chains.pop(tid, None)
                self._trace_touch_mono.pop(tid, None)
                self._trace_truncated.pop(tid, None)

    async def get_active(self) -> Dict[str, Any]:
        now = _utcnow()
        async with self._lock:
            signals: Dict[str, Any] = {}
            stale_organs: List[str] = []
            for oid, sig in self._latest_by_organ.items():
                if _stale_signal(sig, now, self.window_sec):
                    stale_organs.append(oid)
                else:
                    signals[oid] = sig.model_dump(mode="json")
            for oid in stale_organs:
                self._latest_by_organ.pop(oid, None)
        return {"as_of": now.isoformat(), "signals": signals}

    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        tid = _normalize_trace_id(trace_id)
        if len(tid) != 32 or any(c not in "0123456789abcdef" for c in tid):
            return None
        async with self._lock:
            self._evict_trace_ttl_locked()
            lst = self._chains.get(tid)
            if not lst:
                return None
            ordered = sorted(lst, key=lambda s: s.observed_at)
            truncated = self._trace_truncated.get(tid, False)
        chain = [_chain_item(s) for s in ordered]
        gaps: List[str] = []
        if truncated:
            gaps.append("trace_truncated_max_signals_per_trace")
        return {
            "trace_id": tid,
            "chain": chain,
            "complete": not truncated,
            "gaps": gaps,
        }
