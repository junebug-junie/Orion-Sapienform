"""Hub in-memory cache for cortex-exec ``CognitionTracePayload`` (Runtime Trace Nexus A4)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.cortex.types import StepExecutionResult
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

logger = logging.getLogger("orion-hub.cognition_trace_cache")


def _redacted_step(step: StepExecutionResult) -> Dict[str, Any]:
    result = step.result if isinstance(step.result, dict) else {}
    services = [str(k) for k in result.keys()]
    artifacts = step.artifacts if isinstance(step.artifacts, dict) else {}
    prompt_template_present = any(
        key in artifacts or key in result
        for key in ("prompt_template", "prompt_template_id", "template_id")
    )
    return {
        "step_name": step.step_name,
        "order": step.order,
        "status": step.status,
        "latency_ms": step.latency_ms,
        "services": services,
        "error_present": bool(step.error),
        "prompt_template_present": prompt_template_present,
    }


class CognitionTraceCache:
    """
    Subscribes to ``orion:cognition:trace``, keeps ``CognitionTracePayload`` per ``correlation_id``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        subscribe_channel: str,
        max_entries: int,
        ttl_sec: float,
        api_debug: bool,
    ) -> None:
        self.enabled = enabled
        self.subscribe_channel = subscribe_channel
        self.max_entries = int(max_entries)
        self.ttl_sec = float(ttl_sec)
        self.api_debug = bool(api_debug)

        self._lock = asyncio.Lock()
        self._entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        self._task: Optional[asyncio.Task] = None
        self._bus: Optional[OrionBusAsync] = None

    async def start(self, bus: OrionBusAsync) -> None:
        if not self.enabled:
            logger.info("cognition_trace_cache disabled via settings.")
            return
        if not bus or not bus.enabled:
            logger.warning("cognition_trace_cache not started (bus unavailable).")
            return
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-cognition-trace-cache")

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
        logger.info("Subscribing to cognition trace channel: %s", self.subscribe_channel)
        try:
            async with self._bus.subscribe(self.subscribe_channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    await self._handle_message(msg)
        except asyncio.CancelledError:
            logger.info("cognition_trace_cache task cancelled.")
        except Exception as exc:
            logger.error("cognition_trace_cache loop failed: %s", exc, exc_info=True)

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
        payload = env.payload
        if not isinstance(payload, dict):
            return
        try:
            trace = CognitionTracePayload.model_validate(payload)
        except Exception:
            logger.debug("cognition_trace_cache skip invalid CognitionTracePayload kind=%s", env.kind)
            return
        corr = str(env.correlation_id or trace.correlation_id or "").strip()
        if not corr:
            logger.debug("cognition_trace_cache skip missing correlation_id kind=%s", env.kind)
            return
        await self.put(corr, trace, otel_trace_id=None)

    async def put(
        self,
        correlation_id: str,
        trace: CognitionTracePayload,
        *,
        otel_trace_id: Optional[str] = None,
    ) -> None:
        corr = str(correlation_id or "").strip()
        if not corr:
            return
        async with self._lock:
            self._entries[corr] = {
                "trace": trace,
                "otel_trace_id": otel_trace_id,
                "touch_mono": time.monotonic(),
                "truncated": False,
            }
            self._entries.move_to_end(corr)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._evict_ttl_locked()

    def _evict_ttl_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl_sec
        for corr in list(self._entries.keys()):
            touched = self._entries[corr].get("touch_mono", 0.0)
            if touched < cutoff:
                self._entries.pop(corr, None)

    def _entry_locked(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        self._evict_ttl_locked()
        entry = self._entries.get(correlation_id)
        if entry is None:
            return None
        entry["touch_mono"] = time.monotonic()
        self._entries.move_to_end(correlation_id)
        return entry

    async def get_redacted(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        corr = str(correlation_id or "").strip()
        if not corr:
            return None
        async with self._lock:
            entry = self._entry_locked(corr)
            if entry is None:
                return None
            trace: CognitionTracePayload = entry["trace"]
            truncated = bool(entry.get("truncated"))
            otel_trace_id = entry.get("otel_trace_id")
        steps = [_redacted_step(s) for s in (trace.steps or [])]
        gaps: List[str] = []
        if truncated:
            gaps.append("trace_truncated_max_entries")
        return {
            "correlation_id": corr,
            "verb": trace.verb,
            "mode": trace.mode,
            "steps": steps,
            "recall_used": bool(trace.recall_used),
            "final_text_present": bool(trace.final_text and str(trace.final_text).strip()),
            "otel_trace_id": otel_trace_id,
            "metadata": dict(trace.metadata or {}),
            "complete": not truncated,
            "gaps": gaps,
        }

    async def get_debug(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        if not self.api_debug:
            return await self.get_redacted(correlation_id)
        corr = str(correlation_id or "").strip()
        if not corr:
            return None
        async with self._lock:
            entry = self._entry_locked(corr)
            if entry is None:
                return None
            trace: CognitionTracePayload = entry["trace"]
            truncated = bool(entry.get("truncated"))
            otel_trace_id = entry.get("otel_trace_id")
        gaps: List[str] = []
        if truncated:
            gaps.append("trace_truncated_max_entries")
        steps_out: List[Dict[str, Any]] = []
        for step in trace.steps or []:
            redacted = _redacted_step(step)
            redacted["logs"] = list(step.logs or [])[:20]
            artifact_keys = list((step.artifacts or {}).keys())[:20]
            redacted["artifact_keys"] = artifact_keys
            steps_out.append(redacted)
        return {
            "correlation_id": corr,
            "verb": trace.verb,
            "mode": trace.mode,
            "steps": steps_out,
            "recall_used": bool(trace.recall_used),
            "final_text_present": bool(trace.final_text and str(trace.final_text).strip()),
            "final_text": trace.final_text,
            "otel_trace_id": otel_trace_id,
            "metadata": dict(trace.metadata or {}),
            "recall_debug": dict(trace.recall_debug or {}),
            "complete": not truncated,
            "gaps": gaps,
        }
