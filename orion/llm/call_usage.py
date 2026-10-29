"""Per-call token ledger: read token counts from a reply, queue the event, publish it.

Shared by orion-llm-gateway (every call it places) and the harness governor's warm FCC relay
(FCC-path turns only). Contract: ``orion/schemas/llm_call_usage.py``.

Three pieces:

* ``token_counts(head, tail)`` -- pure. Reads ``usage`` and llama.cpp ``timings`` from a parsed
  reply dict, or from the raw bytes of a stream (SSE). A stream needs its HEAD as well as its
  tail: an Anthropic stream carries the prompt counts in ``message_start`` (first event) and the
  completion count in ``message_delta`` (last events). Live 2026-10-29 against circe llama.cpp:
  an OpenAI stream has no ``usage`` chunk (no ``stream_options``) but its last chunk has
  ``timings``; an Anthropic reply has ``usage`` and no ``timings``.
* ``UsageSink`` -- a bounded, thread-safe queue. ``offer`` never blocks and never raises, so the
  serving path cannot be slowed by the ledger. When full, the OLDEST event is dropped and counted.
* ``run_usage_publisher`` -- drains the sink onto the bus. A publish failure drops that batch
  (logged with a count); telemetry never backs up into serving.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger("orion.llm.call_usage")

DEFAULT_SINK_MAXLEN = 4096
# How much of a stream's start and end the producers keep to read the counts.
STREAM_HEAD_BYTES = 4096
STREAM_TAIL_BYTES = 4096


@dataclass(frozen=True)
class TokenCounts:
    prompt_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    token_source: str = "none"
    decode_tps: Optional[float] = None


def _nonneg(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0 or f == float("inf"):
        return None
    return f


def _from_usage(usage: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(prompt incl. cached, cached, completion) from one ``usage`` block, either format."""
    if any(k in usage for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")):
        # Anthropic: input_tokens EXCLUDES cache reads/writes.
        inp = _nonneg(usage.get("input_tokens"))
        read = _nonneg(usage.get("cache_read_input_tokens"))
        made = _nonneg(usage.get("cache_creation_input_tokens"))
        prompt = None if inp is None else inp + (read or 0) + (made or 0)
        return prompt, (read if inp is not None else None), _nonneg(usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return (
        _nonneg(usage.get("prompt_tokens")),
        _nonneg(details.get("cached_tokens")),
        _nonneg(usage.get("completion_tokens")),
    )


def _iter_objects(payload: Any) -> Iterable[dict]:
    """Every JSON object in a reply: the dict itself, or each SSE ``data:`` line / JSON body."""
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", "ignore")
    elif isinstance(payload, str):
        text = payload
    else:
        return
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            yield obj
            return
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue  # a line cut by the head/tail window
        if isinstance(obj, dict):
            yield obj


def _usages(obj: dict) -> Iterable[dict]:
    if isinstance(obj.get("usage"), dict):
        yield obj["usage"]
    message = obj.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        yield message["usage"]  # Anthropic message_start


def token_counts(*payloads: Any) -> TokenCounts:
    """Counts from a reply. Pass the parsed body, or a stream's head then tail (order matters:
    a later ``usage`` value overrides an earlier one, as ``message_delta`` follows ``message_start``)."""
    prompt = cached = completion = None
    found_usage = False
    timings: Optional[dict] = None
    for payload in payloads:
        for obj in _iter_objects(payload):
            for usage in _usages(obj):
                p, c, o = _from_usage(usage)
                if p is not None or (o is not None and obj.get("type") != "message_start"):
                    found_usage = True
                if p is not None:
                    prompt, cached = p, c
                # message_start's output_tokens is a placeholder (0/1); the real count is in
                # message_delta. A stream cut before message_delta leaves completion unknown.
                if o is not None and obj.get("type") != "message_start":
                    completion = o
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
    decode_tps = None
    if timings is not None:
        decode_tps = _positive_float(timings.get("predicted_per_second"))
        t_new, t_cache, t_out = (_nonneg(timings.get(k)) for k in ("prompt_n", "cache_n", "predicted_n"))
        t_prompt = None if t_new is None else t_new + (t_cache or 0)
        if prompt is None and t_prompt is not None:
            prompt, cached = t_prompt, t_cache
        elif cached is None and t_cache is not None and prompt is not None:
            cached = t_cache
        if completion is None:
            completion = t_out
    if found_usage:
        source = "usage"
    elif timings is not None and (prompt is not None or completion is not None):
        source = "timings"
    else:
        source = "none"
    return TokenCounts(prompt, cached, completion, source, decode_tps)


class UsageSink:
    """Bounded, thread-safe. ``offer`` never blocks and never raises."""

    def __init__(self, maxlen: int = DEFAULT_SINK_MAXLEN) -> None:
        self._lock = threading.Lock()
        self._items: deque = deque(maxlen=max(1, int(maxlen)))
        self.dropped = 0

    def offer(self, event: Any) -> None:
        try:
            with self._lock:
                if len(self._items) == self._items.maxlen:
                    self.dropped += 1
                self._items.append(event)
        except Exception:  # noqa: BLE001 -- never break a reply
            pass

    def drain(self, limit: int = 256) -> list:
        with self._lock:
            out = []
            while self._items and len(out) < limit:
                out.append(self._items.popleft())
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def new_call_id() -> str:
    return uuid.uuid4().hex


def _envelope_correlation(corr: Optional[str]) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(corr)) if corr else None
    except ValueError:
        return None


async def publish_batch(bus: Any, events: list, *, source: Any, channel: str) -> int:
    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.schemas.llm_call_usage import LLM_CALL_USAGE_KIND

    sent = 0
    for event in events:
        corr = _envelope_correlation(getattr(event, "correlation_id", None))
        kwargs = {"correlation_id": corr} if corr is not None else {}
        envelope = BaseEnvelope(kind=LLM_CALL_USAGE_KIND, source=source,
                                payload=event.model_dump(mode="json"), **kwargs)
        await bus.publish(channel, envelope)
        sent += 1
    return sent


async def run_usage_publisher(
    bus: Any,
    sink: UsageSink,
    *,
    source: Any,
    channel: Optional[str] = None,
    interval_sec: float = 2.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    from orion.schemas.llm_call_usage import LLM_CALL_USAGE_CHANNEL

    channel = channel or LLM_CALL_USAGE_CHANNEL
    stop = stop or asyncio.Event()
    reported_dropped = 0
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.2, float(interval_sec)))
        except asyncio.TimeoutError:
            pass
        while True:
            batch = sink.drain()
            if not batch:
                break
            try:
                await publish_batch(bus, batch, source=source, channel=channel)
            except Exception:  # noqa: BLE001
                logger.warning("llm_call_usage_publish_failed dropped=%d", len(batch), exc_info=True)
                break
        if sink.dropped != reported_dropped:
            logger.warning("llm_call_usage_sink_overflow dropped_total=%d", sink.dropped)
            reported_dropped = sink.dropped
        if stop.is_set():
            return


__all__ = [
    "STREAM_HEAD_BYTES",
    "STREAM_TAIL_BYTES",
    "TokenCounts",
    "UsageSink",
    "new_call_id",
    "publish_batch",
    "run_usage_publisher",
    "token_counts",
]
