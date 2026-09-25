"""The gateway reporting on its own inference calls (Layer 1 of the llm_inference lane).

Every bus-RPC chat call the gateway answers is classified by what actually
happened -- served, backend failed, gateway refused, request unusable -- and
counted into a fixed window. Once per window the counts are published as one
grammar trace (``llm_gateway.inference:<gateway>:<window_id>``) on
``orion:grammar:event``: one atom per serving node plus a closing atom that is
sent even when the window saw no calls, so "no traffic" and "gateway gone" stay
distinguishable downstream.

Why the gateway and nobody else: a backend failure comes back to the caller as
a normal reply whose text is ``[Error: llamacpp failed: ...]`` (llm_backend.py
has ~16 such returns), so caller-side RPC health counts it as a success and
host GPU pressure reads an idle, broken backend as calm. Only this process sees
the call fail.

Bounded by construction: counts, latency percentiles and token totals only.
No prompt text, no response text, no correlation ids.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.cognition.cortex_payload_extract import looks_like_error_text
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.llm_inference_projection import (
    LLM_INFERENCE_SOURCE_SERVICE,
    LLM_INFERENCE_TRACE_PREFIX,
    OUTCOME_SERVED,
    REFUSAL_CLASSES,
    REQUEST_INVALID_CLASSES,
    ROLE_NODE_WINDOW,
    ROLE_WINDOW_COMPLETED,
    UPSTREAM_FAILURE_CLASSES,
)

logger = logging.getLogger("orion-llm-gateway.grammar")

_MAX_LATENCY_SAMPLES = 512
_MAX_LABELS = 8
_UNROUTED = "unrouted"
_HTTP_STATUS_RE = re.compile(r"(client|server) error '(\d{3})")


def classify_outcome(result: Any) -> str:
    """Name what happened to one call, from the result dict the gateway replies with.

    Structured ``raw.error`` wins; otherwise the backend's in-text error framing is
    detected with the repo's canonical detector and sub-classed by its fixed wording.
    """
    if not isinstance(result, dict):
        return "upstream_error"
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    err = str(raw.get("error") or "").strip()
    if err:
        if err in REFUSAL_CLASSES or err in REQUEST_INVALID_CLASSES:
            return err
        return "upstream_error"
    text = str(result.get("text") or "")
    if not text.strip():
        if str(result.get("reasoning_content") or result.get("inline_think_content") or "").strip():
            return OUTCOME_SERVED
        # Counted for inspection only, not as a backend failure: whether an empty
        # completion is common at rest has not been measured live yet.
        return "upstream_empty"
    if not looks_like_error_text(text):
        return OUTCOME_SERVED
    low = text.strip().lower()
    if "attachments could not be read" in low:
        return "request_invalid"
    if "not configured" in low:
        return "route_not_configured"
    if "timed out" in low or "timeout" in low:
        return "upstream_timeout"
    if ("404" in low and "not found" in low) or " 404 at " in low:
        return "upstream_not_found"
    status = _HTTP_STATUS_RE.search(low)
    if status:
        return "upstream_http_5xx" if status.group(1) == "server" else "upstream_http_4xx"
    if "connection refused" in low or "connecterror" in low or "all connection attempts failed" in low:
        return "upstream_connect"
    return "upstream_error"


def node_hint(served_by: str | None) -> str:
    """Gateway worker labels are ``{node}-worker[-lane][-N]``. The reducer decides
    whether the hint names a real field node; the gateway only groups by it."""
    if not served_by or not str(served_by).strip():
        return _UNROUTED
    return str(served_by).strip().lower().split("-worker")[0] or _UNROUTED


def _usage_tokens(result: Any) -> tuple[int, int]:
    raw = result.get("raw") if isinstance(result, dict) and isinstance(result.get("raw"), dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    out: list[int] = []
    for key in ("prompt_tokens", "completion_tokens"):
        try:
            out.append(max(0, int(usage.get(key) or 0)))
        except (TypeError, ValueError):
            out.append(0)
    return out[0], out[1]


def _percentile(sorted_vals: list[int], q: float) -> int | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


@dataclass
class _NodeBucket:
    calls: int = 0
    classes: dict[str, int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    served_latency_ms: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, *, outcome: str, served_by: str | None, elapsed_ms: int, tokens: tuple[int, int]) -> None:
        self.calls += 1
        self.classes[outcome] = self.classes.get(outcome, 0) + 1
        label = str(served_by or "").strip()
        if label and label not in self.labels and len(self.labels) < _MAX_LABELS:
            self.labels.append(label)
        if outcome == OUTCOME_SERVED:
            if len(self.served_latency_ms) >= _MAX_LATENCY_SAMPLES:
                self.served_latency_ms.pop(0)
            self.served_latency_ms.append(max(0, int(elapsed_ms)))
            self.prompt_tokens += tokens[0]
            self.completion_tokens += tokens[1]

    def count(self, classes: frozenset[str]) -> int:
        return sum(n for name, n in self.classes.items() if name in classes)

    def summary(self, node: str) -> str:
        lat = sorted(self.served_latency_ms)
        classes = "|".join(f"{k}:{v}" for k, v in sorted(self.classes.items())) or "none"
        labels = "|".join(self.labels) or "none"
        return (
            f"node={node} calls={self.calls} served={self.classes.get(OUTCOME_SERVED, 0)} "
            f"upstream_failed={self.count(UPSTREAM_FAILURE_CLASSES)} "
            f"refused={self.count(REFUSAL_CLASSES)} request_invalid={self.count(REQUEST_INVALID_CLASSES)} "
            f"p50_ms={_percentile(lat, 0.5)} p95_ms={_percentile(lat, 0.95)} "
            f"prompt_tokens={self.prompt_tokens} completion_tokens={self.completion_tokens} "
            f"workers={labels} classes={classes}"
        )


class InferenceWindowRecorder:
    """Counts calls into the current window. ``record`` is called on the event loop
    after each reply is built; ``drain`` swaps the window out atomically."""

    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, _NodeBucket] = {}
        self._window_start = self._clock()

    def record(self, result: Any, *, served_by: str | None, elapsed_s: float) -> str:
        outcome = classify_outcome(result)
        with self._lock:
            bucket = self._buckets.setdefault(node_hint(served_by), _NodeBucket())
            bucket.add(
                outcome=outcome,
                served_by=served_by,
                elapsed_ms=int(round(max(0.0, elapsed_s) * 1000)),
                tokens=_usage_tokens(result),
            )
        return outcome

    def drain(self) -> tuple[float, float, dict[str, _NodeBucket]]:
        with self._lock:
            start, end = self._window_start, self._clock()
            buckets, self._buckets = self._buckets, {}
            self._window_start = end
        return start, end, buckets


def _window_id(start_ts: float) -> str:
    return datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_window_events(
    *,
    gateway_node: str,
    window_start: float,
    window_end: float,
    buckets: dict[str, _NodeBucket],
) -> list[GrammarEventV1]:
    gw = (gateway_node or "gateway").strip().lower().replace(":", "_") or "gateway"
    trace_id = f"{LLM_INFERENCE_TRACE_PREFIX}{gw}:{_window_id(window_start)}"
    emitted_at = datetime.fromtimestamp(window_end, tz=timezone.utc)
    dims = ["inference", "llm"]
    provenance = GrammarProvenanceV1(
        source_service=LLM_INFERENCE_SOURCE_SERVICE,
        source_component="inference_window",
        source_trace_id=trace_id,
    )

    def _event(idx: int, role: str, summary: str, text_value: str) -> GrammarEventV1:
        event_id = f"{trace_id}:{idx:02d}:{role}"
        return GrammarEventV1(
            event_id=event_id,
            event_kind="atom_emitted",
            trace_id=trace_id,
            emitted_at=emitted_at,
            observed_at=emitted_at,
            layer="inference",
            dimensions=dims,
            atom=GrammarAtomV1(
                atom_id=event_id,
                trace_id=trace_id,
                atom_type="observation",
                semantic_role=role,
                layer="inference",
                dimensions=dims,
                summary=summary,
                text_value=text_value,
                confidence=1.0,
                salience=0.3,
            ),
            provenance=provenance,
        )

    events = [
        _event(i, ROLE_NODE_WINDOW, bucket.summary(node), node)
        for i, (node, bucket) in enumerate(sorted(buckets.items()))
    ]
    total = sum(b.calls for b in buckets.values())
    events.append(
        _event(
            len(events),
            ROLE_WINDOW_COMPLETED,
            f"gateway={gw} calls={total} nodes={len(buckets)} window_sec={max(0.0, window_end - window_start):.1f}",
            gw,
        )
    )
    return events


_recorder: InferenceWindowRecorder | None = None


def get_recorder() -> InferenceWindowRecorder:
    global _recorder
    if _recorder is None:
        _recorder = InferenceWindowRecorder()
    return _recorder


def reset_recorder_for_tests() -> None:
    global _recorder
    _recorder = None


async def run_window_publisher(
    bus: Any,
    *,
    gateway_node: str,
    window_sec: float,
    stop: asyncio.Event | None = None,
) -> None:
    """Flush one window every ``window_sec``. Publishing failures are logged and the
    window is dropped -- telemetry must never back up into the serving path."""
    from orion.grammar.publish import publish_grammar_event

    recorder = get_recorder()
    stop = stop or asyncio.Event()
    interval = max(5.0, float(window_sec))
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        start, end, buckets = recorder.drain()
        try:
            for event in build_window_events(
                gateway_node=gateway_node, window_start=start, window_end=end, buckets=buckets
            ):
                await publish_grammar_event(bus, event, source_name=LLM_INFERENCE_SOURCE_SERVICE)
        except Exception:  # noqa: BLE001
            logger.warning("llm_gateway_grammar_publish_failed window_start=%s", start, exc_info=True)
