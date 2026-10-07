"""The gateway reporting on its own inference calls (Layer 1 of the llm_inference lane).

Every chat call the gateway answers -- bus RPC and the HTTP passthroughs
(``/v1/messages``, ``/v1/chat/completions``) alike -- is classified by what
actually happened -- served, backend failed, gateway refused, request unusable --
and counted into a fixed window. Once per window the counts are published as one
grammar trace (``llm_gateway.inference:<gateway>:<window_id>``) on
``orion:grammar:event``: one atom per serving node plus a closing atom that is
sent even when the window saw no calls, so "no traffic" and "gateway gone" stay
distinguishable downstream.

Why the gateway and nobody else: a backend failure comes back to the caller as
a normal ``llm.chat.result`` reply (since 2026-10-07 with empty content and a
typed ``raw.error``; before that, text ``[Error: llamacpp failed: ...]`` and
``raw={}``), so caller-side RPC health counts it as a success and host GPU
pressure reads an idle, broken backend as calm. Only this process sees the call
fail. The ``[Error: ...]`` text sniffing below remains for the gateway's own
non-upstream refusals (vision, unconfigured route) that still use that framing.

Two clocks per call, per granted GPU-pool role (gpu-pool stage 6.2, 2026-09-30):
``wait`` is lease request -> grant (the line), ``model`` is grant -> reply (the
worker). They are disjoint intervals by construction (``CallClock``), so "the GPU
is slow" and "the line is long" no longer read as one number. ``decode_tps`` is
llama.cpp's own ``timings.predicted_per_second`` -- a per-token speed, so a 0.3 s
classifier and a 120 s agent turn on the same role are comparable. The old mixed
``p50_ms``/``p95_ms`` (queue wait + model time, per machine) are retired.

Two covariates ride along so a per-role baseline is not fooled by normal slot sharing
(stage 7 input, 2026-09-30): ``busy`` -- this gateway's calls in flight on the granted
role at grant, this one included (pool_placement occupancy), with decode speed split
``solo`` (busy == 1) vs ``shared``; and llama.cpp's own ``timings.prompt_n``/``cache_n``
(prompt tokens processed vs reused from the KV cache) summed per role.

Bounded by construction: counts, clock percentiles and token totals only.
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
from typing import Any, Callable

from orion.cognition.cortex_payload_extract import looks_like_error_text
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.llm_inference_projection import (
    LLM_INFERENCE_TRACE_PREFIX,
    OUTCOME_SERVED,
    REFUSAL_CLASSES,
    REQUEST_INVALID_CLASSES,
    ROLE_NODE_WINDOW,
    ROLE_WINDOW_COMPLETED,
    UPSTREAM_FAILURE_CLASSES,
)

logger = logging.getLogger("orion-llm-gateway.grammar")

# A module-level literal (same value as the contract constant, pinned by a test) so
# static producer-catalog scans can resolve this file's GrammarProvenanceV1 identity.
SOURCE_SERVICE = "orion-llm-gateway"

_MAX_LATENCY_SAMPLES = 512
_MAX_LABELS = 8
_MAX_ROLES = 8
_UNROUTED = "unrouted"
# A call that never held a grant (pool refused, plan error, crash before the lease).
UNGRANTED_ROLE = "ungranted"
_ROLE_SAFE_RE = re.compile(r"[^a-z0-9_.-]")
_DECODE_TPS_RE = re.compile(rb'"predicted_per_second"\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)')
_PROMPT_N_RE = re.compile(rb'"prompt_n"\s*:\s*([0-9]+)')
_CACHE_N_RE = re.compile(rb'"cache_n"\s*:\s*([0-9]+)')
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
        if err in REFUSAL_CLASSES or err in REQUEST_INVALID_CLASSES or err == "gateway_exception":
            return err
        if err in UPSTREAM_FAILURE_CLASSES:
            # llm_backend's typed upstream failures (upstream_http_5xx, upstream_not_found, ...)
            # already carry the contract's own class name.
            return err
        if err == "timeout":
            # the granted node was still working when the caller's budget ran out
            return "upstream_timeout"
        return "upstream_error"
    text = str(result.get("text") or "")
    if not text.strip():
        if str(result.get("reasoning_content") or result.get("inline_think_content") or "").strip():
            return OUTCOME_SERVED
        # Counted for inspection only, not as a backend failure: whether an empty
        # completion is common at rest has not been measured live yet.
        return "upstream_empty"
    low = text.strip().lower()
    # Every gateway-originated failure text is framed "[Error: ..." (llm_backend.py).
    # Model prose that merely starts with "Error:" or explains a "connection refused"
    # is an answer, not a failure -- review finding, 2026-09-25. The canonical
    # detector stays as a guard so a framing change there cannot widen this.
    if not low.startswith("[error:") or not looks_like_error_text(text):
        return OUTCOME_SERVED
    if "attachments could not be read" in low or "cannot accept image" in low:
        # the caller sent something the route cannot take; the backend was never asked
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


# HTTP passthrough outcomes that are not in the contract's class sets: counted for
# inspection only (like upstream_empty), never as a backend failure or a refusal.
CLIENT_GONE = "client_gone"


def classify_http_outcome(status_code: int, *, context_overflow: bool = False) -> str:
    """Name what happened to one HTTP passthrough call from the upstream's status code.

    Same classes as the bus path: 2xx served, 4xx the caller's request (llama.cpp's
    context overflow is its own class), 5xx the backend failing."""
    if context_overflow:
        return "context_overflow"
    if 200 <= int(status_code) < 400:
        return OUTCOME_SERVED
    if 400 <= int(status_code) < 500:
        return "upstream_http_4xx"
    return "upstream_http_5xx"


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


def _percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


_OTHER_WORKER = "other"


def _ms(seconds: float) -> int:
    return int(round(max(0.0, seconds) * 1000))


@dataclass
class CallClock:
    """The two clocks of one call, kept disjoint by construction.

    ``waiting()`` when a pool acquire starts, ``granted(role)`` when the lease yields,
    ``replied()`` when the upstream reply (or the stream's end) is back. A call that
    re-leases (context overflow, class clamp) sums each wait and each grant->reply
    interval; ``role`` is the last granted role. A call that never got a grant has a
    ``wait_ms`` and no ``model_ms``. ``None`` means "never measured", not zero.
    """

    clock: Callable[[], float] = time.monotonic
    wait_ms: int | None = None
    model_ms: int | None = None
    role: str | None = None
    # this gateway's calls in flight on ``role`` at the (last) grant, this one included
    busy_at_grant: int | None = None
    # the GPU pool's host node (config ``host.name``): where a call that never got a grant
    # files its wait, so "the line is long" survives the reducer's known-node filter
    pool_node: str | None = None
    _wait_from: float | None = None
    _model_from: float | None = None

    def waiting(self) -> None:
        self._close_model()
        self._wait_from = self.clock()

    def granted(self, role: str | None, busy: int | None = None) -> None:
        now = self.clock()
        self.busy_at_grant = busy if busy is None else max(1, int(busy))
        if self._wait_from is not None:
            self.wait_ms = (self.wait_ms or 0) + _ms(now - self._wait_from)
            self._wait_from = None
        self._model_from = now
        self.role = str(role or "").strip() or None

    def not_granted(self) -> None:
        if self._wait_from is not None:
            self.wait_ms = (self.wait_ms or 0) + _ms(self.clock() - self._wait_from)
            self._wait_from = None

    def replied(self) -> None:
        self._close_model()

    def _close_model(self) -> None:
        if self._model_from is not None:
            self.model_ms = (self.model_ms or 0) + _ms(self.clock() - self._model_from)
            self._model_from = None

    def close(self) -> None:
        """Close whichever interval is still open (a crash mid-wait or mid-call)."""
        self.not_granted()
        self._close_model()


def role_key(role: str | None) -> str:
    raw = _ROLE_SAFE_RE.sub("", str(role or "").strip().lower())
    return raw or UNGRANTED_ROLE


@dataclass(frozen=True)
class ReplyTimings:
    """llama.cpp's own ``timings`` for one reply. Every field None when not reported."""

    decode_tps: float | None = None
    prompt_n: int | None = None  # prompt tokens processed this call
    cache_n: int | None = None  # prompt tokens reused from the slot's KV cache


def _last_int(pattern: "re.Pattern[bytes]", data: bytes) -> int | None:
    found = pattern.findall(data)
    return int(found[-1]) if found else None


def _nonneg_int(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def timings_from(payload: Any) -> ReplyTimings:
    """Decode speed and prompt-cache counts from a reply body (dict, bytes, or a stream's tail)."""
    prompt_n = cache_n = None
    if isinstance(payload, dict):
        timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
        prompt_n, cache_n = _nonneg_int(timings.get("prompt_n")), _nonneg_int(timings.get("cache_n"))
    elif isinstance(payload, (bytes, bytearray, str)):
        data = payload.encode("utf-8", "ignore") if isinstance(payload, str) else bytes(payload)
        prompt_n, cache_n = _last_int(_PROMPT_N_RE, data), _last_int(_CACHE_N_RE, data)
    return ReplyTimings(decode_tps=decode_tps_from(payload), prompt_n=prompt_n, cache_n=cache_n)


def decode_tps_from(payload: Any) -> float | None:
    """llama.cpp's own decode speed (``timings.predicted_per_second``) from a reply body.

    Accepts the parsed reply dict (bus path: ``result["raw"]``) or raw bytes/str (HTTP
    passthrough body, or a stream's tail, where the last SSE chunk carries ``timings``).
    None when the backend did not report it -- never derived from wall time, which would
    mix prompt processing and queueing into a per-token speed."""
    value: Any = None
    if isinstance(payload, dict):
        timings = payload.get("timings")
        if isinstance(timings, dict):
            value = timings.get("predicted_per_second")
    elif isinstance(payload, (bytes, bytearray, str)):
        data = payload.encode("utf-8", "ignore") if isinstance(payload, str) else bytes(payload)
        matches = _DECODE_TPS_RE.findall(data)
        value = matches[-1].decode() if matches else None
    try:
        tps = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if tps is None or tps != tps or tps <= 0 or tps == float("inf"):
        return None
    return tps


def _fmt_tps(value: float | None) -> str | None:
    return None if value is None else f"{value:.1f}"


@dataclass
class _RoleBucket:
    calls: int = 0
    # of ``calls``, how many came through the HTTP passthroughs (the rest are bus RPC)
    http_calls: int = 0
    classes: dict[str, int] = field(default_factory=dict)
    # every call that waited on the pool (granted or not): the line
    wait_ms: list[int] = field(default_factory=list)
    # served calls only: the worker. A timed-out call's model time is its budget, not a speed.
    model_ms: list[int] = field(default_factory=list)
    decode_tps: list[float] = field(default_factory=list)
    # decode speed banded by occupancy at grant: alone on the role vs sharing it
    decode_tps_solo: list[float] = field(default_factory=list)
    decode_tps_shared: list[float] = field(default_factory=list)
    # every granted call's occupancy at grant (this gateway's calls in flight on the role)
    busy: list[int] = field(default_factory=list)
    # served calls reporting llama.cpp prompt counts: processed vs reused from the KV cache
    prompt_n: int = 0
    cache_n: int = 0
    cache_reports: int = 0

    @staticmethod
    def _push(samples: list, value) -> None:
        if len(samples) >= _MAX_LATENCY_SAMPLES:
            samples.pop(0)
        samples.append(value)

    def add(self, *, outcome: str, timing: "CallClock | None", timings: ReplyTimings | None,
            http: bool = False) -> None:
        self.calls += 1
        self.http_calls += int(bool(http))
        self.classes[outcome] = self.classes.get(outcome, 0) + 1
        busy = timing.busy_at_grant if timing is not None else None
        if timing is not None and timing.wait_ms is not None:
            self._push(self.wait_ms, int(timing.wait_ms))
        if busy is not None:
            self._push(self.busy, int(busy))
        if outcome != OUTCOME_SERVED:
            return
        if timing is not None and timing.model_ms is not None:
            self._push(self.model_ms, int(timing.model_ms))
        timings = timings or ReplyTimings()
        if timings.decode_tps is not None:
            self._push(self.decode_tps, float(timings.decode_tps))
            if busy is not None:
                self._push(self.decode_tps_solo if busy <= 1 else self.decode_tps_shared, float(timings.decode_tps))
        if timings.prompt_n is not None and timings.cache_n is not None:
            self.prompt_n += timings.prompt_n
            self.cache_n += timings.cache_n
            self.cache_reports += 1

    def count(self, classes: frozenset[str]) -> int:
        return sum(n for name, n in self.classes.items() if name in classes)

    def summary(self, role: str, slots: int | None = None) -> str:
        """``role[k:v|k:v]`` -- one token, no spaces/commas/semicolons, so it rides the node
        atom's kv summary and a role can never be split from its node by a batch boundary."""
        wait, model, tps = sorted(self.wait_ms), sorted(self.model_ms), sorted(self.decode_tps)
        busy = sorted(self.busy)
        fields = [
            ("calls", self.calls),
            ("http_calls", self.http_calls),
            ("served", self.classes.get(OUTCOME_SERVED, 0)),
            ("upstream_failed", self.count(UPSTREAM_FAILURE_CLASSES)),
            ("refused", self.count(REFUSAL_CLASSES)),
            ("request_invalid", self.count(REQUEST_INVALID_CLASSES)),
            ("wait_p50_ms", _percentile(wait, 0.5)),
            ("wait_p95_ms", _percentile(wait, 0.95)),
            ("model_p50_ms", _percentile(model, 0.5)),
            ("model_p95_ms", _percentile(model, 0.95)),
            ("decode_tps_p50", _fmt_tps(_percentile(tps, 0.5))),
            ("decode_tps_n", len(tps)),
            ("decode_tps_solo_p50", _fmt_tps(_percentile(sorted(self.decode_tps_solo), 0.5))),
            ("decode_tps_solo_n", len(self.decode_tps_solo)),
            ("decode_tps_shared_p50", _fmt_tps(_percentile(sorted(self.decode_tps_shared), 0.5))),
            ("decode_tps_shared_n", len(self.decode_tps_shared)),
            ("busy_p50", _percentile(busy, 0.5)),
            ("busy_max", busy[-1] if busy else None),
            ("slots", slots if slots and slots > 0 else None),
            ("prompt_n", self.prompt_n if self.cache_reports else None),
            ("cache_n", self.cache_n if self.cache_reports else None),
            ("cache_reports", self.cache_reports),
        ]
        body = "|".join(f"{k}:{v}" for k, v in fields if v is not None)
        return f"{role}[{body}]"


def _counts(counts: dict[str, int]) -> str:
    return "|".join(f"{k}:{v}" for k, v in sorted(counts.items()) if v > 0) or "none"


@dataclass
class _NodeBucket:
    calls: int = 0
    classes: dict[str, int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Per worker label: calls sent upstream (served + upstream failure) and upstream
    # failures (2026-09-29), so the substrate can say which lane is failing. Bounded
    # to _MAX_LABELS labels; the rest are counted under _OTHER_WORKER.
    worker_attempted: dict[str, int] = field(default_factory=dict)
    worker_failed: dict[str, int] = field(default_factory=dict)
    # granted GPU-pool role -> that role's two clocks and decode speed (stage 6.2). Bounded
    # to _MAX_ROLES; the pool has five roles today.
    roles: dict[str, _RoleBucket] = field(default_factory=dict)

    def add(
        self,
        *,
        outcome: str,
        served_by: str | None,
        tokens: tuple[int, int],
        timing: CallClock | None = None,
        timings: ReplyTimings | None = None,
        http: bool = False,
        roles: bool = True,
        counts: bool = True,
    ) -> None:
        """``roles``/``counts``: which half to record (an ungranted call's node counts and its
        role clock go to different buckets; see InferenceWindowRecorder.record_outcome)."""
        if roles:
            self.add_role(outcome=outcome, timing=timing, timings=timings, http=http)
        if http or not counts:
            # HTTP passthrough calls reach the per-role clocks only (stage 6.2, record-only).
            # The node-level counts below feed inference_failure_pressure, a live field channel
            # whose population has been bus-RPC calls since 2026-09-25; widening it to the
            # passthroughs is a metric-definition change, not part of this stage.
            return
        self.calls += 1
        self.classes[outcome] = self.classes.get(outcome, 0) + 1
        label = str(served_by or "").strip()
        if label and label not in self.labels and len(self.labels) < _MAX_LABELS:
            self.labels.append(label)
        upstream_failed = outcome in UPSTREAM_FAILURE_CLASSES
        if label and (outcome == OUTCOME_SERVED or upstream_failed):
            key = label if (label in self.worker_attempted or len(self.worker_attempted) < _MAX_LABELS) else _OTHER_WORKER
            self.worker_attempted[key] = self.worker_attempted.get(key, 0) + 1
            if upstream_failed:
                self.worker_failed[key] = self.worker_failed.get(key, 0) + 1
        if outcome == OUTCOME_SERVED:
            self.prompt_tokens += tokens[0]
            self.completion_tokens += tokens[1]

    def add_role(self, *, outcome: str, timing: CallClock | None, timings: ReplyTimings | None,
                 http: bool) -> None:
        role = role_key(timing.role if timing is not None else None)
        if role not in self.roles and len(self.roles) >= _MAX_ROLES:
            role = _OTHER_WORKER
        self.roles.setdefault(role, _RoleBucket()).add(outcome=outcome, timing=timing, timings=timings, http=http)

    def count(self, classes: frozenset[str]) -> int:
        return sum(n for name, n in self.classes.items() if name in classes)

    def summary(self, node: str, role_slots: dict[str, int] | None = None) -> str:
        slots = {role_key(k): v for k, v in (role_slots or {}).items()}
        classes = "|".join(f"{k}:{v}" for k, v in sorted(self.classes.items())) or "none"
        labels = "|".join(self.labels) or "none"
        return (
            f"node={node} calls={self.calls} served={self.classes.get(OUTCOME_SERVED, 0)} "
            f"upstream_failed={self.count(UPSTREAM_FAILURE_CLASSES)} "
            f"refused={self.count(REFUSAL_CLASSES)} request_invalid={self.count(REQUEST_INVALID_CLASSES)} "
            f"prompt_tokens={self.prompt_tokens} completion_tokens={self.completion_tokens} "
            f"workers={labels} classes={classes} "
            f"worker_attempted={_counts(self.worker_attempted)} "
            f"worker_failed={_counts(self.worker_failed)} "
            f"roles={''.join(b.summary(r, slots.get(r)) for r, b in sorted(self.roles.items())) or 'none'}"
        )


class InferenceWindowRecorder:
    """Counts calls into the current window. ``record`` is called on the event loop
    after each reply is built; ``drain`` swaps the window out atomically."""

    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, _NodeBucket] = {}
        self._window_start = self._clock()

    def record(self, result: Any, *, served_by: str | None, timing: CallClock | None = None) -> str:
        """Bus path: classify the reply dict, read tokens and decode speed from its ``raw``."""
        outcome = classify_outcome(result)
        raw = result.get("raw") if isinstance(result, dict) and isinstance(result.get("raw"), dict) else {}
        self.record_outcome(
            outcome,
            served_by=served_by,
            timing=timing,
            tokens=_usage_tokens(result),
            timings=timings_from(raw),
        )
        return outcome

    def record_outcome(
        self,
        outcome: str,
        *,
        served_by: str | None,
        timing: CallClock | None = None,
        tokens: tuple[int, int] = (0, 0),
        timings: ReplyTimings | None = None,
        http: bool = False,
    ) -> str:
        """An already-classified call (the HTTP passthroughs classify by status code and pass
        ``http=True``: counted in the per-role clocks only, never in the node counts that feed
        inference_failure_pressure)."""
        if timing is not None:
            timing.close()
        node = node_hint(served_by)
        pool_node = str(timing.pool_node or "").strip().lower() if timing is not None else ""
        with self._lock:
            if timing is not None and timing.role is None and pool_node and pool_node != node:
                # Never granted: no worker, so the node counts stay where they always were
                # (unrouted -> unattributed), and the wait goes to the pool's host node's
                # "ungranted" role -- the line is that pool's line.
                if not http:
                    self._buckets.setdefault(node, _NodeBucket()).add(
                        outcome=outcome, served_by=served_by, tokens=tokens, timing=timing,
                        timings=timings, http=http, roles=False)
                self._buckets.setdefault(pool_node, _NodeBucket()).add_role(
                    outcome=outcome, timing=timing, timings=timings, http=http)
            else:
                self._buckets.setdefault(node, _NodeBucket()).add(
                    outcome=outcome, served_by=served_by, tokens=tokens, timing=timing,
                    timings=timings, http=http)
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
    role_slots: dict[str, int] | None = None,
) -> list[GrammarEventV1]:
    """``role_slots``: the pool's slot count per role at window end (None -> omitted)."""
    gw = (gateway_node or "gateway").strip().lower().replace(":", "_") or "gateway"
    trace_id = f"{LLM_INFERENCE_TRACE_PREFIX}{gw}:{_window_id(window_start)}"
    emitted_at = datetime.fromtimestamp(window_end, tz=timezone.utc)
    dims = ["inference", "llm"]
    provenance = GrammarProvenanceV1(
        source_service=SOURCE_SERVICE,
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
        _event(i, ROLE_NODE_WINDOW, bucket.summary(node, role_slots), node)
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
    slots_provider: Callable[[], Any] | None = None,
) -> None:
    """Flush one window every ``window_sec``. Publishing failures are logged and the
    window is dropped -- telemetry must never back up into the serving path.

    ``slots_provider`` (async, -> {role: slots}) is read once per window, off the serving
    path; a failure only omits ``slots`` from that window."""
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
        role_slots: dict[str, int] | None = None
        if slots_provider is not None and buckets:
            try:
                role_slots = await slots_provider()
            except Exception:  # noqa: BLE001
                logger.warning("llm_gateway_grammar_slots_unavailable", exc_info=True)
        events = build_window_events(
            gateway_node=gateway_node, window_start=start, window_end=end, buckets=buckets,
            role_slots=role_slots,
        )
        sent = 0
        try:
            for event in events:
                await publish_grammar_event(bus, event, source_name=SOURCE_SERVICE)
                sent += 1
        except Exception:  # noqa: BLE001
            logger.warning(
                "llm_gateway_grammar_publish_failed window_start=%s dropped=%d of %d",
                start, len(events) - sent, len(events), exc_info=True,
            )
